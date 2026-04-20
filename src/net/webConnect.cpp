/**
 * @file webConnect.cpp
 * @author shawn
 * @date 2026/3/31
 * @brief Remote PvP socket transport implementation.
 */
#include "../../include/gomoku/net/webConnect.h"
#include "../../include/gomoku/net/webconnect_protocol.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace gomoku {
namespace {

#ifdef _WIN32
using SocketHandle = SOCKET;
using SocketLength = int;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
using SocketLength = socklen_t;
constexpr SocketHandle kInvalidSocket = -1;
#endif

enum class WaitResult {
    Ready,
    Timeout,
    Error
};

[[nodiscard]] std::string trimCopy(std::string text) {
    text.erase(text.begin(), std::ranges::find_if(text, [](const unsigned char ch) {
        return !std::isspace(ch);
    }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](const unsigned char ch) {
        return !std::isspace(ch);
    }).base(), text.end());
    return text;
}

[[nodiscard]] bool parseInt(const std::string_view token, int& value) {
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    return ec == std::errc{} && ptr == token.data() + token.size();
}

[[nodiscard]] std::string normalizeToken(std::string token) {
    std::ranges::transform(token, token.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return token;
}

// [[nodiscard]] std::string stoneToken(const Stone stone) {
//     switch (stone) {
//         case Stone::BLACK:
//             return "BLACK";
//         case Stone::WHITE:
//             return "WHITE";
//         case Stone::EMPTY:
//         default:
//             return "EMPTY";
//     }
// }

[[nodiscard]] std::optional<Stone> stoneFromToken(std::string token) {
    token = normalizeToken(std::move(token));
    if (token == "BLACK") return Stone::BLACK;
    if (token == "WHITE") return Stone::WHITE;
    if (token == "EMPTY") return Stone::EMPTY;
    return std::nullopt;
}

[[nodiscard]] std::string socketErrorString() {
#ifdef _WIN32
    const int code = WSAGetLastError();
    return std::system_category().message(code) + " (WSA " + std::to_string(code) + ")";
#else
    return std::strerror(errno);
#endif
}

bool ensureSocketRuntime(std::string& error) {
#ifdef _WIN32
    static const int kStartupResult = [] {
        WSADATA data{};
        return WSAStartup(MAKEWORD(2, 2), &data);
    }();
    if (kStartupResult != 0) {
        error = "WSAStartup failed with code " + std::to_string(kStartupResult);
        return false;
    }
#else
    (void)error;
#endif
    return true;
}

void closeSocket(SocketHandle& socket) {
    if (socket == kInvalidSocket) {
        return;
    }
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
    socket = kInvalidSocket;
}

class SocketOwner {
public:
    SocketOwner() = default;
    explicit SocketOwner(const SocketHandle socket)
        : socket_(socket) {}

    ~SocketOwner() {
        closeSocket(socket_);
    }

    SocketOwner(const SocketOwner&) = delete;
    SocketOwner& operator=(const SocketOwner&) = delete;

    SocketOwner(SocketOwner&& other) noexcept
        : socket_(std::exchange(other.socket_, kInvalidSocket)) {}

    SocketOwner& operator=(SocketOwner&& other) noexcept {
        if (this != &other) {
            closeSocket(socket_);
            socket_ = std::exchange(other.socket_, kInvalidSocket);
        }
        return *this;
    }

    [[nodiscard]] bool valid() const noexcept {
        return socket_ != kInvalidSocket;
    }

    [[nodiscard]] SocketHandle native() const noexcept {
        return socket_;
    }

    void reset(const SocketHandle socket = kInvalidSocket) {
        closeSocket(socket_);
        socket_ = socket;
    }

private:
    SocketHandle socket_ = kInvalidSocket;
};

WaitResult waitForReadable(const SocketHandle socket,
                           const std::optional<std::chrono::milliseconds> timeout,
                           std::string& error) {
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(socket, &read_set);

    timeval tv{};
    const timeval* tv_ptr = nullptr;
    if (timeout.has_value()) {
        const auto safe_timeout = std::max(timeout->count(), std::chrono::milliseconds::rep{0});
        tv.tv_sec = static_cast<long>(safe_timeout / 1000);
        tv.tv_usec = static_cast<long>((safe_timeout % 1000) * 1000);
        tv_ptr = &tv;
    }

#ifdef _WIN32
    const int ready = select(0, &read_set, nullptr, nullptr, tv_ptr);
#else
    const int ready = select(socket + 1, &read_set, nullptr, nullptr, tv_ptr);
#endif
    if (ready > 0) {
        return WaitResult::Ready;
    }
    if (ready == 0) {
        return WaitResult::Timeout;
    }

    error = "select() failed while waiting for socket readability: " + socketErrorString();
    return WaitResult::Error;
}

bool setReuseAddress(const SocketHandle socket, std::string& error) {
    constexpr int enabled = 1;
#ifdef _WIN32
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&enabled),
                   static_cast<int>(sizeof(enabled))) != 0) {
#else
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0) {
#endif
        error = "setsockopt(SO_REUSEADDR) failed: " + socketErrorString();
        return false;
    }
    return true;
}

bool sendAll(const SocketHandle socket, const std::string_view payload, std::string& error) {
    std::size_t sent_total = 0;
    while (sent_total < payload.size()) {
#ifdef _WIN32
        const int sent = send(socket,
                              payload.data() + static_cast<std::ptrdiff_t>(sent_total),
                              static_cast<int>(payload.size() - sent_total),
                              0);
#else
        const auto sent = send(socket,
                               payload.data() + static_cast<std::ptrdiff_t>(sent_total),
                               payload.size() - sent_total,
                               0);
#endif
        if (sent <= 0) {
            error = "send() failed: " + socketErrorString();
            return false;
        }
        sent_total += static_cast<std::size_t>(sent);
    }
    return true;
}

[[nodiscard]] std::string endpointToString(const sockaddr* address, const SocketLength length) {
    std::array<char, NI_MAXHOST> host{};
    std::array<char, NI_MAXSERV> service{};
    if (getnameinfo(address,
                    length,
                    host.data(),
                    static_cast<SocketLength>(host.size()),
                    service.data(),
                    static_cast<SocketLength>(service.size()),
                    NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
        return {};
    }
    return std::string(host.data()) + ":" + service.data();
}

bool queryEndpoint(const SocketHandle socket, const bool local, std::string& out) {
    sockaddr_storage address{};
    SocketLength length = sizeof(address);
    const int rc = local
        ? getsockname(socket, reinterpret_cast<sockaddr*>(&address), &length)
        : getpeername(socket, reinterpret_cast<sockaddr*>(&address), &length);
    if (rc != 0) {
        out.clear();
        return false;
    }

    out = endpointToString(reinterpret_cast<const sockaddr*>(&address), length);
    return !out.empty();
}

[[nodiscard]] std::string buildSnapshotLine(const GameSession& session) {
    std::ostringstream stream;
    const auto& moves = session.move_history();
    stream << "SNAPSHOT " << moves.size() << ' ';
    if (moves.empty()) {
        stream << '-';
        return stream.str();
    }

    bool first = true;
    for (const auto& [x, y] : moves) {
        if (!first) {
            stream << ';';
        }
        stream << x << ',' << y;
        first = false;
    }
    return stream.str();
}

bool parseSnapshotMoves(const std::string& encoded_moves,
                        std::vector<std::pair<int, int>>& parsed_moves,
                        std::string& error) {
    const std::string payload = trimCopy(encoded_moves);
    parsed_moves.clear();
    if (payload.empty() || payload == "-") {
        return true;
    }

    std::size_t begin = 0;
    while (begin < payload.size()) {
        const std::size_t end = payload.find(';', begin);
        const std::string_view move = end == std::string::npos
            ? std::string_view(payload).substr(begin)
            : std::string_view(payload).substr(begin, end - begin);
        const std::size_t comma = move.find(',');
        if (comma == std::string_view::npos) {
            error = "Invalid snapshot move token: " + std::string(move);
            return false;
        }

        int x = 0;
        int y = 0;
        if (!parseInt(move.substr(0, comma), x) || !parseInt(move.substr(comma + 1), y)) {
            error = "Invalid snapshot coordinates: " + std::string(move);
            return false;
        }
        parsed_moves.emplace_back(x, y);

        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }

    return true;
}

bool applySnapshot(GameSession& session,
                   const std::size_t declared_move_count,
                   const std::string& encoded_moves,
                   std::string& error) {
    std::vector<std::pair<int, int>> parsed_moves;
    if (!parseSnapshotMoves(encoded_moves, parsed_moves, error)) {
        return false;
    }

    if (declared_move_count != parsed_moves.size()) {
        error = "Snapshot move count mismatch. Declared=" + std::to_string(declared_move_count) +
                 ", payload=" + std::to_string(parsed_moves.size());
        return false;
    }

    GameSession staged(session.board().getSize(), session.saves_dir());
    staged.start(SessionMode::PVP);
    for (const auto& [x, y] : parsed_moves) {
        if (!staged.human_move(x, y, false)) {
            error = "Snapshot replay rejected move (" + std::to_string(x) + "," + std::to_string(y) + ")";
            return false;
        }
    }

    session = std::move(staged);
    return true;
}

} // namespace

namespace net {

bool applyRemoteMove(GameSession& session,
                     const Stone remote_stone,
                     const int x,
                     const int y,
                     std::string& error) {
    if (session.board().getCurrentPlayer() != remote_stone) {
        error = "Rejected remote MOVE at (" + std::to_string(x) + "," + std::to_string(y) +
                "): it is not the remote player's turn.";
        return false;
    }
    if (!session.human_move(x, y)) {
        error = "Remote move rejected at (" + std::to_string(x) + "," + std::to_string(y) + ")";
        return false;
    }
    return true;
}

} // namespace net

struct webConnect::Impl {
    SocketOwner listener;
    SocketOwner peer;
    bool hosting = false;
    bool connected = false;
    bool handshake_complete = false;
    bool hello_received = false;
    Stone local_stone = Stone::BLACK;
    Stone remote_stone = Stone::WHITE;
    std::string inbound_buffer;
    std::string local_endpoint;
    std::string remote_endpoint;
    std::string last_error;

    bool sendLine(const std::string& line) {
        if (!peer.valid()) {
            last_error = "Peer socket is not connected.";
            return false;
        }
        return sendAll(peer.native(), line + "\n", last_error);
    }

    bool handleLine(GameSession& session, const std::string& raw_line) {
        std::istringstream stream(raw_line);
        std::string command;
        if (!(stream >> command)) {
            return false;
        }

        command = normalizeToken(std::move(command));

        if (command == "HELLO") {
            int version = 0;
            int board_size = 0;
            std::string host_color_token;
            if (!(stream >> version >> board_size >> host_color_token)) {
                last_error = "Malformed HELLO packet: " + raw_line;
                return false;
            }
            if (version != 1) {
                last_error = "Unsupported protocol version: " + std::to_string(version);
                return false;
            }
            if (board_size != session.board().getSize()) {
                last_error = "Board size mismatch. Local=" + std::to_string(session.board().getSize()) +
                             ", remote=" + std::to_string(board_size);
                return false;
            }

            const auto host_color = stoneFromToken(std::move(host_color_token));
            if (!host_color || *host_color == Stone::EMPTY) {
                last_error = "HELLO packet specified an invalid host color.";
                return false;
            }

            session.start(SessionMode::PVP);
            remote_stone = *host_color;
            local_stone = (*host_color == Stone::BLACK) ? Stone::WHITE : Stone::BLACK;
            hello_received = true;
            handshake_complete = false;
            last_error.clear();
            return true;
        }

        if (command == "READY") {
            if (!hosting && !hello_received) {
                last_error = "Received READY before HELLO.";
                return false;
            }
            handshake_complete = true;
            last_error.clear();
            return true;
        }

        if (command == "MOVE") {
            if (!handshake_complete) {
                last_error = "Received MOVE before the handshake completed.";
                return false;
            }
            int x = 0;
            int y = 0;
            if (!(stream >> x >> y)) {
                last_error = "Malformed MOVE packet: " + raw_line;
                return false;
            }
            if (!net::applyRemoteMove(session, remote_stone, x, y, last_error)) {
                return false;
            }
            last_error.clear();
            return true;
        }

        if (command == "UNDO") {
            if (!session.undo()) {
                last_error = "Remote requested undo, but there are no moves to undo.";
                return false;
            }
            last_error.clear();
            return true;
        }

        if (command == "RESET") {
            session.start(SessionMode::PVP);
            last_error.clear();
            return true;
        }

        if (command == "SNAPSHOT") {
            std::size_t move_count = 0;
            if (!(stream >> move_count)) {
                last_error = "Malformed SNAPSHOT packet: " + raw_line;
                return false;
            }

            std::string payload;
            std::getline(stream, payload);
            payload = trimCopy(std::move(payload));
            if (!applySnapshot(session, move_count, payload, last_error)) {
                return false;
            }
            if (!hosting && !hello_received) {
                last_error = "Received SNAPSHOT before HELLO.";
                return false;
            }
            if (!hosting) {
                handshake_complete = true;
            }
            last_error.clear();
            return true;
        }

        last_error = "Unsupported packet: " + raw_line;
        return false;
    }
};

webConnect::webConnect(GameSession& session)
    : session_(&session),
      impl_(std::make_unique<Impl>()) {}

webConnect::~webConnect() = default;
webConnect::webConnect(webConnect&&) noexcept = default;
webConnect& webConnect::operator=(webConnect&&) noexcept = default;

bool webConnect::openHost(const std::uint16_t port, const std::string& bind_address) const {
    disconnect();
    if (!ensureSocketRuntime(impl_->last_error)) {
        return false;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* results = nullptr;
    const std::string service = std::to_string(port);
    const char* host = (bind_address.empty() || bind_address == "0.0.0.0") ? nullptr : bind_address.c_str();
    if (const int lookup_rc = getaddrinfo(host, service.c_str(), &hints, &results); lookup_rc != 0) {
        impl_->last_error = "getaddrinfo() failed for host bind: " + std::string(gai_strerror(lookup_rc));
        return false;
    }

    SocketOwner listener;
    for (const addrinfo* current = results; current != nullptr; current = current->ai_next) {
        SocketOwner candidate(socket(current->ai_family, current->ai_socktype, current->ai_protocol));
        if (!candidate.valid()) {
            continue;
        }
        if (!setReuseAddress(candidate.native(), impl_->last_error)) {
            continue;
        }
        if (bind(candidate.native(), current->ai_addr, static_cast<SocketLength>(current->ai_addrlen)) != 0) {
            impl_->last_error = "bind() failed: " + socketErrorString();
            continue;
        }
        if (listen(candidate.native(), 1) != 0) {
            impl_->last_error = "listen() failed: " + socketErrorString();
            continue;
        }

        listener = std::move(candidate);
        break;
    }
    freeaddrinfo(results);

    if (!listener.valid()) {
        if (impl_->last_error.empty()) {
            impl_->last_error = "Unable to open a listening socket.";
        }
        return false;
    }

    session_->start(SessionMode::PVP);
    impl_->listener = std::move(listener);
    impl_->hosting = true;
    impl_->connected = false;
    impl_->handshake_complete = false;
    impl_->hello_received = false;
    impl_->local_stone = Stone::BLACK;
    impl_->remote_stone = Stone::WHITE;
    impl_->inbound_buffer.clear();
    queryEndpoint(impl_->listener.native(), true, impl_->local_endpoint);
    impl_->remote_endpoint.clear();
    impl_->last_error.clear();
    return true;
}

bool webConnect::waitForPeer(const std::optional<std::chrono::milliseconds> timeout) const {
    if (!impl_->listener.valid()) {
        impl_->last_error = "Host socket is not open.";
        return false;
    }

    if (const auto wait_result = waitForReadable(impl_->listener.native(), timeout, impl_->last_error); wait_result != WaitResult::Ready) {
        return false;
    }

    sockaddr_storage address{};
    SocketLength length = sizeof(address);
    SocketOwner accepted(accept(impl_->listener.native(),
                                reinterpret_cast<sockaddr*>(&address),
                                &length));
    if (!accepted.valid()) {
        impl_->last_error = "accept() failed: " + socketErrorString();
        return false;
    }

    impl_->peer = std::move(accepted);
    impl_->connected = true;
    impl_->remote_endpoint = endpointToString(reinterpret_cast<const sockaddr*>(&address), length);
    queryEndpoint(impl_->peer.native(), true, impl_->local_endpoint);

    if (const std::string hello = "HELLO 1 " + std::to_string(session_->board().getSize()) + ' ' + "BLACK"; !impl_->sendLine(hello) || !syncSnapshot()) {
        disconnect();
        return false;
    }

    impl_->handshake_complete = true;
    impl_->last_error.clear();
    return true;
}

bool webConnect::connectTo(const std::string& host, const std::uint16_t port) const {
    disconnect();
    if (!ensureSocketRuntime(impl_->last_error)) {
        return false;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    const std::string service = std::to_string(port);
    if (const int lookup_rc = getaddrinfo(host.c_str(), service.c_str(), &hints, &results); lookup_rc != 0) {
        impl_->last_error = "getaddrinfo() failed for remote host: " + std::string(gai_strerror(lookup_rc));
        return false;
    }

    SocketOwner peer;
    for (const addrinfo* current = results; current != nullptr; current = current->ai_next) {
        SocketOwner candidate(socket(current->ai_family, current->ai_socktype, current->ai_protocol));
        if (!candidate.valid()) {
            continue;
        }
        if (connect(candidate.native(), current->ai_addr, static_cast<SocketLength>(current->ai_addrlen)) == 0) {
            peer = std::move(candidate);
            break;
        }
        impl_->last_error = "connect() failed: " + socketErrorString();
    }
    freeaddrinfo(results);

    if (!peer.valid()) {
        if (impl_->last_error.empty()) {
            impl_->last_error = "Unable to connect to remote host.";
        }
        return false;
    }

    impl_->peer = std::move(peer);
    impl_->hosting = false;
    impl_->connected = true;
    impl_->handshake_complete = false;
    impl_->hello_received = false;
    impl_->local_stone = Stone::WHITE;
    impl_->remote_stone = Stone::BLACK;
    impl_->inbound_buffer.clear();
    queryEndpoint(impl_->peer.native(), true, impl_->local_endpoint);
    queryEndpoint(impl_->peer.native(), false, impl_->remote_endpoint);
    impl_->last_error.clear();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!impl_->handshake_complete && impl_->connected) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            break;
        }

        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const bool handled = pump(remaining);
        if (!handled && !impl_->connected) {
            break;
        }
        if (!handled && !impl_->last_error.empty()) {
            break;
        }
    }

    if (!impl_->handshake_complete) {
        if (impl_->last_error.empty()) {
            impl_->last_error = "Timed out waiting for remote handshake completion.";
        }
        disconnect();
        return false;
    }

    if (!impl_->sendLine("READY")) {
        disconnect();
        return false;
    }

    impl_->last_error.clear();
    return true;
}

bool webConnect::sendLocalMove(const int x, const int y) const {
    if (!impl_->connected) {
        impl_->last_error = "Cannot send move without an active peer.";
        return false;
    }
    if (!isLocalTurn()) {
        impl_->last_error = "It is not the local player's turn.";
        return false;
    }
    if (!session_->human_move(x, y)) {
        impl_->last_error = "Local move was rejected by the game session.";
        return false;
    }

    if (!impl_->sendLine("MOVE " + std::to_string(x) + ' ' + std::to_string(y))) {
        session_->undo();
        impl_->last_error = "Failed to send move; local move was rolled back. " + impl_->last_error;
        return false;
    }

    impl_->last_error.clear();
    return true;
}

bool webConnect::requestUndo() const {
    if (!impl_->connected) {
        impl_->last_error = "Cannot request undo without an active peer.";
        return false;
    }
    if (session_->move_history().empty()) {
        impl_->last_error = "There are no moves to undo.";
        return false;
    }
    if (!impl_->sendLine("UNDO")) {
        return false;
    }
    if (!session_->undo()) {
        impl_->last_error = "Local undo failed after sending UNDO.";
        return false;
    }
    impl_->last_error.clear();
    return true;
}

bool webConnect::requestReset() const {
    if (!impl_->connected) {
        impl_->last_error = "Cannot request reset without an active peer.";
        return false;
    }
    if (!impl_->sendLine("RESET")) {
        return false;
    }
    session_->start(SessionMode::PVP);
    impl_->last_error.clear();
    return true;
}

bool webConnect::syncSnapshot() const {
    if (!impl_->connected) {
        impl_->last_error = "Cannot sync snapshot without an active peer.";
        return false;
    }
    const bool ok = impl_->sendLine(buildSnapshotLine(*session_));
    if (ok) {
        impl_->last_error.clear();
    }
    return ok;
}

bool webConnect::pump(const std::chrono::milliseconds timeout) const {
    if (!impl_->connected || !impl_->peer.valid()) {
        impl_->last_error = "Peer socket is not connected.";
        return false;
    }

    if (const auto wait_result = waitForReadable(impl_->peer.native(), timeout, impl_->last_error); wait_result != WaitResult::Ready) {
        return false;
    }

    std::array<char, 2048> buffer{};
#ifdef _WIN32
    const int received = recv(impl_->peer.native(), buffer.data(), static_cast<int>(buffer.size()), 0);
#else
    const auto received = recv(impl_->peer.native(), buffer.data(), buffer.size(), 0);
#endif
    if (received == 0) {
        impl_->last_error = "Remote peer closed the connection.";
        disconnect();
        return false;
    }
    if (received < 0) {
        impl_->last_error = "recv() failed: " + socketErrorString();
        disconnect();
        return false;
    }

    impl_->inbound_buffer.append(buffer.data(), static_cast<std::size_t>(received));

    bool handled_packet = false;
    while (true) {
        const std::size_t newline = impl_->inbound_buffer.find('\n');
        if (newline == std::string::npos) {
            break;
        }

        std::string line = impl_->inbound_buffer.substr(0, newline);
        impl_->inbound_buffer.erase(0, newline + 1);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        line = trimCopy(std::move(line));
        if (line.empty()) {
            continue;
        }

        if (!impl_->handleLine(*session_, line)) {
            return false;
        }
        handled_packet = true;
    }

    return handled_packet;
}

void webConnect::disconnect() const {
    impl_->listener.reset();
    impl_->peer.reset();
    impl_->hosting = false;
    impl_->connected = false;
    impl_->handshake_complete = false;
    impl_->hello_received = false;
    impl_->inbound_buffer.clear();
    impl_->local_endpoint.clear();
    impl_->remote_endpoint.clear();
}

bool webConnect::isHosting() const noexcept {
    return impl_->hosting;
}

bool webConnect::isConnected() const noexcept {
    return impl_->connected;
}

bool webConnect::isLocalTurn() const {
    return session_ != nullptr && session_->board().getCurrentPlayer() == impl_->local_stone;
}

Stone webConnect::localStone() const noexcept {
    return impl_->local_stone;
}

Stone webConnect::remoteStone() const noexcept {
    return impl_->remote_stone;
}

std::string webConnect::localEndpoint() const {
    return impl_->local_endpoint;
}

std::string webConnect::remoteEndpoint() const {
    return impl_->remote_endpoint;
}

const std::string& webConnect::lastError() const noexcept {
    return impl_->last_error;
}

} // namespace gomoku
