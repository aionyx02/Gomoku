#include "gomoku/core/game_session.h"
#include "gomoku/net/webConnect.h"
#include "gomoku/net/webconnect_protocol.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

using gomoku::GameSession;
using gomoku::SessionMode;
using gomoku::Stone;
using gomoku::webConnect;

#ifdef _WIN32
using TestSocketHandle = SOCKET;
using TestSocketLength = int;
constexpr TestSocketHandle kInvalidTestSocket = INVALID_SOCKET;
#else
using TestSocketHandle = int;
using TestSocketLength = socklen_t;
constexpr TestSocketHandle kInvalidTestSocket = -1;
#endif

bool expect(const bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

void closeTestSocket(const TestSocketHandle socket) {
#ifdef _WIN32
    if (socket != INVALID_SOCKET) {
        closesocket(socket);
    }
#else
    if (socket >= 0) {
        close(socket);
    }
#endif
}

[[nodiscard]] std::string testSocketErrorString() {
#ifdef _WIN32
    const int code = WSAGetLastError();
    return std::system_category().message(code) + " (WSA " + std::to_string(code) + ")";
#else
    return std::strerror(errno);
#endif
}

bool ensureTestSocketRuntime(std::string& error) {
#ifdef _WIN32
    static const int startup_result = [] {
        WSADATA data{};
        return WSAStartup(MAKEWORD(2, 2), &data);
    }();
    if (startup_result != 0) {
        error = "WSAStartup failed with code " + std::to_string(startup_result);
        return false;
    }
#else
    (void)error;
#endif
    return true;
}

class ScopedTestSocket {
public:
    ScopedTestSocket() = default;

    explicit ScopedTestSocket(const TestSocketHandle socket)
        : socket_(socket) {}

    ~ScopedTestSocket() {
        reset();
    }

    ScopedTestSocket(const ScopedTestSocket&) = delete;
    ScopedTestSocket& operator=(const ScopedTestSocket&) = delete;

    ScopedTestSocket(ScopedTestSocket&& other) noexcept
        : socket_(other.release()) {}

    ScopedTestSocket& operator=(ScopedTestSocket&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    [[nodiscard]] bool valid() const {
        return socket_ != kInvalidTestSocket;
    }

    [[nodiscard]] TestSocketHandle get() const {
        return socket_;
    }

    TestSocketHandle release() {
        const TestSocketHandle current = socket_;
        socket_ = kInvalidTestSocket;
        return current;
    }

    void reset(const TestSocketHandle next = kInvalidTestSocket) {
        closeTestSocket(socket_);
        socket_ = next;
    }

private:
    TestSocketHandle socket_ = kInvalidTestSocket;
};

bool sendAllForTest(const TestSocketHandle socket, const std::string& payload, std::string& error) {
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
            error = "send() failed: " + testSocketErrorString();
            return false;
        }
        sent_total += static_cast<std::size_t>(sent);
    }
    return true;
}

bool openLoopbackListener(ScopedTestSocket& listener, std::uint16_t& port, std::string& error) {
    if (!ensureTestSocketRuntime(error)) {
        return false;
    }

    ScopedTestSocket socket_owner(socket(AF_INET, SOCK_STREAM, 0));
    if (!socket_owner.valid()) {
        error = "socket() failed: " + testSocketErrorString();
        return false;
    }

    constexpr int enabled = 1;
#ifdef _WIN32
    if (setsockopt(socket_owner.get(),
                   SOL_SOCKET,
                   SO_REUSEADDR,
                   reinterpret_cast<const char*>(&enabled),
                   static_cast<int>(sizeof(enabled))) != 0) {
#else
    if (setsockopt(socket_owner.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0) {
#endif
        error = "setsockopt(SO_REUSEADDR) failed: " + testSocketErrorString();
        return false;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(socket_owner.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        error = "bind() failed: " + testSocketErrorString();
        return false;
    }
    if (listen(socket_owner.get(), 1) != 0) {
        error = "listen() failed: " + testSocketErrorString();
        return false;
    }

    sockaddr_in bound{};
    TestSocketLength bound_length = sizeof(bound);
    if (getsockname(socket_owner.get(), reinterpret_cast<sockaddr*>(&bound), &bound_length) != 0) {
        error = "getsockname() failed: " + testSocketErrorString();
        return false;
    }

    port = ntohs(bound.sin_port);
    listener = std::move(socket_owner);
    return true;
}

bool test_remote_move_rejects_out_of_turn_packets() {
    GameSession session(15);
    session.start(SessionMode::PVP);

    std::string error;
    if (!expect(!gomoku::net::applyRemoteMove(session, Stone::WHITE, 7, 7, error),
                "remote WHITE move should be rejected while BLACK is to play")) {
        return false;
    }
    if (!expect(error.find("not the remote player's turn") != std::string::npos,
                "out-of-turn remote move should report turn validation failure")) {
        return false;
    }
    if (!expect(session.move_history().empty(), "rejected remote move must not mutate the session")) {
        return false;
    }

    if (!expect(session.human_move(0, 0, false), "local opening move should succeed")) {
        return false;
    }

    error.clear();
    if (!expect(gomoku::net::applyRemoteMove(session, Stone::WHITE, 1, 1, error),
                "remote WHITE move should be accepted on WHITE's turn")) {
        return false;
    }
    if (!expect(session.move_history().size() == 2, "accepted remote move should be recorded")) {
        return false;
    }
    return expect(session.board().getCurrentPlayer() == Stone::BLACK,
                  "turn should advance back to BLACK after a valid remote WHITE move");
}

bool test_deserialize_rejects_invalid_board_size_without_mutating_state() {
    namespace fs = std::filesystem;

    const fs::path temp_dir = fs::temp_directory_path() / "gomoku_regression_tests";
    fs::create_directories(temp_dir);
    const fs::path save_path = temp_dir / "invalid_size.gomoku";

    {
        std::ofstream save(save_path);
        save << "mode PVP\n";
        save << "size 0\n";
    }

    GameSession session(15);
    session.start(SessionMode::PVP);
    if (!expect(session.human_move(3, 3, false), "baseline move should succeed")) {
        return false;
    }

    if (!expect(!session.deserialize(save_path.string()), "deserialize should reject board size 0")) {
        return false;
    }
    if (!expect(session.last_persistence_error().find("supported range") != std::string::npos,
                "invalid size should surface a descriptive error")) {
        return false;
    }
    if (!expect(session.board().getSize() == 15, "failed deserialize must preserve the existing board size")) {
        return false;
    }
    return expect(session.move_history().size() == 1,
                  "failed deserialize must preserve the existing move history");
}

bool test_deserialize_rejects_partial_replay_without_mutating_state() {
    namespace fs = std::filesystem;

    const fs::path temp_dir = fs::temp_directory_path() / "gomoku_regression_tests";
    fs::create_directories(temp_dir);
    const fs::path save_path = temp_dir / "partial_replay.gomoku";

    {
        std::ofstream save(save_path);
        save << "mode PVP\n";
        save << "size 15\n";
        save << "7 7\n";
        save << "7 7\n";
    }

    GameSession session(15);
    session.start(SessionMode::PVP);
    if (!expect(session.human_move(2, 2, false), "baseline move should succeed")) {
        return false;
    }

    if (!expect(!session.deserialize(save_path.string()), "deserialize should reject an invalid replay")) {
        return false;
    }
    if (!expect(session.last_persistence_error().find("Illegal move") != std::string::npos,
                "invalid replay should report the illegal move")) {
        return false;
    }
    if (!expect(session.move_history().size() == 1, "failed deserialize must not partially replace the move history")) {
        return false;
    }
    return expect(session.last_move() == std::optional<std::pair<int, int>>{{2, 2}},
                  "failed deserialize must preserve the last move");
}

bool test_snapshot_packet_rejects_partial_replay_without_mutating_state() {
    GameSession session(15);
    session.start(SessionMode::PVP);
    if (!expect(session.human_move(4, 4, false), "baseline move should succeed")) {
        return false;
    }

    std::string error;
    if (!expect(!gomoku::net::applySnapshotPacket(session, 2, "7,7;7,7", error),
                "snapshot replay with an illegal second move should be rejected")) {
        return false;
    }
    if (!expect(error.find("Snapshot replay rejected move") != std::string::npos,
                "invalid snapshot replay should report illegal move details")) {
        return false;
    }
    if (!expect(session.move_history().size() == 1,
                "failed snapshot replay must not partially replace move history")) {
        return false;
    }
    return expect(session.last_move() == std::optional<std::pair<int, int>>{{4, 4}},
                  "failed snapshot replay must preserve last move");
}

bool test_snapshot_packet_rejects_move_count_mismatch_without_mutating_state() {
    GameSession session(15);
    session.start(SessionMode::PVP);
    if (!expect(session.human_move(1, 1, false), "baseline move should succeed")) {
        return false;
    }

    std::string error;
    if (!expect(!gomoku::net::applySnapshotPacket(session, 1, "7,7;7,8", error),
                "snapshot with declared count mismatch should be rejected")) {
        return false;
    }
    if (!expect(error.find("move count mismatch") != std::string::npos,
                "count mismatch should surface a descriptive error")) {
        return false;
    }
    if (!expect(session.move_history().size() == 1,
                "count-mismatch snapshot must not overwrite existing move history")) {
        return false;
    }
    return expect(session.last_move() == std::optional<std::pair<int, int>>{{1, 1}},
                  "count-mismatch snapshot must preserve the existing last move");
}

bool test_connect_to_rejects_handshake_when_snapshot_is_invalid() {
    ScopedTestSocket listener;
    std::string setup_error;
    std::uint16_t port = 0;
    if (!openLoopbackListener(listener, port, setup_error)) {
        return expect(false, "failed to create loopback listener for handshake test: " + setup_error);
    }

    std::string server_error;
    std::thread server([listener_socket = listener.release(), &server_error] {
        ScopedTestSocket listener_owner(listener_socket);
        sockaddr_in client_address{};
        TestSocketLength client_length = sizeof(client_address);
        ScopedTestSocket accepted(accept(listener_owner.get(),
                                        reinterpret_cast<sockaddr*>(&client_address),
                                        &client_length));
        listener_owner.reset();
        if (!accepted.valid()) {
            server_error = "accept() failed: " + testSocketErrorString();
            return;
        }

        const std::string payload = "HELLO 1 15 BLACK\nSNAPSHOT 2 7,7;7,7\n";
        sendAllForTest(accepted.get(), payload, server_error);
    });

    GameSession session(15);
    webConnect connection(session);
    const bool connected = connection.connectTo("127.0.0.1", port);
    server.join();

    if (!expect(server_error.empty(), "mock server should not fail while sending handshake payload")) {
        return false;
    }
    if (!expect(!connected, "connectTo should fail when handshake snapshot is invalid")) {
        return false;
    }
    return expect(connection.lastError().find("Snapshot") != std::string::npos,
                  "failed handshake should preserve snapshot validation error");
}

bool test_serialize_reports_error_when_save_directory_is_invalid() {
    namespace fs = std::filesystem;

    const fs::path temp_dir = fs::temp_directory_path() / "gomoku_regression_tests";
    fs::create_directories(temp_dir);
    const fs::path invalid_save_root = temp_dir / "save_dir_blocker.tmp";
    {
        std::ofstream blocker(invalid_save_root);
        blocker << "not a directory";
    }

    GameSession session(15, invalid_save_root.string());
    session.start(SessionMode::PVP);
    const std::string serialized = session.serialize();
    if (!expect(serialized.empty(), "serialize should fail when save root is not a directory")) {
        return false;
    }
    return expect(session.last_persistence_error().find("Failed to create save directory") != std::string::npos,
                  "serialize failure should preserve a descriptive persistence error");
}

} // namespace

int main() {
    bool ok = true;
    ok = test_remote_move_rejects_out_of_turn_packets() && ok;
    ok = test_deserialize_rejects_invalid_board_size_without_mutating_state() && ok;
    ok = test_deserialize_rejects_partial_replay_without_mutating_state() && ok;
    ok = test_snapshot_packet_rejects_partial_replay_without_mutating_state() && ok;
    ok = test_snapshot_packet_rejects_move_count_mismatch_without_mutating_state() && ok;
    ok = test_connect_to_rejects_handshake_when_snapshot_is_invalid() && ok;
    ok = test_serialize_reports_error_when_save_directory_is_invalid() && ok;

    if (!ok) {
        return 1;
    }

    std::cout << "All regression tests passed.\n";
    return 0;
}
