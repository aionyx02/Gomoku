/**
 * @file webConnect.h
 * @author shawn
 * @date 2026/3/31
 * @brief Remote PvP transport for Gomoku.
 */
#ifndef GOMOKU_WEBCONNECT_H
#define GOMOKU_WEBCONNECT_H

#include "game_session.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace gomoku {

class webConnect {
public:
    explicit webConnect(GameSession& session);
    ~webConnect();

    webConnect(const webConnect&) = delete;
    webConnect& operator=(const webConnect&) = delete;
    webConnect(webConnect&&) noexcept;
    webConnect& operator=(webConnect&&) noexcept;

    bool openHost(std::uint16_t port, std::string bind_address = "0.0.0.0");
    bool waitForPeer(std::optional<std::chrono::milliseconds> timeout = std::nullopt);
    bool connectTo(const std::string& host, std::uint16_t port);

    bool sendLocalMove(int x, int y);
    bool requestUndo();
    bool requestReset();
    bool syncSnapshot();
    bool pump(std::chrono::milliseconds timeout = std::chrono::milliseconds{0});

    void disconnect();

    [[nodiscard]] bool isHosting() const noexcept;
    [[nodiscard]] bool isConnected() const noexcept;
    [[nodiscard]] bool isLocalTurn() const;
    [[nodiscard]] Stone localStone() const noexcept;
    [[nodiscard]] Stone remoteStone() const noexcept;
    [[nodiscard]] std::string localEndpoint() const;
    [[nodiscard]] std::string remoteEndpoint() const;
    [[nodiscard]] const std::string& lastError() const noexcept;

private:
    struct Impl;

    GameSession* session_ = nullptr;
    std::unique_ptr<Impl> impl_;
};

using WebConnect = webConnect;

} // namespace gomoku

#endif // GOMOKU_WEBCONNECT_H
