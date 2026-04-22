/**
 * @file webConnect.h
 * @author shawn
 * @date 2026/3/31
 * @brief Remote PvP transport for Gomoku.
 */
#ifndef GOMOKU_WEBCONNECT_H
#define GOMOKU_WEBCONNECT_H

#include "../core/game_session.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gomoku {

class webConnect {
public:
    explicit webConnect(GameSession& session);
    ~webConnect();

    webConnect(const webConnect&) = delete;
    webConnect& operator=(const webConnect&) = delete;
    webConnect(webConnect&&) noexcept;
    webConnect& operator=(webConnect&&) noexcept;

    bool openHost(std::uint16_t port, const std::string& bind_address = "0.0.0.0") const;
    bool waitForPeer(std::optional<std::chrono::milliseconds> timeout = std::nullopt) const;
    bool connectTo(const std::string& host, std::uint16_t port) const;

    bool sendLocalMove(int x, int y) const;
    bool requestUndo() const;
    bool requestSkipTurn() const;
    bool requestReset() const;
    bool syncConfig(const SessionRules& rules) const;
    bool syncSnapshot() const;
    bool pump(std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) const;

    void disconnect() const;

    [[nodiscard]] bool isHosting() const noexcept;
    [[nodiscard]] bool isConnected() const noexcept;
    [[nodiscard]] bool isLocalTurn() const;
    [[nodiscard]] Stone localStone() const noexcept;
    [[nodiscard]] Stone remoteStone() const noexcept;
    [[nodiscard]] std::string localEndpoint() const;
    [[nodiscard]] std::string remoteEndpoint() const;
    [[nodiscard]] std::vector<std::string> shareableEndpoints() const;
    [[nodiscard]] const std::string& lastError() const noexcept;

private:
    struct Impl;

    GameSession* session_ = nullptr;
    std::unique_ptr<Impl> impl_;
};

using WebConnect = webConnect;

} // namespace gomoku

#endif // GOMOKU_WEBCONNECT_H
