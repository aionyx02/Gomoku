/**
 * @file game_session.h
 * @author shawn
 * @date 2026/3/31
 * @brief Game flow/session controller shared by UI or network adapters.
 */
#ifndef GOMOKU_GAME_SESSION_H
#define GOMOKU_GAME_SESSION_H

#include "ai_player.h"
#include "engine.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gomoku {

enum class SessionMode : uint8_t {
    PVP = 0,
    PVE
};

class GameSession {
public:
    explicit GameSession(int board_size = 15,
                         std::string source_dir = ".",
                         std::string preferred_python = "python",
                         std::string saves_dir = "");

    void start(SessionMode next_mode);
    void reset();

    bool human_move(int x, int y);
    bool ai_move();

    [[nodiscard]] const Board& board() const;
    [[nodiscard]] GameStatus status() const;
    [[nodiscard]] SessionMode mode() const;
    [[nodiscard]] const std::string& ai_status_text() const;
    [[nodiscard]] bool ai_used_fallback() const;
    [[nodiscard]] std::optional<std::pair<int, int>> last_move() const;
    [[nodiscard]] const std::vector<std::pair<int, int>>& move_history() const;

    bool undo();

    // Returns the path to the saved file, or empty string on failure.
    std::string serialize() const;
    bool deserialize(const std::string& filepath);

    [[nodiscard]] const std::string& saves_dir() const;

private:
    int board_size_ = 15;
    SessionMode mode_ = SessionMode::PVP;
    Board board_;
    ai::Player ai_player_;
    std::string ai_status_text_ = "AI: disabled in PvP";
    bool ai_used_fallback_ = false;
    std::optional<std::pair<int, int>> last_move_;
    std::vector<std::pair<int, int>> move_history_;
    std::string saves_dir_;
};

} // namespace gomoku

#endif
