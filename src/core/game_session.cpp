#include "gomoku/game_session.h"

#include <algorithm>
#include <string>
#include <utility>

namespace gomoku {

GameSession::GameSession(const int board_size,
                         std::string source_dir,
                         std::string preferred_python)
    : board_size_(std::max(1, board_size)),
      board_(board_size_),
      ai_player_(std::move(source_dir), std::move(preferred_python)) {
    reset();
}

void GameSession::start(const SessionMode next_mode) {
    mode_ = next_mode;
    reset();
}

void GameSession::reset() {
    board_ = Board(board_size_);
    last_move_.reset();
    ai_used_fallback_ = false;

    if (mode_ == SessionMode::PVE) {
        ai_status_text_ = "AI: ready (model expected at python/models/az_prompt_smoke/gomoku_model.pt)";
    } else {
        ai_status_text_ = "AI: disabled in PvP";
    }
}

bool GameSession::human_move(const int x, const int y) {
    if (board_.getStatus() != GameStatus::PLAYING) {
        return false;
    }

    if (mode_ == SessionMode::PVE && board_.getCurrentPlayer() != Stone::BLACK) {
        return false;
    }

    if (!board_.placeStone(x, y)) {
        return false;
    }

    last_move_ = std::pair{x, y};
    return true;
}

bool GameSession::ai_move() {
    if (mode_ != SessionMode::PVE) {
        ai_status_text_ = "AI: disabled in PvP";
        return false;
    }

    if (board_.getStatus() != GameStatus::PLAYING || board_.getCurrentPlayer() != Stone::WHITE) {
        return false;
    }

    const auto [move, used_fallback, diagnostic] = ai_player_.makeMove(board_);
    ai_used_fallback_ = used_fallback;

    if (!move) {
        ai_status_text_ = "AI failed: " + diagnostic;
        return false;
    }

    const auto [x, y] = *move;
    if (!board_.placeStone(x, y)) {
        ai_status_text_ = "AI failed: suggested move is invalid";
        ai_used_fallback_ = true;
        return false;
    }

    last_move_ = std::pair{x, y};
    ai_status_text_ = used_fallback
        ? "AI(fallback): move (" + std::to_string(x) + "," + std::to_string(y) + ")"
        : "AI(model): move (" + std::to_string(x) + "," + std::to_string(y) + ")";

    if (used_fallback && !diagnostic.empty()) {
        ai_status_text_ += " | reason: " + diagnostic;
    }

    return true;
}

const Board& GameSession::board() const {
    return board_;
}

GameStatus GameSession::status() const {
    return board_.getStatus();
}

SessionMode GameSession::mode() const {
    return mode_;
}

const std::string& GameSession::ai_status_text() const {
    return ai_status_text_;
}

bool GameSession::ai_used_fallback() const {
    return ai_used_fallback_;
}

std::optional<std::pair<int, int>> GameSession::last_move() const {
    return last_move_;
}

} // namespace gomoku
