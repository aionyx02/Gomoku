#include "gomoku/game_session.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace gomoku {

GameSession::GameSession(const int board_size,
                         std::string source_dir,
                         std::string preferred_python,
                         std::string saves_dir)
    : board_size_(std::max(1, board_size)),
      board_(board_size_),
      ai_player_(std::move(source_dir), std::move(preferred_python)),
      saves_dir_(saves_dir.empty() ? "saves" : std::move(saves_dir)) {
    reset();
}

void GameSession::start(const SessionMode next_mode) {
    mode_ = next_mode;
    reset();
}

void GameSession::reset() {
    board_ = Board(board_size_);
    last_move_.reset();
    move_history_.clear();
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
    move_history_.emplace_back(x, y);
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
    move_history_.emplace_back(x, y);
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

const std::vector<std::pair<int, int>>& GameSession::move_history() const {
    return move_history_;
}

const std::string& GameSession::saves_dir() const {
    return saves_dir_;
}

std::string GameSession::serialize() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(saves_dir_, ec);
    if (ec) return {};

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm_info{};
#ifdef _WIN32
    localtime_s(&tm_info, &t);
#else
    localtime_r(&t, &tm_info);
#endif
    std::ostringstream name;
    name << std::put_time(&tm_info, "save_%Y%m%d_%H%M%S.gomoku");
    const std::string filepath = (fs::path(saves_dir_) / name.str()).string();

    std::ofstream f(filepath);
    if (!f) return {};

    f << "mode " << (mode_ == SessionMode::PVE ? "PVE" : "PVP") << "\n";
    f << "size " << board_size_ << "\n";
    for (const auto& [x, y] : move_history_) {
        f << x << " " << y << "\n";
    }
    return filepath;
}

bool GameSession::deserialize(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f) return false;

    std::string key, val;
    if (!(f >> key >> val) || key != "mode") return false;
    const SessionMode file_mode = (val == "PVE") ? SessionMode::PVE : SessionMode::PVP;

    int file_size{};
    if (!(f >> key >> file_size) || key != "size") return false;

    board_size_ = file_size;
    mode_       = file_mode;
    board_      = Board(board_size_);
    move_history_.clear();
    last_move_.reset();
    ai_used_fallback_ = false;

    int x{}, y{};
    while (f >> x >> y) {
        if (!board_.placeStone(x, y)) break;
        move_history_.emplace_back(x, y);
    }
    last_move_ = move_history_.empty()
        ? std::nullopt
        : std::optional{move_history_.back()};

    ai_status_text_ = (mode_ == SessionMode::PVE)
        ? "AI: ready (model expected at python/models/az_prompt_smoke/gomoku_model.pt)"
        : "AI: disabled in PvP";
    return true;
}

bool GameSession::undo() {
    if (move_history_.empty()) return false;

    if (mode_ == SessionMode::PVP) {
        const auto [x, y] = move_history_.back();
        move_history_.pop_back();
        board_.undoStone(x, y);
    } else {
        // PvE: undo steps until it is human (BLACK)'s turn, max 2 steps
        for (int i = 0; i < 2 && !move_history_.empty(); ++i) {
            const auto [x, y] = move_history_.back();
            move_history_.pop_back();
            board_.undoStone(x, y);
            if (board_.getCurrentPlayer() == Stone::BLACK) break;
        }
    }

    last_move_ = move_history_.empty()
        ? std::nullopt
        : std::optional{move_history_.back()};
    return true;
}

} // namespace gomoku
