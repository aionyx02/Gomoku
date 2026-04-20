#include "../../include/gomoku/core/game_session.h"
#include "../../include/gomoku/audio/voice.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace gomoku {

namespace {

constexpr int kMinSerializedBoardSize = 1;
constexpr int kMaxSerializedBoardSize = 255;

void playStoneAudioIfEnabled(const bool play_audio) {
    if (play_audio) {
        voice::placeStoneSound();
    }
}

[[nodiscard]] bool parseModeToken(const std::string& token, SessionMode& mode) {
    if (token == "PVE") {
        mode = SessionMode::PVE;
        return true;
    }
    if (token == "PVP") {
        mode = SessionMode::PVP;
        return true;
    }
    return false;
}

[[nodiscard]] std::string makeAiStatusText(const SessionMode mode) {
    return mode == SessionMode::PVE
        ? "AI: ready (C++ reward engine)"
        : "AI: disabled in PvP";
}

} // namespace

GameSession::GameSession(const int board_size,
                         std::string saves_dir)
    : board_size_(std::max(1, board_size)),
      board_(board_size_),
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

    ai_status_text_ = makeAiStatusText(mode_);
}

bool GameSession::human_move(const int x, const int y, const bool play_audio) {
    if (board_.getStatus() != GameStatus::PLAYING) {
        return false;
    }

    if (mode_ == SessionMode::PVE && board_.getCurrentPlayer() != Stone::BLACK) {
        return false;
    }

    if (!board_.placeStone(x, y)) {
        return false;
    }

    playStoneAudioIfEnabled(play_audio);
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

    const auto [move, used_fallback, diagnostic] = gomoku::ai::Player::makeMove(board_);
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

    playStoneAudioIfEnabled(true);
    last_move_ = std::pair{x, y};
    move_history_.emplace_back(x, y);
    ai_status_text_ = "AI(reward): move (" + std::to_string(x) + "," + std::to_string(y) + ")";

    if (!diagnostic.empty()) {
        ai_status_text_ += " | " + diagnostic;
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
    last_persistence_error_.clear();

    std::error_code ec;
    fs::create_directories(saves_dir_, ec);
    if (ec) {
        last_persistence_error_ = "Failed to create save directory: " + saves_dir_ + " (" + ec.message() + ")";
        return {};
    }

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) %
                        std::chrono::seconds{1};
    std::tm tm_info{};
#ifdef _WIN32
    localtime_s(&tm_info, &t);
#else
    localtime_r(&t, &tm_info);
#endif
    std::ostringstream base_name;
    base_name << "save_" << std::put_time(&tm_info, "%Y%m%d_%H%M%S")
              << '_' << std::setw(3) << std::setfill('0') << millis.count();

    fs::path filepath = fs::path(saves_dir_) / (base_name.str() + ".gomoku");
    for (int collision_index = 1;; ++collision_index) {
        if (!fs::exists(filepath, ec)) {
            break;
        }
        if (ec) {
            last_persistence_error_ = "Failed to check save file collision: " + filepath.string() + " (" + ec.message() + ")";
            return {};
        }
        std::ostringstream collision_name;
        collision_name << base_name.str() << '_' << collision_index << ".gomoku";
        filepath = fs::path(saves_dir_) / collision_name.str();
        ec.clear();
    }

    std::ofstream f(filepath.string());
    if (!f) {
        last_persistence_error_ = "Failed to open save file: " + filepath.string();
        return {};
    }

    f << "mode " << (mode_ == SessionMode::PVE ? "PVE" : "PVP") << "\n";
    f << "size " << board_size_ << "\n";
    for (const auto& [x, y] : move_history_) {
        f << x << " " << y << "\n";
    }

    f.flush();
    if (!f) {
        last_persistence_error_ = "Failed to write save file: " + filepath.string();
        return {};
    }

    return filepath.string();
}

bool GameSession::deserialize(const std::string& filepath) {
    last_persistence_error_.clear();

    std::ifstream f(filepath);
    if (!f) {
        last_persistence_error_ = "Failed to open save file: " + filepath;
        return false;
    }

    std::string line;
    if (!std::getline(f, line)) {
        last_persistence_error_ = "Save file is missing the mode header.";
        return false;
    }

    std::istringstream mode_line(line);
    std::string mode_value;
    if (std::string mode_key; !(mode_line >> mode_key >> mode_value) || mode_key != "mode") {
        last_persistence_error_ = "Invalid mode header in save file.";
        return false;
    }
    auto parsed_mode = SessionMode::PVP;
    if (!parseModeToken(mode_value, parsed_mode)) {
        last_persistence_error_ = "Unsupported mode token in save file: " + mode_value;
        return false;
    }
    std::string extra_token;
    if (mode_line >> extra_token) {
        last_persistence_error_ = "Mode header contains unexpected trailing data.";
        return false;
    }

    if (!std::getline(f, line)) {
        last_persistence_error_ = "Save file is missing the board size header.";
        return false;
    }

    std::istringstream size_line(line);
    int parsed_size = 0;
    if (std::string size_key; !(size_line >> size_key >> parsed_size) || size_key != "size") {
        last_persistence_error_ = "Invalid size header in save file.";
        return false;
    }
    if (size_line >> extra_token) {
        last_persistence_error_ = "Size header contains unexpected trailing data.";
        return false;
    }
    if (parsed_size < kMinSerializedBoardSize || parsed_size > kMaxSerializedBoardSize) {
        last_persistence_error_ = "Board size out of supported range: " + std::to_string(parsed_size);
        return false;
    }

    Board parsed_board(parsed_size);
    std::vector<std::pair<int, int>> parsed_history;

    int line_number = 2;
    while (std::getline(f, line)) {
        ++line_number;

        std::istringstream move_line(line);
        int x = 0;
        int y = 0;
        if (!(move_line >> x >> y)) {
            last_persistence_error_ = "Invalid move entry at line " + std::to_string(line_number) + ".";
            return false;
        }
        if (move_line >> extra_token) {
            last_persistence_error_ = "Move entry has unexpected trailing data at line " + std::to_string(line_number) + ".";
            return false;
        }
        if (!parsed_board.placeStone(x, y)) {
            last_persistence_error_ = "Illegal move in save file at line " + std::to_string(line_number) + ".";
            return false;
        }
        parsed_history.emplace_back(x, y);
    }

    if (!f.eof()) {
        last_persistence_error_ = "Failed while reading save file: " + filepath;
        return false;
    }

    board_size_ = parsed_size;
    mode_ = parsed_mode;
    board_ = std::move(parsed_board);
    move_history_ = std::move(parsed_history);
    last_move_ = move_history_.empty()
        ? std::nullopt
        : std::optional{move_history_.back()};
    ai_used_fallback_ = false;
    ai_status_text_ = makeAiStatusText(mode_);
    return true;
}

const std::string& GameSession::last_persistence_error() const {
    return last_persistence_error_;
}


bool GameSession::skipTurn() {
    return board_.skipTurn();
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
