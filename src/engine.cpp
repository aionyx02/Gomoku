/**
 * @file engine.cpp
 * @author aionyx
 * @date 2026/3/19
 * @brief Gomoku core engine implementation (self-contained state machine)
 *
 * * Under the hood:
 * - Memory Layout: Uses a flattened 1D std::vector for a 2D grid. Guarantees contiguous
 *   memory allocation, maximizing CPU L1/L2 cache hit rates during MCTS rollouts.
 * - System Calls / Interactions: STRICTLY NONE. Zero I/O, purely CPU-bound calculations.
 * - Resource Impact: O(1) time complexity for moves and win-checking. Copying the board
 *   is an O(N) memory block copy (where N = 225), highly optimized for tree search.
 */
#include "gomoku/engine.h"

namespace gomoku {

    Board::Board(int s)
        : size_(s),
          grid_(s * s, Stone::EMPTY),
          current_player_(Stone::BLACK),
          status_(GameStatus::PLAYING) {}

    bool Board::placeStone(const int x, const int y) {
        if (status_ != GameStatus::PLAYING || x < 0 || x >= size_ || y < 0 || y >= size_)
            return false;

        const int index = y * size_ + x;

        if (grid_[index] != Stone::EMPTY)
            return false;

        grid_[index] = current_player_;

        if (checkWinCondition(x, y)) {
            status_ = (current_player_ == Stone::BLACK) ? GameStatus::BLACK_WIN : GameStatus::WHITE_WIN;
        } else {
            bool has_empty_cell = false;
            for (const Stone stone : grid_) {
                if (stone == Stone::EMPTY) {
                    has_empty_cell = true;
                    break;
                }
            }

            if (!has_empty_cell) {
                status_ = GameStatus::DRAW;
            } else {
                current_player_ = (current_player_ == Stone::BLACK) ? Stone::WHITE : Stone::BLACK;
            }
        }

        return true;
    }

    Stone Board::getStone(const int x, const int y) const {
        if (x < 0 || x >= size_ || y < 0 || y >= size_)
            return Stone::EMPTY;

        return grid_[y * size_ + x];
    }

    Stone Board::getCurrentPlayer() const {
        return current_player_;
    }

    GameStatus Board::getStatus() const {
        return status_;
    }

    bool Board::checkWinCondition(const int lastX, const int lastY) const {
        const Stone target = grid_[lastY * size_ + lastX];
        if (target == Stone::EMPTY)
            return false;

        static constexpr int dx[] = {1, 0, 1, 1};
        static constexpr int dy[] = {0, 1, 1, -1};

        for (int i = 0; i < 4; ++i) {
            int count = 1;

            for (int step = 1; step < 5; ++step) {
                if (getStone(lastX + dx[i] * step, lastY + dy[i] * step) == target)
                    ++count;
                else
                    break;
            }

            for (int step = 1; step < 5; ++step) {
                if (getStone(lastX - dx[i] * step, lastY - dy[i] * step) == target)
                    ++count;
                else
                    break;
            }

            if (count >= 5)
                return true;
        }

        return false;
    }
}
