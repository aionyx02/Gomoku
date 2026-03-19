/**
 * @file main.cpp
 * @author shawn
 * @date 2026/3/19
 * @brief 
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */
#include "gomoku/main.h"

namespace gomoku {
    Board::Board(int s) : size_(s), grid_(s * s, Stone::EMPTY) {

    };

    bool Board::placeStone(const int x,const int y,const Stone s) {
        if (x < 0 || x >= size_ || y < 0 || y >= size_)
            return false;

        const int index = y * size_ + x;

        if (grid_[index] != Stone::EMPTY)
            return false;
        grid_[index] = s;

        return true;
    }

    Stone Board::getStone(int x, int y) const {
        if (x < 0 || x >= size_ || y < 0 || y >= size_)
            return Stone::EMPTY;

        return grid_[y * size_ +x];
    }

    bool GameEngine::checkWin(const Board& board,const int lastX,const int lastY) {
        const Stone target = board.getStone(lastX, lastY);
        if (target == Stone::EMPTY)
            return false;

        static int dx[] = {1, 0, 1, 1};
        static int dy[] = {0, 1, 1, -1};

        for (int i = 0; i < 4; ++i) {
            int count = 1;

            for (int step = 1; step < 5; ++step) {
                if (board.getStone(lastX + dx[i] * step, lastY + dy[i] * step) == target)
                    ++count;
                else
                    break;
            }

            for (int step = 1; step < 5; ++step) {
                if (board.getStone(lastX - dx[i] * step, lastY - dy[i] * step) == target)
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
