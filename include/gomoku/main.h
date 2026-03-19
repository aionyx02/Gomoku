/**
 * @file main.h
 * @author shawn
 * @date 2026/3/19
 * @brief the file is to set the base struct about gomoku
 *
 * * Under the hood:
 * - Memory Layout:
 * - System Calls / Interactions:
 * - Resource Impact: 
 */
#ifndef GOMOKU_MAIN_H
#define GOMOKU_MAIN_H

#include <vector>
#include <cstdint>

namespace gomoku {
    enum class Stone : uint8_t { EMPTY = 0, BLACK, WHITE };

    class Board {
    private:
        int size_;
        std::vector<Stone> grid_;
    public:
        explicit Board(int s = 15);
        bool placeStone(int x, int y, Stone s);
        [[nodiscard]] Stone getStone(int x, int y) const;
    };

    class GameEngine {
    public:
        static bool checkWin(const Board& board, int lastX, int lastY);
    };

}

#endif