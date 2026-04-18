/**
 * @file engine.h
 * @author aionyx
 * @date 2026/3/19
 * @brief Gomoku core engine declaration (self-contained state machine)
 */
#ifndef GOMOKU_ENGINE_H
#define GOMOKU_ENGINE_H

#include <cstdint>
#include <vector>

namespace gomoku {
    enum class Stone : uint8_t { EMPTY = 0, BLACK, WHITE };

    enum class GameStatus : uint8_t {
        PLAYING = 0,
        BLACK_WIN,
        WHITE_WIN,
        DRAW
    };

    class Board {
    private:
        int size_;
        std::vector<Stone> grid_;
        Stone current_player_;
        GameStatus status_;

        [[nodiscard]] bool checkWinCondition(int lastX, int lastY) const;

    public:
        explicit Board(int s = 15);

        // Place a stone for the current player and auto-transition state.
        bool placeStone(int x, int y);

        // Remove the stone at (x,y) and restore turn/status to PLAYING.
        bool undoStone(int x, int y);

        // Forfeit the current player's turn (no stone placed).
        bool skipTurn();

        [[nodiscard]] Stone getStone(int x, int y) const;
        [[nodiscard]] Stone getCurrentPlayer() const;
        [[nodiscard]] GameStatus getStatus() const;
        [[nodiscard]] int getSize() const { return size_; }
    };
}

#endif
