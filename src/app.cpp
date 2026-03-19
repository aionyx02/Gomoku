/**
 * @file app.cpp
 * @author shawn
 * @date 2026/3/19
 * @brief 
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */

#include <iostream>
#include "gomoku/main.h"

using namespace gomoku;

void drawBoard(const Board& board,const int size) {
    std::cout << "  ";
    for (int i = 0; i < size; ++i) std::cout << (i % 10) << " ";
    std::cout << "\n";

    for (int y = 0; y < size; ++y) {
        std::cout << (y % 10) << " ";
        for (int x = 0; x < size; ++x) {
            Stone s = board.getStone(x, y);
            if (s == Stone::BLACK) std::cout << "X ";
            else if (s == Stone::WHITE) std::cout << "O ";
            else std::cout << ". ";
        }
        std::cout << "\n";
    }
}

int main() {
    constexpr int boardSize = 15;
    Board board(boardSize);
    Stone currentPlayer = Stone::BLACK;

    std::cout << "=== Gomoku ===\n";
    std::cout << "Commands: [x] [y]. Input -1 to exit.\n";

    while (true) {
        drawBoard(board, boardSize);
        std::cout << (currentPlayer == Stone::BLACK ? "BLACK (X)" : "WHITE (O)") << " turn: ";

        int x, y;
        if (!(std::cin >> x) || x == -1) break;
        if (!(std::cin >> y)) break;

        if (!board.placeStone(x, y, currentPlayer)) {
            std::cout << "Invalid move! Try again.\n";
            continue;
        }

        if (gomoku::GameEngine::checkWin(board, x, y)) {
            drawBoard(board, boardSize);
            std::cout << "Game Over! " << (currentPlayer == Stone::BLACK ? "BLACK" : "WHITE") << " wins!\n";
            break;
        }

        currentPlayer = (currentPlayer == Stone::BLACK) ? Stone::WHITE : Stone::BLACK;
    }

    std::cout << "Test finished.\n";
    return 0;
}