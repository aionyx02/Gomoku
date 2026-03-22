/**
 * @file app.cpp
 * @author shawn
 * @date 2026/3/19
 * @brief the file is to play the gomoku game
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */

#include "gomoku/main.h"
#include "gomoku/ui_controller.h"

using namespace gomoku;

int main() {
    gomoku::Board board(15);
    UI::Controller controller(board);

    controller.Start();

    return 0;
}