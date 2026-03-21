/**
 * @file ui_controller.h
 * @author shawn
 * @date 2026/3/22
 * @brief 
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */
#ifndef GOMOKU_UI_CONTROLLER_H
#define GOMOKU_UI_CONTROLLER_H
#include "main.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

namespace UI {
    class Controller {
    public:
        explicit Controller(gomoku::Board& board);

        void Start();
    private:
        gomoku::Board& board;
        ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();

        int active_index = 0;
        int current_x = 0;
        int current_y = 0;

        ftxui::Component RenderFrontPage();
        ftxui::Component RenderGameBoard();
        ftxui::Component RenderEndPage();

        ftxui::Element RenderGrid();
    };
}
#endif //GOMOKU_UI_CONTROLLER_H