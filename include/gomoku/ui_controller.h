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
        const std::vector<std::string> menu_entries = {"開始新遊戲", "結束遊戲",};

        int menu_selected = 0;
        int active_index = 0;
        int current_x = 0;
        int current_y = 0;

        gomoku::Stone current_player = gomoku::Stone::BLACK;

        ftxui::Component RenderFrontPage();
        ftxui::Component RenderGameBoard();
        ftxui::Component RenderEndPage();

        [[nodiscard]]ftxui::Element RenderGrid() const;
    };
}
#endif //GOMOKU_UI_CONTROLLER_H