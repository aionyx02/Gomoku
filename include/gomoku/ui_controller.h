/**
 * @file ui_controller.h
 * @author shawn
 * @date 2026/3/22
 * @brief Terminal UI controller for Gomoku.
 */
#ifndef GOMOKU_UI_CONTROLLER_H
#define GOMOKU_UI_CONTROLLER_H

#include "engine.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <string>
#include <vector>

namespace UI {
    class Controller {
    public:
        explicit Controller(gomoku::Board& board);

        void Start();

    private:
        gomoku::Board& board;
        ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();
        std::vector<std::string> menu_entries = {"Start Game", "Exit"};

        int menu_selected = 0;
        int active_index = 0;
        int current_x = 0;
        int current_y = 0;

        ftxui::Component RenderFrontPage();
        ftxui::Component RenderGameBoard();
        ftxui::Component RenderEndPage();

        [[nodiscard]] ftxui::Element RenderGrid() const;
    };
}

#endif
