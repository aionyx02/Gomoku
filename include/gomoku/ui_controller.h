/**
 * @file ui_controller.h
 * @author shawn
 * @date 2026/3/22
 * @brief Terminal UI controller for Gomoku.
 */
#ifndef GOMOKU_UI_CONTROLLER_H
#define GOMOKU_UI_CONTROLLER_H

#include "engine.h"
#include <memory>
#include <string>
#include <vector>

namespace ftxui {
    class ComponentBase;
    class Node;
    using Component = std::shared_ptr<ComponentBase>;
    using Element = std::shared_ptr<Node>;
}

namespace UI {
    class Controller {
    public:
        explicit Controller(gomoku::Board& board);
        ~Controller();

        void Start();

    private:
        struct ScreenState;
        gomoku::Board& board;
        std::unique_ptr<ScreenState> screen_state;
        std::vector<std::string> menu_entries = {"Start Game (PvP)", "Start Game (PvE)", "Exit"};

        int menu_selected = 0;
        int active_index = 0;
        int current_x = 0;
        int current_y = 0;

        ftxui::Component RenderFrontPage();
        ftxui::Component RenderGameBoard();
        ftxui::Component RenderGameAIBoard();
        ftxui::Component RenderEndPage();

        [[nodiscard]] ftxui::Element RenderGrid() const;
    };
}

#endif
