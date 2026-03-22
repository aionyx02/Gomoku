#include "gomoku/ui_controller.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"  // for Component, Components
#include "ftxui/component/component_options.hpp"  // for ButtonOption, CheckboxOption, MenuOption
#include "ftxui/dom/elements.hpp"  // for Element
#include "ftxui/util/ref.hpp"
#include <vector>
#include <string>

namespace UI {
    using namespace ftxui;

    Controller::Controller(gomoku::Board& board) : board(board) {};

    void Controller::Start() {
        auto container = Container::Tab({
            this->RenderFrontPage(),
            this->RenderGameBoard(),
            this->RenderEndPage()
        }, &active_index);

        this->screen.Loop(container);
    };
    Component Controller::RenderFrontPage() {

        const auto menu = Menu(&menu_entries, &menu_selected);

        auto component = Renderer(menu, [menu] {
           return vbox({
                text("===Gomoku===") | hcenter | bold,
               separator(),
               menu->Render() | hcenter,
           }) | border | center;
        });

        component |= CatchEvent([this](Event event) -> bool {
            if (event == Event::Return) {
                if (this->menu_selected == 0) {
                    this->active_index = 1;
                    return true;
                }
                else if (this->menu_selected == 1) {
                    this->screen.Exit();
                    return true;
                }
            }
            return false;
        });

        return component;
    };
    Component Controller::RenderGameBoard() {
        auto comp = Make<ComponentBase>();


        comp -> OnEvent = [this](Event event) -> bool {
            bool handled = false;

            if (event == Event::ArrowUp) {
                this->current_y = std::max(0, this->current_y - 1);
                handled = true;
            }
            if (event == Event::ArrowDown) {
                this->current_y = std::min(14, this->current_y + 1);
                handled = true;
            }
            if (event == Event::ArrowLeft) {
                this->current_x = std::max(0, this->current_x - 1);
                handled = true;
            }
            if (event == Event::ArrowRight) {
                this->current_x = std::min(14, this->current_x + 1);
                handled = true;
            }


            if (event == Event::Return || event == Event::Character(' ')) {
                if (this->board.getStone(this->current_x, this->current_y) == gomoku::Stone::EMPTY) {
                    if (this->board.placeStone(current_x, current_y, current_player)) {
                        if (gomoku::GameEngine::checkWin(this->board, current_x, current_y)) {
                            this->active_index = 2;
                        }
                        else
                            current_player = (current_player == gomoku::Stone::BLACK) ? gomoku::Stone::WHITE : gomoku::Stone::BLACK;
                    }
                }
                handled = true;
            }
            return handled;
        };

        comp->Render = [this]() {
            return RenderGrid();
        };

        comp->SetActiveChild(true);

        return comp;
    };

    Component Controller::RenderEndPage() {
        auto container = Container::Vertical({
            Button("回首頁", [this] {
                this->active_index = 0;
            }),
            Button("在玩一局", [this] {
                this->active_index = 1;
            })
        });

        return Renderer(container, [container, this] {
            return vbox({
                text("遊戲結束") | hcenter | bold,
                separator(),
                text(this->current_player == gomoku::Stone::BLACK ? "黑棋勝利!" : "白棋勝利!") | hcenter | color(Color::Green),
                separator(),
                container->Render() | hcenter
            }) | border | center;
        });
    };

    Element Controller::RenderGrid() const {
        Elements rows;
        for (int y = 0; y < 15; ++y) {
            Elements cols;
            for (int x = 0; x < 15; ++x) {
                const std::string cell = (board.getStone(x, y) == gomoku::Stone::EMPTY) ? " + " :
                                   (board.getStone(x, y) == gomoku::Stone::BLACK) ? " ○ " : " ● ";
                auto element = text(cell);

                if (x == current_x && y == current_y)
                    element |= bgcolor(Color::Blue);
                cols.push_back(element | border);
            }
            rows.push_back(hbox(std::move(cols)));
        }
        return vbox(std::move(rows)) | center;
    };
}