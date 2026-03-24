#include "gomoku/ui_controller.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/component_options.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/util/ref.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace UI {
    using namespace ftxui;

    namespace {
        constexpr int kBoardSize = 15;
    }

    class InteractiveBoard : public ComponentBase {
    public:
        std::function<Element()> renderLogic;
        std::function<bool(Event)> eventLogic;

        [[nodiscard]] Element OnRender() override {
            return renderLogic();
        }

        [[nodiscard]] bool Focusable() const override {
            return true;
        }

        bool OnEvent(const Event event) override {
            return eventLogic(event);
        }
    };

    Controller::Controller(gomoku::Board& board) : board(board) {}

    void Controller::Start() {
        const auto container = Container::Tab({
            this->RenderFrontPage(),
            this->RenderGameBoard(),
            this->RenderEndPage()
        }, &active_index);

        this->screen.Loop(container);
    }

    Component Controller::RenderFrontPage() {
        const auto menu = Menu(&menu_entries, &menu_selected);

        auto component = Renderer(menu, [menu] {
            return vbox({
                text("=== Gomoku ===") | hcenter | bold | color(Color::Cyan),
                separator(),
                menu->Render() | hcenter,
                separator(),
                text("Use arrow keys to move, Enter/Space to place") | dim | hcenter
            }) | border | center;
        });

        component |= CatchEvent([this](const Event& event) -> bool {
            if (event != Event::Return)
                return false;

            if (this->menu_selected == 0) {
                this->active_index = 1;
                return true;
            }

            if (this->menu_selected == 1) {
                this->screen.Exit();
                return true;
            }

            return false;
        });

        return component;
    }

    Component Controller::RenderGameBoard() {
        auto comp = std::make_shared<InteractiveBoard>();

        comp->renderLogic = [this]() -> Element {
            return RenderGrid();
        };

        comp->eventLogic = [this](const Event& event) -> bool {
            if (event == Event::ArrowUp) {
                this->current_y = std::max(0, this->current_y - 1);
                return true;
            }
            if (event == Event::ArrowDown) {
                this->current_y = std::min(kBoardSize - 1, this->current_y + 1);
                return true;
            }
            if (event == Event::ArrowLeft) {
                this->current_x = std::max(0, this->current_x - 1);
                return true;
            }
            if (event == Event::ArrowRight) {
                this->current_x = std::min(kBoardSize - 1, this->current_x + 1);
                return true;
            }

            if (event == Event::Return || event == Event::Character(' ')) {
                if (this->board.placeStone(this->current_x, this->current_y)) {
                    if (this->board.getStatus() != gomoku::GameStatus::PLAYING)
                        this->active_index = 2;
                    return true;
                }
            }

            return false;
        };

        return comp;
    }

    Component Controller::RenderEndPage() {
        auto container = Container::Vertical({
            Button("Back to menu", [this] {
                this->active_index = 0;
                this->board = gomoku::Board(kBoardSize);
                this->current_x = 0;
                this->current_y = 0;
            }),
            Button("Play again", [this] {
                this->active_index = 1;
                this->board = gomoku::Board(kBoardSize);
                this->current_x = kBoardSize / 2;
                this->current_y = kBoardSize / 2;
            })
        });

        return Renderer(container, [container, this] {
            std::string result_text = "Draw";
            if (const gomoku::GameStatus status = this->board.getStatus(); status == gomoku::GameStatus::BLACK_WIN)
                result_text = "Black win";
            else if (status == gomoku::GameStatus::WHITE_WIN)
                result_text = "White win";
            else if (status == gomoku::GameStatus::PLAYING)
                result_text = "Playing";

            return vbox({
                text("Game Over") | hcenter | bold | color(Color::Red),
                separator(),
                text("Result: " + result_text) | hcenter | color(Color::Yellow),
                separator(),
                container->Render() | hcenter
            }) | border | center;
        });
    }

    Element Controller::RenderGrid() const {
        const gomoku::Stone current = board.getCurrentPlayer();
        const std::string turn_text = (current == gomoku::Stone::BLACK)
            ? "Current player: Black"
            : "Current player: White";
        const Color turn_color = (current == gomoku::Stone::BLACK) ? Color::Red : Color::White;

        auto status_bar = text(turn_text) | bold | color(turn_color) | hcenter;
        Elements rows;

        for (int y = 0; y < kBoardSize; ++y) {
            Elements cols;
            for (int x = 0; x < kBoardSize; ++x) {
                const auto stone = board.getStone(x, y);
                const std::string cell = (stone == gomoku::Stone::EMPTY) ? " + "
                    : (stone == gomoku::Stone::BLACK) ? " ○ " : " ● ";

                auto element = text(cell);
                if (x == current_x && y == current_y) {
                    element |= bgcolor(Color::Blue) | color(Color::White);
                } else if (stone == gomoku::Stone::BLACK) {
                    element |= color(Color::Red);
                } else if (stone == gomoku::Stone::WHITE) {
                    element |= color(Color::White);
                }
                cols.push_back(element);
            }
            rows.push_back(hbox(std::move(cols)));
        }

        auto board_ui = vbox(std::move(rows)) | hcenter;

        return vbox({
            status_bar,
            separator(),
            board_ui
        }) | border | center;
    }
}
