#include "gomoku/ui_controller.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/component_options.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/util/ref.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace UI {
    using namespace ftxui;

    namespace {
        constexpr int kBoardSize = 15;

#ifdef GOMOKU_SOURCE_DIR
        constexpr const char* kSourceDir = GOMOKU_SOURCE_DIR;
#else
        constexpr const char* kSourceDir = ".";
#endif

#ifdef GOMOKU_PYTHON_EXECUTABLE
        constexpr const char* kPythonExecutable = GOMOKU_PYTHON_EXECUTABLE;
#else
        constexpr const char* kPythonExecutable = "python";
#endif

        char StoneToSymbol(const gomoku::Stone stone) {
            if (stone == gomoku::Stone::BLACK)
                return 'B';
            if (stone == gomoku::Stone::WHITE)
                return 'W';
            return '.';
        }

        std::string SerializeBoard(const gomoku::Board& board) {
            const int size = board.getSize();
            std::string serialized;
            serialized.reserve(size * size + size - 1);

            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x)
                    serialized.push_back(StoneToSymbol(board.getStone(x, y)));
                if (y + 1 < size)
                    serialized.push_back('|');
            }

            return serialized;
        }

        std::optional<std::pair<int, int> > QueryAIMove(const gomoku::Board& board) {
            const char current_symbol = StoneToSymbol(board.getCurrentPlayer());
            const std::string board_text = SerializeBoard(board);
            const std::string script_path = std::string(kSourceDir) + "/runModelAndReturnPoint.py";
            const std::string model_path = std::string(kSourceDir) + "/gomoku_model.pt";

            const std::string command =
                "\"" + std::string(kPythonExecutable) + "\" "
                "\"" + script_path + "\" "
                "--board-size " + std::to_string(board.getSize()) + " "
                "--model-path \"" + model_path + "\" "
                "--current " + std::string(1, current_symbol) + " "
                "--board \"" + board_text + "\" 2>&1";

#if defined(_WIN32)
            FILE* pipe = _popen(command.c_str(), "r");
#else
            FILE* pipe = popen(command.c_str(), "r");
#endif
            if (pipe == nullptr)
                return std::nullopt;

            std::array<char, 256> buffer {};
            std::string output;
            while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
                output += buffer.data();

#if defined(_WIN32)
            const int exit_code = _pclose(pipe);
#else
            const int exit_code = pclose(pipe);
#endif
            if (exit_code != 0)
                return std::nullopt;

            std::istringstream parser(output);
            int x = -1;
            int y = -1;
            if (!(parser >> x >> y))
                return std::nullopt;

            const int size = board.getSize();
            if (x < 0 || x >= size || y < 0 || y >= size)
                return std::nullopt;

            return std::make_pair(x, y);
        }

        std::optional<std::pair<int, int> > FindFallbackMove(const gomoku::Board& board) {
            const int size = board.getSize();
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    if (board.getStone(x, y) == gomoku::Stone::EMPTY)
                        return std::make_pair(x, y);
                }
            }
            return std::nullopt;
        }
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

    struct Controller::ScreenState {
        ScreenInteractive screen = ScreenInteractive::Fullscreen();
    };

    Controller::Controller(gomoku::Board& board)
        : board(board),
          screen_state(std::make_unique<ScreenState>()) {}

    Controller::~Controller() = default;

    void Controller::Start() {
        const auto container = Container::Tab({
            this->RenderFrontPage(),
            this->RenderGameBoard(),
            this->RenderGameAIBoard(),
            this->RenderEndPage()
        }, &active_index);

        this->screen_state->screen.Loop(container);
    }
    Component Controller::RenderFrontPage() {
        const auto menu = Menu(&menu_entries, &menu_selected);

        auto component = Renderer(menu, [menu] {
            return vbox({
                text("=== Gomoku===") | hcenter | bold | color(Color::Cyan),
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
                this->active_index = 2;
                return true;
            }
            if (this->menu_selected == 2) {
                this->screen_state->screen.Exit();
                return true;
            }

            return false;
        });

        return component;
    }
    Component Controller::RenderGameAIBoard() {
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
                        this->active_index = 3;

                    if (this->board.getStatus() == gomoku::GameStatus::PLAYING) {
                        bool ai_placed = false;

                        if (const auto ai_move = QueryAIMove(this->board); ai_move.has_value()) {
                            const auto [ai_x, ai_y] = ai_move.value();
                            ai_placed = this->board.placeStone(ai_x, ai_y);
                            if (ai_placed) {
                                this->current_x = ai_x;
                                this->current_y = ai_y;
                            }
                        }

                        if (!ai_placed) {
                            if (const auto fallback = FindFallbackMove(this->board); fallback.has_value()) {
                                const auto [fallback_x, fallback_y] = fallback.value();
                                if (this->board.placeStone(fallback_x, fallback_y)) {
                                    this->current_x = fallback_x;
                                    this->current_y = fallback_y;
                                }
                            }
                        }

                        if (this->board.getStatus() != gomoku::GameStatus::PLAYING)
                            this->active_index = 3;
                    }

                    return true;
                }
            }

            return false;
        };

        return comp;
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
                        this->active_index = 3;
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
