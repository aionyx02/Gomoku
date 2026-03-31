#include "gomoku/ui_controller.h"

#include "gomoku/ai_player.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef GOMOKU_SOURCE_DIR
constexpr auto kSourceDir = GOMOKU_SOURCE_DIR;
#else
constexpr const char* kSourceDir = ".";
#endif

#ifdef GOMOKU_PYTHON_EXECUTABLE
constexpr auto kPythonExecutable = GOMOKU_PYTHON_EXECUTABLE;
#else
constexpr const char* kPythonExecutable = "python";
#endif

namespace UI {
using namespace ftxui;

namespace {

constexpr int kMenuTab = 0;
constexpr int kPvpTab = 1;
constexpr int kPveTab = 2;
constexpr int kResultTab = 3;
constexpr int kDefaultBoardSize = 15;

enum class GameMode {
    Pvp,
    Pve
};

class InteractiveBoard final : public ComponentBase {
public:
    std::function<Element()> render_logic;
    std::function<bool(Event)> event_logic;

    Element OnRender() override {
        return render_logic();
    }

    [[nodiscard]] bool Focusable() const override {
        return true;
    }

    bool OnEvent(const Event event) override {
        return event_logic(event);
    }
};

std::string gameResultText(const gomoku::GameStatus status) {
    switch (status) {
        case gomoku::GameStatus::BLACK_WIN:
            return "Black win";
        case gomoku::GameStatus::WHITE_WIN:
            return "White win";
        case gomoku::GameStatus::DRAW:
            return "Draw";
        case gomoku::GameStatus::PLAYING:
        default:
            return "Playing";
    }
}

Element framedStoneCell(const std::string& symbol, const Color symbol_color) {
    return hbox({
        text("  "),
        text(symbol) | color(symbol_color) | bold,
        text(" ")
    }) | size(WIDTH, EQUAL, 5) | hcenter;
}

Element stoneCellElement(const gomoku::Stone stone) {
    switch (stone) {
        case gomoku::Stone::WHITE:
            return framedStoneCell("●", Color::White);
        case gomoku::Stone::BLACK:
            return framedStoneCell("○", Color::Red);
        case gomoku::Stone::EMPTY:
        default:
            return text("  +  ") | color(Color::GrayDark) | size(WIDTH, EQUAL, 5) | hcenter;
    }
}

} // namespace

struct Controller::Impl {
    struct ScreenState {
        ScreenInteractive screen = ScreenInteractive::Fullscreen();
    };

    explicit Impl(gomoku::Board& game_board)
        : board(game_board),
          screen_state(std::make_unique<ScreenState>()),
          ai_player(kSourceDir, kPythonExecutable),
          board_size(std::max(1, game_board.getSize())) {
        backToMenu();
    }

    void Start() {
        const auto container = Container::Tab({
            renderFrontPage(),
            renderGameBoard(false),
            renderGameBoard(true),
            renderEndPage()
        }, &active_index);

        screen_state->screen.Loop(container);
    }

    [[nodiscard]] int boardLimit() const {
        return std::max(0, board.getSize() - 1);
    }

    void centerCursor() {
        const int center = board.getSize() / 2;
        current_x = center;
        current_y = center;
    }

    void startGame(const GameMode next_mode) {
        current_mode = next_mode;
        board = gomoku::Board(board_size);
        centerCursor();
        ai_used_fallback = false;

        if (current_mode == GameMode::Pve) {
            active_index = kPveTab;
            ai_status_text = "AI: ready (model expected at gomoku_model.pt)";
        } else {
            active_index = kPvpTab;
            ai_status_text = "AI: disabled in PvP";
        }
    }

    void backToMenu() {
        board = gomoku::Board(board_size);
        active_index = kMenuTab;
        current_x = 0;
        current_y = 0;
        ai_status_text = "AI: ready";
        ai_used_fallback = false;
    }

    bool handleCursorMove(const Event& event) {
        const int limit = boardLimit();
        if (event == Event::ArrowUp) {
            current_y = std::max(0, current_y - 1);
            return true;
        }
        if (event == Event::ArrowDown) {
            current_y = std::min(limit, current_y + 1);
            return true;
        }
        if (event == Event::ArrowLeft) {
            current_x = std::max(0, current_x - 1);
            return true;
        }
        if (event == Event::ArrowRight) {
            current_x = std::min(limit, current_x + 1);
            return true;
        }
        return false;
    }

    void tryMoveToResult() {
        if (board.getStatus() != gomoku::GameStatus::PLAYING) {
            active_index = kResultTab;
        }
    }

    bool runAiTurn() {
        const auto [move, used_fallback, diagnostic] = ai_player.makeMove(board);
        ai_used_fallback = used_fallback;

        if (!move) {
            ai_status_text = "AI failed: " + diagnostic;
            return false;
        }

        const auto [x, y] = *move;
        if (!board.placeStone(x, y)) {
            ai_status_text = "AI failed: suggested move is invalid";
            ai_used_fallback = true;
            return false;
        }

        current_x = x;
        current_y = y;
        ai_status_text = used_fallback
            ? "AI(fallback): move (" + std::to_string(x) + "," + std::to_string(y) + ")"
            : "AI(model): move (" + std::to_string(x) + "," + std::to_string(y) + ")";

        if (used_fallback && !diagnostic.empty()) {
            ai_status_text += " | reason: " + diagnostic;
        }

        tryMoveToResult();
        return true;
    }

    Component renderFrontPage() {
        auto menu = Menu(&menu_entries, &menu_selected);

        auto component = Renderer(menu, [menu] {
            return vbox({
                text("=== Gomoku ===") | hcenter | bold | color(Color::Cyan),
                separator(),
                menu->Render() | hcenter,
                separator(),
                text("Use arrow keys to move, Enter/Space to place") | dim | hcenter
            }) | border | center;
        });

        component |= CatchEvent([this](const Event& event) {
            if (event != Event::Return) {
                return false;
            }

            if (menu_selected == 0) {
                startGame(GameMode::Pvp);
                return true;
            }
            if (menu_selected == 1) {
                startGame(GameMode::Pve);
                return true;
            }
            if (menu_selected == 2) {
                screen_state->screen.Exit();
                return true;
            }

            return false;
        });

        return component;
    }

    Component renderGameBoard(const bool has_ai) {
        auto component = std::make_shared<InteractiveBoard>();
        component->render_logic = [this] { return renderGrid(); };
        component->event_logic = [this, has_ai](const Event& event) {
            if (handleCursorMove(event)) {
                return true;
            }

            if (event != Event::Return && event != Event::Character(' ')) {
                return false;
            }

            if (!board.placeStone(current_x, current_y)) {
                return false;
            }

            tryMoveToResult();
            if (has_ai && board.getStatus() == gomoku::GameStatus::PLAYING) {
                runAiTurn();
            }

            return true;
        };

        return component;
    }

    Component renderEndPage() {
        auto container = Container::Vertical({
            Button("Back to menu", [this] { backToMenu(); }),
            Button("Play again", [this] { startGame(current_mode); })
        });

        return Renderer(container, [container, this] {
            return vbox({
                text("Game Over") | hcenter | bold | color(Color::Red),
                separator(),
                text("Result: " + gameResultText(board.getStatus())) | hcenter | color(Color::Yellow),
                separator(),
                container->Render() | hcenter
            }) | border | center;
        });
    }

    [[nodiscard]] Element renderGrid() const {
        const bool is_black_turn = board.getCurrentPlayer() == gomoku::Stone::BLACK;
        auto status_bar = text(is_black_turn ? "Current player: Black" : "Current player: White") |
                          bold |
                          color(is_black_turn ? Color::Red : Color::White) |
                          hcenter;

        auto ai_bar = text(ai_status_text) | hcenter;
        if (active_index == kPveTab) {
            ai_bar |= ai_used_fallback ? color(Color::Yellow) : color(Color::Green);
        } else {
            ai_bar |= dim;
        }

        Elements rows;
        const int size = board.getSize();
        for (int y = 0; y < size; ++y) {
            Elements columns;
            for (int x = 0; x < size; ++x) {
                const auto stone = board.getStone(x, y);
                auto cell = stoneCellElement(stone);

                if (x == current_x && y == current_y) {
                    if (stone == gomoku::Stone::WHITE) {
                        cell |= underlined;
                    } else {
                        cell |= bgcolor(Color::Blue);
                    }
                }

                columns.push_back(std::move(cell));
            }
            rows.push_back(hbox(std::move(columns)));
        }

        return vbox({
            status_bar,
            ai_bar,
            separator(),
            vbox(std::move(rows)) | hcenter
        }) | border | center;
    }

    gomoku::Board& board;
    std::unique_ptr<ScreenState> screen_state;
    gomoku::ai::Player ai_player;

    std::vector<std::string> menu_entries = {
        "Start Game (PvP)",
        "Start Game (PvE)",
        "Exit"
    };

    int menu_selected = 0;
    int active_index = kMenuTab;
    int current_x = 0;
    int current_y = 0;
    int board_size = kDefaultBoardSize;

    GameMode current_mode = GameMode::Pvp;
    std::string ai_status_text = "AI: ready";
    bool ai_used_fallback = false;
};

Controller::Controller(gomoku::Board& board)
    : impl_(std::make_unique<Impl>(board)) {}

Controller::~Controller() = default;

void Controller::Start() const {
    impl_->Start();
}

} // namespace UI
