#include "gomoku/ui_controller.h"

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

namespace UI {
using namespace ftxui;

namespace {

constexpr int kMenuTab = 0;
constexpr int kPvpTab = 1;
constexpr int kPveTab = 2;
constexpr int kResultTab = 3;

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

    explicit Impl(gomoku::GameSession& game_session)
        : session(game_session),
          screen_state(std::make_unique<ScreenState>()) {
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
        return std::max(0, session.board().getSize() - 1);
    }

    void centerCursor() {
        const int center = session.board().getSize() / 2;
        current_x = center;
        current_y = center;
    }

    void startGame(const gomoku::SessionMode next_mode) {
        session.start(next_mode);
        centerCursor();

        if (next_mode == gomoku::SessionMode::PVE) {
            active_index = kPveTab;
        } else {
            active_index = kPvpTab;
        }
    }

    void backToMenu() {
        session.reset();
        active_index = kMenuTab;
        current_x = 0;
        current_y = 0;
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
        if (session.status() != gomoku::GameStatus::PLAYING) {
            active_index = kResultTab;
        }
    }

    bool runAiTurn() {
        if (!session.ai_move()) {
            return false;
        }

        if (const auto move = session.last_move()) {
            current_x = move->first;
            current_y = move->second;
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
                startGame(gomoku::SessionMode::PVP);
                return true;
            }
            if (menu_selected == 1) {
                startGame(gomoku::SessionMode::PVE);
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

            if (!session.human_move(current_x, current_y)) {
                return false;
            }

            tryMoveToResult();
            if (has_ai && session.status() == gomoku::GameStatus::PLAYING) {
                runAiTurn();
            }

            return true;
        };

        return component;
    }

    Component renderEndPage() {
        auto container = Container::Vertical({
            Button("Back to menu", [this] { backToMenu(); }),
            Button("Play again", [this] { startGame(session.mode()); })
        });

        return Renderer(container, [container, this] {
            return vbox({
                text("Game Over") | hcenter | bold | color(Color::Red),
                separator(),
                text("Result: " + gameResultText(session.status())) | hcenter | color(Color::Yellow),
                separator(),
                container->Render() | hcenter
            }) | border | center;
        });
    }

    [[nodiscard]] Element renderGrid() const {
        const auto& board = session.board();
        const bool is_black_turn = board.getCurrentPlayer() == gomoku::Stone::BLACK;
        auto status_bar = text(is_black_turn ? "Current player: Black" : "Current player: White") |
                          bold |
                          color(is_black_turn ? Color::Red : Color::White) |
                          hcenter;

        auto ai_bar = text(session.ai_status_text()) | hcenter;
        if (active_index == kPveTab) {
            ai_bar |= session.ai_used_fallback() ? color(Color::Yellow) : color(Color::Green);
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

    gomoku::GameSession& session;
    std::unique_ptr<ScreenState> screen_state;

    std::vector<std::string> menu_entries = {
        "Start Game (PvP)",
        "Start Game (PvE)",
        "Exit"
    };

    int menu_selected = 0;
    int active_index = kMenuTab;
    int current_x = 0;
    int current_y = 0;
};

Controller::Controller(gomoku::GameSession& session)
    : impl_(std::make_unique<Impl>(session)) {}

Controller::~Controller() = default;

void Controller::Start() const {
    impl_->Start();
}

} // namespace UI
