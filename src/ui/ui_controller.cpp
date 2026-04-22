#include "../../include/gomoku/ui/ui_controller_internal.h"

namespace UI::detail {
using namespace ftxui;

std::string gameResultText(const gomoku::GameStatus status) {
    switch (status) {
        case gomoku::GameStatus::BLACK_WIN:
            return "Black Win";
        case gomoku::GameStatus::WHITE_WIN:
            return "White Win";
        case gomoku::GameStatus::DRAW:
            return "Draw";
        case gomoku::GameStatus::PLAYING:
        default:
            return "Playing";
    }
}

std::string stoneText(const gomoku::Stone stone) {
    switch (stone) {
        case gomoku::Stone::BLACK:
            return "Black";
        case gomoku::Stone::WHITE:
            return "White";
        case gomoku::Stone::EMPTY:
        default:
            return "None";
    }
}

Element framedStoneCell(const std::string& symbol, const Color symbol_color) {
    return hbox({
        text(" "),
        text(symbol) | color(symbol_color) | bold,
        text(" ")
    }) | size(WIDTH, EQUAL, 3) | hcenter;
}

Element stoneCellElement(const gomoku::Stone stone) {
    switch (stone) {
        case gomoku::Stone::WHITE:
            return framedStoneCell("●", Color::White);
        case gomoku::Stone::BLACK:
            return framedStoneCell("○", Color::Red);
        case gomoku::Stone::EMPTY:
        default:
            return text(" + ") | color(Color::GrayDark) | size(WIDTH, EQUAL, 3) | hcenter;
    }
}

std::string trimCopy(std::string value) {
    const auto is_space = [](const unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::ranges::find_if(value, [&](const unsigned char ch) {
        return !is_space(ch);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](const unsigned char ch) {
        return !is_space(ch);
    }).base(), value.end());
    return value;
}

std::optional<std::uint16_t> parsePortValue(const std::string& raw_value) {
    const std::string value = trimCopy(raw_value);
    if (value.empty()) {
        return std::nullopt;
    }

    int port = 0;
    if (const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), port);
        ec != std::errc{} || ptr != value.data() + value.size()) {
        return std::nullopt;
    }
    if (port < 1 || port > 65535) {
        return std::nullopt;
    }
    return static_cast<std::uint16_t>(port);
}

Color remoteStatusColor(const bool connected, const bool waiting) {
    if (connected) {
        return Color::Green;
    }
    if (waiting) {
        return Color::Yellow;
    }
    return Color::Orange1;
}

} // namespace UI::detail

namespace UI {
using namespace ftxui;

Controller::Impl::Impl(gomoku::GameSession& game_session)
    : session(game_session),
      network(game_session),
      screen_state(std::make_unique<ScreenState>()) {
    ui_tick_stop_.store(false);
    ui_tick_ = std::thread([this] {
        while (!ui_tick_stop_.load()) {
            std::this_thread::sleep_for(detail::kUiTickInterval);
            if (screen_state) {
                screen_state->screen.PostEvent(Event::Custom);
            }
        }
    });
    backToMenu();
}

Controller::Impl::~Impl() {
    ui_tick_stop_.store(true);
    if (ui_tick_.joinable()) {
        ui_tick_.join();
    }
    stopReplayAuto();
    stopTimer();
}

void Controller::Impl::Start() {
    const auto container = Container::Tab({
        renderFrontPage(),
        renderGameBoard(false),
        renderGameBoard(true),
        renderEndPage(),
        renderSetupPage(),
        renderLoadGamePage(),
        renderRemoteHostPage(),
        renderRemoteJoinPage(),
        renderRemoteWaitPage(),
        renderReplayPage()
    }, &active_index);

    const auto root = CatchEvent(container, [this](const Event& event) {
        if (event == Event::Custom) {
            return handleNetworkTick();
        }
        return false;
    });

    screen_state->screen.Loop(root);
}

Controller::Controller(gomoku::GameSession& session)
    : impl_(std::make_unique<Impl>(session)) {}

Controller::~Controller() = default;

void Controller::Start() const {
    impl_->Start();
}

} // namespace UI
