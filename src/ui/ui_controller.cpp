#include "../../include/gomoku/ui/ui_controller.h"

#include "../../include/gomoku/net/webConnect.h"

#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UI {
using namespace ftxui;

namespace {

constexpr int kMenuTab = 0;
constexpr int kPvpTab = 1;
constexpr int kPveTab = 2;
constexpr int kResultTab = 3;
constexpr int kSetupTab      = 4;
constexpr int kLoadTab       = 5;
constexpr int kRemoteHostTab = 6;
constexpr int kRemoteJoinTab = 7;
constexpr int kRemoteWaitTab = 8;
constexpr int kReplayTab     = 9;

constexpr std::chrono::milliseconds kUiTickInterval{100};

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
    if (const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), port); ec != std::errc{} || ptr != value.data() + value.size()) {
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

} // namespace

struct Controller::Impl {
    struct ScreenState {
        ScreenInteractive screen = ScreenInteractive::Fullscreen();
    };

    explicit Impl(gomoku::GameSession& game_session)
        : session(game_session),
          network(game_session),
          screen_state(std::make_unique<ScreenState>()) {
        ui_tick_stop_.store(false);
        ui_tick_ = std::thread([this] {
            while (!ui_tick_stop_.load()) {
                std::this_thread::sleep_for(kUiTickInterval);
                if (screen_state) {
                    screen_state->screen.PostEvent(Event::Custom);
                }
            }
        });
        backToMenu();
    }

    ~Impl() {
        ui_tick_stop_.store(true);
        if (ui_tick_.joinable()) {
            ui_tick_.join();
        }
        stopReplayAuto();
        stopTimer();
    }

    void Start() {
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

    [[nodiscard]] int boardLimit() const {
        return std::max(0, session.board().getSize() - 1);
    }

    void centerCursor() {
        const int center = session.board().getSize() / 2;
        current_x = center;
        current_y = center;
    }

    void syncCursorToSession() {
        if (const auto move = session.last_move()) {
            current_x = move->first;
            current_y = move->second;
            return;
        }
        centerCursor();
    }

    void updateRemoteStatus(const std::string& override_text = {}) {
        if (!override_text.empty()) {
            remote_status_text_ = override_text;
            return;
        }

        if (remote_waiting_for_peer_) {
            const std::string endpoint = network.localEndpoint().empty()
                ? ("port " + remote_port_input_)
                : network.localEndpoint();
            remote_status_text_ = "Waiting for peer on " + endpoint;
            return;
        }

        if (network.isConnected()) {
            remote_status_text_ = "Remote connected | You: " + stoneText(network.localStone());
            if (!network.remoteEndpoint().empty()) {
                remote_status_text_ += " | Peer: " + network.remoteEndpoint();
            }
            if (session.status() == gomoku::GameStatus::PLAYING) {
                remote_status_text_ += network.isLocalTurn()
                    ? " | Your turn"
                    : " | Opponent turn";
            }
            return;
        }

        if (!network.lastError().empty()) {
            remote_status_text_ = "Remote error: " + network.lastError();
            return;
        }

        remote_status_text_ = "Remote session idle";
    }

    void shutdownRemoteSession() {
        network.disconnect();
        remote_mode_ = false;
        remote_waiting_for_peer_ = false;
        remote_status_text_.clear();
        show_save_menu_ = false;
    }

    void clearLocalStatus() {
        local_status_text_.clear();
        local_status_color_ = Color::GrayDark;
    }

    void setLocalStatus(std::string message, const Color color = Color::Red) {
        local_status_text_ = std::move(message);
        local_status_color_ = color;
    }

    void startGame(const gomoku::SessionMode next_mode) {
        shutdownRemoteSession();
        clearLocalStatus();
        session.start(next_mode);
        centerCursor();

        if (next_mode == gomoku::SessionMode::PVE) {
            active_index = kPveTab;
        } else {
            active_index = kPvpTab;
        }

        if (settings_timer_enabled) {
            startTimer();
        }
    }

    void backToMenu() {
        stopTimer();
        consecutive_timeouts_ = 0;
        shutdownRemoteSession();
        clearLocalStatus();
        session.reset();
        active_index = kMenuTab;
        current_x = 0;
        current_y = 0;
    }

    void openRemoteHostPage() {
        shutdownRemoteSession();
        clearLocalStatus();
        session.start(gomoku::SessionMode::PVP);
        centerCursor();
        remote_status_text_ = "Choose a port and start hosting.";
        active_index = kRemoteHostTab;
    }

    void openRemoteJoinPage() {
        shutdownRemoteSession();
        clearLocalStatus();
        session.start(gomoku::SessionMode::PVP);
        centerCursor();
        remote_status_text_ = "Enter the host IP and port.";
        active_index = kRemoteJoinTab;
    }

    bool beginHosting() {
        clearLocalStatus();
        const auto port = parsePortValue(remote_port_input_);
        if (!port) {
            updateRemoteStatus("Port must be between 1 and 65535.");
            return false;
        }

        network.disconnect();
        session.start(gomoku::SessionMode::PVP);
        centerCursor();
        remote_mode_ = true;
        remote_waiting_for_peer_ = true;

        if (!network.openHost(*port)) {
            remote_mode_ = false;
            remote_waiting_for_peer_ = false;
            updateRemoteStatus("Host failed: " + network.lastError());
            return false;
        }

        updateRemoteStatus();
        active_index = kRemoteWaitTab;
        return true;
    }

    bool beginJoin() {
        clearLocalStatus();
        const std::string host = trimCopy(remote_host_input_);
        if (host.empty()) {
            updateRemoteStatus("Host IP or hostname cannot be empty.");
            return false;
        }

        const auto port = parsePortValue(remote_port_input_);
        if (!port) {
            updateRemoteStatus("Port must be between 1 and 65535.");
            return false;
        }

        network.disconnect();
        session.start(gomoku::SessionMode::PVP);
        centerCursor();
        remote_mode_ = true;
        remote_waiting_for_peer_ = false;

        if (!network.connectTo(host, *port)) {
            remote_mode_ = false;
            updateRemoteStatus("Join failed: " + network.lastError());
            return false;
        }

        updateRemoteStatus();
        syncCursorToSession();
        active_index = kPvpTab;
        handleNetworkTick();
        return true;
    }

    bool handleNetworkTick() {
        bool changed = false;

        if (remote_mode_ && remote_waiting_for_peer_) {
            if (network.waitForPeer(std::chrono::milliseconds{0})) {
                remote_waiting_for_peer_ = false;
                updateRemoteStatus();
                syncCursorToSession();
                active_index = kPvpTab;
                changed = true;
            } else if (!network.lastError().empty()) {
                updateRemoteStatus();
                changed = true;
            }
        }

        if (remote_mode_ && !remote_waiting_for_peer_ && network.isConnected()) {
            const auto previous_move_count = session.move_history().size();
            const auto previous_status = session.status();

            if (const bool handled = network.pump(std::chrono::milliseconds{0}); handled ||
                                                                                 previous_move_count != session.move_history().size() ||
                                                                                 previous_status != session.status()) {
                syncCursorToSession();
                updateRemoteStatus();
                changed = true;
            } else if (!network.lastError().empty()) {
                updateRemoteStatus();
                changed = true;
            }
        } else if (remote_mode_ && !remote_waiting_for_peer_ && !network.isConnected() && !network.lastError().empty()) {
            updateRemoteStatus();
            changed = true;
        }

        if (remote_mode_) {
            if (session.status() != gomoku::GameStatus::PLAYING &&
                active_index != kResultTab &&
                active_index != kRemoteWaitTab) {
                active_index = kResultTab;
                changed = true;
            }

            if (session.status() == gomoku::GameStatus::PLAYING && active_index == kResultTab) {
                active_index = kPvpTab;
                syncCursorToSession();
                changed = true;
            }
        }

        return changed;
    }

    void stopReplayAuto() {
        replay_auto_ = false;
        replay_auto_stop_.store(true);
        if (replay_auto_thread_.joinable()) {
            replay_auto_thread_.join();
        }
    }

    void startReplayAuto() {
        stopReplayAuto();
        replay_auto_ = true;
        replay_auto_stop_.store(false);
        replay_auto_thread_ = std::thread([this] {
            while (!replay_auto_stop_.load()) {
                for (int i = 0; i < 10 && !replay_auto_stop_.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!replay_auto_stop_.load()) {
                    screen_state->screen.PostEvent(Event::Special("replay_tick"));
                }
            }
        });
    }

    void stopTimer() {
        timer_stop_.store(true);
        if (timer_thread_.joinable()) {
            timer_thread_.join();
        }
        timer_remaining_ = 0;
    }

    void startTimer() {
        stopTimer();
        int secs = 20;
        try { secs = std::stoi(settings_timer_seconds_str_); } catch (...) {}
        if (secs <= 0) secs = 20;
        timer_remaining_ = secs;
        timer_stop_.store(false);
        timer_thread_ = std::thread([this] {
            while (!timer_stop_.load()) {
                for (int i = 0; i < 10 && !timer_stop_.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                if (!timer_stop_.load()) {
                    screen_state->screen.PostEvent(Event::Special("timer_tick"));
                }
            }
        });
    }

    void handleTimerTimeout(const bool has_ai) {
        if (session.status() != gomoku::GameStatus::PLAYING) return;
        session.skipTurn();
        ++consecutive_timeouts_;

        if (has_ai && session.status() == gomoku::GameStatus::PLAYING) {
            runAiTurn();
        }

        if (consecutive_timeouts_ >= 3) {
            consecutive_timeouts_ = 0;
            stopTimer();
            show_save_menu_ = true;
            save_menu_selected_ = 0;
            return;
        }

        if (settings_timer_enabled && session.status() == gomoku::GameStatus::PLAYING) {
            startTimer();
        }
    }

    void enterReplay() {
        stopReplayAuto();
        replay_history_ = session.move_history();
        replay_step_    = static_cast<int>(replay_history_.size());
        active_index    = kReplayTab;
    }

    [[nodiscard]] Element renderReplayGrid() const {
        const int board_size = session.board().getSize();
        const int max_step   = static_cast<int>(replay_history_.size());

        std::unordered_map<int, int> pos_to_step;
        pos_to_step.reserve(replay_step_);
        for (int i = 0; i < replay_step_; ++i) {
            const auto [x, y] = replay_history_[i];
            pos_to_step[y * board_size + x] = i + 1;
        }

        auto status_bar = text("Replay  Step " + std::to_string(replay_step_) +
                               " / " + std::to_string(max_step))
                          | hcenter | bold | color(Color::Cyan);

        Elements rows;
        for (int y = 0; y < board_size; ++y) {
            Elements columns;
            for (int x = 0; x < board_size; ++x) {
                const auto it = pos_to_step.find(y * board_size + x);
                Element cell;
                if (it != pos_to_step.end()) {
                    const int  step     = it->second;
                    const bool is_black = (step % 2 == 1);
                    const auto s        = std::to_string(step);
                    std::string padded;
                    if (s.size() == 1)      padded = "  " + s + "  ";
                    else if (s.size() == 2) padded = " "  + s + "  ";
                    else                    padded = s    + "  ";
                    cell = text(padded)
                           | color(is_black ? Color::Red : Color::White)
                           | bold;
                } else {
                    cell = text("  +  ")
                           | color(Color::GrayDark);
                }
                columns.push_back(std::move(cell));
            }
            rows.push_back(hbox(std::move(columns)));
        }

        auto bottom_bar = hbox({
            text(" [R Rewind] ") | bold,
            filler(),
            text(" [<- Prev] ") | (replay_step_ > 0 ? bold : dim),
            filler(),
            replay_auto_
                ? (text(" [Space Stop] ") | color(Color::Green) | bold)
                : (text(" [Space Play] ") | bold),
            filler(),
            text(" [Next ->] ") | (replay_step_ < max_step ? bold : dim),
            filler(),
            text(" [Esc Back] ") | bold,
        });

        return vbox({
            status_bar,
            separator(),
            vbox(std::move(rows)) | hcenter,
            separator(),
            bottom_bar
        }) | border | center;
    }

    Component renderReplayPage() {
        auto component = std::make_shared<InteractiveBoard>();
        component->render_logic = [this] { return renderReplayGrid(); };
        component->event_logic  = [this](const Event& event) {
            const int max_step = static_cast<int>(replay_history_.size());

            if (event == Event::Special("replay_tick")) {
                if (replay_auto_ && replay_step_ < max_step) {
                    ++replay_step_;
                    if (replay_step_ == max_step) {
                        stopReplayAuto();
                    }
                }
                return true;
            }
            if (event == Event::Escape) {
                stopReplayAuto();
                result_selected_ = 0;
                active_index = kResultTab;
                return true;
            }
            if (event == Event::ArrowLeft) {
                if (replay_step_ > 0) --replay_step_;
                return true;
            }
            if (event == Event::ArrowRight) {
                if (replay_step_ < max_step) ++replay_step_;
                return true;
            }
            if (event == Event::Character('r') || event == Event::Character('R')) {
                replay_step_ = 0;
                stopReplayAuto();
                startReplayAuto();
                return true;
            }
            if (event == Event::Character(' ')) {
                if (replay_auto_) {
                    stopReplayAuto();
                } else {
                    startReplayAuto();
                }
                return true;
            }
            return false;
        };

        return component;
    }

    void refreshSavesList() {
        namespace fs = std::filesystem;
        save_files_.clear();
        load_selected_ = 0;
        load_directory_error_ = false;
        clearLocalStatus();

        const fs::path saves_path(session.saves_dir());
        std::error_code ec;
        const bool exists = fs::exists(saves_path, ec);
        if (ec) {
            load_directory_error_ = true;
            setLocalStatus("Unable to inspect save directory: " + saves_path.string() + " (" + ec.message() + ")");
            return;
        }
        if (!exists) {
            return;
        }

        const bool is_directory = fs::is_directory(saves_path, ec);
        if (ec) {
            load_directory_error_ = true;
            setLocalStatus("Unable to inspect save directory: " + saves_path.string() + " (" + ec.message() + ")");
            return;
        }
        if (!is_directory) {
            load_directory_error_ = true;
            setLocalStatus("Save path is not a directory: " + saves_path.string());
            return;
        }

        for (fs::directory_iterator it(saves_path, ec), end; !ec && it != end; it.increment(ec)) {
            if (const auto& entry = *it; entry.path().extension() == ".gomoku") {
                save_files_.push_back(entry.path().string());
            }
        }
        if (ec) {
            save_files_.clear();
            load_directory_error_ = true;
            setLocalStatus("Unable to read save directory: " + saves_path.string() + " (" + ec.message() + ")");
            return;
        }

        std::ranges::sort(save_files_, std::greater<>());
    }

    Component renderLoadGamePage() {
        auto component = std::make_shared<InteractiveBoard>();

        component->render_logic = [this] {
            Elements content = {
                text("Load Game") | hcenter | bold | color(Color::Cyan),
                separator()
            };

            if (save_files_.empty()) {
                if (!load_directory_error_) {
                    content.push_back(text("No saves found in: " + session.saves_dir()) | dim | hcenter);
                }
                if (!local_status_text_.empty()) {
                    content.push_back(separator());
                    content.push_back(text(local_status_text_) | color(local_status_color_) | hcenter);
                }
                content.push_back(separator());
                content.push_back(text("Esc Back to menu") | dim | hcenter);
                return vbox(std::move(content)) | border | center;
            }

            Elements items;
            for (int i = 0; i < static_cast<int>(save_files_.size()); ++i) {
                namespace fs = std::filesystem;
                auto label = fs::path(save_files_[i]).stem().string();
                auto item = text("  " + label + "  ");
                if (i == load_selected_) {
                    item |= inverted;
                }
                items.push_back(item);
            }

            content.push_back(vbox(std::move(items)));
            if (!local_status_text_.empty()) {
                content.push_back(separator());
                content.push_back(text(local_status_text_) | color(local_status_color_) | hcenter);
            }
            content.push_back(separator());
            content.push_back(text("Arrow keys Select  Enter Load  Esc Back") | dim | hcenter);
            return vbox(std::move(content)) | border | center;
        };

        component->event_logic = [this](const Event& event) {
            if (event == Event::Escape) {
                active_index = kMenuTab;
                return true;
            }
            if (event == Event::ArrowUp) {
                if (!save_files_.empty()) {
                    load_selected_ = std::max(0, load_selected_ - 1);
                }
                return true;
            }
            if (event == Event::ArrowDown) {
                if (!save_files_.empty()) {
                    load_selected_ = std::min(static_cast<int>(save_files_.size()) - 1, load_selected_ + 1);
                }
                return true;
            }
            if (event == Event::Return && !save_files_.empty()) {
                if (session.deserialize(save_files_[load_selected_])) {
                    shutdownRemoteSession();
                    clearLocalStatus();
                    syncCursorToSession();
                    active_index = (session.mode() == gomoku::SessionMode::PVE) ? kPveTab : kPvpTab;
                    if (session.status() != gomoku::GameStatus::PLAYING) {
                        result_selected_ = 0;
                        active_index = kResultTab;
                    }
                } else {
                    setLocalStatus("Load failed: " + session.last_persistence_error());
                }
                return true;
            }
            return false;
        };

        return component;
    }

    bool trySaveSessionWithFeedback() {
        if (const std::string save_path = session.serialize(); !save_path.empty()) {
            clearLocalStatus();
            setStatusMsg("Game saved!", 1500);
            return true;
        }

        const std::string detail = session.last_persistence_error().empty()
            ? "Unknown persistence error."
            : session.last_persistence_error();
        const std::string message = "Save failed: " + detail;
        setLocalStatus(message);
        setStatusMsg(message, 3000);
        return false;
    }

    bool handleSaveMenuEvent(const Event& event) {
        constexpr int kSaveMenuItems = 4;

        if (event == Event::ArrowUp) {
            save_menu_selected_ = (save_menu_selected_ + kSaveMenuItems - 1) % kSaveMenuItems;
            return true;
        }
        if (event == Event::ArrowDown) {
            save_menu_selected_ = (save_menu_selected_ + 1) % kSaveMenuItems;
            return true;
        }
        if (event == Event::Escape || event == Event::Character('e') || event == Event::Character('E')) {
            show_save_menu_ = false;
            return true;
        }
        // Hotkey shortcuts: S for Save (only), L for Leave
        if (event == Event::Character('s') || event == Event::Character('S')) {
            // S key directly executes Save
            if (trySaveSessionWithFeedback()) {
                show_save_menu_ = false;
            }
            return true;
        }
        if (event == Event::Character('l') || event == Event::Character('L')) {
            // Leave
            show_save_menu_ = false;
            backToMenu();
            return true;
        }
        if (event == Event::Return) {
            switch (save_menu_selected_) {
                case 0: // Save & Leave
                    if (trySaveSessionWithFeedback()) {
                        show_save_menu_ = false;
                        backToMenu();
                    }
                    break;
                case 1: // Save
                    if (trySaveSessionWithFeedback()) {
                        show_save_menu_ = false;
                    }
                    break;
                case 2: // Leave
                    show_save_menu_ = false;
                    backToMenu();
                    break;
                case 3: // Cancel
                    show_save_menu_ = false;
                    break;
                default: ;
            }
            return true;
        }
        return true; // absorb all keys while submenu is open
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
            result_selected_ = 0;
            active_index = kResultTab;
        }
    }

    bool runAiTurn() {
        if (!session.ai_move()) {
            return false;
        }

        syncCursorToSession();
        tryMoveToResult();
        return true;
    }

    void setStatusMsg(const std::string& msg, const int ms = 1500) {
        status_msg_ = msg;
        status_msg_until_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    }

    [[nodiscard]] std::string activeStatusMsg() const {
        if (!status_msg_.empty() && std::chrono::steady_clock::now() < status_msg_until_) {
            return status_msg_;
        }
        return {};
    }

    Component renderFrontPage() {
        auto menu = Menu(&menu_entries, &menu_selected);

        auto component = Renderer(menu, [menu] {
            return vbox({
                text("=== Gomoku ===") | hcenter | bold | color(Color::Cyan),
                separator(),
                menu->Render() | hcenter,
                separator(),
                text(" use arrow keys to move, ENTER to select ") | dim | hcenter
            }) | border | center;
        });

        component |= CatchEvent([this, menu](const Event& event) {
            const int menu_count = static_cast<int>(menu_entries.size());
            if (menu_count > 0 && event == Event::ArrowUp && menu_selected == 0) {
                return menu->OnEvent(Event::End);
            }
            if (menu_count > 0 && event == Event::ArrowDown && menu_selected == menu_count - 1) {
                return menu->OnEvent(Event::Home);
            }

            if (event != Event::Return) {
                return false;
            }

            if (menu_selected == 0) {
                pending_mode_ = gomoku::SessionMode::PVP;
                settings_focus_ = 1;
                previous_tab = kMenuTab;
                active_index = kSetupTab;
                return true;
            }
            if (menu_selected == 1) {
                pending_mode_ = gomoku::SessionMode::PVE;
                settings_focus_ = 1;
                previous_tab = kMenuTab;
                active_index = kSetupTab;
                return true;
            }
            if (menu_selected == 2) {
                openRemoteHostPage();
                return true;
            }
            if (menu_selected == 3) {
                openRemoteJoinPage();
                return true;
            }
            if (menu_selected == 4) {
                pending_mode_.reset();
                settings_focus_ = 0;
                previous_tab = kMenuTab;
                active_index = kSetupTab;
                return true;
            }
            if (menu_selected == 5) {
                refreshSavesList();
                active_index = kLoadTab;
                return true;
            }
            if (menu_selected == 6) {
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
            if (event == Event::Special("timer_tick")) {
                if (settings_timer_enabled && !show_save_menu_ &&
                    session.status() == gomoku::GameStatus::PLAYING) {
                    if (timer_remaining_ > 0) {
                        --timer_remaining_;
                    }
                    if (timer_remaining_ == 0) {
                        handleTimerTimeout(has_ai);
                    }
                }
                return true;
            }

            if (show_save_menu_) {
                return handleSaveMenuEventWrapper(event);
            }

            if (handleCursorMove(event)) {
                return true;
            }

            if (remote_mode_ && (event == Event::Character('q') || event == Event::Character('Q'))) {
                backToMenu();
                return true;
            }

            if (event == Event::Character('l') || event == Event::Character('L')) {
                if (remote_mode_) {
                    updateRemoteStatus("Save/Leave is disabled during remote games. Press Q to disconnect.");
                    return true;
                }
                show_save_menu_ = true;
                save_menu_selected_ = 0;
                return true;
            }

            if (event == Event::Character('s') || event == Event::Character('S')) {
                pending_mode_.reset();
                settings_focus_ = 0;
                previous_tab = active_index;
                active_index = kSetupTab;
                return true;
            }

            if (event == Event::Character('u') || event == Event::Character('U')) {
                if (!settings_undo_enabled) {
                    setStatusMsg("Undo is disabled (enable in Setup)", 2000);
                    return true;
                }

                if (remote_mode_) {
                    if (!network.requestUndo()) {
                        updateRemoteStatus("Undo failed: " + network.lastError());
                        return true;
                    }
                    syncCursorToSession();
                    updateRemoteStatus();
                    if (session.status() == gomoku::GameStatus::PLAYING && active_index == kResultTab) {
                        active_index = kPvpTab;
                    }
                    return true;
                }

                if (!session.undo()) {
                    setStatusMsg("No moves to undo", 1500);
                } else {
                    setStatusMsg("Move undone", 1000);
                    syncCursorToSession();
                    if (settings_timer_enabled && session.status() == gomoku::GameStatus::PLAYING) {
                        startTimer();
                    }
                }
                return true;
            }

            if (event != Event::Return && event != Event::Character(' ')) {
                return false;
            }

            if (remote_mode_) {
                if (!network.isConnected()) {
                    updateRemoteStatus("Remote peer is not connected.");
                    return true;
                }
                if (!network.isLocalTurn()) {
                    updateRemoteStatus("Waiting for the remote player's move.");
                    return true;
                }
                if (!network.sendLocalMove(current_x, current_y)) {
                    updateRemoteStatus("Move failed: " + network.lastError());
                    return true;
                }
                clearLocalStatus();
                syncCursorToSession();
                updateRemoteStatus();
                tryMoveToResult();
                return true;
            }

            if (!session.human_move(current_x, current_y)) {
                setStatusMsg("Cell already occupied", 1500);
                return true;
            }

            clearLocalStatus();
            consecutive_timeouts_ = 0;
            stopTimer();
            tryMoveToResult();
            if (has_ai && session.status() == gomoku::GameStatus::PLAYING) {
                runAiTurn();
            }
            if (settings_timer_enabled && session.status() == gomoku::GameStatus::PLAYING) {
                startTimer();
            }

            return true;
        };

        return component;
    }

    bool handleSaveMenuEventWrapper(const Event& event) {
        if (show_save_menu_) {
            return handleSaveMenuEvent(event);
        }
        return false;
    }

    Component renderSetupPage() {
        CheckboxOption cb_opt;
        cb_opt.transform = [](const EntryState& state) {
            auto prefix = text(state.state ? "[x] " : "[ ] ");
            auto label = text(state.label);
            if (state.focused) {
                prefix |= inverted;
                label |= inverted;
            }
            return hbox({prefix, label}) | flex;
        };

        auto undo_checkbox  = Checkbox("Undo",       &settings_undo_enabled,  cb_opt);
        auto timer_checkbox = Checkbox("Time Limit", &settings_timer_enabled, cb_opt);

        InputOption inp_opt = InputOption::Default();
        inp_opt.on_change = [this] {
            std::string filtered;
            for (const char c : settings_timer_seconds_str_) {
                if (c >= '0' && c <= '9') filtered += c;
            }
            if (filtered.size() > 3) filtered.resize(3);
            settings_timer_seconds_str_ = filtered;
        };
        auto timer_input = Input(&settings_timer_seconds_str_, "secs", inp_opt);

        ButtonOption btn_opt;
        btn_opt.transform = [](const EntryState& state) {
            auto element = text(state.label) | flex;
            if (state.focused) {
                element |= inverted;
            }
            return element;
        };
        auto back_button = Button("Ok", [this] {
            if (pending_mode_.has_value()) {
                const auto mode = *pending_mode_;
                pending_mode_.reset();
                startGame(mode);
            } else {
                active_index = previous_tab;
            }
        }, btn_opt);

        auto checkboxes = Container::Vertical({ undo_checkbox, timer_checkbox, timer_input });
        auto container  = Container::Vertical({ checkboxes, back_button }, &settings_focus_);

        return Renderer(container, [this, undo_checkbox, timer_checkbox, timer_input, back_button] {
            auto timer_row = hbox({
                text("    "),
                timer_input->Render() | size(WIDTH, EQUAL, 4),
                text(" sec")
            });
            if (!settings_timer_enabled) timer_row = timer_row | dim;

            auto how_to_play = vbox({
                text("\u2500\u2500 How to Play \u2500\u2500") | bold | color(Color::Cyan),
                text("\u00b7 Get 5 in a row to win") | dim,
                text("  (horizontal / vertical / diagonal)") | dim,
                text("\u00b7 Black \u25cb goes first, alternate") | dim,
                text(""),
                text("\u2191\u2193\u2190\u2192 Move   Enter/Space Place") | dim,
                text("U Undo   S Setup   L Leave/Save") | dim,
            });

            auto setup_box = vbox({
                text("\u2500\u2500 Setup \u2500\u2500") | bold | color(Color::Cyan),
                undo_checkbox->Render(),
                hbox({ timer_checkbox->Render(), timer_row }),
            });

            const auto settings_box = vbox({
                text("Briefing") | hcenter | bold | color(Color::Cyan),
                separator(),
                how_to_play,
                separator(),
                setup_box,
                separator(),
                back_button->Render()
            }) | border | center;

            return dbox({ renderGrid(), settings_box | clear_under | center });
        });
    }

    Component renderEndPage() {
        auto component = std::make_shared<InteractiveBoard>();

        component->render_logic = [this] {
            const std::vector<std::string> result_labels = {
                "Back to menu", "Play again", "View Replay"
            };
            Elements items;
            for (int i = 0; i < static_cast<int>(result_labels.size()); ++i) {
                auto item = text("  " + result_labels[i] + "  ");
                if (i == result_selected_) item |= inverted;
                items.push_back(std::move(item));
            }

            Elements box_content = {
                text("Game Over") | hcenter | bold | color(Color::Red),
                separator(),
                text("Result: " + gameResultText(session.status())) | hcenter | color(Color::Cyan),
                separator()
            };
            if (remote_mode_) {
                box_content.push_back(text(remote_status_text_) |
                    color(remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) | hcenter);
                box_content.push_back(separator());
            }
            box_content.push_back(vbox(std::move(items)));
            box_content.push_back(separator());
            box_content.push_back(text(" \u2191\u2193 Move  Enter Confirm ") | dim | hcenter);

            const auto result_box = vbox(std::move(box_content)) | border | center;
            return dbox({ renderGrid(), result_box | clear_under | center });
        };

        component->event_logic = [this](const Event& event) {
            constexpr int kNumOptions = 3;
            if (event == Event::ArrowUp) {
                result_selected_ = (result_selected_ - 1 + kNumOptions) % kNumOptions;
                return true;
            }
            if (event == Event::ArrowDown) {
                result_selected_ = (result_selected_ + 1) % kNumOptions;
                return true;
            }
            if (event == Event::Return) {
                if (result_selected_ == 0) {
                    backToMenu();
                } else if (result_selected_ == 1) {
                    if (remote_mode_) {
                        if (!network.isConnected()) { updateRemoteStatus("Remote peer is not connected."); return true; }
                        if (!network.requestReset()) { updateRemoteStatus("Reset failed: " + network.lastError()); return true; }
                        syncCursorToSession();
                        updateRemoteStatus();
                        active_index = kPvpTab;
                    } else {
                        startGame(session.mode());
                    }
                } else {
                    enterReplay();
                }
                return true;
            }
            return false;
        };

        return component;
    }

    Component renderRemoteHostPage() {
        auto port_input = Input(&remote_port_input_, "7777");
        auto start_button = Button("Start Hosting", [this] { beginHosting(); });
        auto back_button = Button("Back", [this] { backToMenu(); });

        const auto container = Container::Vertical({port_input, start_button, back_button});
        auto component = Renderer(container, [port_input, start_button, back_button, this] {
            return vbox({
                text("Host Remote Game") | hcenter | bold | color(Color::Cyan),
                separator(),
                text("Port") | bold,
                port_input->Render() | border,
                separator(),
                text(remote_status_text_) |
                    color(remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) |
                    hcenter,
                separator(),
                start_button->Render(),
                back_button->Render()
            }) | border | center;
        });

        component |= CatchEvent([this](const Event& event) {
            if (event == Event::Escape) {
                backToMenu();
                return true;
            }
            return false;
        });

        return component;
    }

    Component renderRemoteJoinPage() {
        auto host_input = Input(&remote_host_input_, "127.0.0.1");
        auto port_input = Input(&remote_port_input_, "7777");
        auto connect_button = Button("Connect", [this] { beginJoin(); });
        auto back_button = Button("Back", [this] { backToMenu(); });

        const auto container = Container::Vertical({host_input, port_input, connect_button, back_button});
        auto component = Renderer(container, [host_input, port_input, connect_button, back_button, this] {
            return vbox({
                text("Join Remote Game") | hcenter | bold | color(Color::Cyan),
                separator(),
                text("Host") | bold,
                host_input->Render() | border,
                separator(),
                text("Port") | bold,
                port_input->Render() | border,
                separator(),
                text(remote_status_text_) |
                    color(remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) |
                    hcenter,
                separator(),
                connect_button->Render(),
                back_button->Render()
            }) | border | center;
        });

        component |= CatchEvent([this](const Event& event) {
            if (event == Event::Escape) {
                backToMenu();
                return true;
            }
            return false;
        });

        return component;
    }

    Component renderRemoteWaitPage() {
        auto cancel_button = Button("Cancel", [this] { backToMenu(); });
        auto component = Renderer(cancel_button, [cancel_button, this] {
            return vbox({
                text("Waiting For Remote Player") | hcenter | bold | color(Color::Cyan),
                separator(),
                text(remote_status_text_) |
                    color(remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) |
                    hcenter,
                separator(),
                text("Share this address with the other player:") | hcenter,
                text(network.localEndpoint().empty() ? ("port " + remote_port_input_) : network.localEndpoint()) |
                    bold |
                    hcenter,
                separator(),
                cancel_button->Render() | hcenter,
                text("Esc Cancel") | dim | hcenter
            }) | border | center;
        });

        component |= CatchEvent([this](const Event& event) {
            if (event == Event::Escape) {
                backToMenu();
                return true;
            }
            return false;
        });

        return component;
    }

    [[nodiscard]] Element renderGrid() const {
        const auto& board = session.board();

        // Top info bar: Mode (left) | Timer status (right); overridden by remote/local status
        Element info_bar;
        if (remote_mode_) {
            info_bar = text(remote_status_text_) | hcenter |
                       color(remoteStatusColor(network.isConnected(), remote_waiting_for_peer_));
        } else if (!local_status_text_.empty()) {
            info_bar = text(local_status_text_) | hcenter | color(local_status_color_);
        } else {
            const std::string mode_str = "Mode: " + std::string(active_index == kPveTab ? "PvE" : "PvP");
            Element timer_status;
            if (!settings_timer_enabled) {
                timer_status = text("Timer Off") | dim;
            } else {
                const bool urgent = (timer_remaining_ <= 5);
                timer_status = hbox({
                    text("Time remaining: "),
                    text(std::to_string(timer_remaining_) + "s") | bold |
                        color(urgent ? Color::Red : Color::Yellow)
                });
            }
            info_bar = hbox({ text(" " + mode_str), filler(), timer_status, text(" ") });
        }

        // Current player bar
        const bool is_black_turn = board.getCurrentPlayer() == gomoku::Stone::BLACK;
        auto player_bar = text(is_black_turn ? "Current player: Black" : "Current player: White") |
                          bold | color(is_black_turn ? Color::Red : Color::White) | hcenter;

        // AI status bar
        auto ai_bar = text(session.ai_status_text()) | hcenter;
        if (active_index == kPveTab) {
            ai_bar |= session.ai_used_fallback() ? color(Color::Yellow) : color(Color::Green);
        } else {
            ai_bar |= dim;
        }

        Elements rows;
        const int size = board.getSize();
        const bool in_game = (active_index == kPvpTab || active_index == kPveTab);
        const auto last_mv = session.last_move();
        for (int y = 0; y < size; ++y) {
            Elements columns;
            for (int x = 0; x < size; ++x) {
                const auto stone = board.getStone(x, y);
                auto cell = stoneCellElement(stone);

                const bool is_last_move = last_mv.has_value() && last_mv->first == x && last_mv->second == y;

                if (x == current_x && y == current_y && !show_save_menu_ && in_game) {
                    cell |= bgcolor(Color::Blue);
                } else if (is_last_move && in_game) {
                    if (stone == gomoku::Stone::WHITE) {
                        cell |= bgcolor(Color::GrayDark) | underlined;
                    } else {
                        cell |= bgcolor(Color::GrayDark);
                    }
                }

                columns.push_back(std::move(cell));
            }
            rows.push_back(hbox(std::move(columns)));
        }

        // Hint line: urgent timer > timed status message > default controls
        // Use fixed width to prevent layout shift when content changes
        Element hint_line;
        std::string hint_text;
        bool hint_is_status = false;
        
        if (settings_timer_enabled && timer_remaining_ > 0 && timer_remaining_ <= 3) {
            hint_text = "  Hurry up! You'll get skipped!  ";
            hint_is_status = true;
        } else {
            if (const auto msg = activeStatusMsg(); !msg.empty()) {
                hint_text = "  " + msg + "  ";
                hint_is_status = true;
            } else {
                hint_text = "↑↓←→ Move  |  [Enter]/[Space] Place   ";
            }
        }
        
        // Pad or truncate to fixed width to prevent layout shift
        if (constexpr int kHintLineWidth = 45; static_cast<int>(hint_text.length()) < kHintLineWidth) {
            const int padding = kHintLineWidth - static_cast<int>(hint_text.length());
            const int left_pad = padding / 2;
            const int right_pad = padding - left_pad;
            hint_text = std::string(left_pad, ' ') + hint_text + std::string(right_pad, ' ');
        }
        
        hint_line = text(hint_text);
        if (hint_is_status) {
            hint_line |= color(Color::Yellow);
        } else {
            hint_line |= dim;
        }
        hint_line |= hcenter;

        auto bottom_bar = remote_mode_
            ? hbox({
                text(" [S]Setup ") | bold,
                filler(),
                text(" [U]Undo ") | (settings_undo_enabled ? bold : dim),
                filler(),
                text(" [Q]Disconnect ") | bold
            })
            : hbox({
                text(" [S]Setup ") | bold,
                filler(),
                text(" [U]Undo ") | (settings_undo_enabled ? bold : dim),
                filler(),
                text(" [L]Leave/Save ") | bold
            });

        auto board_element = vbox({
            info_bar,
            player_bar,
            ai_bar,
            separator(),
            vbox(std::move(rows)) | hcenter,
            separator(),
            hint_line,
            bottom_bar
        }) | border | center;

        if (!show_save_menu_) {
            return board_element;
        }

        const std::vector<std::string> save_menu_labels = {
            "Save & Leave",
            "[S] Save",
            "[L] Leave",
            "[ESC] Back"
        };

        Elements menu_items;
        for (int i = 0; i < static_cast<int>(save_menu_labels.size()); ++i) {
            auto item = text("  " + save_menu_labels[i] + "  ");
            if (i == save_menu_selected_) {
                item |= inverted;
            }
            menu_items.push_back(item);
        }

        const auto overlay = vbox({
            text("Leave Game?") | hcenter | bold | color(Color::Cyan),
            separator(),
            vbox(std::move(menu_items)),
            separator(),
            text(" ↑↓ Move  Enter Confirm ") | dim | hcenter
        }) | border | bgcolor(Color::Black) | center;

        return dbox({ board_element, overlay | clear_under | center });
    }

    gomoku::GameSession& session;
    gomoku::webConnect network;
    std::unique_ptr<ScreenState> screen_state;
    std::thread ui_tick_;
    std::atomic<bool> ui_tick_stop_{false};

    std::vector<std::string> menu_entries = {
        "Start Game (PvP)",
        "Start Game (PvE)",
        "Host Remote Game",
        "Join Remote Game",
        "Setup",
        "Load Game",
        "Exit"
    };

    int menu_selected = 0;
    int active_index = kMenuTab;
    int previous_tab = kMenuTab;
    int current_x = 0;
    int current_y = 0;

    // Setup state (wired up in Phase C / Phase F)
    bool settings_undo_enabled  = true;
    int  settings_focus_        = 0;
    std::optional<gomoku::SessionMode> pending_mode_;

    // Move timer state
    bool                settings_timer_enabled      = false;
    std::string         settings_timer_seconds_str_ = "20";
    int                 timer_remaining_            = 0;
    int                 consecutive_timeouts_       = 0;
    std::atomic<bool>   timer_stop_{true};
    std::thread         timer_thread_;

    // Save/Leave submenu state
    bool show_save_menu_     = false;
    int  save_menu_selected_ = 0;
    int  result_selected_    = 0;

    // Status hint message (transient, time-based)
    std::string status_msg_;
    std::chrono::steady_clock::time_point status_msg_until_{};

    std::vector<std::string> save_files_;
    int load_selected_ = 0;

    bool remote_mode_ = false;
    bool remote_waiting_for_peer_ = false;
    std::string remote_status_text_;
    std::string local_status_text_;
    Color local_status_color_ = Color::GrayDark;
    bool load_directory_error_ = false;
    std::string remote_host_input_ = "127.0.0.1";
    std::string remote_port_input_ = "7777";

    std::vector<std::pair<int,int>> replay_history_;
    int                replay_step_ = 0;
    bool               replay_auto_ = false;
    std::atomic<bool>  replay_auto_stop_{true};
    std::thread        replay_auto_thread_;
};

Controller::Controller(gomoku::GameSession& session)
    : impl_(std::make_unique<Impl>(session)) {}

Controller::~Controller() = default;

void Controller::Start() const {
    impl_->Start();
}

} // namespace UI
