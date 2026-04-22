#ifndef GOMOKU_UI_CONTROLLER_INTERNAL_H
#define GOMOKU_UI_CONTROLLER_INTERNAL_H

#include "../../include/gomoku/net/webConnect.h"
#include "../../include/gomoku/ui/ui_controller.h"

#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace UI::detail {

inline constexpr int kMenuTab = 0;
inline constexpr int kPvpTab = 1;
inline constexpr int kPveTab = 2;
inline constexpr int kResultTab = 3;
inline constexpr int kSetupTab = 4;
inline constexpr int kLoadTab = 5;
inline constexpr int kRemoteHostTab = 6;
inline constexpr int kRemoteJoinTab = 7;
inline constexpr int kRemoteWaitTab = 8;
inline constexpr int kReplayTab = 9;

inline constexpr std::chrono::milliseconds kUiTickInterval{100};

class InteractiveBoard final : public ftxui::ComponentBase {
public:
    std::function<ftxui::Element()> render_logic;
    std::function<bool(ftxui::Event)> event_logic;

    ftxui::Element OnRender() override {
        return render_logic();
    }

    [[nodiscard]] bool Focusable() const override {
        return true;
    }

    bool OnEvent(const ftxui::Event event) override {
        return event_logic(event);
    }
};

std::string gameResultText(gomoku::GameStatus status);
std::string stoneText(gomoku::Stone stone);
ftxui::Element framedStoneCell(const std::string& symbol, ftxui::Color symbol_color);
ftxui::Element stoneCellElement(gomoku::Stone stone);
std::string trimCopy(std::string value);
std::optional<std::uint16_t> parsePortValue(const std::string& raw_value);
ftxui::Color remoteStatusColor(bool connected, bool waiting);

} // namespace UI::detail

namespace UI {

struct Controller::Impl {
    struct ScreenState {
        ftxui::ScreenInteractive screen = ftxui::ScreenInteractive::Fullscreen();
    };

    explicit Impl(gomoku::GameSession& game_session);
    ~Impl();

    void Start();

    [[nodiscard]] int boardLimit() const;
    void centerCursor();
    void syncCursorToSession();
    void updateRemoteStatus(const std::string& override_text = {});
    void shutdownRemoteSession();
    void clearLocalStatus();
    void setLocalStatus(std::string message, ftxui::Color color = ftxui::Color::Red);
    [[nodiscard]] gomoku::SessionRules currentUiRules() const;
    void syncSettingsFromSession();
    [[nodiscard]] bool shouldRunTimer() const;
    void syncTimerToSessionState();
    [[nodiscard]] std::vector<std::string> shareableHostEndpoints() const;
    [[nodiscard]] std::string waitingEndpointSummary() const;
    bool commitSetupChanges();
    void resetTimeoutState();
    void startGame(gomoku::SessionMode next_mode);
    void backToMenu();
    void openRemoteHostPage();
    void openRemoteJoinPage();
    bool beginHosting();
    bool beginJoin();
    bool handleNetworkTick();

    void stopReplayAuto();
    void startReplayAuto();
    void stopTimer();
    void startTimer();
    void handleTimerTimeout(bool has_ai);
    void enterReplay();
    [[nodiscard]] ftxui::Element renderReplayGrid() const;
    ftxui::Component renderReplayPage();
    void refreshSavesList();
    ftxui::Component renderLoadGamePage();
    bool trySaveSessionWithFeedback();
    bool handleSaveMenuEvent(const ftxui::Event& event);
    bool handleCursorMove(const ftxui::Event& event);
    void tryMoveToResult();
    bool runAiTurn();
    void setStatusMsg(const std::string& msg, int ms = 1500);
    [[nodiscard]] std::string activeStatusMsg() const;
    ftxui::Component renderFrontPage();
    ftxui::Component renderGameBoard(bool has_ai);
    bool handleSaveMenuEventWrapper(const ftxui::Event& event);
    ftxui::Component renderSetupPage();
    ftxui::Component renderEndPage();
    ftxui::Component renderRemoteHostPage();
    ftxui::Component renderRemoteJoinPage();
    ftxui::Component renderRemoteWaitPage();
    [[nodiscard]] ftxui::Element renderGrid() const;

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
    int active_index = detail::kMenuTab;
    int previous_tab = detail::kMenuTab;
    int current_x = 0;
    int current_y = 0;

    bool settings_undo_enabled = true;
    int settings_focus_ = 0;
    std::optional<gomoku::SessionMode> pending_mode_;

    bool settings_timer_enabled = false;
    std::string settings_timer_seconds_str_ = "20";
    int timer_remaining_ = 0;
    int consecutive_timeouts_ = 0;
    std::atomic<bool> timer_stop_{true};
    std::thread timer_thread_;

    bool show_save_menu_ = false;
    int save_menu_selected_ = 0;
    int result_selected_ = 0;

    std::string status_msg_;
    std::chrono::steady_clock::time_point status_msg_until_{};

    std::vector<std::string> save_files_;
    int load_selected_ = 0;

    bool remote_mode_ = false;
    bool remote_waiting_for_peer_ = false;
    std::string remote_status_text_;
    std::string local_status_text_;
    ftxui::Color local_status_color_ = ftxui::Color::GrayDark;
    bool load_directory_error_ = false;
    std::string remote_host_input_ = "127.0.0.1";
    std::string remote_port_input_ = "7777";

    std::vector<std::pair<int, int>> replay_history_;
    int replay_step_ = 0;
    bool replay_auto_ = false;
    std::atomic<bool> replay_auto_stop_{true};
    std::thread replay_auto_thread_;
};

} // namespace UI

#endif // GOMOKU_UI_CONTROLLER_INTERNAL_H
