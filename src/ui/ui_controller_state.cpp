#include "../../include/gomoku/ui/ui_controller_internal.h"

#include "../../include/gomoku/audio/voice.h"

namespace UI {
using namespace ftxui;

int Controller::Impl::boardLimit() const {
    return std::max(0, session.board().getSize() - 1);
}

void Controller::Impl::centerCursor() {
    const int center = session.board().getSize() / 2;
    current_x = center;
    current_y = center;
}

void Controller::Impl::syncCursorToSession() {
    if (const auto move = session.last_move()) {
        current_x = move->first;
        current_y = move->second;
        return;
    }
    centerCursor();
}

void Controller::Impl::updateRemoteStatus(const std::string& override_text) {
    if (!override_text.empty()) {
        remote_status_text_ = override_text;
        return;
    }

    if (remote_waiting_for_peer_) {
        remote_status_text_ = "Waiting for peer on " + waitingEndpointSummary();
        return;
    }

    if (network.isConnected()) {
        remote_status_text_ = "Remote connected | You: " + detail::stoneText(network.localStone());
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

void Controller::Impl::shutdownRemoteSession() {
    network.disconnect();
    remote_mode_ = false;
    remote_waiting_for_peer_ = false;
    remote_status_text_.clear();
    show_save_menu_ = false;
}

void Controller::Impl::clearLocalStatus() {
    local_status_text_.clear();
    local_status_color_ = Color::GrayDark;
}

void Controller::Impl::setLocalStatus(std::string message, const Color color) {
    local_status_text_ = std::move(message);
    local_status_color_ = color;
}

gomoku::SessionRules Controller::Impl::currentUiRules() const {
    int seconds = 20;
    try {
        seconds = std::stoi(settings_timer_seconds_str_);
    } catch (...) {
        seconds = 20;
    }
    if (seconds <= 0) {
        seconds = 20;
    }

    return gomoku::SessionRules{
        .undo_enabled = settings_undo_enabled,
        .timer_enabled = settings_timer_enabled,
        .timer_seconds = seconds,
    };
}

void Controller::Impl::syncSettingsFromSession() {
    const auto&[undo_enabled, timer_enabled, timer_seconds] = session.rules();
    settings_undo_enabled = undo_enabled;
    settings_timer_enabled = timer_enabled;
    settings_timer_seconds_str_ = std::to_string(timer_seconds);
}

bool Controller::Impl::shouldRunTimer() const {
    if (const auto& rules = session.rules(); !rules.timer_enabled || show_save_menu_ || session.status() != gomoku::GameStatus::PLAYING) {
        return false;
    }

    if (remote_mode_) {
        return !remote_waiting_for_peer_ && network.isConnected() && network.isLocalTurn();
    }

    return active_index == detail::kPvpTab || active_index == detail::kPveTab;
}

void Controller::Impl::syncTimerToSessionState() {
    if (!shouldRunTimer()) {
        stopTimer();
        return;
    }

    if (!timer_stop_.load()) {
        return;
    }

    startTimer();
}

std::vector<std::string> Controller::Impl::shareableHostEndpoints() const {
    auto endpoints = network.shareableEndpoints();
    if (!endpoints.empty()) {
        return endpoints;
    }

    if (const std::string endpoint = network.localEndpoint(); !endpoint.empty() &&
                                                              endpoint.find("0.0.0.0:") == std::string::npos &&
                                                              endpoint.find("[::]:") == std::string::npos) {
        endpoints.push_back(endpoint);
    }

    if (endpoints.empty() && !remote_port_input_.empty()) {
        endpoints.push_back("127.0.0.1:" + remote_port_input_);
    }
    return endpoints;
}

std::string Controller::Impl::waitingEndpointSummary() const {
    const auto endpoints = shareableHostEndpoints();
    if (endpoints.empty()) {
        return "port " + remote_port_input_;
    }
    if (endpoints.size() == 1) {
        return endpoints.front();
    }
    return endpoints.front() + " +" + std::to_string(endpoints.size() - 1) + " more";
}

bool Controller::Impl::commitSetupChanges() {
    const auto requested_rules = currentUiRules();
    const auto previous_rules = session.rules();

    if (!remote_mode_ || remote_waiting_for_peer_ || !network.isConnected()) {
        session.setRules(requested_rules);
        syncSettingsFromSession();
        stopTimer();
        syncTimerToSessionState();
        return true;
    }

    if (requested_rules == previous_rules) {
        syncSettingsFromSession();
        syncTimerToSessionState();
        return true;
    }

    if (!network.syncConfig(requested_rules)) {
        syncSettingsFromSession();
        setLocalStatus("Setup sync failed: " + network.lastError());
        setStatusMsg("Setup sync failed", 2500);
        return false;
    }

    clearLocalStatus();
    session.setRules(requested_rules);
    syncSettingsFromSession();
    stopTimer();
    syncTimerToSessionState();
    updateRemoteStatus("Remote setup synchronized.");
    setStatusMsg("Remote setup synchronized", 1500);
    return true;
}

void Controller::Impl::resetTimeoutState() {
    consecutive_timeouts_ = 0;
    stopTimer();
}

void Controller::Impl::startGame(const gomoku::SessionMode next_mode) {
    stopAiDelay();
    shutdownRemoteSession();
    clearLocalStatus();
    resetTimeoutState();
    session.start(next_mode);
    syncSettingsFromSession();
    centerCursor();

    if (next_mode == gomoku::SessionMode::PVE) {
        active_index = detail::kPveTab;
    } else {
        active_index = detail::kPvpTab;
    }

    syncTimerToSessionState();
}

void Controller::Impl::backToMenu() {
    stopAiDelay();
    stopTimer();
    consecutive_timeouts_ = 0;
    shutdownRemoteSession();
    clearLocalStatus();
    session.reset();
    syncSettingsFromSession();
    active_index = detail::kMenuTab;
    current_x = 0;
    current_y = 0;
}

void Controller::Impl::openRemoteHostPage() {
    shutdownRemoteSession();
    clearLocalStatus();
    session.start(gomoku::SessionMode::PVP);
    centerCursor();
    remote_status_text_ = "Choose a port and start hosting.";
    active_index = detail::kRemoteHostTab;
}

void Controller::Impl::openRemoteJoinPage() {
    shutdownRemoteSession();
    clearLocalStatus();
    session.start(gomoku::SessionMode::PVP);
    centerCursor();
    remote_status_text_ = "Enter the host IP and port.";
    active_index = detail::kRemoteJoinTab;
}

bool Controller::Impl::beginHosting() {
    clearLocalStatus();
    resetTimeoutState();
    const auto port = detail::parsePortValue(remote_port_input_);
    if (!port) {
        updateRemoteStatus("Port must be between 1 and 65535.");
        return false;
    }

    session.setRules(currentUiRules());
    syncSettingsFromSession();
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
    active_index = detail::kRemoteWaitTab;
    return true;
}

bool Controller::Impl::beginJoin() {
    clearLocalStatus();
    resetTimeoutState();
    const std::string host = detail::trimCopy(remote_host_input_);
    if (host.empty()) {
        updateRemoteStatus("Host IP or hostname cannot be empty.");
        return false;
    }

    const auto port = detail::parsePortValue(remote_port_input_);
    if (!port) {
        updateRemoteStatus("Port must be between 1 and 65535.");
        return false;
    }

    session.setRules(currentUiRules());
    syncSettingsFromSession();
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
    syncSettingsFromSession();
    syncCursorToSession();
    active_index = detail::kPvpTab;
    syncTimerToSessionState();
    handleNetworkTick();
    return true;
}

bool Controller::Impl::handleNetworkTick() {
    bool changed = false;

    if (remote_mode_ && remote_waiting_for_peer_) {
        if (network.waitForPeer(std::chrono::milliseconds{0})) {
            remote_waiting_for_peer_ = false;
            updateRemoteStatus();
            syncSettingsFromSession();
            syncCursorToSession();
            active_index = detail::kPvpTab;
            syncTimerToSessionState();
            changed = true;
        } else if (!network.lastError().empty()) {
            updateRemoteStatus();
            changed = true;
        }
    }

    if (remote_mode_ && !remote_waiting_for_peer_ && network.isConnected()) {
        const auto previous_move_count = session.move_history().size();
        const auto previous_status = session.status();
        const auto previous_rules = session.rules();
        const auto previous_turn = session.board().getCurrentPlayer();

        if (const bool handled = network.pump(std::chrono::milliseconds{0});
            handled ||
            previous_move_count != session.move_history().size() ||
            previous_status != session.status() ||
            previous_turn != session.board().getCurrentPlayer() ||
            previous_rules != session.rules()) {
            if (previous_rules != session.rules()) {
                syncSettingsFromSession();
                clearLocalStatus();
                stopTimer();
                setStatusMsg("Remote setup updated", 1500);
            }
            syncCursorToSession();
            updateRemoteStatus();
            syncTimerToSessionState();
            changed = true;
        } else if (!network.lastError().empty()) {
            updateRemoteStatus();
            changed = true;
        }
    } else if (remote_mode_ && !remote_waiting_for_peer_ && !network.isConnected() && !network.lastError().empty()) {
        stopTimer();
        updateRemoteStatus();
        changed = true;
    }

    if (remote_mode_) {
        if (session.status() != gomoku::GameStatus::PLAYING &&
            active_index != detail::kResultTab &&
            active_index != detail::kRemoteWaitTab) {
            stopTimer();
            tryMoveToResult();
            changed = true;
        }

        if (session.status() == gomoku::GameStatus::PLAYING && active_index == detail::kResultTab) {
            active_index = detail::kPvpTab;
            syncCursorToSession();
            syncTimerToSessionState();
            changed = true;
        }
    }

    return changed;
}

void Controller::Impl::stopReplayAuto() {
    replay_auto_ = false;
    replay_auto_stop_.store(true);
    if (replay_auto_thread_.joinable()) {
        replay_auto_thread_.join();
    }
}

void Controller::Impl::startReplayAuto() {
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

void Controller::Impl::stopAiDelay() {
    ai_delay_stop_.store(true);
    if (ai_delay_thread_.joinable()) {
        ai_delay_thread_.join();
    }
}

void Controller::Impl::startAiDelay() {
    stopAiDelay();
    ai_delay_stop_.store(false);
    ai_delay_thread_ = std::thread([this] {
        for (int i = 0; i < 10 && !ai_delay_stop_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!ai_delay_stop_.load()) {
            screen_state->screen.PostEvent(Event::Special("ai_move"));
        }
    });
}

void Controller::Impl::stopTimer() {
    timer_stop_.store(true);
    if (timer_thread_.joinable()) {
        timer_thread_.join();
    }
    timer_remaining_ = 0;
}

void Controller::Impl::startTimer() {
    stopTimer();
    const int secs = session.rules().timer_seconds;
    settings_timer_seconds_str_ = std::to_string(secs);
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

void Controller::Impl::handleTimerTimeout(const bool has_ai) {
    if (session.status() != gomoku::GameStatus::PLAYING) {
        return;
    }

    if (remote_mode_) {
        if (!network.requestSkipTurn()) {
            updateRemoteStatus("Time limit sync failed: " + network.lastError());
            syncTimerToSessionState();
            return;
        }
        updateRemoteStatus();
    } else {
        session.skipTurn();
    }

    ++consecutive_timeouts_;

    if (!remote_mode_ && has_ai && session.status() == gomoku::GameStatus::PLAYING) {
        startAiDelay();
    }

    if (consecutive_timeouts_ >= 3) {
        consecutive_timeouts_ = 0;
        stopTimer();
        show_save_menu_ = true;
        save_menu_selected_ = 0;
        return;
    }

    syncTimerToSessionState();
}

void Controller::Impl::enterReplay() {
    stopReplayAuto();
    replay_history_ = session.move_history();
    replay_step_ = static_cast<int>(replay_history_.size());
    active_index = detail::kReplayTab;
}

void Controller::Impl::refreshSavesList() {
    namespace fs = std::filesystem;

    save_file_infos_.clear();
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

    std::vector<std::string> save_files;
    for (fs::directory_iterator it(saves_path, ec), end; !ec && it != end; it.increment(ec)) {
        if (const auto& entry = *it; entry.path().extension() == ".gomoku") {
            save_files.push_back(entry.path().string());
        }
    }
    if (ec) {
        load_directory_error_ = true;
        setLocalStatus("Unable to read save directory: " + saves_path.string() + " (" + ec.message() + ")");
        return;
    }

    std::ranges::sort(save_files, std::greater<>());

    save_file_infos_.reserve(save_files.size());
    for (const auto& path : save_files) {
        save_file_infos_.push_back(gomoku::GameSession::peekSaveFile(path));
    }
}

bool Controller::Impl::trySaveSessionWithFeedback() {
    namespace fs = std::filesystem;

    if (const std::string save_path = session.serialize(); !save_path.empty()) {
        setStatusMsg("Saved " + fs::path(save_path).filename().string(), 2000);
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

bool Controller::Impl::handleSaveMenuEvent(const Event& event) {
    constexpr int kSaveMenuItems = 4;

    if (event == Event::ArrowUp) {
        save_menu_selected_ = (save_menu_selected_ + kSaveMenuItems - 1) % kSaveMenuItems;
        voice::menuMoveSound();
        return true;
    }
    if (event == Event::ArrowDown) {
        save_menu_selected_ = (save_menu_selected_ + 1) % kSaveMenuItems;
        voice::menuMoveSound();
        return true;
    }
    if (event == Event::Escape || event == Event::Character('e') || event == Event::Character('E')) {
        voice::selectedSound();
        show_save_menu_ = false;
        return true;
    }
    if (event == Event::Character('s') || event == Event::Character('S')) {
        voice::selectedSound();
        if (trySaveSessionWithFeedback()) {
            show_save_menu_ = false;
        }
        return true;
    }
    if (event == Event::Character('l') || event == Event::Character('L')) {
        voice::selectedSound();
        show_save_menu_ = false;
        backToMenu();
        return true;
    }
    if (event == Event::Return) {
        voice::selectedSound();
        switch (save_menu_selected_) {
            case 0:
                if (trySaveSessionWithFeedback()) {
                    show_save_menu_ = false;
                    backToMenu();
                }
                break;
            case 1:
                if (trySaveSessionWithFeedback()) {
                    show_save_menu_ = false;
                }
                break;
            case 2:
                show_save_menu_ = false;
                backToMenu();
                break;
            case 3:
                show_save_menu_ = false;
                break;
            default:
                break;
        }
        return true;
    }
    return true;
}

bool Controller::Impl::handleCursorMove(const Event& event) {
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

void Controller::Impl::tryMoveToResult() {
    const auto status = session.status();
    if (status == gomoku::GameStatus::PLAYING) {
        return;
    }

    if (status != gomoku::GameStatus::DRAW) {
        const auto winning_stone = (status == gomoku::GameStatus::BLACK_WIN)
            ? gomoku::Stone::BLACK
            : gomoku::Stone::WHITE;

        if (remote_mode_) {
            if (winning_stone == network.localStone()) {
                voice::victorySound();
            } else {
                voice::defeatSound();
            }
        } else if (active_index == detail::kPveTab) {
            if (winning_stone == gomoku::Stone::BLACK) {
                voice::victorySound();
            } else {
                voice::defeatSound();
            }
        } else {
            voice::victorySound();
        }
    }

    result_selected_ = 0;
    active_index = detail::kResultTab;
}

bool Controller::Impl::runAiTurn() {
    if (!session.ai_move()) {
        return false;
    }

    syncCursorToSession();
    tryMoveToResult();
    return true;
}

void Controller::Impl::setStatusMsg(const std::string& msg, const int ms) {
    status_msg_ = msg;
    status_msg_until_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
}

std::string Controller::Impl::activeStatusMsg() const {
    if (!status_msg_.empty() && std::chrono::steady_clock::now() < status_msg_until_) {
        return status_msg_;
    }
    return {};
}

bool Controller::Impl::handleSaveMenuEventWrapper(const Event& event) {
    if (show_save_menu_) {
        return handleSaveMenuEvent(event);
    }
    return false;
}

} // namespace UI
