#include "../../include/gomoku/ui/ui_controller_internal.h"
#include "../../include/gomoku/audio/voice.h"

namespace UI {
using namespace ftxui;

Element Controller::Impl::renderReplayGrid() const {
    const int board_size = session.board().getSize();
    const int max_step = static_cast<int>(replay_history_.size());

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
                const int step = it->second;
                const bool is_black = (step % 2 == 1);
                const auto s = std::to_string(step);
                std::string padded;
                if (s.size() == 1) {
                    padded = "  " + s + "  ";
                } else if (s.size() == 2) {
                    padded = " " + s + "  ";
                } else {
                    padded = s + "  ";
                }
                cell = text(padded)
                       | color(is_black ? Color::Red : Color::White)
                       | bold;
            } else {
                cell = text("  +  ") | color(Color::GrayDark);
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

Component Controller::Impl::renderReplayPage() {
    auto component = std::make_shared<detail::InteractiveBoard>();
    component->render_logic = [this] { return renderReplayGrid(); };
    component->event_logic = [this](const Event& event) {
        const int max_step = static_cast<int>(replay_history_.size());

        if (event == Event::Special("replay_tick")) {
            if (replay_auto_ && replay_step_ < max_step) {
                ++replay_step_;
                voice::placeStoneSound();
                if (replay_step_ == max_step) {
                    stopReplayAuto();
                }
            }
            return true;
        }
        if (event == Event::Escape) {
            voice::selectedSound();
            stopReplayAuto();
            result_selected_ = 0;
            active_index = detail::kResultTab;
            return true;
        }
        if (event == Event::ArrowLeft) {
            if (replay_step_ > 0) {
                --replay_step_;
            }
            voice::menuMoveSound();
            return true;
        }
        if (event == Event::ArrowRight) {
            if (replay_step_ < max_step) {
                ++replay_step_;
                voice::placeStoneSound();
            } else {
                voice::menuMoveSound();
            }
            return true;
        }
        if (event == Event::Character('r') || event == Event::Character('R')) {
            voice::selectedSound();
            replay_step_ = 0;
            stopReplayAuto();
            startReplayAuto();
            return true;
        }
        if (event == Event::Character(' ')) {
            voice::selectedSound();
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

Component Controller::Impl::renderLoadGamePage() {
    auto component = std::make_shared<detail::InteractiveBoard>();

    component->render_logic = [this] {
        Elements content = {
            text("Load Game") | hcenter | bold | color(Color::Cyan),
            separator()
        };

        if (save_file_infos_.empty()) {
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

        content.push_back(hbox({
            text(" filename           ") | bold,
            text("| mode ") | bold,
            text("| status   ") | bold,
        }));
        content.push_back(separator());

        for (int i = 0; i < static_cast<int>(save_file_infos_.size()); ++i) {
            const auto& info = save_file_infos_[i];
            constexpr int kFilenameWidth = 20;
            std::string name = " " + info.filename;
            if (static_cast<int>(name.size()) > kFilenameWidth) {
                name = name.substr(0, kFilenameWidth - 1) + "~";
            }
            while (static_cast<int>(name.size()) < kFilenameWidth) {
                name += ' ';
            }
            auto row = hbox({
                text(name),
                text("| " + info.mode + "  "),
                text("| " + info.status + " "),
            });
            if (i == load_selected_) {
                row |= inverted;
            }
            content.push_back(std::move(row));
        }

        if (!local_status_text_.empty()) {
            content.push_back(separator());
            content.push_back(text(local_status_text_) | color(local_status_color_) | hcenter);
        }
        content.push_back(separator());
        content.push_back(text("\u2191\u2193 Select  Enter Load  Esc Back") | dim | hcenter);
        return vbox(std::move(content)) | border | center;
    };

    component->event_logic = [this](const Event& event) {
        if (event == Event::Escape) {
            voice::selectedSound();
            active_index = detail::kMenuTab;
            return true;
        }
        if (event == Event::ArrowUp) {
            if (!save_files_.empty()) {
                load_selected_ = std::max(0, load_selected_ - 1);
            }
            voice::menuMoveSound();
            return true;
        }
        if (event == Event::ArrowDown) {
            if (!save_files_.empty()) {
                load_selected_ = std::min(static_cast<int>(save_files_.size()) - 1, load_selected_ + 1);
            }
            voice::menuMoveSound();
            return true;
        }
        if (event == Event::Return && !save_files_.empty()) {
            voice::selectedSound();
            if (session.deserialize(save_files_[load_selected_])) {
                shutdownRemoteSession();
                clearLocalStatus();
                syncCursorToSession();
                active_index = (session.mode() == gomoku::SessionMode::PVE) ? detail::kPveTab : detail::kPvpTab;
                syncSettingsFromSession();
                if (session.status() != gomoku::GameStatus::PLAYING) {
                    result_selected_ = 0;
                    active_index = detail::kResultTab;
                } else {
                    syncTimerToSessionState();
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

Component Controller::Impl::renderFrontPage() {
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
            voice::menuMoveSound();
            return menu->OnEvent(Event::End);
        }
        if (menu_count > 0 && event == Event::ArrowDown && menu_selected == menu_count - 1) {
            voice::menuMoveSound();
            return menu->OnEvent(Event::Home);
        }

        if (event == Event::ArrowUp || event == Event::ArrowDown) {
            voice::menuMoveSound();
            return false;
        }

        if (event != Event::Return) {
            return false;
        }

        voice::selectedSound();

        if (menu_selected == 0) {
            pending_mode_ = gomoku::SessionMode::PVP;
            settings_focus_ = 1;
            previous_tab = detail::kMenuTab;
            active_index = detail::kSetupTab;
            return true;
        }
        if (menu_selected == 1) {
            pending_mode_ = gomoku::SessionMode::PVE;
            settings_focus_ = 1;
            previous_tab = detail::kMenuTab;
            active_index = detail::kSetupTab;
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
            previous_tab = detail::kMenuTab;
            active_index = detail::kSetupTab;
            return true;
        }
        if (menu_selected == 5) {
            refreshSavesList();
            active_index = detail::kLoadTab;
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

Component Controller::Impl::renderGameBoard(const bool has_ai) {
    auto component = std::make_shared<detail::InteractiveBoard>();
    component->render_logic = [this] { return renderGrid(); };
    component->event_logic = [this, has_ai](const Event& event) {
        if (event == Event::Special("ai_move")) {
            if (has_ai && session.status() == gomoku::GameStatus::PLAYING) {
                runAiTurn();
                syncTimerToSessionState();
            }
            return true;
        }

        if (event == Event::Special("timer_tick")) {
            if (shouldRunTimer()) {
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
            voice::selectedSound();
            backToMenu();
            return true;
        }

        if (event == Event::Character('l') || event == Event::Character('L')) {
            voice::selectedSound();
            if (remote_mode_) {
                updateRemoteStatus("Save/Leave is disabled during remote games. Press Q to disconnect.");
                return true;
            }
            show_save_menu_ = true;
            save_menu_selected_ = 0;
            return true;
        }

        if (event == Event::Character('s') || event == Event::Character('S')) {
            voice::selectedSound();
            pending_mode_.reset();
            settings_focus_ = 0;
            previous_tab = active_index;
            active_index = detail::kSetupTab;
            return true;
        }

        if (event == Event::Character('u') || event == Event::Character('U')) {
            voice::selectedSound();
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
                syncTimerToSessionState();
                if (session.status() == gomoku::GameStatus::PLAYING && active_index == detail::kResultTab) {
                    active_index = detail::kPvpTab;
                }
                return true;
            }

            if (!session.undo()) {
                setStatusMsg("No moves to undo", 1500);
            } else {
                setStatusMsg("Move undone", 1000);
                syncCursorToSession();
                syncTimerToSessionState();
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
            voice::placeStoneSound();
            clearLocalStatus();
            syncCursorToSession();
            updateRemoteStatus();
            tryMoveToResult();
            syncTimerToSessionState();
            return true;
        }

        if (!session.human_move(current_x, current_y)) {
            setStatusMsg("Cell already occupied", 1500);
            return true;
        }

        voice::placeStoneSound();
        clearLocalStatus();
        consecutive_timeouts_ = 0;
        stopTimer();
        tryMoveToResult();
        if (has_ai && session.status() == gomoku::GameStatus::PLAYING) {
            startAiDelay();
        }
        syncTimerToSessionState();

        return true;
    };

    return component;
}

Component Controller::Impl::renderSetupPage() {
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

    auto undo_checkbox = Checkbox("Undo", &settings_undo_enabled, cb_opt);
    auto timer_checkbox = Checkbox("Time Limit", &settings_timer_enabled, cb_opt);

    InputOption inp_opt = InputOption::Default();
    inp_opt.on_change = [this] {
        std::string filtered;
        for (const char c : settings_timer_seconds_str_) {
            if (c >= '0' && c <= '9') {
                filtered += c;
            }
        }
        if (filtered.size() > 3) {
            filtered.resize(3);
        }
        settings_timer_seconds_str_ = filtered;
    };
    auto timer_input = Input(&settings_timer_seconds_str_, "secs", inp_opt);

    ButtonOption btn_opt;
    btn_opt.transform = [](const EntryState& state) {
        auto element = text(state.label) | hcenter | flex;
        if (state.focused) {
            element |= inverted;
        }
        return element;
    };
    auto back_button = Button("Ok", [this] {
        if (!commitSetupChanges()) {
            return;
        }
        if (pending_mode_.has_value()) {
            const auto mode = *pending_mode_;
            pending_mode_.reset();
            startGame(mode);
        } else {
            active_index = previous_tab;
        }
    }, btn_opt);

    auto checkboxes = Container::Vertical({undo_checkbox, timer_checkbox, timer_input});
    auto container = Container::Vertical({checkboxes, back_button}, &settings_focus_);

    auto component = Renderer(container, [this, undo_checkbox, timer_checkbox, timer_input, back_button] {
        auto timer_row = hbox({
            text("    "),
            timer_input->Render() | size(WIDTH, EQUAL, 4),
            text(" sec ")
        });
        if (!settings_timer_enabled) {
            timer_row = timer_row | dim;
        }

        const auto how_to_play = vbox({
            text("\u2500\u2500 How to Play \u2500\u2500") | bold | color(Color::Cyan) | hcenter,
            text(" \u00b7 Get 5 in a row to win") | dim,
            text("  (horizontal / vertical / diagonal) ") | dim,
            text(" \u00b7 Black X goes first, alternate") | dim,
            text(""),
            text(" \u2191\u2193\u2190\u2192 to Move, Enter/Space to Place ") | dim,
            text(" Press [S]/[U]/[L] for more options ") | dim,
        });

        const auto setup_box = vbox({
            text("\u2500\u2500 Setup \u2500\u2500") | bold | color(Color::Cyan) | hcenter,
            undo_checkbox->Render(),
            hbox({timer_checkbox->Render(), timer_row}),
        });

        Elements settings_content = {
            text("Briefing") | hcenter | bold | color(Color::Cyan),
            separator(),
            how_to_play,
            separator(),
            setup_box,
        };
        if (!local_status_text_.empty()) {
            settings_content.push_back(separator());
            settings_content.push_back(text(local_status_text_) | color(local_status_color_) | hcenter);
        }
        settings_content.push_back(separator());
        settings_content.push_back(back_button->Render() | hcenter);

        const auto settings_box = vbox(std::move(settings_content)) | border | center;
        return dbox({renderGrid(), settings_box | clear_under | center});
    });

    component |= CatchEvent([](const Event& event) {
        if (event == Event::ArrowUp || event == Event::ArrowDown) {
            voice::menuMoveSound();
            return false;
        }
        if (event == Event::Return || event == Event::Character(' ')) {
            voice::selectedSound();
            return false;
        }
        return false;
    });

    return component;
}

Component Controller::Impl::renderEndPage() {
    auto component = std::make_shared<detail::InteractiveBoard>();

    component->render_logic = [this] {
        const std::vector<std::string> result_labels = {
            "Back to menu", "Play again", "Save Game", "View Replay"
        };
        Elements items;
        for (int i = 0; i < static_cast<int>(result_labels.size()); ++i) {
            auto item = text("  " + result_labels[i] + "  ");
            if (i == result_selected_) {
                item |= inverted;
            }
            items.push_back(std::move(item));
        }

        Elements box_content = {
            text("Game Over") | hcenter | bold | color(Color::Red),
            separator(),
            text("Result: " + detail::gameResultText(session.status())) | hcenter | bold | color(Color::Cyan),
            separator()
        };
        box_content.push_back(vbox(std::move(items)));
        box_content.push_back(separator());
        box_content.push_back(text(" \u2191\u2193 Move  Enter Confirm ") | dim | hcenter);

        const auto result_box = vbox(std::move(box_content)) | border | center;
        return dbox({renderGrid(), result_box | clear_under | center});
    };

    component->event_logic = [this](const Event& event) {
        constexpr int kNumOptions = 4;
        if (event == Event::ArrowUp) {
            result_selected_ = (result_selected_ - 1 + kNumOptions) % kNumOptions;
            voice::menuMoveSound();
            return true;
        }
        if (event == Event::ArrowDown) {
            result_selected_ = (result_selected_ + 1) % kNumOptions;
            voice::menuMoveSound();
            return true;
        }
        if (event == Event::Return) {
            voice::selectedSound();
            if (result_selected_ == 0) {
                backToMenu();
            } else if (result_selected_ == 1) {
                if (remote_mode_) {
                    if (!network.isConnected()) {
                        updateRemoteStatus("Remote peer is not connected.");
                        return true;
                    }
                    if (!network.requestReset()) {
                        updateRemoteStatus("Reset failed: " + network.lastError());
                        return true;
                    }
                    syncCursorToSession();
                    updateRemoteStatus();
                    active_index = detail::kPvpTab;
                } else {
                    startGame(session.mode());
                }
            } else if (result_selected_ == 2) {
                trySaveSessionWithFeedback();
            } else {
                enterReplay();
            }
            return true;
        }
        return false;
    };

    return component;
}

Component Controller::Impl::renderRemoteHostPage() {
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
                color(detail::remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) |
                hcenter,
            separator(),
            start_button->Render(),
            back_button->Render()
        }) | border | center;
    });

    component |= CatchEvent([this](const Event& event) {
        if (event == Event::ArrowUp || event == Event::ArrowDown) {
            voice::menuMoveSound();
            return false;
        }
        if (event == Event::Escape) {
            voice::selectedSound();
            backToMenu();
            return true;
        }
        return false;
    });

    return component;
}

Component Controller::Impl::renderRemoteJoinPage() {
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
                color(detail::remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) |
                hcenter,
            separator(),
            connect_button->Render(),
            back_button->Render()
        }) | border | center;
    });

    component |= CatchEvent([this](const Event& event) {
        if (event == Event::ArrowUp || event == Event::ArrowDown) {
            voice::menuMoveSound();
            return false;
        }
        if (event == Event::Escape) {
            voice::selectedSound();
            backToMenu();
            return true;
        }
        return false;
    });

    return component;
}

Component Controller::Impl::renderRemoteWaitPage() {
    auto cancel_button = Button("Cancel", [this] { backToMenu(); });
    auto component = Renderer(cancel_button, [cancel_button, this] {
        Elements endpoint_lines;
        for (const auto& endpoint : shareableHostEndpoints()) {
            endpoint_lines.push_back(text(endpoint) | bold | hcenter);
        }
        if (endpoint_lines.empty()) {
            endpoint_lines.push_back(text("port " + remote_port_input_) | bold | hcenter);
        }

        return vbox({
            text("Waiting For Remote Player") | hcenter | bold | color(Color::Cyan),
            separator(),
            text(remote_status_text_) |
                color(detail::remoteStatusColor(network.isConnected(), remote_waiting_for_peer_)) |
                hcenter,
            separator(),
            text("Share one of these addresses with the other player:") | hcenter,
            vbox(std::move(endpoint_lines)),
            separator(),
            cancel_button->Render() | hcenter,
            text("Esc Cancel") | dim | hcenter
        }) | border | center;
    });

    component |= CatchEvent([this](const Event& event) {
        if (event == Event::Escape) {
            voice::selectedSound();
            backToMenu();
            return true;
        }
        return false;
    });

    return component;
}

Element Controller::Impl::renderGrid() const {
    const auto& board = session.board();

    const int src_tab = [&]() -> int {
        if (active_index == detail::kSetupTab) {
            if (previous_tab != detail::kMenuTab) {
                return previous_tab;
            }
            if (pending_mode_.has_value()) {
                return (*pending_mode_ == gomoku::SessionMode::PVE)
                    ? detail::kPveTab
                    : detail::kPvpTab;
            }
            return detail::kPvpTab;
        }
        if (active_index == detail::kResultTab) {
            if (remote_mode_) {
                return detail::kPvpTab;
            }
            return (session.mode() == gomoku::SessionMode::PVE)
                ? detail::kPveTab
                : detail::kPvpTab;
        }
        return active_index;
    }();

    Element info_bar;
    if (!local_status_text_.empty()) {
        info_bar = text(local_status_text_) | hcenter | color(local_status_color_);
    } else {
        std::string mode_label;
        if (remote_mode_) {
            const auto endpoint = network.remoteEndpoint();
            mode_label = endpoint.empty() ? "Remote" : "Remote(" + endpoint + ")";
        } else if (src_tab == detail::kPveTab) {
            mode_label = "PvE (AI)";
        } else {
            mode_label = "PvP";
        }

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
        info_bar = hbox({text(" Mode: " + mode_label), filler(), timer_status, text(" ")});
    }

    const bool is_black_turn = board.getCurrentPlayer() == gomoku::Stone::BLACK;
    Element opponent_bar = text("");
    bool show_opponent_bar = false;
    if (src_tab == detail::kPveTab) {
        show_opponent_bar = true;
        opponent_bar = hbox({
            text(" Opponent(AI): White") | bold | color(Color::White),
            filler(),
            text("You: Black ") | bold | color(Color::Red)
        });
    } else if (remote_mode_ && network.isConnected()) {
        show_opponent_bar = true;
        const auto local_stone = network.localStone();
        const auto opponent_stone = (local_stone == gomoku::Stone::BLACK)
            ? gomoku::Stone::WHITE
            : gomoku::Stone::BLACK;
        opponent_bar = hbox({
            text(" Opponent: " + detail::stoneText(opponent_stone))
                | bold | color(opponent_stone == gomoku::Stone::BLACK ? Color::Red : Color::White),
            filler(),
            text("You: " + detail::stoneText(local_stone) + " ")
                | bold | color(local_stone == gomoku::Stone::BLACK ? Color::Red : Color::White)
        });
    }

    std::string player_suffix;
    if (src_tab == detail::kPveTab) {
        player_suffix = is_black_turn ? " (You)" : " (AI)";
    } else if (remote_mode_ && network.isConnected()) {
        player_suffix = (board.getCurrentPlayer() == network.localStone())
            ? " (You)"
            : " (Opponent)";
    }
    const std::string player_name = is_black_turn ? "Black" : "White";
    auto player_bar = text("Current player: " + player_name + player_suffix) |
                      bold | color(is_black_turn ? Color::Red : Color::White) | hcenter;

    Elements rows;
    const int size = board.getSize();
    const bool in_game = (active_index == detail::kPvpTab || active_index == detail::kPveTab);
    const auto last_mv = session.last_move();
    for (int y = 0; y < size; ++y) {
        Elements columns;
        for (int x = 0; x < size; ++x) {
            const auto stone = board.getStone(x, y);
            auto cell = detail::stoneCellElement(stone);

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

    Element hint_line;
    std::string hint_text;
    bool hint_is_status = false;

    if (settings_timer_enabled && timer_remaining_ > 0 && timer_remaining_ <= 3) {
        hint_text = "  Hurry up! You'll get skipped!  ";
        hint_is_status = true;
    } else if (remote_mode_ && !network.lastError().empty()) {
        hint_text = "  " + network.lastError() + "  ";
        hint_is_status = true;
    } else if (const auto msg = activeStatusMsg(); !msg.empty()) {
        hint_text = "  " + msg + "  ";
        hint_is_status = true;
    } else {
        hint_text = "\u2191\u2193\u2190\u2192 Move   [Enter]/[Space] Place";
    }

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

    Elements ai_section;
    if (src_tab == detail::kPveTab) {
        const std::string& ai_text = session.ai_status_text();
        const auto newline = ai_text.find('\n');
        const std::string ai_line1 = newline != std::string::npos ? ai_text.substr(0, newline) : ai_text;
        const std::string ai_line2 = newline != std::string::npos ? ai_text.substr(newline + 1) : "";
        ai_section.push_back(separator());
        ai_section.push_back(text(ai_line1) | hcenter | dim);
        if (!ai_line2.empty()) {
            ai_section.push_back(text(ai_line2) | hcenter | dim);
        }
    }

    Elements board_rows = {info_bar};
    if (show_opponent_bar) {
        board_rows.push_back(opponent_bar);
    }
    board_rows.push_back(player_bar);
    board_rows.push_back(separator());
    board_rows.push_back(vbox(std::move(rows)) | hcenter);
    board_rows.push_back(separator());
    board_rows.push_back(hint_line);
    board_rows.push_back(bottom_bar);
    for (auto& element : ai_section) {
        board_rows.push_back(std::move(element));
    }

    auto board_element = vbox(std::move(board_rows)) | border | center;

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
        text(" \u2191\u2193 Move  Enter Confirm ") | dim | hcenter
    }) | border | bgcolor(Color::Black) | center;

    return dbox({board_element, overlay | clear_under | center});
}

} // namespace UI
