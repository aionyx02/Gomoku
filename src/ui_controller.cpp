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
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace UI {
    using namespace ftxui;
    namespace fs = std::filesystem;

    namespace {
        constexpr int kBoardSize = 15;

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

        struct AIMoveResult {
            std::optional<std::pair<int, int> > move;
            std::string reason;
        };

        std::string QuoteArg(const std::string& value) {
            std::string escaped;
            escaped.reserve(value.size());
            for (const char ch : value) {
                if (ch == '"')
                    escaped += "\\\"";
                else
                    escaped.push_back(ch);
            }
            return "\"" + escaped + "\"";
        }

        fs::path GetExecutableDir() {
#if defined(_WIN32)
            std::array<char, 4096> buffer {};
            if (const DWORD len = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size())); len > 0 && len < buffer.size())
                return fs::path(std::string(buffer.data(), len)).parent_path();
#endif
            std::error_code ec;
            const fs::path current = fs::current_path(ec);
            if (!ec)
                return current;
            return fs::path(".");
        }

        std::vector<fs::path> CollectBaseDirs() {
            std::vector<fs::path> bases;
            const auto add_unique = [&bases](const fs::path& raw) {
                if (raw.empty())
                    return;
                std::error_code ec;
                fs::path normalized = fs::absolute(raw, ec);
                if (ec)
                    normalized = raw.lexically_normal();
                else
                    normalized = normalized.lexically_normal();

                const std::string key = normalized.generic_string();
                const bool exists = std::ranges::any_of(bases
                                                        ,
                                                        [&key](const fs::path& existing) { return existing.generic_string() == key; }
                );
                if (!exists)
                    bases.push_back(normalized);
            };

            const fs::path exe_dir = GetExecutableDir();
            add_unique(fs::path(kSourceDir));
            add_unique(exe_dir);
            add_unique(exe_dir.parent_path());
            add_unique(fs::current_path());
            add_unique(fs::current_path().parent_path());
            return bases;
        }

        std::optional<fs::path> FindFileInBases(
            const std::vector<fs::path>& bases,
            const std::string& filename
        ) {
            for (const auto& base : bases) {
                const fs::path candidate = (base / filename).lexically_normal();
                if (std::error_code ec; fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec))
                    return candidate;
            }
            return std::nullopt;
        }

        std::string BuildSearchedPathsSummary(
            const std::vector<fs::path>& bases,
            const std::string& filename
        ) {
            std::string joined;
            for (const auto& base : bases) {
                if (!joined.empty())
                    joined += ", ";
                joined += (base / filename).generic_string();
            }
            return joined;
        }

        bool IsPathLikeCommand(const std::string& candidate) {
            return candidate.find('\\') != std::string::npos
                || candidate.find('/') != std::string::npos
                || candidate.find(':') != std::string::npos;
        }

        std::string NormalizeKey(std::string value) {
            std::ranges::transform(value
                                   ,
                                   value.begin(),
                                   [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
            );
            return value;
        }

        std::vector<std::string> ResolvePythonCommands(const std::vector<fs::path>& bases) {
            std::vector<std::string> resolved;
            std::vector<std::string> visited_keys;
            const auto push_unique = [&resolved, &visited_keys](const std::string& key, const std::string& value) {
                if (value.empty())
                    return;
                const std::string lowered = NormalizeKey(key);
                if (std::ranges::find(visited_keys, lowered) != visited_keys.end())
                    return;
                visited_keys.push_back(lowered);
                resolved.push_back(value);
            };

            std::vector<std::string> candidates;

            if (const char* env_py = std::getenv("GOMOKU_PYTHON_EXECUTABLE"); env_py != nullptr && env_py[0] != '\0')
                candidates.emplace_back(env_py);

            candidates.emplace_back(kPythonExecutable);

            for (const auto& base : bases) {
                candidates.push_back((base / ".gomoku-python/Scripts/python.exe").generic_string());
                candidates.push_back((base / ".gomoku-python/python.exe").generic_string());
                candidates.push_back((base / ".venv/Scripts/python.exe").generic_string());
                candidates.push_back((base / ".venv/bin/python3").generic_string());
                candidates.push_back((base / ".venv/bin/python").generic_string());
                candidates.push_back((base / "Scripts/python.exe").generic_string());
                candidates.push_back((base / "python/bin/python3").generic_string());
                candidates.push_back((base / "python/bin/python").generic_string());
            }

            candidates.emplace_back("python");
            candidates.emplace_back("python3");

            for (auto candidate : candidates) {
                if (candidate.empty())
                    continue;


                if (candidate.size() >= 2 && candidate.front() == '"' && candidate.back() == '"')
                    candidate = candidate.substr(1, candidate.size() - 2);

                const bool looks_like_path = IsPathLikeCommand(candidate);
                const std::string key = looks_like_path ? fs::path(candidate).generic_string() : candidate;

                if (!looks_like_path) {
                    push_unique(key, candidate);
                    continue;
                }

                if (std::error_code ec; fs::exists(fs::path(candidate), ec))
                    push_unique(
                        key,
                        fs::path(candidate).lexically_normal().generic_string()
                    );
            }

            return resolved;
        }

        std::string CompactMessage(const std::string &text, const size_t max_len = 140) {
            std::string compact;
            compact.reserve(text.size());
            bool in_space = false;
            for (const unsigned char ch : text) {
                if (std::isspace(ch) != 0) {
                    if (!in_space && !compact.empty())
                        compact.push_back(' ');
                    in_space = true;
                    continue;
                }
                compact.push_back(static_cast<char>(ch));
                in_space = false;
            }

            if (compact.size() <= max_len)
                return compact;

            return compact.substr(0, max_len - 3) + "...";
        }

        std::optional<std::pair<int, int> > ParseMoveFromOutput(const std::string& output) {
            std::istringstream lines(output);
            std::string line;
            std::optional<std::pair<int, int> > parsed_move;
            while (std::getline(lines, line)) {
                std::istringstream parser(line);
                int x = -1;
                int y = -1;
                if (std::string extra; (parser >> x >> y) && !(parser >> extra))
                    parsed_move = std::make_pair(x, y);
            }
            return parsed_move;
        }

#if defined(_WIN32)
        int RunProcessCapture(const std::string& cmdline, std::string& output, std::string* startup_error = nullptr) {
            SECURITY_ATTRIBUTES sa {};
            sa.nLength = sizeof(sa);
            sa.bInheritHandle = TRUE;

            HANDLE read_pipe = nullptr;
            HANDLE write_pipe = nullptr;

            if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
                // if (startup_error != nullptr)
                //     *startup_error = "CreatePipe failed";
                return -1;
            }
            if (!SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0)) {
                CloseHandle(read_pipe);
                CloseHandle(write_pipe);
                // if (startup_error != nullptr)
                //     *startup_error = "SetHandleInformation failed";
                return -1;
            }

            STARTUPINFOA si {};
            si.cb = sizeof(si);
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdOutput = write_pipe;
            si.hStdError = write_pipe;
            si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

            PROCESS_INFORMATION pi {};
            std::vector<char> mutable_cmd(cmdline.begin(), cmdline.end());
            mutable_cmd.push_back('\0');

            const BOOL ok = CreateProcessA(
                nullptr,
                mutable_cmd.data(),
                nullptr,
                nullptr,
                TRUE,
                CREATE_NO_WINDOW,
                nullptr,
                nullptr,
                &si,
                &pi
            );

            CloseHandle(write_pipe);
            //
            // if (!ok) {
            //     CloseHandle(read_pipe);
            //     if (startup_error != nullptr) {
            //         const DWORD error = GetLastError();
            //         *startup_error = "CreateProcess failed (" + std::to_string(error) + ")";
            //     }
            //     return -1;
            // }

            std::array<char, 4096> buffer {};
            DWORD bytes_read = 0;
            while (ReadFile(read_pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read, nullptr)
                   && bytes_read > 0) {
                output.append(buffer.data(), bytes_read);
            }

            CloseHandle(read_pipe);
            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exit_code = 1;
            GetExitCodeProcess(pi.hProcess, &exit_code);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            return static_cast<int>(exit_code);
        }
#endif

        char StoneToSymbol(const gomoku::Stone stone) {
            if (stone == gomoku::Stone::BLACK)
                return 'B';
            if (stone == gomoku::Stone::WHITE)
                return 'W';
            return '.';
        }

        std::string SerializeBoardFlat(const gomoku::Board& board) {
            const int size = board.getSize();
            std::string serialized;
            serialized.reserve(size * size);

            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x)
                    serialized.push_back(StoneToSymbol(board.getStone(x, y)));
            }

            return serialized;
        }

        AIMoveResult QueryAIMove(const gomoku::Board& board) {
            const char current_symbol = StoneToSymbol(board.getCurrentPlayer());
            const auto base_dirs = CollectBaseDirs();
            const auto script_path = [&]() -> std::optional<fs::path> {
                if (auto p = FindFileInBases(base_dirs, "runModelAndReturnPoint.py"); p.has_value())
                    return p;
                return FindFileInBases(base_dirs, "python/runModelAndReturnPoint.py");
            }();
            if (!script_path.has_value()) {
                return AIMoveResult{ std::nullopt,
                    "runModelAndReturnPoint.py not found; searched: "
                    + CompactMessage(BuildSearchedPathsSummary(base_dirs, "runModelAndReturnPoint.py"), 220)
                };
            }

            const auto model_path = FindFileInBases(base_dirs, "gomoku_model.pt");
            if (!model_path.has_value()) {
                return AIMoveResult{
                    std::nullopt,
                    "gomoku_model.pt not found; searched: "
                    + CompactMessage(BuildSearchedPathsSummary(base_dirs, "gomoku_model.pt"), 220),
                };
            }

            const std::string board_text = SerializeBoardFlat(board);
            const auto python_commands = ResolvePythonCommands(base_dirs);
            if (python_commands.empty()) {
                return AIMoveResult{
                    std::nullopt,
                    "no usable python command was found (checked env, .gomoku-python, .venv, PATH)"
                };
            }

            std::vector<std::string> attempt_summaries;
            attempt_summaries.reserve(python_commands.size());

#if defined(_WIN32)
            SetCurrentDirectoryA(fs::path(kSourceDir).string().c_str());
#else
            {
                std::error_code ec;
                fs::current_path(fs::path(kSourceDir), ec);
            }
#endif

            for (const auto& python_command : python_commands) {
                attempt_summaries.push_back(
                    "[try] py=" + python_command
                    + " script=" + script_path->generic_string()
                    + " model=" + model_path->generic_string()
                );

                std::string output;
                int exit_code = 0;

#if defined(_WIN32)
                const std::string process_command =
                    QuoteArg(python_command) + " " +
                    QuoteArg(script_path->generic_string()) + " "
                    "--board-size " + std::to_string(board.getSize()) + " "
                    "--model-path " + QuoteArg(model_path->generic_string()) + " "
                    "--current " + std::string(1, current_symbol) + " "
                    "--board " + QuoteArg(board_text);

                std::string startup_error;
                exit_code = RunProcessCapture(process_command, output, &startup_error);
                if (exit_code == -1) {
                    attempt_summaries.push_back(
                        "python=" + CompactMessage(python_command, 80) +
                        " -> cannot spawn process (" + CompactMessage(startup_error, 120) + ")"
                    );
                    continue;
                }
#else
                const std::string command =
                    QuoteArg(python_command) + " " +
                    QuoteArg(script_path->generic_string()) + " "
                    "--board-size " + std::to_string(board.getSize()) + " "
                    "--model-path " + QuoteArg(model_path->generic_string()) + " "
                    "--current " + std::string(1, current_symbol) + " "
                    "--board " + QuoteArg(board_text) + " 2>&1";

                FILE* pipe = popen(command.c_str(), "r");
                if (pipe == nullptr) {
                    attempt_summaries.push_back(
                        "python=" + CompactMessage(python_command, 80) + " -> cannot spawn process"
                    );
                    continue;
                }

                std::array<char, 256> buffer {};
                while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
                    output += buffer.data();

                exit_code = pclose(pipe);
#endif
                if (exit_code != 0) {
                    const std::string detail = CompactMessage(output);
                    attempt_summaries.push_back(
                        "python=" + CompactMessage(python_command, 80) + " -> exit=" + std::to_string(exit_code)
                        + (detail.empty() ? "" : (": " + detail))
                    );
                    continue;
                }

                const auto parsed_move = ParseMoveFromOutput(output);
                if (!parsed_move.has_value()) {
                    const std::string detail = CompactMessage(output);
                    const std::string reason = detail.empty()
                        ? "cannot parse AI move from script output (python=" + python_command + ")"
                        : "cannot parse AI move from script output (python=" + python_command + "): " + detail;
                    return AIMoveResult{std::nullopt, reason};
                }

                const auto [x, y] = parsed_move.value();

                if (const int size = board.getSize(); x < 0 || x >= size || y < 0 || y >= size) {
                    return AIMoveResult{
                        std::nullopt,
                        "AI move out of board range: (" + std::to_string(x) + ", " + std::to_string(y) + ")",
                    };
                }

                return AIMoveResult{std::make_pair(x, y), ""};
            }

            std::string summary;
            for (const auto& item : attempt_summaries) {
                if (!summary.empty())
                    summary += " | ";
                summary += item;
            }
            {
                const std::string& full_summary =  summary;
                if (FILE* log = fopen((fs::path(kSourceDir) / "ai_debug.log").string().c_str(), "w")) {
                    fprintf(log, "%s\n", full_summary.c_str());
                    fclose(log);
                }
                return AIMoveResult{
                    std::nullopt,
                    "all python candidates failed: " + CompactMessage(full_summary, 1200)
                };
            }
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
                this->board = gomoku::Board(kBoardSize);
                this->current_x = kBoardSize / 2;
                this->current_y = kBoardSize / 2;
                this->ai_status_text = "AI: ready (model expected at gomoku_model.pt)";
                this->ai_used_fallback = false;
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
                        std::string fallback_reason = "unknown AI error";

                        if (const auto [move, reason] = QueryAIMove(this->board); move.has_value()) {
                            const auto [ai_x, ai_y] = move.value();
                            ai_placed = this->board.placeStone(ai_x, ai_y);
                            if (ai_placed) {
                                this->current_x = ai_x;
                                this->current_y = ai_y;
                                this->ai_status_text = "AI(model): move (" + std::to_string(ai_x) + ", " +
                                                       std::to_string(ai_y) + ")";
                                this->ai_used_fallback = false;
                            } else {
                                fallback_reason = "model returned an occupied or invalid cell";
                            }
                        } else {
                            fallback_reason = reason;
                        }

                        if (!ai_placed) {
                            if (const auto fallback = FindFallbackMove(this->board); fallback.has_value()) {
                                if (const auto [fallback_x, fallback_y] = fallback.value(); this->board.placeStone(fallback_x, fallback_y)) {
                                    this->current_x = fallback_x;
                                    this->current_y = fallback_y;
                                    this->ai_status_text = "AI(fallback): move (" + std::to_string(fallback_x) +
                                                           ", " + std::to_string(fallback_y) + ") | reason: " +
                                                           fallback_reason;
                                    this->ai_used_fallback = true;
                                }
                            } else {
                                this->ai_status_text = "AI failed: no legal fallback move";
                                this->ai_used_fallback = true;
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
                this->ai_status_text = "AI: ready";
                this->ai_used_fallback = false;
            }),
            Button("Play again", [this] {
                this->active_index = 1;
                this->board = gomoku::Board(kBoardSize);
                this->current_x = kBoardSize / 2;
                this->current_y = kBoardSize / 2;
                this->ai_status_text = "AI: ready";
                this->ai_used_fallback = false;
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
        auto ai_bar = text(this->ai_status_text) | hcenter;
        if (this->active_index == 2) {
            ai_bar |= this->ai_used_fallback ? color(Color::Yellow) : color(Color::Green);
        } else {
            ai_bar |= dim;
        }
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
            ai_bar,
            separator(),
            board_ui
        }) | border | center;
    }
}
