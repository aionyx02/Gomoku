#include "gomoku/ui_controller.h"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include <array>
#include <cctype>
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

        // ── helpers ──────────────────────────────────────────────────────────

        std::string QuoteArg(const std::string& s) {
            std::string out = "\"";
            for (const char ch : s) { if (ch == '"') out += "\\\""; else out += ch; }
            return out + "\"";
        }

        std::string CompactMessage(const std::string& text, const size_t max_len = 140) {
            std::string out;
            bool sp = false;
            for (const char ch : text) {
                if (std::isspace(ch)) { if (!sp && !out.empty()) out += ' '; sp = true; }
                else { out += ch; sp = false; }
            }
            return out.size() <= max_len ? out : out.substr(0, max_len - 3) + "...";
        }

        // ── path / python discovery ───────────────────────────────────────────

        fs::path ExeDir() {
#if defined(_WIN32)
            std::array<char, 4096> buf{};
            if (const DWORD n = GetModuleFileNameA(nullptr, buf.data(), (DWORD)buf.size()); n > 0 && n < buf.size())
                return fs::path(std::string(buf.data(), n)).parent_path();
#endif
            std::error_code ec;
            return fs::current_path(ec);
        }

        std::vector<fs::path> BaseDirs() {
            std::vector<fs::path> out;
            auto add = [&](fs::path p) {
                std::error_code ec;
                p = fs::absolute(p, ec).lexically_normal();
                const std::string key = p.generic_string();
                for (auto& x : out) if (x.generic_string() == key) return;
                out.push_back(p);
            };
            const fs::path exe = ExeDir();
            add(fs::path(kSourceDir));
            add(exe);
            add(exe.parent_path());
            add(fs::current_path());
            add(fs::current_path().parent_path());
            return out;
        }

        std::optional<fs::path> FindFile(const std::vector<fs::path>& bases, const std::string& name) {
            for (auto& b : bases) {
                fs::path c = (b / name).lexically_normal();
                if (std::error_code ec; fs::exists(c, ec) && fs::is_regular_file(c, ec)) return c;
            }
            return std::nullopt;
        }

        std::string SearchedPaths(const std::vector<fs::path>& bases, const std::string& name) {
            std::string out;
            for (auto& b : bases) { if (!out.empty()) out += ", "; out += (b / name).generic_string(); }
            return out;
        }

        std::vector<std::string> PythonCommands(const std::vector<fs::path>& bases) {
            std::vector<std::string> cmds, seen;
            auto push = [&](std::string cmd) {
                if (cmd.empty()) return;
                // strip surrounding quotes
                if (cmd.size() >= 2 && cmd.front() == '"' && cmd.back() == '"')
                    cmd = cmd.substr(1, cmd.size() - 2);
                // normalize key
                std::string key = cmd;
                for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (std::ranges::find(seen, key) != seen.end()) return;
                seen.push_back(key);
                if (cmd.find_first_of("/\\:") != std::string::npos) {
                    if (std::error_code ec; !fs::exists(cmd, ec)) return;
                    cmd = fs::path(cmd).lexically_normal().generic_string();
                }
                cmds.push_back(cmd);
            };
            if (const char* e = std::getenv("GOMOKU_PYTHON_EXECUTABLE"); e && *e) push(e);
            push(kPythonExecutable);
            for (auto& b : bases) {
                for (auto rel : {".gomoku-python/Scripts/python.exe",
                                 ".gomoku-python/python.exe",
                                 ".venv/Scripts/python.exe",
                                 ".venv/bin/python3",
                                 ".venv/bin/python",
                                 "Scripts/python.exe",
                                 "python/bin/python3",
                                 "python/bin/python"})
                    push((b / rel).generic_string());
            }
            push("python"); push("python3");
            return cmds;
        }

        // ── process execution ────────────────────────────────────────────────

#if defined(_WIN32)
        int RunProcess(const std::string& cmdline, std::string& output) {
            SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
            HANDLE rp{}, wp{};
            if (!CreatePipe(&rp, &wp, &sa, 0)) return -1;
            SetHandleInformation(rp, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOA si{};
            si.cb = sizeof(si); si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdOutput = si.hStdError = wp;
            si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

            PROCESS_INFORMATION pi{};
            std::vector<char> cmd(cmdline.begin(), cmdline.end()); cmd.push_back('\0');
            if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                CloseHandle(rp); CloseHandle(wp); return -1;
            }
            CloseHandle(wp);

            std::array<char, 4096> buf{};
            DWORD n{};
            while (ReadFile(rp, buf.data(), (DWORD)buf.size(), &n, nullptr) && n) output.append(buf.data(), n);
            CloseHandle(rp);
            WaitForSingleObject(pi.hProcess, INFINITE);
            DWORD ec = 1; GetExitCodeProcess(pi.hProcess, &ec);
            CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
            return static_cast<int>(ec);
        }
#endif

        // ── board helpers ────────────────────────────────────────────────────

        char StoneChar(const gomoku::Stone s) {
            return s == gomoku::Stone::BLACK ? 'B' : s == gomoku::Stone::WHITE ? 'W' : '.';
        }

        std::string SerializeBoard(const gomoku::Board& b) {
            const int sz = b.getSize();
            std::string out; out.reserve(sz * sz);
            for (int y = 0; y < sz; ++y)
                for (int x = 0; x < sz; ++x)
                    out += StoneChar(b.getStone(x, y));
            return out;
        }

        std::optional<std::pair<int,int>> ParseMove(const std::string& output) {
            std::istringstream ss(output); std::string line;
            std::optional<std::pair<int,int>> result;
            while (std::getline(ss, line)) {
                std::istringstream ls(line); int y; std::string extra;
                if (int x; (ls >> x >> y) && !(ls >> extra)) result = {x, y};
            }
            return result;
        }

        // ── AI query ─────────────────────────────────────────────────────────

        struct AIMoveResult { std::optional<std::pair<int,int>> move; std::string reason; };

        AIMoveResult QueryAIMove(const gomoku::Board& board) {
            auto bases = BaseDirs();

            auto script = FindFile(bases, "runModelAndReturnPoint.py");
            if (!script) script = FindFile(bases, "python/runModelAndReturnPoint.py");
            if (!script) return {std::nullopt, "runModelAndReturnPoint.py not found; searched: "
                                 + CompactMessage(SearchedPaths(bases, "runModelAndReturnPoint.py"), 220)};

            auto model = FindFile(bases, "gomoku_model.pt");
            if (!model) return {std::nullopt, "gomoku_model.pt not found; searched: "
                                + CompactMessage(SearchedPaths(bases, "gomoku_model.pt"), 220)};

            auto pycmds = PythonCommands(bases);
            if (pycmds.empty()) return {std::nullopt, "no usable python command found"};

            const std::string board_text = SerializeBoard(board);
            const char cur = StoneChar(board.getCurrentPlayer());
            const std::string args =
                "--board-size " + std::to_string(board.getSize()) + " "
                "--model-path " + QuoteArg(model->generic_string()) + " "
                "--current " + std::string(1, cur) + " "
                "--board " + QuoteArg(board_text);

#if defined(_WIN32)
            SetCurrentDirectoryA(fs::path(kSourceDir).string().c_str());
#else
            { std::error_code ec; fs::current_path(fs::path(kSourceDir), ec); }
#endif

            std::vector<std::string> log;
            for (auto& py : pycmds) {
                std::string output;
#if defined(_WIN32)
                std::string cmd = QuoteArg(py) + " " + QuoteArg(script->generic_string()) + " " + args;
                int rc = RunProcess(cmd, output);
#else
                std::string cmd = QuoteArg(py) + " " + QuoteArg(script->generic_string()) + " " + args + " 2>&1";
                FILE* pipe = popen(cmd.c_str(), "r");
                if (!pipe) { log.push_back("py=" + py + " -> cannot spawn"); continue; }
                std::array<char, 256> buf{};
                while (fgets(buf.data(), (int)buf.size(), pipe)) output += buf.data();
                int rc = pclose(pipe);
#endif
                if (rc != 0) { log.push_back("py=" + py + " exit=" + std::to_string(rc) + ": " + CompactMessage(output)); continue; }

                auto move = ParseMove(output);
                if (!move) return {std::nullopt, "cannot parse AI move (py=" + py + "): " + CompactMessage(output)};

                auto [x, y] = *move;
                if (int sz = board.getSize(); x < 0 || x >= sz || y < 0 || y >= sz)
                    return {std::nullopt, "AI move out of range: (" + std::to_string(x) + "," + std::to_string(y) + ")"};

                return {std::make_pair(x, y), ""};
            }

            std::string summary;
            for (auto& s : log) { if (!summary.empty()) summary += " | "; summary += s; }
            if (FILE* f = fopen((fs::path(kSourceDir) / "ai_debug.log").string().c_str(), "w")) {
                fprintf(f, "%s\n", summary.c_str()); fclose(f);
            }
            return {std::nullopt, "all python candidates failed: " + CompactMessage(summary, 1200)};
        }

        std::optional<std::pair<int,int>> FallbackMove(const gomoku::Board& b) {
            const int sz = b.getSize();
            for (int y = 0; y < sz; ++y)
                for (int x = 0; x < sz; ++x)
                    if (b.getStone(x, y) == gomoku::Stone::EMPTY) return {{x, y}};
            return std::nullopt;
        }
    }

    // ── InteractiveBoard ─────────────────────────────────────────────────────

    class InteractiveBoard : public ComponentBase {
    public:
        std::function<Element()>    renderLogic;
        std::function<bool(Event)>  eventLogic;
        Element OnRender() override { return renderLogic(); }
        [[nodiscard]] bool Focusable() const override { return true; }
        bool OnEvent(const Event e) override { return eventLogic(e); }
    };

    // ── Controller ───────────────────────────────────────────────────────────

    struct Controller::ScreenState {
        ScreenInteractive screen = ScreenInteractive::Fullscreen();
    };

    Controller::Controller(gomoku::Board& board)
        : board(board), screen_state(std::make_unique<ScreenState>()) {}

    Controller::~Controller() = default;

    void Controller::Start() {
        const auto container = Container::Tab({
            RenderFrontPage(),
            RenderGameBoard(/*has_ai=*/false),
            RenderGameBoard(/*has_ai=*/true),
            RenderEndPage()
        }, &active_index);
        screen_state->screen.Loop(container);
    }

    // Shared arrow-key cursor movement
    bool Controller::HandleMove(const Event& event) {
        if (event == Event::ArrowUp)    { current_y = std::max(0, current_y - 1); return true; }
        if (event == Event::ArrowDown)  { current_y = std::min(kBoardSize - 1, current_y + 1); return true; }
        if (event == Event::ArrowLeft)  { current_x = std::max(0, current_x - 1); return true; }
        if (event == Event::ArrowRight) { current_x = std::min(kBoardSize - 1, current_x + 1); return true; }
        return false;
    }

    Component Controller::RenderFrontPage() {
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
        component |= CatchEvent([this](const Event& e) -> bool {
            if (e != Event::Return) return false;
            if (menu_selected == 0) { active_index = 1; return true; }
            if (menu_selected == 1) {
                active_index = 2;
                board = gomoku::Board(kBoardSize);
                current_x = current_y = kBoardSize / 2;
                ai_status_text = "AI: ready (model expected at gomoku_model.pt)";
                ai_used_fallback = false;
                return true;
            }
            if (menu_selected == 2) { screen_state->screen.Exit(); return true; }
            return false;
        });
        return component;
    }

    // Single factory for both PvP and PvAI boards
    Component Controller::RenderGameBoard(bool has_ai) {
        auto comp = std::make_shared<InteractiveBoard>();
        comp->renderLogic = [this]() { return RenderGrid(); };
        comp->eventLogic  = [this, has_ai](const Event& event) -> bool {
            if (HandleMove(event)) return true;

            if (event != Event::Return && event != Event::Character(' ')) return false;
            if (!board.placeStone(current_x, current_y)) return false;

            if (board.getStatus() != gomoku::GameStatus::PLAYING) { active_index = 3; return true; }

            if (has_ai) {
                bool placed = false;
                std::string fallback_reason = "unknown AI error";

                if (auto [move, reason] = QueryAIMove(board); move) {
                    if (auto [ax, ay] = *move; (placed = board.placeStone(ax, ay))) {
                        current_x = ax; current_y = ay;
                        ai_status_text = "AI(model): move (" + std::to_string(ax) + "," + std::to_string(ay) + ")";
                        ai_used_fallback = false;
                    } else { fallback_reason = "model returned an occupied or invalid cell"; }
                } else { fallback_reason = reason; }

                if (!placed) {
                    if (auto fb = FallbackMove(board)) {
                        if (auto [fx, fy] = *fb; board.placeStone(fx, fy)) {
                            current_x = fx; current_y = fy;
                            ai_status_text = "AI(fallback): move (" + std::to_string(fx) + "," + std::to_string(fy)
                                             + ") | reason: " + fallback_reason;
                            ai_used_fallback = true;
                        }
                    } else { ai_status_text = "AI failed: no legal fallback move"; ai_used_fallback = true; }
                }

                if (board.getStatus() != gomoku::GameStatus::PLAYING) active_index = 3;
            }
            return true;
        };
        return comp;
    }

    Component Controller::RenderEndPage() {
        auto reset = [this](const int next_index) {
            active_index   = next_index;
            board          = gomoku::Board(kBoardSize);
            current_x      = next_index == 0 ? 0 : kBoardSize / 2;
            current_y      = next_index == 0 ? 0 : kBoardSize / 2;
            ai_status_text = "AI: ready";
            ai_used_fallback = false;
        };
        auto container = Container::Vertical({
            Button("Back to menu", [reset] { reset(0); }),
            Button("Play again",   [reset] { reset(1); })
        });
        return Renderer(container, [container, this] {
            const auto status = board.getStatus();
            const std::string result =
                status == gomoku::GameStatus::BLACK_WIN ? "Black win" :
                status == gomoku::GameStatus::WHITE_WIN ? "White win" :
                status == gomoku::GameStatus::PLAYING   ? "Playing"   : "Draw";
            return vbox({
                text("Game Over") | hcenter | bold | color(Color::Red),
                separator(),
                text("Result: " + result) | hcenter | color(Color::Yellow),
                separator(),
                container->Render() | hcenter
            }) | border | center;
        });
    }

    Element Controller::RenderGrid() const {
        const auto cur   = board.getCurrentPlayer();
        const bool black = cur == gomoku::Stone::BLACK;
        auto status_bar = text(black ? "Current player: Black" : "Current player: White")
                          | bold | color(black ? Color::Red : Color::White) | hcenter;

        auto ai_bar = text(ai_status_text) | hcenter;
        if (active_index == 2)
            ai_bar |= ai_used_fallback ? color(Color::Yellow) : color(Color::Green);
        else
            ai_bar |= dim;

        Elements rows;
        for (int y = 0; y < kBoardSize; ++y) {
            Elements cols;
            for (int x = 0; x < kBoardSize; ++x) {
                const auto stone = board.getStone(x, y);
                const std::string cell = stone == gomoku::Stone::EMPTY ? " + "
                                 : stone == gomoku::Stone::BLACK ? " ○ " : " ● ";
                auto el = text(cell);
                if (x == current_x && y == current_y) el |= bgcolor(Color::Blue) | color(Color::White);
                else if (stone == gomoku::Stone::BLACK) el |= color(Color::Red);
                else if (stone == gomoku::Stone::WHITE) el |= color(Color::White);
                cols.push_back(el);
            }
            rows.push_back(hbox(std::move(cols)));
        }

        return vbox({status_bar, ai_bar, separator(), vbox(std::move(rows)) | hcenter}) | border | center;
    }
}