#include "gomoku/ai_player.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace gomoku::ai {
namespace fs = std::filesystem;

namespace {

char stoneChar(const gomoku::Stone stone) {
    if (stone == gomoku::Stone::BLACK) {
        return 'B';
    }
    if (stone == gomoku::Stone::WHITE) {
        return 'W';
    }
    return '.';
}

std::string quoteArg(const std::string& value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            out += "\\\"";
        } else {
            out += ch;
        }
    }
    return out + "\"";
}

std::string compactMessage(const std::string& text, const size_t max_len = 180) {
    std::string out;
    bool has_space = false;
    for (const char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!has_space && !out.empty()) {
                out += ' ';
            }
            has_space = true;
            continue;
        }
        out += ch;
        has_space = false;
    }

    if (out.size() <= max_len) {
        return out;
    }
    return out.substr(0, max_len - 3) + "...";
}

fs::path executableDir() {
#if defined(_WIN32)
    std::array<char, 4096> buffer{};
    if (const DWORD size = GetModuleFileNameA(nullptr, buffer.data(), static_cast<DWORD>(buffer.size())); size > 0 && size < buffer.size()) {
        return fs::path(std::string(buffer.data(), size)).parent_path();
    }
#endif

    std::error_code ec;
    return fs::current_path(ec);
}

std::vector<fs::path> baseDirs(const std::string& source_dir) {
    std::vector<fs::path> paths;
    const auto add = [&paths](fs::path value) {
        std::error_code ec;
        value = fs::absolute(value, ec).lexically_normal();
        const auto key = value.generic_string();
        for (const auto& existing : paths) {
            if (existing.generic_string() == key) {
                return;
            }
        }
        paths.push_back(std::move(value));
    };

    const fs::path exe_dir = executableDir();
    add(source_dir);
    add(exe_dir);
    add(exe_dir.parent_path());
    add(fs::current_path());
    add(fs::current_path().parent_path());
    return paths;
}

std::optional<fs::path> findFile(const std::vector<fs::path>& bases, const std::string& name) {
    for (const auto& base : bases) {
        const auto candidate = (base / name).lexically_normal();
        if (std::error_code ec; fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
            return candidate;
        }
    }
    return std::nullopt;
}

std::string searchedPaths(const std::vector<fs::path>& bases, const std::string& name) {
    std::string output;
    for (const auto& base : bases) {
        if (!output.empty()) {
            output += ", ";
        }
        output += (base / name).generic_string();
    }
    return output;
}

void writeDebugLog(const std::string& source_dir, const std::string& content) {
    std::error_code ec;
    fs::path debug_log_path = fs::path(source_dir) / "ai_debug.log";
    debug_log_path = fs::absolute(debug_log_path, ec).lexically_normal();
    if (FILE* file = fopen(debug_log_path.string().c_str(), "w")) {
        fputs(content.c_str(), file);
        fclose(file);
    }
}

std::vector<std::string> pythonCommands(const std::vector<fs::path>& bases, std::string preferred_python) {
    std::vector<std::string> commands;
    std::vector<std::string> seen;

    const auto push = [&commands, &seen](std::string command) {
        if (command.empty()) {
            return;
        }
        if (command.size() >= 2 && command.front() == '"' && command.back() == '"') {
            command = command.substr(1, command.size() - 2);
        }

        std::string key = command;
        for (auto& ch : key) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        if (std::ranges::find(seen, key) != seen.end()) {
            return;
        }
        seen.push_back(std::move(key));

        if (command.find_first_of("/\\:") != std::string::npos) {
            if (std::error_code ec; !fs::exists(command, ec)) {
                return;
            }
            command = fs::path(command).lexically_normal().generic_string();
        }
        commands.push_back(std::move(command));
    };

    if (const char* env_py = std::getenv("GOMOKU_PYTHON_EXECUTABLE"); env_py && *env_py) {
        push(env_py);
    }
    push(std::move(preferred_python));

    static constexpr std::array kCandidateExecutables = {
        ".gomoku-python/Scripts/python.exe",
        ".gomoku-python/python.exe",
        ".venv/Scripts/python.exe",
        ".venv/bin/python3",
        ".venv/bin/python",
        "Scripts/python.exe",
        "python/bin/python3",
        "python/bin/python"
    };
    for (const auto& base : bases) {
        for (const auto* relative : kCandidateExecutables) {
            push((base / relative).generic_string());
        }
    }

    push("python");
    push("python3");
    return commands;
}

std::string serializeBoard(const gomoku::Board& board) {
    const int size = board.getSize();
    std::string out;
    out.reserve(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            out += stoneChar(board.getStone(x, y));
        }
    }
    return out;
}

std::optional<std::pair<int, int>> parseMove(const std::string& output) {
    std::istringstream ss(output);
    std::string line;
    std::optional<std::pair<int, int>> parsed_move;

    while (std::getline(ss, line)) {
        std::istringstream line_stream(line);
        int x = 0;
        int y = 0;
        if (std::string trailing; (line_stream >> x >> y) && !(line_stream >> trailing)) {
            parsed_move = std::pair{x, y};
        }
    }

    return parsed_move;
}

#if defined(_WIN32)
int runProcess(const std::string& command_line, std::string& output) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        return -1;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA startup_info{};
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdOutput = write_pipe;
    startup_info.hStdError = write_pipe;
    startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION process_info{};
    std::vector<char> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back('\0');

    if (!CreateProcessA(nullptr,
                        mutable_command.data(),
                        nullptr,
                        nullptr,
                        TRUE,
                        CREATE_NO_WINDOW,
                        nullptr,
                        nullptr,
                        &startup_info,
                        &process_info)) {
        CloseHandle(read_pipe);
        CloseHandle(write_pipe);
        return -1;
    }

    CloseHandle(write_pipe);

    std::array<char, 4096> buffer{};
    DWORD read = 0;
    while (ReadFile(read_pipe, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr) && read > 0) {
        output.append(buffer.data(), read);
    }

    CloseHandle(read_pipe);
    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD exit_code = 1;
    GetExitCodeProcess(process_info.hProcess, &exit_code);
    CloseHandle(process_info.hProcess);
    CloseHandle(process_info.hThread);

    return static_cast<int>(exit_code);
}
#endif

struct RawModelResult {
    std::optional<std::pair<int, int>> move;
    std::string reason;
};

RawModelResult queryModelMove(const gomoku::Board& board,
                              const std::string& source_dir,
                              const std::string& preferred_python) {
    const auto bases = baseDirs(source_dir);

    std::optional<fs::path> script;
    for (const auto* candidate : {
             "python/inference/runModelAndReturnPoint.py",
             "python/runModelAndReturnPoint.py",
             "runModelAndReturnPoint.py"
         }) {
        script = findFile(bases, candidate);
        if (script) {
            break;
        }
    }
    if (!script) {
        const std::string reason =
            "runModelAndReturnPoint.py not found in expected paths: python/inference, python, project root";
        writeDebugLog(source_dir, reason + "\n");
        return {
            std::nullopt,
            reason
        };
    }

    static constexpr std::array kModelCandidates = {
        "best.pt",
        "python/models/az_prompt_smoke/best.pt",
        "python/models/az_prompt_smoke/gomoku_model.pt",
        "python/models/az_prompt_smoke/publish.pt",
        "python/models/alphazero/best.pt",
        "python/models/alphazero/gomoku_alphazero_best.pt",
        "python/models/alphazero_bootstrap/gomoku_alphazero_bootstrap.pt",
        "gomoku_model.pt",
        "gomoku_model.pt.bak",
        "python/models/gomoku_model.pt"
    };

    std::optional<fs::path> model;
    for (const auto* candidate : kModelCandidates) {
        model = findFile(bases, candidate);
        if (model) {
            break;
        }
    }
    if (!model) {
        std::string searched_summary;
        std::string debug_detail = "[model-search]\n";
        for (const auto* candidate : kModelCandidates) {
            if (!searched_summary.empty()) {
                searched_summary += " || ";
            }
            searched_summary += compactMessage(searchedPaths(bases, candidate), 220);
            debug_detail += "[candidate] ";
            debug_detail += candidate;
            debug_detail += "\n";
            debug_detail += searchedPaths(bases, candidate);
            debug_detail += "\n";
        }
        const std::string reason =
            "no model checkpoint found; searched: " + compactMessage(searched_summary, 1200);
        writeDebugLog(source_dir, debug_detail + "\n[reason] " + reason + "\n");
        return {
            std::nullopt,
            reason
        };
    }

    const auto candidates = pythonCommands(bases, preferred_python);
    if (candidates.empty()) {
        return {std::nullopt, "no usable python command found"};
    }

    const std::string board_text = serializeBoard(board);
    const char current = stoneChar(board.getCurrentPlayer());
    const std::string args =
        "--board-size " + std::to_string(board.getSize()) + " "
        "--model-path " + quoteArg(model->generic_string()) + " "
        "--current " + std::string(1, current) + " "
        "--board " + quoteArg(board_text);

    std::vector<std::string> logs;
    logs.reserve(candidates.size());
    std::string debug_log;

    for (const auto& python : candidates) {
        std::string output;
        debug_log += "[try] py=" + python +
                     " script=" + script->generic_string() +
                     " model=" + model->generic_string() + "\n";
#if defined(_WIN32)
        const std::string command =
            quoteArg(python) + " " + quoteArg(script->generic_string()) + " " + args;
        const int rc = runProcess(command, output);
#else
        const std::string command =
            quoteArg(python) + " " + quoteArg(script->generic_string()) + " " + args + " 2>&1";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            logs.push_back("py=" + python + " -> cannot spawn");
            continue;
        }
        std::array<char, 256> buffer{};
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
            output += buffer.data();
        }
        const int rc = pclose(pipe);
#endif

        if (rc != 0) {
            logs.push_back("py=" + python + " exit=" + std::to_string(rc) + ": " + compactMessage(output));
            debug_log += "exit=" + std::to_string(rc) + "\n";
            if (output.empty()) {
                debug_log += "(no output)\n";
            } else {
                debug_log += output;
                if (output.back() != '\n') {
                    debug_log += "\n";
                }
            }
            debug_log += "\n";
            continue;
        }

        const auto parsed = parseMove(output);
        if (!parsed) {
            writeDebugLog(source_dir, debug_log + "[parse-error]\n" + output + "\n");
            return {
                std::nullopt,
                "cannot parse AI move (py=" + python + "): " + compactMessage(output)
            };
        }

        const auto [x, y] = *parsed;
        if (const int size = board.getSize(); x < 0 || x >= size || y < 0 || y >= size) {
            writeDebugLog(source_dir, debug_log + "[range-error] output:\n" + output + "\n");
            return {
                std::nullopt,
                "AI move out of range: (" + std::to_string(x) + "," + std::to_string(y) + ")"
            };
        }

        if (board.getStone(x, y) != gomoku::Stone::EMPTY) {
            writeDebugLog(source_dir, debug_log + "[occupied-error] output:\n" + output + "\n");
            return {
                std::nullopt,
                "AI move is not empty: (" + std::to_string(x) + "," + std::to_string(y) + ")"
            };
        }

        return {std::pair{x, y}, ""};
    }

    std::string summary;
    for (const auto& entry : logs) {
        if (!summary.empty()) {
            summary += " | ";
        }
        summary += entry;
    }

    writeDebugLog(source_dir, debug_log + "[summary] " + summary + "\n");

    return {
        std::nullopt,
        "all python candidates failed: " + compactMessage(summary, 1200)
    };
}

std::optional<std::pair<int, int>> fallbackMove(const gomoku::Board& board) {
    const int size = board.getSize();
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.getStone(x, y) == gomoku::Stone::EMPTY) {
                return std::pair{x, y};
            }
        }
    }
    return std::nullopt;
}

} // namespace

Player::Player(std::string source_dir, std::string preferred_python)
    : source_dir_(std::move(source_dir)),
      preferred_python_(std::move(preferred_python)) {}

MoveResult Player::makeMove(const Board& board) const {
    auto [model_move, reason] = queryModelMove(board, source_dir_, preferred_python_);
    if (model_move) {
        return {.move = model_move, .used_fallback = false, .diagnostic = ""};
    }

    const auto fallback = fallbackMove(board);
    if (!fallback) {
        if (reason.empty()) {
            reason = "no legal fallback move";
        }
        return {.move = std::nullopt, .used_fallback = true, .diagnostic = std::move(reason)};
    }

    return {.move = fallback, .used_fallback = true, .diagnostic = std::move(reason)};
}

} // namespace gomoku::ai
