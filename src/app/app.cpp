/**
 * @file app.cpp
 * @author shawn
 * @date 2026/3/19
 * @brief the file is to play the gomoku game
 *
 * * Under the hood:
 * - Memory Layout:
 * - System Calls / Interactions:
 * - Resource Impact:
 */

#include "../../include/gomoku/core/game_session.h"
#include "../../include/gomoku/ui/ui_controller.h"
#include "../../include/gomoku/audio/voice.h"

#include <iostream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

bool prepareInteractiveTerminal() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
    const HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    DWORD output_mode = 0;
    DWORD input_mode = 0;
    if (output == INVALID_HANDLE_VALUE || input == INVALID_HANDLE_VALUE ||
        GetConsoleMode(output, &output_mode) == 0 ||
        GetConsoleMode(input, &input_mode) == 0) {
        std::cerr << "Gomoku needs an interactive terminal. Run Gomoku_Project.exe "
                  << "from Windows Terminal or PowerShell instead of an IDE output window.\n";
        return false;
    }

    constexpr DWORD kEnableVirtualTerminalProcessing = 0x0004;
    constexpr DWORD kDisableNewlineAutoReturn = 0x0008;
    constexpr DWORD kEnableVirtualTerminalInput = 0x0200;
    output_mode |= kEnableVirtualTerminalProcessing | kDisableNewlineAutoReturn;
    input_mode |= kEnableVirtualTerminalInput;
    if (SetConsoleMode(output, output_mode) == 0 || SetConsoleMode(input, input_mode) == 0) {
        std::cerr << "Gomoku could not enable virtual terminal support. Please use Windows Terminal.\n";
        return false;
    }
#endif
    return true;
}

} // namespace

int main() {
    if (!prepareInteractiveTerminal()) {
        return 1;
    }

    gomoku::GameSession session(15);
    const UI::Controller controller(session);

    voice::initAudioSystem();
    voice::backGroundMusic();

    controller.Start();

    voice::cleanupAudioSystem();
    return 0;
}
