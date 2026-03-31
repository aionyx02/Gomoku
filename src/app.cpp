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

#include "gomoku/game_session.h"
#include "gomoku/ui_controller.h"
#include "gomoku/voice.h"

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


int main() {

    gomoku::GameSession session(15, kSourceDir, kPythonExecutable);
    const UI::Controller controller(session);

    gameVoice.initAudioSystem();
    gameVoice.backGroundMusic();

    controller.Start();

    gameVoice.cleanupAudioSystem();
    return 0;
}
