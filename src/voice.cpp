/**
 * @file voice.cpp
 * @author shawn
 * @date 2026/3/31
 * @brief 
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */
#include "../include/gomoku/voice.h"

#define MINIAUDIO_IMPLEMENTATION
#include "../include/gomoku/miniaudio.h"

#include <iostream>

static ma_engine engine;
static ma_sound g_bgm;
static bool g_isInitalized = false;

voice gameVoice;

bool voice::initAudioSystem() {
    if (g_isInitalized) {
        return true;
    }

    ma_result result = ma_engine_init(nullptr, &engine);
    if (result != MA_SUCCESS) {
        std::cerr << "Failed to initialize audio engine: " << result << std::endl;
        return false;
    }

    result = ma_sound_init_from_file(&engine,"../voice/backGround.mp3", MA_SOUND_FLAG_STREAM, nullptr, nullptr, &g_bgm);

    if (result == MA_SUCCESS) {
        ma_sound_set_looping(&g_bgm, true);
    }
    g_isInitalized = true;
    return true;
}

void voice::cleanupAudioSystem() {
    if (g_isInitalized) {
        ma_engine_uninit(&engine);
        ma_sound_uninit(&g_bgm);
        g_isInitalized = false;
    }
}

void voice::clickSound() {
    if (!g_isInitalized) return;
    ma_engine_play_sound(&engine, "../voice/click.mp3", nullptr);
}

void voice::backGroundMusic() {
    if (!g_isInitalized) return;
    ma_engine_play_sound(&engine, "../voice/backGround.mp3", nullptr);
}

void voice::placeStoneSound() {
    if (!g_isInitalized) return;
    ma_engine_play_sound(&engine, "../voice/placeStoneVoice.mp3", nullptr);
}

void voice::stopBackgroundMusic() {
    if (!g_isInitalized) return;
    ma_sound_stop(&g_bgm);
}


