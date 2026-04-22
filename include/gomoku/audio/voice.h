/**
 * @file voice.h
 * @author shawn
 * @date 2026/3/31
 * @brief 
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */
#ifndef voice_H
#define voice_H

class voice {
public:
    explicit voice() = default;

    ~voice() = default;

    static bool initAudioSystem();

    static void cleanupAudioSystem();

    static void clickSound();

    static void backGroundMusic();

    static void placeStoneSound();

    static void victorySound();

    static void defeatSound();

    static void stopBackgroundMusic();
private:
};

extern voice gameVoice;

#endif // voice_H