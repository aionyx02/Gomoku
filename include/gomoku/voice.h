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

namespace gomoku {
    class voice {
    public:
        explicit voice() = default;

        ~voice() = default;

        voice(const voice &) = delete;

        voice &operator=(const voice &) = delete;

        voice(voice &&) noexcept = default;

        voice &operator=(voice &&) noexcept = default;

    private:
    };
}

#endif // voice_H