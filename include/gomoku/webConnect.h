/**
 * @file webConnect.h
 * @author shawn
 * @date 2026/3/31
 * @brief 
 *
 * * Under the hood:
 * - Memory Layout: 
 * - System Calls / Interactions: 
 * - Resource Impact: 
 */
#ifndef webConnect_H
#define webConnect_H

namespace gomoku {
    class webConnect {
    public:
        explicit webConnect() = default;

        ~webConnect() = default;

        webConnect(const webConnect &) = delete;

        webConnect &operator=(const webConnect &) = delete;

        webConnect(webConnect &&) noexcept = default;

        webConnect &operator=(webConnect &&) noexcept = default;

    private:
    };
}

#endif // webConnect_H