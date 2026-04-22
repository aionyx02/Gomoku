/**
 * @file ui_controller.h
 * @brief Terminal UI controller for Gomoku.
 */
#ifndef GOMOKU_UI_CONTROLLER_H
#define GOMOKU_UI_CONTROLLER_H

#include <memory>

namespace gomoku {
class GameSession;
} // namespace gomoku

namespace UI {

class Controller {
public:
    explicit Controller(gomoku::GameSession& session);
    ~Controller();

    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    void Start() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace UI

#endif // GOMOKU_UI_CONTROLLER_H
