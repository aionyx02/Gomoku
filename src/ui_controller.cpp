#include "gomoku/ui_controller.h"
#include <vector>
namespace UI {
    using namespace ftxui;

    Controller::Controller(gomoku::Board& board) : board(board) {};

    void Controller::Start() {
        auto container = Container::Tab({
            this->RenderFrontPage(),
            this->RenderGameBoard(),
            this->RenderEndPage()
        }, &active_index);
    };
    ftxui::Component Controller::RenderFrontPage() {

    };
    Component Controller::RenderGameBoard() {
        return Renderer([this] {
            Elements rows;
            for (int x = 0; x < 15; ++x) {
                Elements cols;
                for (int y = 0; y < 15; ++y) {
                    const std::string cell = (board.getStone(x, y) == gomoku::Stone::EMPTY) ? " + " :
                                       (board.getStone(x, y) == gomoku::Stone::BLACK) ? " ○ " : " ● ";
                    auto element = text(cell);

                    if (x == current_x && y == current_y)
                        element |= bgcolor(Color::Blue);
                    cols.push_back(element | border);
                }
                rows.push_back(hbox(std::move(cols)));
            }
            return vbox(std::move(rows)) | center;
        })| CatchEvent([this](Event event) {
            if (event == Event::ArrowUp)
                current_y = std::max(0, current_y - 1);
            if (event == Event::Character(' ')) {
                active_index = 2;
            }
            return true;
        });
    };

    Component Controller::RenderEndPage() {

    };

    Element RenderGrid() {

    };
}