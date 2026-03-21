cmake_minimum_required(VERSION 3.20)
project(Gomoku)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(Gomoku_Project
        include/gomoku/main.h
        src/main.cpp
        src/app.cpp
        src/ui_controller.cpp
)

include(FetchContent)
FetchContent_Declare(ftxui
        GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui
        GIT_TAG v6.1.9
)
FetchContent_MakeAvailable(ftxui)

target_link_libraries(Gomoku_Project PRIVATE
        # Chose a submodule
        ftxui::component
        ftxui::dom
        ftxui::screen
)