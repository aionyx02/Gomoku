cmake_minimum_required(VERSION 3.20)
project(Gomoku LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(ftxui
        GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui
        GIT_TAG v6.1.9
)
FetchContent_MakeAvailable(ftxui)

add_library(gomoku_core STATIC src/core/engine.cpp src/audio/voice.cpp)
target_include_directories(gomoku_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(gomoku_core PRIVATE GOMOKU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")


add_executable(Gomoku_Project
        src/app/app.cpp
        src/ai/ai_player.cpp
        src/core/game_session.cpp
        src/ui/ui_controller.cpp
        src/audio/voice.cpp
        src/net/webConnect.cpp
)

target_include_directories(Gomoku_Project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(Gomoku_Project PRIVATE
        GOMOKU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(Gomoku_Project PRIVATE
        ftxui::component
        ftxui::dom
        ftxui::screen
        gomoku_core
)

if(WIN32)
    target_link_libraries(Gomoku_Project PRIVATE ws2_32)
endif()