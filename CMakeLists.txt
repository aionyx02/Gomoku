cmake_minimum_required(VERSION 3.20)
project(Gomoku LANGUAGES CXX)
if(NOT MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

if(MSVC)
    add_compile_options(/utf-8)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()

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
        src/ui/ui_controller_state.cpp
        src/ui/ui_controller_views.cpp
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

add_custom_command(TARGET Gomoku_Project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:Gomoku_Project>/assets
)

if(WIN32)
    target_link_libraries(Gomoku_Project PRIVATE ws2_32)
endif()

enable_testing()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/regression_tests.cpp)
    add_executable(gomoku_regression_tests
            tests/regression_tests.cpp
            src/ai/ai_player.cpp
            src/core/game_session.cpp
            src/net/webConnect.cpp
    )

    target_include_directories(gomoku_regression_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(gomoku_regression_tests PRIVATE
            GOMOKU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(gomoku_regression_tests PRIVATE gomoku_core)

    if(WIN32)
        target_link_libraries(gomoku_regression_tests PRIVATE ws2_32)
    endif()

    add_test(NAME gomoku_regression_tests COMMAND gomoku_regression_tests)
endif()
