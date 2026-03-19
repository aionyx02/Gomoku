cmake_minimum_required(VERSION 3.20)
project(Gomoku)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(Gomoku_Project
        include/gomoku/main.h
        src/main.cpp
        src/app.cpp
)