// /**
//  * @file bindings.cpp
//  * @author shawn
//  * @date 2026/3/24
//  * @brief
//  *
//  * * Under the hood:
//  * - Memory Layout:
//  * - System Calls / Interactions:
//  * - Resource Impact:
//  */
//
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <algorithm>
// #include <array>
// #include "../../include/gomoku/core/engine.h"
//
// namespace py = pybind11;
//
// PYBIND11_MODULE(gomoku_ai, m) {
//     m.doc() = "Gomoku Core Engine built with C++ and pybind11";
//
//     py::enum_<gomoku::Stone>(m, "Stone")
//         .value("EMPTY", gomoku::Stone::EMPTY)
//         .value("BLACK", gomoku::Stone::BLACK)
//         .value("WHITE", gomoku::Stone::WHITE)
//         .export_values();
//
//     py::enum_<gomoku::GameStatus>(m, "GameStatus")
//         .value("PLAYING", gomoku::GameStatus::PLAYING)
//         .value("BLACK_WIN", gomoku::GameStatus::BLACK_WIN)
//         .value("WHITE_WIN", gomoku::GameStatus::WHITE_WIN)
//         .value("DRAW", gomoku::GameStatus::DRAW)
//         .export_values();
//
//     py::class_<gomoku::Board>(m, "Board")
//         .def(py::init<int>(), py::arg("size") = 15)
//         .def("place_stone", &gomoku::Board::placeStone)
//         .def("get_current_player", &gomoku::Board::getCurrentPlayer)
//         .def("get_status", &gomoku::Board::getStatus)
//         .def("get_stone", &gomoku::Board::getStone)
//         .def("get_observation", [](const gomoku::Board &b) {
//             const int size = b.getSize();
//
//             py::array_t<float> observation({3, size, size});
//             auto r = observation.mutable_unchecked<3>();
//
//             // me is your opponent, opponent is you, because the observation is from the perspective of the current player
//             const gomoku::Stone me = b.getCurrentPlayer();
//             const gomoku::Stone opponent = (me == gomoku::Stone::BLACK) ? gomoku::Stone::WHITE : gomoku::Stone::BLACK;
//             constexpr bool kUseThreatPlane = true;      // false => use constant plane below
//             constexpr float kGlobalPlaneValue = 1.0f;   // used only when kUseThreatPlane == false
//             constexpr std::array<std::pair<int, int>, 4> directions{{
//                 {1, 0}, {0, 1}, {1, 1}, {1, -1}
//             }};
//
//             const auto count_direction = [&](const int x, const int y, const int dx, const int dy, const gomoku::Stone player) {
//                 int count = 0;
//                 for (int step = 1; step < 5; ++step) {
//                     const int nx = x + dx * step;
//                     const int ny = y + dy * step;
//                     if (nx < 0 || nx >= size || ny < 0 || ny >= size) {
//                         break;
//                     }
//                     if (b.getStone(nx, ny) != player) {
//                         break;
//                     }
//                     ++count;
//                 }
//                 return count;
//             };
//
//             const auto open_ends_if_place = [&](const int x, const int y, const int dx, const int dy, const gomoku::Stone player) {
//                 const int left = count_direction(x, y, -dx, -dy, player);
//                 const int right = count_direction(x, y, dx, dy, player);
//
//                 const int lx = x - dx * (left + 1);
//                 const int ly = y - dy * (left + 1);
//                 const int rx = x + dx * (right + 1);
//                 const int ry = y + dy * (right + 1);
//
//                 int open_ends = 0;
//                 if (lx >= 0 && lx < size && ly >= 0 && ly < size &&
//                     b.getStone(lx, ly) == gomoku::Stone::EMPTY) {
//                     ++open_ends;
//                 }
//                 if (rx >= 0 && rx < size && ry >= 0 && ry < size &&
//                     b.getStone(rx, ry) == gomoku::Stone::EMPTY) {
//                     ++open_ends;
//                 }
//                 return open_ends;
//             };
//
//             const auto threat_for_player = [&](const int x, const int y, const gomoku::Stone player) {
//                 if (b.getStone(x, y) != gomoku::Stone::EMPTY) {
//                     return 0.0f;
//                 }
//
//                 float score = 0.0f;
//                 int open_three_count = 0;
//                 int four_plus_count = 0;
//
//                 for (const auto &[dx, dy] : directions) {
//                     const int line_len =
//                         1 + count_direction(x, y, -dx, -dy, player) + count_direction(x, y, dx, dy, player);
//                     if (line_len >= 5) {
//                         return 1.0f;
//                     }
//
//                     const int open_ends = open_ends_if_place(x, y, dx, dy, player);
//                     if (line_len == 4) {
//                         if (open_ends == 2) {
//                             score += 0.90f;
//                             ++four_plus_count;
//                         } else if (open_ends == 1) {
//                             score += 0.55f;
//                             ++four_plus_count;
//                         }
//                     } else if (line_len == 3) {
//                         if (open_ends == 2) {
//                             score += 0.38f;
//                             ++open_three_count;
//                         } else if (open_ends == 1) {
//                             score += 0.18f;
//                         }
//                     } else if (line_len == 2) {
//                         if (open_ends == 2) {
//                             score += 0.10f;
//                         } else if (open_ends == 1) {
//                             score += 0.05f;
//                         }
//                     }
//                 }
//
//                 if (four_plus_count >= 2) {
//                     score += 0.90f;
//                 }
//                 if (open_three_count >= 2) {
//                     score += 0.55f;
//                 }
//                 return std::clamp(score, 0.0f, 1.0f);
//             };
//
//             const auto plane_value = [&](const int x, const int y) {
//                 if constexpr (!kUseThreatPlane) {
//                     return kGlobalPlaneValue;
//                 }
//                 const float attack = threat_for_player(x, y, me);
//                 const float defend = threat_for_player(x, y, opponent);
//                 return std::clamp(0.62f * attack + 0.38f * defend, 0.0f, 1.0f);
//             };
//
//             for (int x = 0; x < size; ++x) {
//                 for (int y = 0; y < size; ++y) {
//                     const gomoku::Stone stone = b.getStone(x, y);
//                     r(0, x, y) = (stone == me) ? 1.0f : 0.0f; // My stones
//                     r(1, x, y) = (stone == opponent) ? 1.0f : 0.0f; // Opponent's stones
//                     r(2, x, y) = plane_value(x, y); // Threat plane (or global constant plane)
//                 }
//              }
//
//              return observation;
//         });
// }
