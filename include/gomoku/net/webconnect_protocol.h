/**
 * @file webconnect_protocol.h
 * @brief Internal protocol helpers shared by the transport and regression tests.
 */
#ifndef GOMOKU_WEBCONNECT_PROTOCOL_H
#define GOMOKU_WEBCONNECT_PROTOCOL_H

#include "../core/game_session.h"

#include <cstddef>
#include <string>

namespace gomoku::net {

[[nodiscard]] bool applyRemoteMove(GameSession& session,
                                   Stone remote_stone,
                                   int x,
                                   int y,
                                   std::string& error);

[[nodiscard]] bool applySnapshotPacket(GameSession& session,
                                       std::size_t declared_move_count,
                                       const std::string& encoded_moves,
                                       std::string& error);

} // namespace gomoku::net

#endif // GOMOKU_WEBCONNECT_PROTOCOL_H
