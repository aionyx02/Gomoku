#include "../../include/gomoku/ai/ai_player.h"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace gomoku::ai {
namespace {

constexpr std::array<std::pair<int, int>, 4> kDirections{{
    {1, 0},
    {0, 1},
    {1, 1},
    {1, -1}
}};

constexpr double kScoreEpsilon = 1e-6;

struct LinePattern {
    int length = 1;
    int open_ends = 0;
    int jump_extensions = 0;
};

struct PlacementReward {
    double total = 0.0;
    int open_threes = 0;
    int fours = 0;
    bool winning = false;
};

struct CandidateReward {
    double total = -std::numeric_limits<double>::infinity();
    double attack = 0.0;
    double defense = 0.0;
    double center = 0.0;
    double local = 0.0;
    bool winning = false;
    bool blocks_win = false;
};

[[nodiscard]] bool inBounds(const Board& board, const int x, const int y) {
    return x >= 0 && x < board.getSize() && y >= 0 && y < board.getSize();
}

[[nodiscard]] Stone opponentOf(const Stone stone) {
    return stone == Stone::BLACK ? Stone::WHITE : Stone::BLACK;
}

[[nodiscard]] bool boardIsEmpty(const Board& board) {
    const int size = board.getSize();
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.getStone(x, y) != Stone::EMPTY) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] int countContiguous(const Board& board,
                                  int x,
                                  int y,
                                  const int dx,
                                  const int dy,
                                  const Stone player) {
    int count = 0;
    x += dx;
    y += dy;

    while (inBounds(board, x, y) && board.getStone(x, y) == player) {
        ++count;
        x += dx;
        y += dy;
    }

    return count;
}

[[nodiscard]] int countJumpExtension(const Board& board,
                                     const int x,
                                     const int y,
                                     const int dx,
                                     const int dy,
                                     const Stone player,
                                     const int contiguous_count) {
    const int gap_x = x + dx * (contiguous_count + 1);
    const int gap_y = y + dy * (contiguous_count + 1);
    if (!inBounds(board, gap_x, gap_y) || board.getStone(gap_x, gap_y) != Stone::EMPTY) {
        return 0;
    }

    return countContiguous(board, gap_x, gap_y, dx, dy, player);
}

[[nodiscard]] LinePattern analyzeDirection(const Board& board,
                                           const int x,
                                           const int y,
                                           const int dx,
                                           const int dy,
                                           const Stone player) {
    const int left = countContiguous(board, x, y, -dx, -dy, player);
    const int right = countContiguous(board, x, y, dx, dy, player);

    LinePattern pattern;
    pattern.length = 1 + left + right;

    const int left_open_x = x - dx * (left + 1);
    if (const int left_open_y = y - dy * (left + 1); inBounds(board, left_open_x, left_open_y) && board.getStone(left_open_x, left_open_y) == Stone::EMPTY) {
        ++pattern.open_ends;
    }

    const int right_open_x = x + dx * (right + 1);
    if (const int right_open_y = y + dy * (right + 1); inBounds(board, right_open_x, right_open_y) && board.getStone(right_open_x, right_open_y) == Stone::EMPTY) {
        ++pattern.open_ends;
    }

    pattern.jump_extensions += countJumpExtension(board, x, y, dx, dy, player, right);
    pattern.jump_extensions += countJumpExtension(board, x, y, -dx, -dy, player, left);
    return pattern;
}

[[nodiscard]] double lineReward(const LinePattern& pattern) {
    if (pattern.length >= 5) {
        return 1'200'000.0;
    }

    double reward = 0.0;
    if (pattern.length == 4) {
        reward = pattern.open_ends == 2 ? 140'000.0 : pattern.open_ends == 1 ? 32'000.0 : 0.0;
    } else if (pattern.length == 3) {
        reward = pattern.open_ends == 2 ? 7'500.0 : pattern.open_ends == 1 ? 1'400.0 : 0.0;
    } else if (pattern.length == 2) {
        reward = pattern.open_ends == 2 ? 520.0 : pattern.open_ends == 1 ? 120.0 : 0.0;
    } else {
        reward = pattern.open_ends == 2 ? 40.0 : pattern.open_ends == 1 ? 12.0 : 0.0;
    }

    if (pattern.jump_extensions > 0) {
        reward += static_cast<double>(pattern.jump_extensions * pattern.jump_extensions) * 180.0;
        if (pattern.length >= 3) {
            reward += static_cast<double>(pattern.jump_extensions) * 950.0;
        }
    }

    return reward;
}

[[nodiscard]] PlacementReward evaluatePlacement(const Board& board,
                                                const int x,
                                                const int y,
                                                const Stone player) {
    PlacementReward reward;
    if (board.getStone(x, y) != Stone::EMPTY) {
        return reward;
    }

    for (const auto& [dx, dy] : kDirections) {
        const LinePattern pattern = analyzeDirection(board, x, y, dx, dy, player);
        reward.total += lineReward(pattern);

        if (pattern.length >= 5) {
            reward.winning = true;
        }
        if (pattern.length == 4 && pattern.open_ends >= 1) {
            ++reward.fours;
        }
        if (pattern.length == 3 && pattern.open_ends == 2) {
            ++reward.open_threes;
        }
    }

    if (reward.fours >= 2) {
        reward.total += 180'000.0;
    }
    if (reward.open_threes >= 2) {
        reward.total += 22'000.0;
    }
    if (reward.winning) {
        reward.total += 4'000'000.0;
    }

    return reward;
}

[[nodiscard]] double centerReward(const Board& board, const int x, const int y) {
    const double center = static_cast<double>(board.getSize() - 1) / 2.0;
    const double dx = static_cast<double>(x) - center;
    const double dy = static_cast<double>(y) - center;
    const double distance = std::hypot(dx, dy);
    const double max_distance = std::max(1.0, std::hypot(center, center));
    return (1.0 - distance / max_distance) * 90.0;
}

[[nodiscard]] double localReward(const Board& board,
                                 const int x,
                                 const int y,
                                 const Stone player) {
    double reward = 0.0;

    for (int offset_y = -2; offset_y <= 2; ++offset_y) {
        for (int offset_x = -2; offset_x <= 2; ++offset_x) {
            if (offset_x == 0 && offset_y == 0) {
                continue;
            }

            const int nx = x + offset_x;
            const int ny = y + offset_y;
            if (!inBounds(board, nx, ny)) {
                continue;
            }

            const Stone stone = board.getStone(nx, ny);
            if (stone == Stone::EMPTY) {
                continue;
            }

            const int distance = std::max(std::abs(offset_x), std::abs(offset_y));
            double contribution = distance == 1 ? 28.0 : 10.0;
            if (stone == player) {
                contribution += 8.0;
            } else {
                contribution += 4.0;
            }
            reward += contribution;
        }
    }

    return reward;
}

[[nodiscard]] CandidateReward evaluateCandidate(const Board& board, const int x, const int y) {
    const Stone player = board.getCurrentPlayer();
    const Stone opponent = opponentOf(player);

    const PlacementReward attack = evaluatePlacement(board, x, y, player);
    const PlacementReward defense = evaluatePlacement(board, x, y, opponent);

    CandidateReward reward;
    reward.attack = attack.total;
    reward.defense = defense.total;
    reward.center = centerReward(board, x, y);
    reward.local = localReward(board, x, y, player);
    reward.winning = attack.winning;
    reward.blocks_win = defense.winning;

    reward.total = attack.total * 1.08 + defense.total * 0.96 + reward.center + reward.local;
    if (attack.fours > 0 && defense.open_threes > 0) {
        reward.total += 18'000.0;
    }
    if (attack.open_threes > 0 && defense.fours > 0) {
        reward.total += 24'000.0;
    }
    if (attack.winning) {
        reward.total += 8'000'000.0;
    }
    if (defense.winning) {
        reward.total += 6'500'000.0;
    }

    return reward;
}

[[nodiscard]] bool shouldReplaceBest(const Board& board,
                                     const int candidate_x,
                                     const int candidate_y,
                                     const CandidateReward& candidate_reward,
                                     const int best_x,
                                     const int best_y,
                                     const CandidateReward& best_reward) {
    if (candidate_reward.total > best_reward.total + kScoreEpsilon) {
        return true;
    }
    if (candidate_reward.total + kScoreEpsilon < best_reward.total) {
        return false;
    }

    if (candidate_reward.attack > best_reward.attack + kScoreEpsilon) {
        return true;
    }
    if (candidate_reward.attack + kScoreEpsilon < best_reward.attack) {
        return false;
    }

    if (candidate_reward.defense > best_reward.defense + kScoreEpsilon) {
        return true;
    }
    if (candidate_reward.defense + kScoreEpsilon < best_reward.defense) {
        return false;
    }

    const double center = static_cast<double>(board.getSize() - 1) / 2.0;
    const double candidate_distance =
        std::abs(static_cast<double>(candidate_x) - center) + std::abs(static_cast<double>(candidate_y) - center);
    const double best_distance =
        std::abs(static_cast<double>(best_x) - center) + std::abs(static_cast<double>(best_y) - center);
    if (candidate_distance + kScoreEpsilon < best_distance) {
        return true;
    }
    if (candidate_distance > best_distance + kScoreEpsilon) {
        return false;
    }

    if (candidate_y != best_y) {
        return candidate_y < best_y;
    }
    return candidate_x < best_x;
}

[[nodiscard]] std::string describeReward(const CandidateReward& reward) {
    std::ostringstream stream;
    stream << "AI model output:  Total-Reward=" << std::llround(reward.total);
    if (reward.winning) {
        stream << "  (WIN)";
    } else if (reward.blocks_win) {
        stream << "  (BLK)";
    }
    stream << "\n"
           << "  Atk=" << std::llround(reward.attack)
           << "   Def=" << std::llround(reward.defense)
           << "   Ctr=" << std::llround(reward.center)
           << "   Loc=" << std::llround(reward.local);
    return stream.str();
}

} // namespace

MoveResult Player::makeMove(const Board& board) {
    if (board.getStatus() != GameStatus::PLAYING) {
        return {.move = std::nullopt, .used_fallback = false, .diagnostic = "board is not in PLAYING state"};
    }

    if (boardIsEmpty(board)) {
        const int center = board.getSize() / 2;
        return {
            .move = std::pair{center, center},
            .used_fallback = false,
            .diagnostic = "reward=open-center"
        };
    }

    std::optional<std::pair<int, int>> best_move;
    CandidateReward best_reward;

    const int size = board.getSize();
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (board.getStone(x, y) != Stone::EMPTY) {
                continue;
            }

            if (const CandidateReward candidate_reward = evaluateCandidate(board, x, y); !best_move ||
                                                                                         shouldReplaceBest(board, x, y, candidate_reward, best_move->first, best_move->second, best_reward)) {
                best_move = std::pair{x, y};
                best_reward = candidate_reward;
            }
        }
    }

    if (!best_move) {
        return {.move = std::nullopt, .used_fallback = false, .diagnostic = "no legal moves"};
    }

    return {
        .move = best_move,
        .used_fallback = false,
        .diagnostic = describeReward(best_reward)
    };
}

} // namespace gomoku::ai
