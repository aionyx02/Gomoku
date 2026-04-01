"""Shared Gomoku board rules and feature helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1

OPPONENT: Dict[int, int] = {BLACK: WHITE, WHITE: BLACK}
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def action_to_rc(action: int, board_size: int) -> Tuple[int, int]:
    return divmod(int(action), int(board_size))


def legal_actions(board: np.ndarray) -> np.ndarray:
    return np.flatnonzero(board.reshape(-1) == EMPTY).astype(np.int32)


def build_observation(board: np.ndarray, current_player: int, input_planes: int = 4) -> np.ndarray:
    input_planes = int(input_planes)
    size = board.shape[0]

    if input_planes <= 3:
        obs = np.empty((3, size, size), dtype=np.float32)
        obs[0] = (board == BLACK).astype(np.float32)
        obs[1] = (board == WHITE).astype(np.float32)
        obs[2] = (board == EMPTY).astype(np.float32)
        return obs

    obs = np.zeros((input_planes, size, size), dtype=np.float32)
    obs[0] = (board == BLACK).astype(np.float32)
    obs[1] = (board == WHITE).astype(np.float32)
    obs[2].fill(1.0 if int(current_player) == BLACK else 0.0)
    obs[3].fill(1.0)
    return obs


def check_win_from(board: np.ndarray, row: int, col: int, player: int) -> bool:
    size = board.shape[0]
    player = int(player)

    for dr, dc in DIRECTIONS:
        count = 1

        for step in range(1, 5):
            rr, cc = row + dr * step, col + dc * step
            if rr < 0 or rr >= size or cc < 0 or cc >= size:
                break
            if board[rr, cc] != player:
                break
            count += 1

        for step in range(1, 5):
            rr, cc = row - dr * step, col - dc * step
            if rr < 0 or rr >= size or cc < 0 or cc >= size:
                break
            if board[rr, cc] != player:
                break
            count += 1

        if count >= 5:
            return True

    return False


def apply_action_on_board(
    board: np.ndarray,
    to_play: int,
    action: int,
) -> Tuple[np.ndarray, int, bool, int]:
    size = board.shape[0]
    row, col = action_to_rc(action, size)
    player = int(to_play)

    if board[row, col] != EMPTY:
        raise ValueError(f"Illegal move ({row}, {col})")

    next_board = board.copy()
    next_board[row, col] = player
    next_to_play = OPPONENT[player]

    if check_win_from(next_board, row, col, player):
        return next_board, next_to_play, True, player

    if np.any(next_board == EMPTY):
        return next_board, next_to_play, False, EMPTY

    return next_board, next_to_play, True, EMPTY


def immediate_winning_actions(board: np.ndarray, player: int) -> np.ndarray:
    actions = legal_actions(board)
    if actions.size == 0:
        return actions

    wins = []
    for action in actions:
        row, col = action_to_rc(action, board.shape[0])
        board[row, col] = int(player)
        is_win = check_win_from(board, row, col, int(player))
        board[row, col] = EMPTY
        if is_win:
            wins.append(int(action))

    return np.array(wins, dtype=np.int32)


def immediate_bonus(board: np.ndarray, player: int, action: int, win_bonus: float, block_bonus: float) -> float:
    action = int(action)
    bonus = 0.0

    my_wins = immediate_winning_actions(board, int(player))
    if my_wins.size > 0 and np.any(my_wins == action):
        bonus += float(win_bonus)

    opp_wins = immediate_winning_actions(board, OPPONENT[int(player)])
    if opp_wins.size > 0 and np.any(opp_wins == action):
        bonus += float(block_bonus)

    return bonus


def terminal_value_for_to_play(winner: int, to_play: int) -> float:
    if int(winner) == EMPTY:
        return 0.0
    return 1.0 if int(winner) == int(to_play) else -1.0


def winner_to_black_value(winner: int) -> float:
    if int(winner) == BLACK:
        return 1.0
    if int(winner) == WHITE:
        return -1.0
    return 0.0
