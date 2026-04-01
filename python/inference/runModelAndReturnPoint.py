import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import traceback
from typing import Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_core import (
    BLACK,
    EMPTY,
    WHITE,
    action_to_rc,
    apply_action_on_board,
    build_observation,
    legal_actions,
    terminal_value_for_to_play,
)
from gomoku_net import build_model_from_state_dict
from mcts_core import NeuralMCTS, pick_argmax_visit

DEFAULT_MCTS_SIMULATIONS = 200
DEFAULT_C_PUCT = 1.5
DEFAULT_MODEL_PATH = "models/az_prompt_smoke/gomoku_model.pt"

SYMBOL_TO_STONE = {".": EMPTY, "B": BLACK, "W": WHITE}
PLAYER_TO_STONE = {"B": BLACK, "W": WHITE}


@dataclass
class InferenceState:
    board: np.ndarray
    to_play: int
    terminal: bool
    winner: int


def apply_action(state: InferenceState, action: int) -> InferenceState:
    next_board, next_to_play, terminal, winner = apply_action_on_board(
        board=state.board,
        to_play=state.to_play,
        action=action,
    )
    return InferenceState(
        board=next_board,
        to_play=next_to_play,
        terminal=terminal,
        winner=winner,
    )


def build_mcts_runner(
    model: torch.nn.Module,
    board_size: int,
    simulations: int,
    c_puct: float,
) -> NeuralMCTS:
    def evaluate_state(state: InferenceState) -> Tuple[np.ndarray, float]:
        input_planes = int(getattr(model, "input_planes", 4))
        obs = build_observation(state.board, state.to_play, input_planes=input_planes)
        with torch.inference_mode():
            policy_logits, value = model(torch.from_numpy(obs).unsqueeze(0).float())
        logits = policy_logits.squeeze(0).cpu().numpy().astype(np.float64)
        return logits, float(value.item())

    return NeuralMCTS(
        action_size=int(board_size) * int(board_size),
        simulations=simulations,
        c_puct=c_puct,
        evaluate_state=evaluate_state,
        next_state=apply_action,
        legal_actions_fn=legal_actions,
        terminal_value_fn=terminal_value_for_to_play,
    )


@lru_cache(maxsize=4)
def load_model(model_path: str, board_size: int) -> torch.nn.Module:
    payload = torch.load(Path(model_path), map_location="cpu")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model, _ = build_model_from_state_dict(state_dict, board_size=board_size)
    return model.eval()


def _resolve_model(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for candidate in (Path.cwd() / p, Path(__file__).resolve().parent / p, PROJECT_ROOT / p):
        if candidate.exists():
            return candidate
    return p


def _policy_argmax_fallback(board: np.ndarray, current_player: int, model_path: str) -> Tuple[int, int]:
    legal = legal_actions(board)
    if legal.size == 0:
        raise ValueError("No legal moves")

    try:
        model = load_model(model_path, board.shape[0])
        input_planes = int(getattr(model, "input_planes", 4))
        obs = build_observation(board, current_player, input_planes=input_planes)
        with torch.inference_mode():
            logits, _ = model(torch.from_numpy(obs).unsqueeze(0).float())
        logits = logits.squeeze(0).cpu().numpy().astype(np.float64)
        masked = np.full_like(logits, -np.inf, dtype=np.float64)
        masked[legal] = logits[legal]
        best_action = int(np.argmax(masked))
    except Exception:
        best_action = int(legal[0])

    return action_to_rc(best_action, board.shape[0])


def _get_best_move_with_mcts(
    board: np.ndarray,
    current_player: int,
    model_path: str,
    simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
) -> Tuple[int, int]:
    state = InferenceState(
        board=board.copy(),
        to_play=int(current_player),
        terminal=False,
        winner=EMPTY,
    )
    model = load_model(model_path, board.shape[0])
    mcts = build_mcts_runner(
        model=model,
        board_size=board.shape[0],
        simulations=simulations,
        c_puct=c_puct,
    )
    root = mcts.run(state, use_root_noise=False)
    best_action = pick_argmax_visit(root)
    return action_to_rc(best_action, board.shape[0])


def get_best_move(board, current_player, model_path) -> tuple:
    """
    Args:
        board: 15x15 numpy array, 1=black, -1=white, 0=empty
        current_player: 1 (black) or -1 (white)
        model_path: checkpoint path
    Returns:
        (row, col)
    """
    board = np.asarray(board, dtype=np.int8)
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError("board must be a square 2D array")
    if int(current_player) not in (BLACK, WHITE):
        raise ValueError("current_player must be 1 or -1")

    resolved = _resolve_model(str(model_path))
    if not resolved.exists():
        raise FileNotFoundError(f"model not found: {resolved}")

    legal = legal_actions(board)
    if legal.size == 0:
        raise ValueError("No legal moves")

    try:
        return _get_best_move_with_mcts(
            board=board,
            current_player=int(current_player),
            model_path=str(resolved),
            simulations=DEFAULT_MCTS_SIMULATIONS,
            c_puct=DEFAULT_C_PUCT,
        )
    except Exception:
        return _policy_argmax_fallback(board, int(current_player), str(resolved))


def parse_board(board_text: str, size: int) -> np.ndarray:
    text = str(board_text)
    rows = text.split("|") if "|" in text else [text[i * size : (i + 1) * size] for i in range(size)]
    if len(rows) != size:
        raise ValueError(f"board row count mismatch: expected {size}, got {len(rows)}")

    board = np.zeros((size, size), dtype=np.int8)
    for row, line in enumerate(rows):
        if len(line) != size:
            raise ValueError(f"row {row} width mismatch")
        for col, symbol in enumerate(line):
            if symbol not in SYMBOL_TO_STONE:
                raise ValueError(f"invalid symbol '{symbol}' at ({row}, {col})")
            board[row, col] = SYMBOL_TO_STONE[symbol]
    return board


def getAIMove(
    board_text,
    current_player,
    model_path=DEFAULT_MODEL_PATH,
    board_size=15,
    mcts_simulations=200,
):
    key = str(current_player).upper()
    if key not in PLAYER_TO_STONE:
        raise ValueError("current_player must be B or W")

    board = parse_board(board_text, board_size)
    player = PLAYER_TO_STONE[key]
    resolved = _resolve_model(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"model not found: {resolved}")

    try:
        row, col = _get_best_move_with_mcts(
            board=board,
            current_player=player,
            model_path=str(resolved),
            simulations=int(mcts_simulations),
            c_puct=DEFAULT_C_PUCT,
        )
    except Exception:
        row, col = _policy_argmax_fallback(board, player, str(resolved))

    # Existing C++ bridge expects output as x y (col row).
    return col, row


def main() -> int:
    parser = argparse.ArgumentParser(description="Return AI move with MCTS + policy/value network")
    parser.add_argument("--board", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mcts-simulations", type=int, default=DEFAULT_MCTS_SIMULATIONS)
    args = parser.parse_args()

    try:
        x, y = getAIMove(
            board_text=args.board,
            current_player=args.current,
            model_path=args.model_path,
            board_size=args.board_size,
            mcts_simulations=args.mcts_simulations,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    print(f"{x} {y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
