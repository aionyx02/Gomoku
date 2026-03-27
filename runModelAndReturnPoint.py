import argparse
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch

from gomoku_net import build_model_from_state_dict

EMPTY = 0
BLACK = 1
WHITE = 2

SYMBOL_TO_STONE = {
    ".": EMPTY,
    "B": BLACK,
    "W": WHITE,
}

PLAYER_TO_STONE = {
    "B": BLACK,
    "W": WHITE,
}


def parse_board(board_text: str, board_size: int) -> np.ndarray:
    rows = board_text.split("|")
    if len(rows) != board_size:
        raise ValueError(f"board row count mismatch: expected {board_size}, got {len(rows)}")

    # Keep the same indexing as the C++ binding: grid[x, y].
    grid = np.zeros((board_size, board_size), dtype=np.int8)
    for y, row in enumerate(rows):
        if len(row) != board_size:
            raise ValueError(f"board col count mismatch in row {y}: expected {board_size}, got {len(row)}")
        for x, symbol in enumerate(row):
            if symbol not in SYMBOL_TO_STONE:
                raise ValueError(f"invalid board symbol '{symbol}' at ({x}, {y})")
            grid[x, y] = SYMBOL_TO_STONE[symbol]

    return grid


def build_observation(grid: np.ndarray, current_player: int) -> np.ndarray:
    opponent = WHITE if current_player == BLACK else BLACK

    obs = np.empty((3, grid.shape[0], grid.shape[1]), dtype=np.float32)
    obs[0] = (grid == current_player).astype(np.float32)
    obs[1] = (grid == opponent).astype(np.float32)
    obs[2] = (grid == EMPTY).astype(np.float32)
    return obs


@lru_cache(maxsize=4)
def load_model(model_path: str, board_size: int) -> torch.nn.Module:
    model_file = Path(model_path)
    state_dict = torch.load(model_file, map_location="cpu")
    model, _ = build_model_from_state_dict(state_dict, board_size=board_size)
    model.eval()
    return model


def getAIMove(board_text: str, current_player: str, model_path: str = "gomoku_model.pt", board_size: int = 15):
    player_key = current_player.upper()
    if player_key not in PLAYER_TO_STONE:
        raise ValueError("current player must be B or W")

    grid = parse_board(board_text, board_size)
    player = PLAYER_TO_STONE[player_key]

    model_file = Path(model_path)
    if not model_file.is_absolute() and not model_file.exists():
        model_file = Path(__file__).resolve().parent / model_file
    if not model_file.exists():
        raise FileNotFoundError(f"model file not found: {model_file}")

    model = load_model(str(model_file.resolve()), board_size)

    obs = build_observation(grid, player)
    tensor_input = torch.from_numpy(obs).unsqueeze(0).float()

    with torch.inference_mode():
        policy, _ = model(tensor_input)

    move_scores = policy.squeeze(0).cpu().numpy().astype(np.float64)
    valid_mask = (grid.reshape(-1) == EMPTY)
    if not np.any(valid_mask):
        raise ValueError("no valid moves")

    move_scores[~valid_mask] = -np.inf
    best_move = int(np.argmax(move_scores))
    x, y = divmod(best_move, board_size)
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Return AI move from gomoku_model.pt")
    parser.add_argument("--board", required=True, help="serialized board rows joined by '|', using . B W")
    parser.add_argument("--current", required=True, help="current player, B or W")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--model-path", default="gomoku_model.pt")
    args = parser.parse_args()

    try:
        x, y = getAIMove(
            board_text=args.board,
            current_player=args.current,
            model_path=args.model_path,
            board_size=args.board_size,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{x} {y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
