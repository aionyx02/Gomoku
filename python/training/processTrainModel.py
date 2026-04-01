import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PYTHON_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PYTHON_ROOT.parent
for base in (PYTHON_ROOT, PROJECT_ROOT):
    base_str = str(base)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)

try:
    from python.gomoku_core import (
        BLACK,
        EMPTY,
        WHITE,
        OPPONENT,
        action_to_rc,
        apply_action_on_board,
        build_observation,
        immediate_bonus,
        legal_actions,
        terminal_value_for_to_play,
        winner_to_black_value,
    )
    from python.gomoku_net import GomokuNet, build_model_from_state_dict, count_parameters
    from python.mcts_core import NeuralMCTS, pick_argmax_visit, visit_policy_from_root
except ModuleNotFoundError:
    from gomoku_core import (
        BLACK,
        EMPTY,
        WHITE,
        OPPONENT,
        action_to_rc,
        apply_action_on_board,
        build_observation,
        immediate_bonus,
        legal_actions,
        terminal_value_for_to_play,
        winner_to_black_value,
    )
    from gomoku_net import GomokuNet, build_model_from_state_dict, count_parameters
    from mcts_core import NeuralMCTS, pick_argmax_visit, visit_policy_from_root

_GOMOKU_AI_LOAD_ERROR: Optional[Exception] = None
try:
    import gomoku_ai
except Exception as exc:  # pragma: no cover - exercised in runtime environments without gomoku_ai.
    gomoku_ai = None
    _GOMOKU_AI_LOAD_ERROR = exc


@dataclass
class GameState:
    board: np.ndarray
    to_play: int
    terminal: bool
    winner: int
    move_count: int


@dataclass
class TrainingSample:
    observation: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.samples: List[TrainingSample] = []
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: TrainingSample) -> None:
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            return
        self.samples[self.cursor] = sample
        self.cursor = (self.cursor + 1) % self.capacity

    def extend(self, items: Sequence[TrainingSample]) -> None:
        for item in items:
            self.add(item)

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.integers(0, len(self.samples), size=int(batch_size))
        obs_batch = np.stack([self.samples[int(i)].observation for i in indices], axis=0).astype(np.float32)
        policy_batch = np.stack([self.samples[int(i)].policy for i in indices], axis=0).astype(np.float32)
        value_batch = np.array([self.samples[int(i)].value for i in indices], dtype=np.float32)
        return obs_batch, policy_batch, value_batch


class CppSelfPlayEngine:
    def __init__(self, board_size: int):
        if gomoku_ai is None:
            detail = f": {_GOMOKU_AI_LOAD_ERROR}" if _GOMOKU_AI_LOAD_ERROR else ""
            raise RuntimeError(f"gomoku_ai module is unavailable{detail}")

        self.gomoku_ai = gomoku_ai
        self.board_size = int(board_size)
        self.board = gomoku_ai.Board(self.board_size)
        self.grid = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.to_play = BLACK
        self.terminal = False
        self.winner = EMPTY
        self.move_count = 0

    def current_state(self) -> GameState:
        return GameState(
            board=self.grid.copy(),
            to_play=self.to_play,
            terminal=self.terminal,
            winner=self.winner,
            move_count=self.move_count,
        )

    def step(self, action: int) -> GameState:
        row, col = action_to_rc(action, self.board_size)
        x, y = col, row
        if not self.board.place_stone(x, y):
            raise ValueError(f"gomoku_ai rejected move ({x}, {y})")

        player = self.to_play
        self.grid[row, col] = player
        self.to_play = OPPONENT[player]
        self.move_count += 1

        status = self.board.get_status()
        if status == self.gomoku_ai.GameStatus.BLACK_WIN:
            self.terminal = True
            self.winner = BLACK
        elif status == self.gomoku_ai.GameStatus.WHITE_WIN:
            self.terminal = True
            self.winner = WHITE
        elif status == self.gomoku_ai.GameStatus.DRAW:
            self.terminal = True
            self.winner = EMPTY
        else:
            self.terminal = False
            self.winner = EMPTY

        return self.current_state()


def apply_action_python(state: GameState, action: int) -> GameState:
    if state.terminal:
        raise ValueError("Cannot apply move on terminal state")

    next_board, next_to_play, terminal, winner = apply_action_on_board(
        board=state.board,
        to_play=state.to_play,
        action=action,
    )
    return GameState(
        board=next_board,
        to_play=next_to_play,
        terminal=terminal,
        winner=winner,
        move_count=state.move_count + 1,
    )


def build_mcts_runner(
    model: torch.nn.Module,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: torch.device,
    rng: np.random.Generator,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> NeuralMCTS:
    def evaluate_state(state: GameState) -> Tuple[np.ndarray, float]:
        input_planes = int(getattr(model, "input_planes", 4))
        obs = build_observation(state.board, state.to_play, input_planes=input_planes)
        with torch.inference_mode():
            policy_logits, value = model(
                torch.from_numpy(obs).unsqueeze(0).to(device, dtype=torch.float32)
            )
        logits = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return logits, float(value.item())

    return NeuralMCTS(
        action_size=int(board_size) * int(board_size),
        simulations=simulations,
        c_puct=c_puct,
        evaluate_state=evaluate_state,
        next_state=apply_action_python,
        legal_actions_fn=legal_actions,
        terminal_value_fn=terminal_value_for_to_play,
        rng=rng,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )


def sample_action(policy: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(policy > 0.0)
    if legal.size == 0:
        raise ValueError("No legal actions in policy")
    probs = policy[legal].astype(np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(legal, p=probs))


def random_symmetry(obs: np.ndarray, policy: np.ndarray, board_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))

    obs_aug = np.rot90(obs, k, axes=(1, 2))
    pi_map = policy.reshape(board_size, board_size)
    pi_aug = np.rot90(pi_map, k, axes=(0, 1))

    if flip:
        obs_aug = np.flip(obs_aug, axis=1)
        pi_aug = np.flip(pi_aug, axis=0)

    return obs_aug.copy(), pi_aug.reshape(-1).copy()


def verify_state_consistency(predicted: GameState, cpp_state: GameState) -> None:
    if predicted.to_play != cpp_state.to_play:
        raise RuntimeError("to_play mismatch between Python and gomoku_ai")
    if predicted.terminal != cpp_state.terminal:
        raise RuntimeError("terminal mismatch between Python and gomoku_ai")
    if predicted.winner != cpp_state.winner:
        raise RuntimeError("winner mismatch between Python and gomoku_ai")
    if not np.array_equal(predicted.board, cpp_state.board):
        raise RuntimeError("board mismatch between Python and gomoku_ai")


def run_self_play_game(
    model: torch.nn.Module,
    board_size: int,
    mcts_simulations: int,
    c_puct: float,
    temperature_moves: int,
    final_temperature: float,
    win_bonus: float,
    block_bonus: float,
    rng: np.random.Generator,
    device: torch.device,
    use_cpp_engine: bool,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> Tuple[List[TrainingSample], int, int]:
    state = GameState(
        board=np.zeros((board_size, board_size), dtype=np.int8),
        to_play=BLACK,
        terminal=False,
        winner=EMPTY,
        move_count=0,
    )
    engine = CppSelfPlayEngine(board_size) if use_cpp_engine else None
    if engine is not None:
        state = engine.current_state()

    mcts = build_mcts_runner(
        model=model,
        board_size=board_size,
        simulations=mcts_simulations,
        c_puct=c_puct,
        device=device,
        rng=rng,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    trajectory: List[Tuple[np.ndarray, np.ndarray, int, float]] = []
    action_size = board_size * board_size

    while not state.terminal:
        root = mcts.run(state, use_root_noise=True)
        temperature = 1.0 if state.move_count < int(temperature_moves) else float(final_temperature)
        policy = visit_policy_from_root(root, action_size, temperature)
        action = sample_action(policy, rng)

        bonus = immediate_bonus(state.board, state.to_play, action, win_bonus=win_bonus, block_bonus=block_bonus)
        input_planes = int(getattr(model, "input_planes", 4))
        trajectory.append(
            (
                build_observation(state.board, state.to_play, input_planes=input_planes),
                policy,
                state.to_play,
                bonus,
            )
        )

        predicted = apply_action_python(state, action)
        if engine is None:
            state = predicted
        else:
            cpp_state = engine.step(action)
            verify_state_consistency(predicted, cpp_state)
            state = cpp_state

    black_value = winner_to_black_value(state.winner)
    samples: List[TrainingSample] = []
    for obs, policy, player, bonus in trajectory:
        base_value = black_value if player == BLACK else -black_value
        shaped_value = float(np.clip(base_value + bonus, -1.0, 1.0))
        samples.append(
            TrainingSample(
                observation=obs.astype(np.float32),
                policy=policy.astype(np.float32),
                value=shaped_value,
            )
        )
    return samples, state.winner, state.move_count


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    steps: int,
    value_loss_weight: float,
    grad_clip: float,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0

    for _ in range(int(steps)):
        obs_np, policy_np, value_np = replay.sample_batch(batch_size, rng)
        obs = torch.from_numpy(obs_np).to(device)
        policy_target = torch.from_numpy(policy_np).to(device)
        value_target = torch.from_numpy(value_np).to(device)

        logits, value_pred = model(obs)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred.squeeze(1), value_target)
        loss = policy_loss + float(value_loss_weight) * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_policy += float(policy_loss.item())
        total_value += float(value_loss.item())

    denom = max(1, int(steps))
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
    }


def run_arena_game(
    candidate_model: torch.nn.Module,
    best_model: torch.nn.Module,
    board_size: int,
    mcts_simulations: int,
    c_puct: float,
    candidate_plays_black: bool,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    state = GameState(
        board=np.zeros((board_size, board_size), dtype=np.int8),
        to_play=BLACK,
        terminal=False,
        winner=EMPTY,
        move_count=0,
    )

    candidate_mcts = build_mcts_runner(
        model=candidate_model,
        board_size=board_size,
        simulations=mcts_simulations,
        c_puct=c_puct,
        device=device,
        rng=rng,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )
    best_mcts = build_mcts_runner(
        model=best_model,
        board_size=board_size,
        simulations=mcts_simulations,
        c_puct=c_puct,
        device=device,
        rng=rng,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )

    while not state.terminal:
        candidate_turn = (state.to_play == BLACK and candidate_plays_black) or (
            state.to_play == WHITE and not candidate_plays_black
        )
        mcts = candidate_mcts if candidate_turn else best_mcts
        root = mcts.run(state, use_root_noise=False)
        action = pick_argmax_visit(root)
        state = apply_action_python(state, action)

    return state.winner


def evaluate_candidate_vs_best(
    candidate_model: torch.nn.Module,
    best_model: torch.nn.Module,
    board_size: int,
    arena_games: int,
    arena_mcts_simulations: int,
    c_puct: float,
    replace_threshold: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, float]:
    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(int(arena_games)):
        candidate_black = (game_idx % 2 == 0)
        winner = run_arena_game(
            candidate_model=candidate_model,
            best_model=best_model,
            board_size=board_size,
            mcts_simulations=arena_mcts_simulations,
            c_puct=c_puct,
            candidate_plays_black=candidate_black,
            rng=rng,
            device=device,
        )
        candidate_won = (winner == BLACK and candidate_black) or (winner == WHITE and not candidate_black)
        candidate_lost = (winner == WHITE and candidate_black) or (winner == BLACK and not candidate_black)

        if candidate_won:
            wins += 1
        elif candidate_lost:
            losses += 1
        else:
            draws += 1

    total = max(1, int(arena_games))
    win_rate = wins / total
    score_rate = (wins + 0.5 * draws) / total
    accepted = win_rate > float(replace_threshold)

    return {
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": float(win_rate),
        "score_rate": float(score_rate),
        "accepted": 1.0 if accepted else 0.0,
    }

def clone_model(model: torch.nn.Module, board_size: int, device: torch.device) -> torch.nn.Module:
    cloned, _ = build_model_from_state_dict(copy.deepcopy(model.state_dict()), board_size=board_size)
    cloned.to(device)
    return cloned


def resolve_device(device_name: str) -> torch.device:
    if device_name and device_name.lower() != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_cpp_engine(mode: str, board_size: int) -> Tuple[bool, str]:
    if mode == "python":
        return False, "forced python engine"
    try:
        engine = CppSelfPlayEngine(board_size)
        _ = engine.current_state()
        return True, "gomoku_ai.pyd loaded"
    except Exception as exc:
        if mode == "cpp":
            raise RuntimeError(f"Cannot use cpp engine: {exc}") from exc
        return False, f"cpp unavailable ({exc}); fallback to python"


def save_checkpoint(
    path: Path,
    best_model: torch.nn.Module,
    iteration: int,
    total_games: int,
    args: argparse.Namespace,
    replay_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": int(iteration),
            "total_self_play_games": int(total_games),
            "best_model_state_dict": best_model.state_dict(),
            "replay_size": int(replay_size),
            "args": vars(args),
        },
        path,
    )


def load_initial_best_model(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, int, int]:
    start_iteration = 1
    total_games = 0

    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location="cpu")
        state_dict = payload.get("best_model_state_dict")
        if state_dict is None:
            raise ValueError("resume checkpoint missing best_model_state_dict")
        best_model, _ = build_model_from_state_dict(state_dict, board_size=args.board_size)
        best_model.to(device)
        start_iteration = int(payload.get("iteration", 0)) + 1
        total_games = int(payload.get("total_self_play_games", 0))
        return best_model, start_iteration, total_games

    if args.init_model:
        payload = torch.load(args.init_model, map_location="cpu")
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
        best_model, _ = build_model_from_state_dict(state_dict, board_size=args.board_size)
        best_model.to(device)
        return best_model, start_iteration, total_games

    best_model = GomokuNet(
        board_size=args.board_size,
        input_planes=4,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        value_hidden_dim=args.value_hidden_dim,
    ).to(device)
    return best_model, start_iteration, total_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero-style self-play trainer for Gomoku.")

    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iteration", type=int, default=12)

    parser.add_argument("--mcts-simulations", type=int, default=400)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.30)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)

    parser.add_argument("--temperature-moves", type=int, default=18)
    parser.add_argument("--final-temperature", type=float, default=1e-3)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 => auto from replay size")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--replay-capacity", type=int, default=200_000)

    parser.add_argument("--arena-games", type=int, default=20)
    parser.add_argument("--arena-mcts-simulations", type=int, default=200)
    parser.add_argument("--replace-threshold", type=float, default=0.55)

    parser.add_argument("--win-in-one-bonus", type=float, default=0.20)
    parser.add_argument("--block-in-one-bonus", type=float, default=0.12)

    parser.add_argument("--checkpoint-every-games", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(PYTHON_ROOT / "models" / "alphazero"))
    parser.add_argument("--weights-name", type=str, default="gomoku_alphazero_best.pt")
    parser.add_argument("--checkpoint-name", type=str, default="gomoku_alphazero_latest.ckpt")
    parser.add_argument(
        "--publish-model-path",
        type=str,
        default=str(PROJECT_ROOT / "python" / "models" / "az_prompt_smoke" / "gomoku_model.pt"),
    )

    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--value-hidden-dim", type=int, default=256)

    parser.add_argument("--self-play-engine", choices=["auto", "cpp", "python"], default="auto")
    parser.add_argument("--no-symmetry-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--init-model", type=str, default="")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    if args.init_model:
        args.init_model = str(Path(args.init_model).resolve())
    if args.resume_checkpoint:
        args.resume_checkpoint = str(Path(args.resume_checkpoint).resolve())

    output_dir = Path(args.output_dir).resolve()
    publish_path = Path(args.publish_model_path).resolve() if args.publish_model_path else None
    weights_path = output_dir / args.weights_name
    latest_ckpt_path = output_dir / args.checkpoint_name

    use_cpp_engine, engine_message = detect_cpp_engine(args.self_play_engine, args.board_size)
    best_model, start_iteration, total_self_play_games = load_initial_best_model(args, device)
    replay = ReplayBuffer(args.replay_capacity)

    print(f"[setup] device={device}")
    print(f"[setup] engine={engine_message}")
    print(f"[setup] best_model_params={count_parameters(best_model):,}")
    print(f"[setup] start_iteration={start_iteration} total_self_play_games={total_self_play_games}")

    end_iteration = start_iteration + int(args.iterations) - 1
    for iteration in range(start_iteration, end_iteration + 1):
        best_model.eval()
        winners = {BLACK: 0, WHITE: 0, EMPTY: 0}
        total_moves = 0
        collected_samples = 0

        for game_idx in range(1, args.games_per_iteration + 1):
            samples, winner, moves = run_self_play_game(
                model=best_model,
                board_size=args.board_size,
                mcts_simulations=args.mcts_simulations,
                c_puct=args.c_puct,
                temperature_moves=args.temperature_moves,
                final_temperature=args.final_temperature,
                win_bonus=args.win_in_one_bonus,
                block_bonus=args.block_in_one_bonus,
                rng=rng,
                device=device,
                use_cpp_engine=use_cpp_engine,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_epsilon=args.dirichlet_epsilon,
            )
            replay.extend(samples)
            if not args.no_symmetry_augment:
                aug = []
                for sample in samples:
                    obs_aug, policy_aug = random_symmetry(sample.observation, sample.policy, args.board_size, rng)
                    aug.append(TrainingSample(obs_aug, policy_aug, sample.value))
                replay.extend(aug)

            total_self_play_games += 1
            winners[winner] += 1
            total_moves += moves
            collected_samples += len(samples)

            print(
                f"[self-play][iter {iteration}] game {game_idx}/{args.games_per_iteration} "
                f"winner={winner} moves={moves} replay={len(replay)} total_games={total_self_play_games}"
            )

            if args.checkpoint_every_games > 0 and total_self_play_games % args.checkpoint_every_games == 0:
                periodic = output_dir / f"checkpoint_game_{total_self_play_games}.ckpt"
                save_checkpoint(
                    path=periodic,
                    best_model=best_model,
                    iteration=iteration,
                    total_games=total_self_play_games,
                    args=args,
                    replay_size=len(replay),
                )
                save_checkpoint(
                    path=latest_ckpt_path,
                    best_model=best_model,
                    iteration=iteration,
                    total_games=total_self_play_games,
                    args=args,
                    replay_size=len(replay),
                )
                torch.save(best_model.state_dict(), weights_path)
                if publish_path:
                    publish_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_model.state_dict(), publish_path)
                print(f"[checkpoint] periodic={periodic}")

        if len(replay) < args.batch_size:
            print(f"[train][iter {iteration}] replay={len(replay)} < batch_size={args.batch_size}, skip training")
            continue

        candidate = clone_model(best_model, board_size=args.board_size, device=device)
        optimizer = torch.optim.AdamW(
            candidate.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else max(1, len(replay) // args.batch_size)
        epoch_losses = []
        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(
                model=candidate,
                optimizer=optimizer,
                replay=replay,
                batch_size=args.batch_size,
                steps=steps_per_epoch,
                value_loss_weight=args.value_loss_weight,
                grad_clip=args.grad_clip,
                device=device,
                rng=rng,
            )
            epoch_losses.append(metrics["loss"])
            print(
                f"[train][iter {iteration}] epoch {epoch}/{args.epochs} "
                f"loss={metrics['loss']:.4f} policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}"
            )

        candidate.eval()
        best_model.eval()
        arena = evaluate_candidate_vs_best(
            candidate_model=candidate,
            best_model=best_model,
            board_size=args.board_size,
            arena_games=args.arena_games,
            arena_mcts_simulations=args.arena_mcts_simulations,
            c_puct=args.c_puct,
            replace_threshold=args.replace_threshold,
            rng=rng,
            device=device,
        )
        accepted = bool(arena["accepted"] > 0.5)

        if accepted:
            best_model.load_state_dict(candidate.state_dict())

        save_checkpoint(
            path=latest_ckpt_path,
            best_model=best_model,
            iteration=iteration,
            total_games=total_self_play_games,
            args=args,
            replay_size=len(replay),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model.state_dict(), weights_path)
        if publish_path:
            publish_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_model.state_dict(), publish_path)

        avg_moves = (total_moves / args.games_per_iteration) if args.games_per_iteration > 0 else 0.0
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(
            f"[summary][iter {iteration}] samples={collected_samples} replay={len(replay)} "
            f"black={winners[BLACK]} white={winners[WHITE]} draw={winners[EMPTY]} avg_moves={avg_moves:.1f} "
            f"avg_loss={avg_loss:.4f}"
        )
        print(
            f"[arena][iter {iteration}] wins={int(arena['wins'])} losses={int(arena['losses'])} "
            f"draws={int(arena['draws'])} win_rate={arena['win_rate']:.3f} "
            f"score_rate={arena['score_rate']:.3f} accepted={accepted}"
        )
        print(f"[save][iter {iteration}] best_weights={weights_path}")
        print(f"[save][iter {iteration}] latest_checkpoint={latest_ckpt_path}")
        if publish_path:
            print(f"[save][iter {iteration}] publish={publish_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
