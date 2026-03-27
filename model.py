import math
from collections import deque

import numpy as np
import torch

import gomoku_ai

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
EPS = 1e-12

EMPTY = 0
BLACK = 1
WHITE = 2

STATUS_PLAYING = 0
STATUS_BLACK_WIN = 1
STATUS_WHITE_WIN = 2
STATUS_DRAW = 3


def _stone_to_int(stone):
    if stone == gomoku_ai.Stone.BLACK:
        return BLACK
    if stone == gomoku_ai.Stone.WHITE:
        return WHITE
    return EMPTY


def _status_to_int(status):
    if status == gomoku_ai.GameStatus.BLACK_WIN:
        return STATUS_BLACK_WIN
    if status == gomoku_ai.GameStatus.WHITE_WIN:
        return STATUS_WHITE_WIN
    if status == gomoku_ai.GameStatus.DRAW:
        return STATUS_DRAW
    return STATUS_PLAYING


def _to_policy_probs_numpy(policy_output):
    if isinstance(policy_output, torch.Tensor):
        policy = (
            policy_output.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
            .numpy()
            .astype(np.float64, copy=False)
        )
    else:
        policy = np.asarray(policy_output, dtype=np.float64).reshape(-1)

    if policy.size == 0:
        return policy

    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)

    if np.all(np.isfinite(policy)) and np.all(policy >= 0.0):
        total = policy.sum()
        if total > 1e-12 and abs(total - 1.0) < 1e-3:
            return policy / total

    shifted = policy - np.max(policy)
    exp_policy = np.exp(shifted)
    exp_sum = exp_policy.sum()
    if exp_sum <= 1e-12:
        return np.full_like(policy, 1.0 / policy.size, dtype=np.float64)
    return exp_policy / exp_sum


def _policy_log_probs_torch(policy_output):
    if policy_output.ndim != 2:
        raise ValueError(
            f"Policy output must be 2D [batch, board_size*board_size], got {policy_output.shape}"
        )

    is_non_negative = bool(torch.all(policy_output >= 0).item())
    if is_non_negative:
        sums = policy_output.sum(dim=1, keepdim=True)
        is_prob_like = bool(torch.all(torch.abs(sums - 1.0) < 1e-3).item())
        if is_prob_like:
            return torch.log(policy_output.clamp_min(1e-12))

    return torch.log_softmax(policy_output, dim=1)


def _augment_state_policy_symmetries(state, policy):
    state_arr = np.asarray(state, dtype=np.float32)
    policy_arr = np.asarray(policy, dtype=np.float32).reshape(-1)

    if state_arr.ndim != 3:
        raise ValueError(f"state must be [C, H, W], got {state_arr.shape}")

    board_size = int(state_arr.shape[1])
    if state_arr.shape[2] != board_size:
        raise ValueError(f"state must be square on spatial dims, got {state_arr.shape}")
    if policy_arr.size != board_size * board_size:
        raise ValueError(
            f"policy size mismatch: expected {board_size * board_size}, got {policy_arr.size}"
        )

    policy_2d = policy_arr.reshape(board_size, board_size)
    augmented = []
    for k in range(4):
        rot_state = np.rot90(state_arr, k=k, axes=(1, 2))
        rot_policy = np.rot90(policy_2d, k=k)
        augmented.append((rot_state.copy(), rot_policy.reshape(-1).copy()))

        # Flip along y-axis for [x, y] board indexing.
        flip_state = np.flip(rot_state, axis=2)
        flip_policy = np.flip(rot_policy, axis=1)
        augmented.append((flip_state.copy(), flip_policy.reshape(-1).copy()))

    return augmented


class _BoardState:

    def __init__(self, size, grid, current_player, status):
        self.size = int(size)
        self.grid = grid
        self.current_player = int(current_player)
        self.status = int(status)

    @classmethod
    def from_cpp_board(cls, board):
        obs = board.get_observation()
        size = int(obs.shape[1])
        grid = np.zeros((size, size), dtype=np.int8)
        for x in range(size):
            for y in range(size):
                grid[x, y] = _stone_to_int(board.get_stone(x, y))
        return cls(
            size=size,
            grid=grid,
            current_player=_stone_to_int(board.get_current_player()),
            status=_status_to_int(board.get_status()),
        )

    def clone(self):
        return _BoardState(
            size=self.size,
            grid=self.grid.copy(),
            current_player=self.current_player,
            status=self.status,
        )

    def get_valid_moves(self):
        if self.status != STATUS_PLAYING:
            return np.zeros(self.size * self.size, dtype=np.float32)
        return (self.grid.reshape(-1) == EMPTY).astype(np.float32)

    def get_observation(self):
        me = self.current_player
        opp = WHITE if me == BLACK else BLACK
        obs = np.empty((3, self.size, self.size), dtype=np.float32)
        obs[0] = (self.grid == me).astype(np.float32)
        obs[1] = (self.grid == opp).astype(np.float32)
        obs[2] = (self.grid == EMPTY).astype(np.float32)
        return obs

    def place(self, action):
        if self.status != STATUS_PLAYING:
            return False

        x, y = divmod(int(action), self.size)
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.grid[x, y] != EMPTY:
            return False

        player = self.current_player
        self.grid[x, y] = player

        if self._check_win(x, y):
            self.status = STATUS_BLACK_WIN if player == BLACK else STATUS_WHITE_WIN
            return True

        if not np.any(self.grid == EMPTY):
            self.status = STATUS_DRAW
            return True

        self.current_player = WHITE if player == BLACK else BLACK
        return True

    def terminal_value(self):
        if self.status == STATUS_DRAW:
            return 0.0
        return -1.0

    def _check_win(self, last_x, last_y):
        target = self.grid[last_x, last_y]
        if target == EMPTY:
            return False

        for dx, dy in DIRECTIONS:
            count = 1

            step = 1
            while step < 5:
                x = last_x + dx * step
                y = last_y + dy * step
                if x < 0 or x >= self.size or y < 0 or y >= self.size:
                    break
                if self.grid[x, y] != target:
                    break
                count += 1
                step += 1

            step = 1
            while step < 5:
                x = last_x - dx * step
                y = last_y - dy * step
                if x < 0 or x >= self.size or y < 0 or y >= self.size:
                    break
                if self.grid[x, y] != target:
                    break
                count += 1
                step += 1

            if count >= 5:
                return True

        return False

    def _count_direction(self, x, y, dx, dy, player):
        count = 0
        step = 1
        while step < 5:
            nx = x + dx * step
            ny = y + dy * step
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                break
            if self.grid[nx, ny] != player:
                break
            count += 1
            step += 1
        return count

    def _line_length_if_place(self, x, y, player, dx, dy):
        left = self._count_direction(x, y, -dx, -dy, player)
        right = self._count_direction(x, y, dx, dy, player)
        return 1 + left + right

    def _open_ends_if_place(self, x, y, player, dx, dy):
        left = self._count_direction(x, y, -dx, -dy, player)
        right = self._count_direction(x, y, dx, dy, player)

        lx = x - dx * (left + 1)
        ly = y - dy * (left + 1)
        rx = x + dx * (right + 1)
        ry = y + dy * (right + 1)

        open_ends = 0
        if 0 <= lx < self.size and 0 <= ly < self.size and self.grid[lx, ly] == EMPTY:
            open_ends += 1
        if 0 <= rx < self.size and 0 <= ry < self.size and self.grid[rx, ry] == EMPTY:
            open_ends += 1
        return open_ends

    def is_winning_action(self, action, player=None):
        if self.status != STATUS_PLAYING:
            return False

        x, y = divmod(int(action), self.size)
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.grid[x, y] != EMPTY:
            return False

        if player is None:
            player = self.current_player
        player = int(player)

        for dx, dy in DIRECTIONS:
            if self._line_length_if_place(x, y, player, dx, dy) >= 5:
                return True
        return False

    def tactical_score(self, action, player=None):
        if self.status != STATUS_PLAYING:
            return 0.0

        x, y = divmod(int(action), self.size)
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return 0.0
        if self.grid[x, y] != EMPTY:
            return 0.0

        if player is None:
            player = self.current_player
        player = int(player)

        score = 0.0
        open_three_count = 0
        four_plus_count = 0
        for dx, dy in DIRECTIONS:
            line_len = self._line_length_if_place(x, y, player, dx, dy)
            if line_len >= 5:
                return 1_000_000.0

            open_ends = self._open_ends_if_place(x, y, player, dx, dy)
            if line_len == 4:
                if open_ends == 2:
                    score += 60_000.0
                    four_plus_count += 1
                elif open_ends == 1:
                    score += 12_000.0
                    four_plus_count += 1
            elif line_len == 3:
                if open_ends == 2:
                    score += 1_500.0
                    open_three_count += 1
                elif open_ends == 1:
                    score += 220.0
            elif line_len == 2:
                if open_ends == 2:
                    score += 80.0
                elif open_ends == 1:
                    score += 16.0

        if four_plus_count >= 2:
            score += 90_000.0
        if open_three_count >= 2:
            score += 6_000.0
        return score

    def get_forced_actions(self):
        valid_actions = np.flatnonzero(self.get_valid_moves())
        if valid_actions.size == 0:
            empty = np.asarray([], dtype=np.int32)
            return empty, empty

        me = self.current_player
        opp = WHITE if me == BLACK else BLACK

        winning = [int(a) for a in valid_actions if self.is_winning_action(int(a), me)]
        if winning:
            return np.asarray(winning, dtype=np.int32), np.asarray([], dtype=np.int32)

        blocks = [int(a) for a in valid_actions if self.is_winning_action(int(a), opp)]
        if blocks:
            return np.asarray([], dtype=np.int32), np.asarray(blocks, dtype=np.int32)

        empty = np.asarray([], dtype=np.int32)
        return empty, empty

    def get_candidate_actions(self, radius=2, max_actions=72):
        if self.status != STATUS_PLAYING:
            return np.asarray([], dtype=np.int32)

        valid_moves = self.get_valid_moves()
        valid_actions = np.flatnonzero(valid_moves)
        if valid_actions.size == 0:
            return np.asarray([], dtype=np.int32)

        occupied = np.argwhere(self.grid != EMPTY)
        if occupied.size == 0:
            center = (self.size // 2) * self.size + (self.size // 2)
            return np.asarray([center], dtype=np.int32)

        radius = max(1, int(radius))
        mask = np.zeros((self.size, self.size), dtype=bool)
        for ox, oy in occupied:
            x0 = max(0, int(ox) - radius)
            x1 = min(self.size, int(ox) + radius + 1)
            y0 = max(0, int(oy) - radius)
            y1 = min(self.size, int(oy) + radius + 1)
            mask[x0:x1, y0:y1] = True

        mask &= (self.grid == EMPTY)
        candidates = np.flatnonzero(mask.reshape(-1))
        if candidates.size == 0:
            return valid_actions.astype(np.int32)

        max_actions = int(max_actions)
        if 0 < max_actions < candidates.size:
            me = self.current_player
            opp = WHITE if me == BLACK else BLACK
            center = (self.size - 1) * 0.5
            scores = np.zeros(candidates.size, dtype=np.float64)
            for i, action in enumerate(candidates):
                x, y = divmod(int(action), self.size)
                attack = self.tactical_score(int(action), me)
                defend = self.tactical_score(int(action), opp)
                center_bias = self.size - (abs(x - center) + abs(y - center))
                scores[i] = attack + 0.92 * defend + 0.15 * center_bias

            top_idx = np.argpartition(scores, -max_actions)[-max_actions:]
            top_scores = scores[top_idx]
            order = np.argsort(top_scores)[::-1]
            candidates = candidates[top_idx[order]]

        return candidates.astype(np.int32)


class TreeNode:

    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0.0
        self.P = float(prior_p)

    def is_leaf(self):
        return len(self.children) == 0

    def isleaf(self):
        return self.is_leaf()

    def is_root(self):
        return self.parent is None

    def isroot(self):
        return self.is_root()

    def expand(self, action_priors):
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, float(prior))

    def select(self, cpuct):
        return max(
            self.children.items(),
            key=lambda item: item[1].get_value(cpuct),
        )

    def get_value(self, cpuct):

        parent_visits = self.parent.n_visits if self.parent is not None else 1
        u = cpuct * self.P * math.sqrt(parent_visits) / (1 + self.n_visits)
        return self.Q + u

    def update(self, leaf_value):

        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent is not None:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


def cloneBoard(board):
    if isinstance(board, _BoardState):
        return board.clone()
    return _BoardState.from_cpp_board(board)


class MCTS:

    def __init__(
        self,
        model,
        simulations=1000,
        cpuct=5.0,
        candidate_radius=2,
        max_candidates=72,
        forced_check_depth=1,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
    ):
        self.model = model
        self.simulations = int(simulations)
        self.cpuct = float(cpuct)
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = int(max_candidates)
        self.forced_check_depth = int(forced_check_depth)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self._device = self._resolve_device()
        self._root = None
        self._root_signature = None

    def reset_tree(self):
        self._root = None
        self._root_signature = None

    def _resolve_device(self):
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    @staticmethod
    def _state_signature(state):
        return state.current_player, state.status, state.grid.tobytes()

    def _prepare_root(self, root_state):
        signature = self._state_signature(root_state)
        if self._root is None or self._root_signature != signature:
            self._root = TreeNode(parent=None, prior_p=1.0)
            self._root_signature = signature
        return self._root

    def _advance_root(self, root_state, action):
        next_state = root_state.clone()
        if not next_state.place(int(action)):
            self.reset_tree()
            return

        if self._root is not None and int(action) in self._root.children:
            next_root = self._root.children[int(action)]
            next_root.parent = None
            self._root = next_root
        else:
            self._root = TreeNode(parent=None, prior_p=1.0)
        self._root_signature = self._state_signature(next_state)

    def _policy_value(self, state):
        obs = torch.as_tensor(
            state.get_observation(),
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)

        with torch.no_grad():
            policy_output, value_output = self.model(obs)

        policy = _to_policy_probs_numpy(policy_output.squeeze(0))
        value = float(value_output.reshape(-1)[0].item())
        value = float(np.clip(value, -1.0, 1.0))
        return policy, value

    @staticmethod
    def _normalize_policy(policy, valid_moves):
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        valid_moves = np.asarray(valid_moves, dtype=np.float64).reshape(-1)

        if policy.size != valid_moves.size:
            raise ValueError("policy and valid_moves size mismatch")

        if not np.all(np.isfinite(policy)):
            policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)

        policy = np.clip(policy, a_min=0.0, a_max=None)
        masked = policy * valid_moves
        total = masked.sum()
        if total <= EPS:
            count = valid_moves.sum()
            if count <= 0:
                return np.zeros_like(valid_moves, dtype=np.float64)
            return valid_moves / count
        return masked / total

    def _build_allowed_mask(self, state, valid_moves, check_forced=True):
        valid_moves = np.asarray(valid_moves, dtype=np.float64).reshape(-1)
        if valid_moves.sum() <= 0:
            return np.zeros_like(valid_moves, dtype=np.float64)

        if check_forced:
            winning, blocking = state.get_forced_actions()
            if winning.size > 0:
                mask = np.zeros_like(valid_moves, dtype=np.float64)
                mask[winning] = 1.0
                return mask
            if blocking.size > 0:
                mask = np.zeros_like(valid_moves, dtype=np.float64)
                mask[blocking] = 1.0
                return mask

        candidates = state.get_candidate_actions(
            radius=self.candidate_radius,
            max_actions=self.max_candidates,
        )
        if candidates.size > 0:
            mask = np.zeros_like(valid_moves, dtype=np.float64)
            mask[candidates] = 1.0
            mask *= valid_moves
            if mask.sum() > 0:
                return mask

        return valid_moves

    def _apply_dirichlet_noise(self, priors):
        priors = np.asarray(priors, dtype=np.float64)
        if priors.size <= 1 or self.dirichlet_epsilon <= 0.0:
            return priors

        alpha = max(1e-3, self.dirichlet_alpha)
        noise = np.random.dirichlet(np.full(priors.size, alpha, dtype=np.float64))
        mixed = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
        mixed = np.clip(mixed, a_min=0.0, a_max=None)
        total = mixed.sum()
        if total <= EPS:
            return priors
        return mixed / total

    def _run_simulation(self, root, root_state, add_root_noise=False):
        node = root
        state = root_state.clone()
        depth = 0

        while not node.is_leaf() and state.status == STATUS_PLAYING:
            action, next_node = node.select(self.cpuct)
            if not state.place(action):
                break
            node = next_node
            depth += 1

        if state.status == STATUS_PLAYING:
            policy, leaf_value = self._policy_value(state)
            valid_moves = state.get_valid_moves()
            allowed_moves = self._build_allowed_mask(
                state,
                valid_moves,
                check_forced=(depth <= self.forced_check_depth),
            )
            policy = self._normalize_policy(policy, allowed_moves)
            actions = np.flatnonzero(allowed_moves)
            if actions.size == 0:
                actions = np.flatnonzero(valid_moves)
                if actions.size == 0:
                    leaf_value = 0.0
                    node.update_recursive(float(leaf_value))
                    return
                policy = self._normalize_policy(policy, valid_moves)

            priors = policy[actions]
            if node.is_root() and add_root_noise:
                priors = self._apply_dirichlet_noise(priors)

            priors_total = np.sum(priors)
            if priors_total <= EPS:
                priors = np.full(actions.size, 1.0 / actions.size, dtype=np.float64)
            else:
                priors = priors / priors_total
            node.expand([(int(a), float(p)) for a, p in zip(actions, priors)])
        else:
            leaf_value = float(state.terminal_value())

        node.update_recursive(float(leaf_value))

    @staticmethod
    def _visits_to_policy(visit_counts, valid_moves, temperature):
        visit_counts = np.asarray(visit_counts, dtype=np.float64)
        valid_moves = np.asarray(valid_moves, dtype=np.float64)

        if visit_counts.size != valid_moves.size:
            raise ValueError("visit_counts and valid_moves size mismatch")
        if valid_moves.sum() <= 0:
            return np.zeros_like(visit_counts, dtype=np.float32)

        if temperature <= 1e-8:
            policy = np.zeros_like(visit_counts, dtype=np.float64)
            legal_visits = visit_counts.copy()
            legal_visits[valid_moves <= 0] = -1.0
            best_action = int(np.argmax(legal_visits))
            policy[best_action] = 1.0
            return policy.astype(np.float32)

        scaled = np.power(np.maximum(visit_counts, 0.0), 1.0 / temperature)
        scaled *= valid_moves
        total = scaled.sum()
        if total <= EPS:
            return (valid_moves / valid_moves.sum()).astype(np.float32)
        return (scaled / total).astype(np.float32)

    def get_action_with_probs(self, realBoard, temperature=1.0, add_root_noise=False):
        root_state = cloneBoard(realBoard)
        board_size = root_state.size

        if root_state.status != STATUS_PLAYING:
            self.reset_tree()
            return -1, np.zeros(board_size * board_size, dtype=np.float32)

        winning, blocking = root_state.get_forced_actions()
        forced_actions = winning if winning.size > 0 else blocking
        if forced_actions.size > 0:
            policy = np.zeros(board_size * board_size, dtype=np.float32)
            share = 1.0 / float(forced_actions.size)
            policy[forced_actions] = share

            if forced_actions.size == 1 or temperature <= 1e-8:
                action = int(forced_actions[0])
            else:
                action = int(np.random.choice(forced_actions))
            self._prepare_root(root_state)
            self._advance_root(root_state, action)
            return action, policy

        root = self._prepare_root(root_state)
        for _ in range(self.simulations):
            self._run_simulation(root, root_state, add_root_noise=bool(add_root_noise))

        valid_moves = root_state.get_valid_moves()
        allowed_mask = self._build_allowed_mask(root_state, valid_moves, check_forced=False)
        visit_counts = np.zeros(board_size * board_size, dtype=np.float64)
        for action, child in root.children.items():
            visit_counts[int(action)] = child.n_visits
        visit_counts *= allowed_mask

        if visit_counts.sum() <= 0:
            valid = np.flatnonzero(allowed_mask)
            if valid.size == 0:
                valid = np.flatnonzero(valid_moves)
            if valid.size == 0:
                self.reset_tree()
                return -1, np.zeros(board_size * board_size, dtype=np.float32)
            policy = np.zeros(board_size * board_size, dtype=np.float32)
            policy[int(valid[0])] = 1.0
            action = int(valid[0])
            self._advance_root(root_state, action)
            return action, policy

        policy = self._visits_to_policy(visit_counts, allowed_mask, temperature)
        if temperature <= 1e-8:
            action = int(np.argmax(policy))
        else:
            action = int(np.random.choice(np.arange(policy.size), p=policy))
        self._advance_root(root_state, action)
        return action, policy

    def getAction(self, realBoard, temperature=1e-8, return_probs=False):
        action, policy = self.get_action_with_probs(
            realBoard,
            temperature=temperature,
            add_root_noise=False,
        )
        if return_probs:
            return action, policy
        return action


def collect_self_play_episode(
    model,
    simulations=400,
    cpuct=5.0,
    board_size=15,
    temperature=1.0,
    temperature_drop_move=10,
    candidate_radius=2,
    max_candidates=72,
    forced_check_depth=1,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
):
    mcts = MCTS(
        model=model,
        simulations=simulations,
        cpuct=cpuct,
        candidate_radius=candidate_radius,
        max_candidates=max_candidates,
        forced_check_depth=forced_check_depth,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    board = gomoku_ai.Board(int(board_size))
    size = int(board.get_observation().shape[1])

    trajectory = []
    move_idx = 0

    while _status_to_int(board.get_status()) == STATUS_PLAYING:
        obs = np.asarray(board.get_observation(), dtype=np.float32)
        current_player = _stone_to_int(board.get_current_player())
        temp = float(temperature) if move_idx < int(temperature_drop_move) else 1e-8

        action, policy = mcts.get_action_with_probs(
            board,
            temperature=temp,
            add_root_noise=(temp > 1e-8),
        )
        if action < 0:
            break

        x, y = divmod(action, size)
        if not board.place_stone(x, y):
            state = _BoardState.from_cpp_board(board)
            legal = np.flatnonzero(state.get_valid_moves())
            if legal.size == 0:
                break
            action = int(legal[0])
            x, y = divmod(action, size)
            board.place_stone(x, y)
            policy = np.zeros(size * size, dtype=np.float32)
            policy[action] = 1.0

        trajectory.append((obs, policy.astype(np.float32), current_player))
        move_idx += 1

    final_status = _status_to_int(board.get_status())
    samples = []
    if final_status in (STATUS_DRAW, STATUS_PLAYING):
        for obs, policy, _ in trajectory:
            samples.append((obs, policy, 0.0))
        return samples

    winner = BLACK if final_status == STATUS_BLACK_WIN else WHITE
    for obs, policy, player in trajectory:
        z = 1.0 if player == winner else -1.0
        samples.append((obs, policy, z))
    return samples


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, state, policy, value):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(policy, dtype=np.float32),
                float(value),
            )
        )
        return 1

    def add_game(self, game_samples, augment_symmetries=False):
        added = 0
        for state, policy, value in game_samples:
            if augment_symmetries:
                for aug_state, aug_policy in _augment_state_policy_symmetries(state, policy):
                    added += self.add(aug_state, aug_policy, value)
            else:
                added += self.add(state, policy, value)
        return added

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("ReplayBuffer is empty.")

        batch_size = min(int(batch_size), len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        states = np.stack([self.buffer[i][0] for i in indices], axis=0).astype(np.float32)
        policies = np.stack([self.buffer[i][1] for i in indices], axis=0).astype(np.float32)
        values = np.asarray([self.buffer[i][2] for i in indices], dtype=np.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)


def train_step(
    model,
    optimizer,
    batch,
    device=None,
    policy_loss_weight=1.0,
    value_loss_weight=1.0,
    max_grad_norm=None,
):
    states, target_policies, target_values = batch

    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            device = torch.device("cpu")

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    target_policies_t = torch.as_tensor(target_policies, dtype=torch.float32, device=device)
    target_values_t = torch.as_tensor(target_values, dtype=torch.float32, device=device).view(-1)

    model.train()
    optimizer.zero_grad()

    policy_output, value_output = model(states_t)
    value_output = value_output.view(-1)
    log_policy = _policy_log_probs_torch(policy_output)

    policy_loss = -(target_policies_t * log_policy).sum(dim=1).mean()
    value_loss = torch.mean((value_output - target_values_t) ** 2)
    loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss

    loss.backward()
    grad_norm = None
    if max_grad_norm is not None and float(max_grad_norm) > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(max_grad_norm),
        )
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "batch_size": int(states_t.shape[0]),
        "grad_norm": float(grad_norm.item()) if grad_norm is not None else None,
    }


def run_training_iteration(
    model,
    optimizer,
    replay_buffer,
    num_self_play_games=1,
    simulations=400,
    cpuct=5.0,
    board_size=15,
    batch_size=128,
    updates_per_iteration=1,
    min_buffer_size=256,
    temperature=1.0,
    temperature_drop_move=10,
    candidate_radius=2,
    max_candidates=72,
    forced_check_depth=1,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    augment_symmetries=True,
    max_grad_norm=None,
    device=None,
):
    game_lengths = []
    raw_samples_collected = 0
    samples_added = 0
    for _ in range(int(num_self_play_games)):
        game_samples = collect_self_play_episode(
            model=model,
            simulations=simulations,
            cpuct=cpuct,
            board_size=board_size,
            temperature=temperature,
            temperature_drop_move=temperature_drop_move,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
            forced_check_depth=forced_check_depth,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )
        raw_samples_collected += len(game_samples)
        samples_added += replay_buffer.add_game(
            game_samples,
            augment_symmetries=augment_symmetries,
        )
        game_lengths.append(len(game_samples))

    updates = []
    if len(replay_buffer) >= int(min_buffer_size):
        for _ in range(int(updates_per_iteration)):
            batch = replay_buffer.sample(batch_size)
            updates.append(
                train_step(
                    model,
                    optimizer,
                    batch,
                    device=device,
                    max_grad_norm=max_grad_norm,
                )
            )

    summary = {
        "games_collected": int(num_self_play_games),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "samples_collected_raw": int(raw_samples_collected),
        "samples_added": int(samples_added),
        "buffer_size": len(replay_buffer),
        "updates_run": len(updates),
    }
    if updates:
        summary["avg_loss"] = float(np.mean([u["loss"] for u in updates]))
        summary["avg_policy_loss"] = float(np.mean([u["policy_loss"] for u in updates]))
        summary["avg_value_loss"] = float(np.mean([u["value_loss"] for u in updates]))
        grad_norms = [u["grad_norm"] for u in updates if u["grad_norm"] is not None]
        if grad_norms:
            summary["avg_grad_norm"] = float(np.mean(grad_norms))
    return summary

