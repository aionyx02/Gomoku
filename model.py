import math
from collections import deque

import numpy as np
import torch

import gomoku_ai

# 遊戲狀態常量
EMPTY = 0  # 空位
BLACK = 1  # 黑子
WHITE = 2  # 白子

# 遊戲狀態常量
STATUS_PLAYING = 0  # 遊戲進行中
STATUS_BLACK_WIN = 1  # 黑子獲勝
STATUS_WHITE_WIN = 2  # 白子獲勝
STATUS_DRAW = 3  # 平局


# 將 C++ 枚舉轉換為 Python 整數
def _stone_to_int(stone):
    """
    將石頭枚舉轉換為整數表示。

    Args:
        stone: gomoku_ai.Stone 枚舉值

    Returns:
        int: 對應的整數 (EMPTY, BLACK, WHITE)
    """
    if stone == gomoku_ai.Stone.BLACK:
        return BLACK
    if stone == gomoku_ai.Stone.WHITE:
        return WHITE
    return EMPTY


def _status_to_int(status):
    """
    將遊戲狀態枚舉轉換為整數表示。

    Args:
        status: gomoku_ai.GameStatus 枚舉值

    Returns:
        int: 對應的整數狀態
    """
    if status == gomoku_ai.GameStatus.BLACK_WIN:
        return STATUS_BLACK_WIN
    if status == gomoku_ai.GameStatus.WHITE_WIN:
        return STATUS_WHITE_WIN
    if status == gomoku_ai.GameStatus.DRAW:
        return STATUS_DRAW
    return STATUS_PLAYING


def _to_policy_probs_numpy(policy_output):
    """
    將神經網絡的策略輸出轉換為概率分佈。

    Args:
        policy_output: 神經網絡輸出的策略值

    Returns:
        np.ndarray: 標準化的概率分佈
    """
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
    """
    計算策略輸出的對數概率。

    Args:
        policy_output: 策略輸出張量 [batch, board_size*board_size]

    Returns:
        torch.Tensor: 對數概率
    """
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


class _BoardState:
    """
    棋盤狀態類，表示 Gomoku 遊戲的當前狀態。
    包含棋盤網格、當前玩家和遊戲狀態。
    """

    def __init__(self, size, grid, current_player, status):
        """
        初始化棋盤狀態。

        Args:
            size (int): 棋盤大小
            grid (np.ndarray): 棋盤網格
            current_player (int): 當前玩家 (BLACK 或 WHITE)
            status (int): 遊戲狀態
        """
        self.size = int(size)
        self.grid = grid
        self.current_player = int(current_player)
        self.status = int(status)

    @classmethod
    def from_cpp_board(cls, board):
        """
        從 C++ 棋盤對象創建 Python 棋盤狀態。

        Args:
            board: C++ 棋盤對象

        Returns:
            _BoardState: Python 棋盤狀態
        """
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
        """
        克隆當前棋盤狀態。

        Returns:
            _BoardState: 克隆的狀態
        """
        return _BoardState(
            size=self.size,
            grid=self.grid.copy(),
            current_player=self.current_player,
            status=self.status,
        )

    def get_valid_moves(self):
        """
        獲取有效的移動位置。

        Returns:
            np.ndarray: 有效移動的掩碼 (1 表示有效)
        """
        if self.status != STATUS_PLAYING:
            return np.zeros(self.size * self.size, dtype=np.float32)
        return (self.grid.reshape(-1) == EMPTY).astype(np.float32)

    def get_observation(self):
        """
        獲取神經網絡的觀察輸入。

        Returns:
            np.ndarray: 形狀為 (3, size, size) 的觀察張量
        """
        me = self.current_player
        opp = WHITE if me == BLACK else BLACK
        obs = np.empty((3, self.size, self.size), dtype=np.float32)
        obs[0] = (self.grid == me).astype(np.float32)  # 自己的位置
        obs[1] = (self.grid == opp).astype(np.float32)  # 對手的位
        obs[2] = (self.grid == EMPTY).astype(np.float32)  # 空位
        return obs

    def place(self, action):
        """
        在指定位置放置石頭。

        Args:
            action (int): 動作索引 (0 到 size*size-1)

        Returns:
            bool: 是否成功放置
        """
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
        """
        獲取終端狀態的值 (用於 MCTS)。

        Returns:
            float: 終端值 (-1.0 表示輸，0.0 表示平局)
        """
        if self.status == STATUS_DRAW:
            return 0.0
        return -1.0

    def _check_win(self, last_x, last_y):
        """
        檢查最後放置的石頭是否導致勝利。

        Args:
            last_x (int): 最後放置的 x 坐標
            last_y (int): 最後放置的 y 坐標

        Returns:
            bool: 是否勝利
        """
        target = self.grid[last_x, last_y]
        if target == EMPTY:
            return False

        directions = ((1, 0), (0, 1), (1, 1), (1, -1))  # 四個方向

        for dx, dy in directions:
            count = 1

            # 正方向檢查
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

            # 負方向檢查
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


class TreeNode:
    """
    MCTS 樹節點類。
    每個節點代表一個遊戲狀態，並包含訪問次數、價值估計等信息。
    """

    def __init__(self, parent, prior_p):
        """
        初始化樹節點。

        Args:
            parent (TreeNode): 父節點
            prior_p (float): 先驗概率
        """
        self.parent = parent
        self.children = {}  # 子節點字典
        self.n_visits = 0  # 訪問次數
        self.Q = 0.0  # 平均價值
        self.P = float(prior_p)  # 先驗概率

    def is_leaf(self):
        """
        檢查是否為葉子節點。

        Returns:
            bool: 是否為葉子節點
        """
        return len(self.children) == 0

    def isleaf(self):
        """
        檢查是否為葉子節點 (別名)。

        Returns:
            bool: 是否為葉子節點
        """
        return self.is_leaf()

    def is_root(self):
        """
        檢查是否為根節點。

        Returns:
            bool: 是否為根節點
        """
        return self.parent is None

    def isroot(self):
        """
        檢查是否為根節點 (別名)。

        Returns:
            bool: 是否為根節點
        """
        return self.is_root()

    def expand(self, action_priors):
        """
        展開節點，添加子節點。

        Args:
            action_priors (list): 動作和先驗概率的列表 [(action, prior), ...]
        """
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, float(prior))

    def select(self, cpuct):
        """
        選擇最佳子節點 (根據 UCB 公式)。

        Args:
            cpuct (float): 探索參數

        Returns:
            tuple: (action, child_node)
        """
        return max(
            self.children.items(),
            key=lambda item: item[1].get_value(cpuct),
        )

    def get_value(self, cpuct):
        """
        計算節點的 UCB 值。

        Args:
            cpuct (float): 探索參數

        Returns:
            float: UCB 值
        """
        parent_visits = self.parent.n_visits if self.parent is not None else 1
        u = cpuct * self.P * math.sqrt(parent_visits) / (1 + self.n_visits)
        return self.Q + u

    def update(self, leaf_value):
        """
        更新節點的價值估計。

        Args:
            leaf_value (float): 葉子節點的值
        """
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """
        遞歸更新節點及其祖先。

        Args:
            leaf_value (float): 葉子節點的值
        """
        if self.parent is not None:
            self.parent.update_recursive(-leaf_value)  # 反轉價值，因為是對手的回合
        self.update(leaf_value)


class MCTS:
    """
    蒙特卡洛樹搜索 (MCTS) 類。
    使用神經網絡指導的 MCTS 來選擇最佳動作。
    """

    def __init__(self, model, simulations=1000, cpuct=5.0):
        """
        初始化 MCTS。

        Args:
            model: 神經網絡模型
            simulations (int): 模擬次數
            cpuct (float): 探索參數
        """
        self.model = model
        self.simulations = int(simulations)
        self.cpuct = float(cpuct)
        self._device = self._resolve_device()

    def _resolve_device(self):
        """
        解析模型的設備 (CPU 或 GPU)。

        Returns:
            torch.device: 設備
        """
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

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
        """
        將策略標準化為有效移動上的概率分佈。

        Args:
            policy (np.ndarray): 原始策略
            valid_moves (np.ndarray): 有效移動掩碼

        Returns:
            np.ndarray: 標準化的策略
        """
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        valid_moves = np.asarray(valid_moves, dtype=np.float64).reshape(-1)

        if policy.size != valid_moves.size:
            raise ValueError("policy and valid_moves size mismatch")

        if not np.all(np.isfinite(policy)):
            policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)

        policy = np.clip(policy, a_min=0.0, a_max=None)
        masked = policy * valid_moves
        total = masked.sum()

        if total <= 1e-12:
            count = valid_moves.sum()
            if count <= 0:
                return np.zeros_like(valid_moves, dtype=np.float64)
            return valid_moves / count

        return masked / total

    def _run_simulation(self, root, root_state):
        """
        運行一次 MCTS 模擬。

        Args:
            root (TreeNode): 根節點
            root_state: 根狀態
        """
        node = root
        state = root_state.clone()

        while not node.is_leaf() and state.status == STATUS_PLAYING:
            action, next_node = node.select(self.cpuct)
            if not state.place(action):
                break
            node = next_node

        if state.status == STATUS_PLAYING:
            policy, leaf_value = self._policy_value(state)
            valid_moves = state.get_valid_moves()
            policy = self._normalize_policy(policy, valid_moves)
            action_priors = [(int(a), float(policy[a])) for a in np.flatnonzero(valid_moves)]
            node.expand(action_priors)
        else:
            leaf_value = float(state.terminal_value())

        node.update_recursive(float(leaf_value))

    @staticmethod
    def _visits_to_policy(visit_counts, valid_moves, temperature):
        """
        將訪問次數轉換為策略概率 (使用溫度參數)。

        Args:
            visit_counts (np.ndarray): 訪問次數
            valid_moves (np.ndarray): 有效移動掩碼
            temperature (float): 溫度參數

        Returns:
            np.ndarray: 策略概率
        """
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

        scaled = np.power(visit_counts, 1.0 / temperature)
        scaled *= valid_moves
        total = scaled.sum()

        if total <= 1e-12:
            return (valid_moves / valid_moves.sum()).astype(np.float32)

        return (scaled / total).astype(np.float32)

    def get_action_with_probs(self, realBoard, temperature=1.0):
        """
        獲取動作及其概率分佈。

        Args:
            realBoard: 棋盤
            temperature (float): 溫度參數

        Returns:
            tuple: (action, policy)
        """
        root_state = self.cloneBoard(realBoard)
        board_size = root_state.size

        if root_state.status != STATUS_PLAYING:
            return -1, np.zeros(board_size * board_size, dtype=np.float32)

        root = TreeNode(parent=None, prior_p=1.0)

        for _ in range(self.simulations):
            self._run_simulation(root, root_state)

        valid_moves = root_state.get_valid_moves()
        visit_counts = np.zeros(board_size * board_size, dtype=np.float64)
        for action, child in root.children.items():
            visit_counts[action] = child.n_visits

        if visit_counts.sum() <= 0:
            valid = np.flatnonzero(root_state.get_valid_moves())
            if valid.size == 0:
                return -1, np.zeros(board_size * board_size, dtype=np.float32)
            policy = np.zeros(board_size * board_size, dtype=np.float32)
            policy[int(valid[0])] = 1.0
            return int(valid[0]), policy

        policy = self._visits_to_policy(visit_counts, valid_moves, temperature)
        action = int(np.random.choice(np.arange(policy.size), p=policy))
        return action, policy

    def getAction(self, realBoard, temperature=1e-8, return_probs=False):
        """
        獲取動作 (兼容舊接口)。

        Args:
            realBoard: 棋盤
            temperature (float): 溫度參數
            return_probs (bool): 是否返回概率

        Returns:
            int or tuple: 動作或 (動作, 概率)
        """
        action, policy = self.get_action_with_probs(realBoard, temperature=temperature)
        if return_probs:
            return action, policy
        return action

    def cloneBoard(self, board):
        """
        克隆棋盤狀態。

        Args:
            board: 棋盤對象

        Returns:
            _BoardState: 克隆的狀態
        """
        if isinstance(board, _BoardState):
            return board.clone()
        return _BoardState.from_cpp_board(board)


def collect_self_play_episode(
    model,
    simulations=400,
    cpuct=5.0,
    board_size=15,
    temperature=1.0,
    temperature_drop_move=10,
):
    """
    收集自我對弈的一個回合數據。

    Args:
        model: 神經網絡模型
        simulations (int): MCTS 模擬次數
        cpuct (float): 探索參數
        board_size (int): 棋盤大小
        temperature (float): 初始溫度
        temperature_drop_move (int): 溫度下降的移動數

    Returns:
        list: 訓練樣本列表 [(state, policy, value), ...]
    """
    mcts = MCTS(model=model, simulations=simulations, cpuct=cpuct)
    board = gomoku_ai.Board(int(board_size))
    size = int(board.get_observation().shape[1])

    trajectory = []
    move_idx = 0

    while _status_to_int(board.get_status()) == STATUS_PLAYING:
        obs = np.asarray(board.get_observation(), dtype=np.float32)
        current_player = _stone_to_int(board.get_current_player())
        temp = float(temperature) if move_idx < int(temperature_drop_move) else 1e-8

        action, policy = mcts.get_action_with_probs(board, temperature=temp)
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
    """
    經驗回放緩衝區類，用於存儲訓練樣本。
    """
    def __init__(self, capacity=50000):
        """
        初始化回放緩衝區。
        
        Args:
            capacity (int): 緩衝區容量
        """
        self.buffer = deque(maxlen=int(capacity))

    def add(self, state, policy, value):
        """
        添加一個訓練樣本。
        
        Args:
            state (np.ndarray): 狀態
            policy (np.ndarray): 策略
            value (float): 價值
        """
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(policy, dtype=np.float32),
                float(value),
            )
        )

    def add_game(self, game_samples):
        """
        添加一個遊戲的所有樣本。
        
        Args:
            game_samples: 遊戲樣本列表
        """
        for state, policy, value in game_samples:
            self.add(state, policy, value)

    def sample(self, batch_size):
        """
        隨機採樣一批訓練樣本。
        
        Args:
            batch_size (int): 批次大小
        
        Returns:
            tuple: (states, policies, values)
        """
        if len(self.buffer) == 0:
            raise ValueError("ReplayBuffer is empty.")

        batch_size = min(int(batch_size), len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        states = np.stack([self.buffer[i][0] for i in indices], axis=0).astype(np.float32)
        policies = np.stack([self.buffer[i][1] for i in indices], axis=0).astype(np.float32)
        values = np.asarray([self.buffer[i][2] for i in indices], dtype=np.float32)
        return states, policies, values

    def __len__(self):
        """
        獲取緩衝區長度。
        
        Returns:
            int: 緩衝區中的樣本數
        """
        return len(self.buffer)


def train_step(
    model,
    optimizer,
    batch,
    device=None,
    policy_loss_weight=1.0,
    value_loss_weight=1.0,
):
    """
    執行一個訓練步驟。
    
    Args:
        model: 神經網絡模型
        optimizer: 優化器
        batch: 訓練批次 (states, policies, values)
        device: 設備
        policy_loss_weight (float): 策略損失權重
        value_loss_weight (float): 價值損失權重
    
    Returns:
        dict: 訓練統計信息
    """
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
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "batch_size": int(states_t.shape[0]),
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
    device=None,
):
    """
    運行一個訓練迭代，包括自我對弈和模型更新。
    
    Args:
        model: 神經網絡模型
        optimizer: 優化器
        replay_buffer: 經驗回放緩衝區
        num_self_play_games (int): 自我對弈遊戲數
        simulations (int): MCTS 模擬次數
        cpuct (float): 探索參數
        board_size (int): 棋盤大小
        batch_size (int): 訓練批次大小
        updates_per_iteration (int): 每次迭代的更新次數
        min_buffer_size (int): 最小緩衝區大小
        temperature (float): 初始溫度
        temperature_drop_move (int): 溫度下降的移動數
        device: 設備
    
    Returns:
        dict: 訓練摘要
    """
    game_lengths = []
    for _ in range(int(num_self_play_games)):
        game_samples = collect_self_play_episode(
            model=model,
            simulations=simulations,
            cpuct=cpuct,
            board_size=board_size,
            temperature=temperature,
            temperature_drop_move=temperature_drop_move,
        )
        replay_buffer.add_game(game_samples)
        game_lengths.append(len(game_samples))

    updates = []
    if len(replay_buffer) >= int(min_buffer_size):
        for _ in range(int(updates_per_iteration)):
            batch = replay_buffer.sample(batch_size)
            updates.append(train_step(model, optimizer, batch, device=device))

    summary = {
        "games_collected": int(num_self_play_games),
        "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "buffer_size": len(replay_buffer),
        "updates_run": len(updates),
    }
    if updates:
        summary["avg_loss"] = float(np.mean([u["loss"] for u in updates]))
        summary["avg_policy_loss"] = float(np.mean([u["policy_loss"] for u in updates]))
        summary["avg_value_loss"] = float(np.mean([u["value_loss"] for u in updates]))
    return summary
