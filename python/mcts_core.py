"""Shared MCTS primitives for training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class SearchNode:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def policy_from_logits(logits: np.ndarray, legal: np.ndarray, action_size: int) -> np.ndarray:
    priors = np.zeros(int(action_size), dtype=np.float64)
    if legal.size == 0:
        return priors

    selected = logits[legal]
    selected = selected - np.max(selected)
    exp = np.exp(selected)
    denom = exp.sum()
    if denom <= 1e-12:
        priors[legal] = 1.0 / legal.size
    else:
        priors[legal] = exp / denom
    return priors


def expand_node(node: SearchNode, priors: np.ndarray, legal: np.ndarray) -> None:
    if node.children:
        return
    for action in legal:
        node.children[int(action)] = SearchNode(prior=float(priors[int(action)]))


def select_child(node: SearchNode, c_puct: float) -> Tuple[int, SearchNode]:
    sqrt_visits = math.sqrt(max(1, node.visit_count))
    best_score = -float("inf")
    best_action = -1
    best_child: Optional[SearchNode] = None

    for action, child in node.children.items():
        q = child.q_value
        u = float(c_puct) * child.prior * sqrt_visits / (1.0 + child.visit_count)
        score = -q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is None:
        raise RuntimeError("MCTS selection failed")
    return best_action, best_child


def backpropagate(path: List[SearchNode], leaf_value: float) -> None:
    value = float(leaf_value)
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def add_root_noise(
    priors: np.ndarray,
    legal: np.ndarray,
    rng: Optional[np.random.Generator],
    alpha: float,
    epsilon: float,
    use_root_noise: bool,
) -> np.ndarray:
    if not use_root_noise:
        return priors
    if rng is None or legal.size <= 1:
        return priors
    if float(alpha) <= 0.0 or float(epsilon) <= 0.0:
        return priors

    noise = rng.dirichlet(np.full(legal.size, float(alpha), dtype=np.float64))
    out = priors.copy()
    out[legal] = (1.0 - float(epsilon)) * out[legal] + float(epsilon) * noise
    return out


def visit_policy_from_root(root: SearchNode, action_size: int, temperature: float) -> np.ndarray:
    policy = np.zeros(int(action_size), dtype=np.float32)
    if not root.children:
        return policy

    actions = np.fromiter(root.children.keys(), dtype=np.int32)
    visits = np.array([root.children[int(a)].visit_count for a in actions], dtype=np.float64)

    if float(temperature) <= 1e-6:
        best = int(actions[int(np.argmax(visits))])
        policy[best] = 1.0
        return policy

    logits = np.log(visits + 1e-10) / float(temperature)
    logits = logits - np.max(logits)
    scaled = np.exp(logits)
    denom = scaled.sum()
    if denom <= 1e-12:
        policy[actions] = 1.0 / actions.size
    else:
        policy[actions] = scaled / denom
    return policy


def pick_argmax_visit(root: SearchNode) -> int:
    if not root.children:
        raise ValueError("Cannot pick move from empty root")
    return max(root.children.keys(), key=lambda a: root.children[int(a)].visit_count)


class NeuralMCTS:
    """Model-guided MCTS that works with any state carrying board/to_play/terminal/winner fields."""

    def __init__(
        self,
        action_size: int,
        simulations: int,
        c_puct: float,
        evaluate_state: Callable[[Any], Tuple[np.ndarray, float]],
        next_state: Callable[[Any, int], Any],
        legal_actions_fn: Callable[[np.ndarray], np.ndarray],
        terminal_value_fn: Callable[[int, int], float],
        rng: Optional[np.random.Generator] = None,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
    ):
        self.action_size = int(action_size)
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.evaluate_state = evaluate_state
        self.next_state = next_state
        self.legal_actions_fn = legal_actions_fn
        self.terminal_value_fn = terminal_value_fn
        self.rng = rng
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)

    def run(self, root_state: Any, use_root_noise: bool) -> SearchNode:
        root = SearchNode(prior=1.0)
        legal_root = self.legal_actions_fn(root_state.board)
        if legal_root.size == 0:
            return root

        logits, _ = self.evaluate_state(root_state)
        priors = policy_from_logits(logits, legal_root, self.action_size)
        priors = add_root_noise(
            priors=priors,
            legal=legal_root,
            rng=self.rng,
            alpha=self.dirichlet_alpha,
            epsilon=self.dirichlet_epsilon,
            use_root_noise=use_root_noise,
        )
        expand_node(root, priors, legal_root)

        for _ in range(self.simulations):
            node = root
            state = root_state
            path = [node]

            while node.children:
                action, node = select_child(node, self.c_puct)
                state = self.next_state(state, action)
                path.append(node)
                if state.terminal:
                    break

            if state.terminal:
                value = self.terminal_value_fn(state.winner, state.to_play)
                backpropagate(path, value)
                continue

            legal = self.legal_actions_fn(state.board)
            if legal.size == 0:
                backpropagate(path, 0.0)
                continue

            logits, leaf_value = self.evaluate_state(state)
            priors = policy_from_logits(logits, legal, self.action_size)
            expand_node(node, priors, legal)
            backpropagate(path, leaf_value)

        return root
