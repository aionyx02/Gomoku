"""Backward-compatible model exports for Gomoku training and inference."""

from gomoku_net import (
    ARCH_ALPHAZERO,
    ARCH_MLP,
    AlphaZeroGomokuNet,
    GomokuNet,
    build_model,
    build_model_from_state_dict,
    count_parameters,
    infer_model_config_from_state_dict,
    quantize_dynamic_linear,
    state_dict_size_mb,
)

__all__ = [
    "ARCH_ALPHAZERO",
    "ARCH_MLP",
    "AlphaZeroGomokuNet",
    "GomokuNet",
    "build_model",
    "build_model_from_state_dict",
    "count_parameters",
    "infer_model_config_from_state_dict",
    "quantize_dynamic_linear",
    "state_dict_size_mb",
]
