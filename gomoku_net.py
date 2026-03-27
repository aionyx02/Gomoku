import copy
import io
import re

import torch
import torch.nn as nn


class GomokuNet(nn.Module):
    def __init__(self, board_size=15, hidden_dim=1033, trunk_layers=2, use_residual=True):
        super().__init__()
        if int(trunk_layers) < 1:
            raise ValueError(f"trunk_layers must be >= 1, got {trunk_layers}")

        in_dim = 3 * board_size * board_size
        out_dim = board_size * board_size
        hidden_dim = int(hidden_dim)
        trunk_layers = int(trunk_layers)

        self.fc = nn.Linear(in_dim, hidden_dim)
        self.trunk = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(trunk_layers - 1))
        self.use_residual = bool(use_residual and trunk_layers > 1)
        self.policy_head = nn.Linear(hidden_dim, out_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc(x))
        for layer in self.trunk:
            updated = torch.relu(layer(h))
            h = h + updated if self.use_residual else updated
        policy = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy, value


def infer_model_config_from_state_dict(state_dict):
    fc_weight = state_dict.get("fc.weight")
    hidden_dim = int(fc_weight.shape[0]) if isinstance(fc_weight, torch.Tensor) else 256

    trunk_ids = []
    for key in state_dict:
        match = re.match(r"^trunk\.(\d+)\.weight$", key)
        if match:
            trunk_ids.append(int(match.group(1)))
    trunk_layers = (max(trunk_ids) + 2) if trunk_ids else 1
    use_residual = trunk_layers > 1

    return {
        "hidden_dim": hidden_dim,
        "trunk_layers": trunk_layers,
        "use_residual": use_residual,
    }


def build_model_from_state_dict(state_dict, board_size=15):
    config = infer_model_config_from_state_dict(state_dict)
    model = GomokuNet(board_size=board_size, **config)
    model.load_state_dict(state_dict)
    return model, config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def quantize_dynamic_linear(model):
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
