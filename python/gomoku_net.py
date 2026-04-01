import copy
import io
import re

import torch
import torch.nn as nn


ARCH_MLP = "mlp"
ARCH_ALPHAZERO = "alphazero_resnet"


class LegacyGomokuMLP(nn.Module):
    """Backward-compatible legacy policy-value MLP."""

    def __init__(self, board_size=15, input_planes=3, hidden_dim=512, trunk_layers=6, use_residual=True):
        super().__init__()
        if int(trunk_layers) < 1:
            raise ValueError(f"trunk_layers must be >= 1, got {trunk_layers}")

        board_size = int(board_size)
        input_planes = int(input_planes)
        hidden_dim = int(hidden_dim)
        trunk_layers = int(trunk_layers)

        in_dim = input_planes * board_size * board_size
        out_dim = board_size * board_size

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
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy_logits, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class GomokuNet(nn.Module):
    """
    AlphaZero-style Policy-Value Network.
    Input: (B, 4, 15, 15) by default.
    Outputs:
      - policy logits: (B, 225)
      - value: (B, 1), tanh range [-1, 1]
    """

    def __init__(self, board_size=15, input_planes=4, channels=128, residual_blocks=6, value_hidden_dim=256):
        super().__init__()
        board_size = int(board_size)
        input_planes = int(input_planes)
        channels = int(channels)
        residual_blocks = int(residual_blocks)
        value_hidden_dim = int(value_hidden_dim)

        if residual_blocks < 1:
            raise ValueError(f"residual_blocks must be >= 1, got {residual_blocks}")

        action_dim = board_size * board_size
        self.board_size = board_size
        self.input_planes = input_planes

        self.conv_in = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(residual_blocks))

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * action_dim, action_dim)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(action_dim, value_hidden_dim)
        self.value_fc2 = nn.Linear(value_hidden_dim, 1)

    def forward(self, x):
        h = torch.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            h = block(h)

        p = torch.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = torch.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value


# Keep an explicit alias for compatibility with older imports.
AlphaZeroGomokuNet = GomokuNet


def _infer_mlp_config(state_dict):
    fc_weight = state_dict.get("fc.weight")
    hidden_dim = int(fc_weight.shape[0]) if isinstance(fc_weight, torch.Tensor) else 512

    trunk_ids = []
    for key in state_dict:
        match = re.match(r"^trunk\.(\d+)\.weight$", key)
        if match:
            trunk_ids.append(int(match.group(1)))
    trunk_layers = (max(trunk_ids) + 2) if trunk_ids else 1
    use_residual = trunk_layers > 1

    # Legacy checkpoints historically used 3 planes.
    return {
        "architecture": ARCH_MLP,
        "input_planes": 3,
        "hidden_dim": hidden_dim,
        "trunk_layers": trunk_layers,
        "use_residual": use_residual,
    }


def _infer_alphazero_config(state_dict):
    conv_weight = state_dict.get("conv_in.weight")
    channels = int(conv_weight.shape[0]) if isinstance(conv_weight, torch.Tensor) else 128
    input_planes = int(conv_weight.shape[1]) if isinstance(conv_weight, torch.Tensor) else 4

    block_ids = []
    for key in state_dict:
        match = re.match(r"^res_blocks\.(\d+)\.conv1\.weight$", key)
        if match:
            block_ids.append(int(match.group(1)))
    residual_blocks = (max(block_ids) + 1) if block_ids else 6

    value_fc1_weight = state_dict.get("value_fc1.weight")
    value_hidden_dim = int(value_fc1_weight.shape[0]) if isinstance(value_fc1_weight, torch.Tensor) else 256

    return {
        "architecture": ARCH_ALPHAZERO,
        "input_planes": input_planes,
        "channels": channels,
        "residual_blocks": residual_blocks,
        "value_hidden_dim": value_hidden_dim,
    }


def infer_model_config_from_state_dict(state_dict):
    if "conv_in.weight" in state_dict:
        return _infer_alphazero_config(state_dict)
    return _infer_mlp_config(state_dict)


def build_model(board_size=15, architecture=ARCH_ALPHAZERO, **kwargs):
    if architecture == ARCH_ALPHAZERO:
        return GomokuNet(board_size=board_size, **kwargs)
    if architecture == ARCH_MLP:
        return LegacyGomokuMLP(board_size=board_size, **kwargs)
    raise ValueError(f"Unsupported architecture: {architecture}")


def build_model_from_state_dict(state_dict, board_size=15):
    config = infer_model_config_from_state_dict(state_dict)
    architecture = config["architecture"]
    args = dict(config)
    del args["architecture"]
    model = build_model(board_size=board_size, architecture=architecture, **args)
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
