import argparse
import copy
import io

import torch
import torch.nn as nn

from model import ReplayBuffer, run_training_iteration


class GomokuNet(nn.Module):
    def __init__(self, board_size=15, hidden_dim=256):
        super().__init__()
        in_dim = 3 * board_size * board_size
        out_dim = board_size * board_size
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, out_dim)  # policy logits
        self.value_head = nn.Linear(hidden_dim, 1)  # value in [-1, 1]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc(x))
        policy = self.policy_head(h)  # [B, board_size*board_size]
        value = torch.tanh(self.value_head(h))  # [B, 1]
        return policy, value


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _state_dict_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 * 1024)


def _quantize_dynamic_linear(model):
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)


def train(
    board_size=15,
    hidden_dim=256,
    iterations=30,
    replay_capacity=50000,
    num_self_play_games=4,
    simulations=400,
    batch_size=128,
    updates_per_iteration=10,
    min_buffer_size=256,
    lr=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GomokuNet(board_size=board_size, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=replay_capacity)

    print(
        f"device={device}, params={_count_parameters(net):,}, "
        f"state_dict_size={_state_dict_size_mb(net):.2f} MB"
    )

    for i in range(iterations):
        summary = run_training_iteration(
            model=net,
            optimizer=optimizer,
            replay_buffer=replay,
            num_self_play_games=num_self_play_games,
            simulations=simulations,
            board_size=board_size,
            batch_size=batch_size,
            updates_per_iteration=updates_per_iteration,
            min_buffer_size=min_buffer_size,
            device=device,
        )
        print(f"[iter {i + 1:03d}/{iterations}] {summary}")

    return net, device


def main():
    parser = argparse.ArgumentParser(description="Train Gomoku model and optionally compress it.")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=256, help="Reduce this to shrink model size.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--self-play-games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-iteration", type=int, default=10)
    parser.add_argument("--min-buffer-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", default="gomoku_model.pt")
    parser.add_argument("--quantize", action="store_true", help="Export int8 dynamic-quantized model.")
    parser.add_argument("--quantized-save-path", default="gomoku_model_int8.pt")

    args = parser.parse_args()

    net, device = train(
        board_size=args.board_size,
        hidden_dim=args.hidden_dim,
        iterations=args.iterations,
        replay_capacity=args.replay_capacity,
        num_self_play_games=args.self_play_games,
        simulations=args.simulations,
        batch_size=args.batch_size,
        updates_per_iteration=args.updates_per_iteration,
        min_buffer_size=args.min_buffer_size,
        lr=args.lr,
    )

    net_cpu = net.to("cpu").eval()
    torch.save(net_cpu.state_dict(), args.save_path)
    fp32_size = _state_dict_size_mb(net_cpu)
    print(f"saved fp32 model -> {args.save_path} ({fp32_size:.2f} MB)")

    if args.quantize:
        qnet = _quantize_dynamic_linear(net_cpu)
        torch.save(qnet.state_dict(), args.quantized_save_path)
        int8_size = _state_dict_size_mb(qnet)
        ratio = (int8_size / fp32_size) if fp32_size > 1e-12 else 1.0
        print(
            f"saved int8 model -> {args.quantized_save_path} "
            f"({int8_size:.2f} MB, {ratio:.2%} of fp32 size)"
        )

    # Move back to the original device only if more work follows in-process.
    net.to(device)


if __name__ == "__main__":
    main()
