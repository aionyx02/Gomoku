import argparse
import random

import numpy as np
import torch

from gomoku_net import GomokuNet, count_parameters, quantize_dynamic_linear, state_dict_size_mb


def _seed_everything(seed):
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    board_size=15,
    hidden_dim=1033,
    trunk_layers=2,
    iterations=30,
    replay_capacity=50000,
    num_self_play_games=4,
    simulations=400,
    cpuct=4.2,
    batch_size=128,
    updates_per_iteration=10,
    min_buffer_size=256,
    temperature=1.0,
    temperature_drop_move=12,
    candidate_radius=2,
    max_candidates=72,
    forced_check_depth=1,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    augment_symmetries=True,
    max_grad_norm=1.5,
    lr=1e-3,
    weight_decay=1e-4,
    min_lr=1e-5,
    use_amp=True,
    policy_label_smoothing=0.03,
    value_loss_type="huber",
    seed=None,
):
    from model import ReplayBuffer, run_training_iteration

    _seed_everything(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(use_amp and device.type == "cuda")

    net = GomokuNet(
        board_size=board_size,
        hidden_dim=hidden_dim,
        trunk_layers=trunk_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    total_updates = max(1, int(iterations) * max(1, int(updates_per_iteration)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=max(0.0, min(float(min_lr), float(lr))),
    )

    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    replay = ReplayBuffer(capacity=replay_capacity)

    print(
        f"device={device}, amp={use_amp}, params={count_parameters(net):,}, "
        f"hidden_dim={hidden_dim}, trunk_layers={trunk_layers}, "
        f"state_dict_size={state_dict_size_mb(net):.2f} MB, "
        f"cpuct={cpuct}, sims={simulations}, augment={augment_symmetries}, "
        f"loss={value_loss_type}, label_smoothing={policy_label_smoothing}, seed={seed}"
    )

    for i in range(iterations):
        summary = run_training_iteration(
            model=net,
            optimizer=optimizer,
            replay_buffer=replay,
            num_self_play_games=num_self_play_games,
            simulations=simulations,
            cpuct=cpuct,
            board_size=board_size,
            batch_size=batch_size,
            updates_per_iteration=updates_per_iteration,
            min_buffer_size=min_buffer_size,
            temperature=temperature,
            temperature_drop_move=temperature_drop_move,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
            forced_check_depth=forced_check_depth,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            augment_symmetries=augment_symmetries,
            max_grad_norm=max_grad_norm,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            policy_label_smoothing=policy_label_smoothing,
            value_loss_type=value_loss_type,
        )
        print(f"[iter {i + 1:03d}/{iterations}] {summary}")

    return net, device


def main():
    parser = argparse.ArgumentParser(description="Train Gomoku model and optionally compress it.")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=1033, help="Lower this to shrink model size.")
    parser.add_argument("--trunk-layers", type=int, default=2, help="Use 1 for the legacy single-hidden-layer model.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--self-play-games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--cpuct", type=float, default=4.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--updates-per-iteration", type=int, default=10)
    parser.add_argument("--min-buffer-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-move", type=int, default=12)
    parser.add_argument("--candidate-radius", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=72)
    parser.add_argument("--forced-check-depth", type=int, default=1)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument(
        "--no-augment-symmetries",
        action="store_true",
        help="Disable 8-way board symmetry augmentation in replay buffer.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--policy-label-smoothing", type=float, default=0.03)
    parser.add_argument("--value-loss-type", choices=("mse", "huber"), default="huber")
    parser.add_argument("--save-path", default="gomoku_model.pt")
    parser.add_argument("--quantize", action="store_true", help="Export int8 dynamic-quantized model.")
    parser.add_argument("--quantized-save-path", default="gomoku_model_int8.pt")

    args = parser.parse_args()

    net, device = train(
        board_size=args.board_size,
        hidden_dim=args.hidden_dim,
        trunk_layers=args.trunk_layers,
        iterations=args.iterations,
        replay_capacity=args.replay_capacity,
        num_self_play_games=args.self_play_games,
        simulations=args.simulations,
        cpuct=args.cpuct,
        batch_size=args.batch_size,
        updates_per_iteration=args.updates_per_iteration,
        min_buffer_size=args.min_buffer_size,
        temperature=args.temperature,
        temperature_drop_move=args.temperature_drop_move,
        candidate_radius=args.candidate_radius,
        max_candidates=args.max_candidates,
        forced_check_depth=args.forced_check_depth,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        augment_symmetries=not args.no_augment_symmetries,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        policy_label_smoothing=args.policy_label_smoothing,
        value_loss_type=args.value_loss_type,
        seed=args.seed,
    )

    net_cpu = net.to("cpu").eval()
    torch.save(net_cpu.state_dict(), args.save_path)
    fp32_size = state_dict_size_mb(net_cpu)
    print(f"saved fp32 model -> {args.save_path} ({fp32_size:.2f} MB)")

    if args.quantize:
        qnet = quantize_dynamic_linear(net_cpu)
        torch.save(qnet.state_dict(), args.quantized_save_path)
        int8_size = state_dict_size_mb(qnet)
        ratio = (int8_size / fp32_size) if fp32_size > 1e-12 else 1.0
        print(
            f"saved int8 model -> {args.quantized_save_path} "
            f"({int8_size:.2f} MB, {ratio:.2%} of fp32 size)"
        )

    # Move back to the original device only if more work follows in-process.
    net.to(device)


if __name__ == "__main__":
    main()
