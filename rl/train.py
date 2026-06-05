# =============================================================================
# 训练主入口 — train.py
#
# 功能:
#   强化学习训练循环，协调自对弈、对手池对战、训练、导出等环节。
#
# 每轮流程:
#   阶段 1 — 自对弈 (25 局): 双方均可放墙，对称对弈
#   阶段 2 — 对手池对战 (~25 局): 前一半新模型不能放墙，后一半双方都能放墙
#   阶段 3 — 训练: 从 buffer 采样 → 更新网络参数
#   阶段 4 — 更新对手池 + 保存 checkpoint + 导出权重
#
# 启动方式:
#   python rl/train.py                    # 从头训练
#   python rl/train.py --resume path      # 从 checkpoint 继续
#   python rl/train.py --export-only path # 只导出权重
# =============================================================================
import os
import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from collections import deque
from config import CONFIG
from model import QuoridorNet, export_weights
from self_play import play_one_game, play_vs_opponent_game


# =============================================================================
# 对手池
# =============================================================================

class OpponentPool:
    """存储历史模型参数，用于对抗训练。"""

    def __init__(self, max_size: int = 50):
        self.state_dicts = []
        self.max_size = max_size

    def add(self, state_dict: dict):
        if len(self.state_dicts) >= self.max_size:
            self.state_dicts.pop(0)
        self.state_dicts.append(copy.deepcopy(state_dict))

    def sample(self) -> dict | None:
        if not self.state_dicts:
            return None
        return random.choice(self.state_dicts)

    def __len__(self) -> int:
        return len(self.state_dicts)


# =============================================================================
# Replay Buffer — 经验回放缓冲区
# =============================================================================

class ReplayBuffer:
    """存储自对弈样本，支持随机采样。"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, samples: list[dict]):
        for s in samples:
            self.buffer.append(s)

    def sample(self, batch_size: int) -> dict:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states   = torch.from_numpy(np.stack([b["encoded_state"] for b in batch]))
        policies = torch.from_numpy(np.stack([b["policy_target"] for b in batch]))
        values   = torch.from_numpy(
            np.array([b["value_target"] for b in batch], dtype=np.float32))

        return {"states": states, "policies": policies, "values": values}

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Checkpoint 读写
# =============================================================================

def save_checkpoint(net: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, buffer: ReplayBuffer, pool: OpponentPool,
                    path: str):
    """保存完整训练状态。"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "buffer": list(buffer.buffer),
        "pool": pool.state_dicts,
    }, path)
    print(f"  [保存] checkpoint -> {path}")


def load_checkpoint(path: str, net: nn.Module,
                    optimizer: optim.Optimizer,
                    buffer_capacity: int) -> tuple[int, ReplayBuffer, OpponentPool]:
    """恢复训练状态，兼容新旧 checkpoint 格式。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt["epoch"]

    buffer = ReplayBuffer(buffer_capacity)
    if "buffer" in ckpt and ckpt["buffer"]:
        for s in ckpt["buffer"]:
            buffer.buffer.append(s)

    pool = OpponentPool()
    if "pool" in ckpt and ckpt["pool"]:
        for sd in ckpt["pool"]:
            pool.state_dicts.append(sd)

    return epoch, buffer, pool


# =============================================================================
# 训练步骤
# =============================================================================

def train_step(net: nn.Module, optimizer: optim.Optimizer,
               batch: dict) -> tuple[float, float]:
    """单步训练，返回 (policy_loss, value_loss)。"""
    states          = batch["states"]
    target_policies = batch["policies"]
    target_values   = batch["values"].unsqueeze(1)

    pred_policies, pred_values = net(states)

    policy_loss = -(target_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
    value_loss = ((pred_values - target_values) ** 2).mean()

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


# =============================================================================
# 游戏阶段常量
# =============================================================================

GAMES_PER_EPOCH   = 50
SELF_PLAY_GAMES    = 25
VS_POOL_NOWALL     = 12   # 新模型不能放墙
VS_POOL_FULL       = 13   # 双方都能放墙

# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quoridor 强化学习训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 文件恢复训练")
    parser.add_argument("--export-only", type=str, default=None,
                        metavar="CHECKPOINT",
                        help="从 checkpoint 导出权重后退出")
    args = parser.parse_args()

    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── 网络 ──
    net = QuoridorNet(config["input_channels"]).to(device)
    opponent_net = QuoridorNet(config["input_channels"]).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"])
    scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])

    # ── buffer / 对手池 ──
    buffer = ReplayBuffer(config["buffer_capacity"])
    pool = OpponentPool()

    start_epoch = 0

    # ── 从 checkpoint 恢复 ──
    if args.resume:
        epoch, buffer, pool = load_checkpoint(
            args.resume, net, optimizer, config["buffer_capacity"])
        start_epoch = epoch + 1
        print(f"恢复训练: 从 epoch {epoch + 1} 继续, "
              f"buffer {len(buffer)}, 对手池 {len(pool)}")

    # ── 仅导出权重 ──
    if args.export_only:
        ckpt = torch.load(args.export_only, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        os.makedirs(config["export_dir"], exist_ok=True)
        path = os.path.join(config["export_dir"], config["export_name"])
        export_weights(net, path)
        print(f"权重导出 -> {path}")
        return

    min_buffer = config["batch_size"] * 4
    steps_per_epoch = max(1, config["samples_per_epoch"] // config["batch_size"])

    print(f"开始训练: {config['epochs']} epochs, "
          f"每轮 {GAMES_PER_EPOCH} 局对弈 ({SELF_PLAY_GAMES} 自对弈 "
          f"+ {VS_POOL_NOWALL} vs-pool-nowall + {VS_POOL_FULL} vs-pool-full), "
          f"训练 {steps_per_epoch} 步, "
          f"batch_size={config['batch_size']}")

    # ── 训练循环 ──
    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*50}", flush=True)

        net.eval()
        total_samples = 0
        global_idx = 0

        # ── 阶段 1: 自对弈 (25 局，双方均可放墙) ──
        print("自对弈...", flush=True)
        for i in range(SELF_PLAY_GAMES):
            samples = play_one_game(net, config)
            buffer.push(samples)
            total_samples += len(samples)
            global_idx += 1

            first = samples[0]
            if any(s["value_target"] > 0 for s in samples):
                won_by = first["turn"] if first["value_target"] > 0 else (3 - first["turn"])
                tag = f"P{won_by}胜"
            else:
                tag = "平局"
            print(f"  对局 {global_idx:>3}/{GAMES_PER_EPOCH}  "
                  f"本局 {len(samples):>3}步  "
                  f"累计 {total_samples:>5}  "
                  f"buffer {len(buffer):>6}  "
                  f"{tag:>5}  [vs-self]",
                  flush=True)

        # ── 阶段 2: 对手池对战 ──
        if len(pool) > 0:
            print("对手池对战...", flush=True)

            # 2a: 新模型不能放墙，对手可以放墙
            for i in range(VS_POOL_NOWALL):
                swap = (i >= VS_POOL_NOWALL // 2)
                net_plays_as = 2 if swap else 1
                tag = "[vs-pool(swap)]" if swap else "[vs-pool]"

                opp_sd = pool.sample()
                opponent_net.load_state_dict(opp_sd)
                opponent_net.eval()

                samples, score = play_vs_opponent_game(
                    net, opponent_net, config,
                    net_plays_as=net_plays_as,
                    no_wall_net=True, no_wall_opp=False)
                buffer.push(samples)
                total_samples += len(samples)
                global_idx += 1

                print(f"  对局 {global_idx:>3}/{GAMES_PER_EPOCH}  "
                      f"本局 {len(samples):>3}样  "
                      f"累计 {total_samples:>5}  "
                      f"buffer {len(buffer):>6}  "
                      f"score={score:+.3f}  {tag}",
                      flush=True)

            # 2b: 双方都能放墙
            for i in range(VS_POOL_FULL):
                swap = (i >= VS_POOL_FULL // 2)
                net_plays_as = 2 if swap else 1
                tag = "[vs-pool-full(swap)]" if swap else "[vs-pool-full]"

                opp_sd = pool.sample()
                opponent_net.load_state_dict(opp_sd)
                opponent_net.eval()

                samples, score = play_vs_opponent_game(
                    net, opponent_net, config,
                    net_plays_as=net_plays_as,
                    no_wall_net=False, no_wall_opp=False)
                buffer.push(samples)
                total_samples += len(samples)
                global_idx += 1

                print(f"  对局 {global_idx:>3}/{GAMES_PER_EPOCH}  "
                      f"本局 {len(samples):>3}样  "
                      f"累计 {total_samples:>5}  "
                      f"buffer {len(buffer):>6}  "
                      f"score={score:+.3f}  {tag}",
                      flush=True)
        else:
            # 对手池为空（首个 epoch）：用剩余额度做对称自对弈
            extra_games = VS_POOL_NOWALL + VS_POOL_FULL
            for i in range(extra_games):
                samples = play_one_game(net, config)
                buffer.push(samples)
                total_samples += len(samples)
                global_idx += 1

                first = samples[0]
                if any(s["value_target"] > 0 for s in samples):
                    won_by = first["turn"] if first["value_target"] > 0 else (3 - first["turn"])
                    tag = f"P{won_by}胜"
                else:
                    tag = "平局"
                print(f"  对局 {global_idx:>3}/{GAMES_PER_EPOCH}  "
                      f"本局 {len(samples):>3}步  "
                      f"累计 {total_samples:>5}  "
                      f"buffer {len(buffer):>6}  "
                      f"{tag:>5}  [vs-self]",
                      flush=True)

        print(f"  [OK] 本轮 {total_samples} 样本, buffer 总大小 {len(buffer)}")

        # ── 训练 ──
        if len(buffer) >= min_buffer:
            net.train()
            sum_p_loss = 0.0
            sum_v_loss = 0.0

            print(f"训练 {steps_per_epoch} 步...", flush=True)
            for step in range(steps_per_epoch):
                batch = buffer.sample(config["batch_size"])
                batch = {k: v.to(device) for k, v in batch.items()}
                pl, vl = train_step(net, optimizer, batch)
                sum_p_loss += pl
                sum_v_loss += vl

                if (step + 1) % max(1, steps_per_epoch // 5) == 0:
                    print(f"  step {step + 1:>3}/{steps_per_epoch}  "
                          f"policy_loss={pl:.4f}  value_loss={vl:.4f}",
                          flush=True)

            print(f"  [OK] 平均 policy_loss={sum_p_loss / steps_per_epoch:.4f}, "
                  f"value_loss={sum_v_loss / steps_per_epoch:.4f}",
                  flush=True)
        else:
            print(f"跳过训练 (buffer {len(buffer)} < {min_buffer})")

        # ── 更新对手池 ──
        pool.add(copy.deepcopy(net.state_dict()))
        print(f"  对手池大小: {len(pool)}")

        # ── 导出权重 ──
        os.makedirs(config["export_dir"], exist_ok=True)
        export_path = os.path.join(config["export_dir"], config["export_name"])
        export_weights(net, export_path)
        print(f"  权重已导出: {export_path}")

        # ── 保存 checkpoint ──
        ckpt_dir = os.path.join(config["export_dir"], "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
        save_checkpoint(net, optimizer, epoch, buffer, pool, ckpt_path)

        # 学习率衰减
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  lr -> {current_lr:.6f}")


if __name__ == "__main__":
    main()
