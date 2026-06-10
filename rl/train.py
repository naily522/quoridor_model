# =============================================================================
# 训练主入口 — train.py
#
# 功能:
#   强化学习训练循环的主脚本，协调自对弈、训练、评估、导出等环节。
#
# 主循环:
#   for each epoch:
#     阶段 1 — 自对弈:  当前网络自对弈 M 局 → 存入 replay buffer
#     阶段 2 — 训练:    从 buffer 采样 mini-batch → 更新网络参数
#     阶段 3 — 评估:    新网络 vs 历史最佳 → 胜率够高才替换 + 导出权重
#
# 损失函数:
#   loss = policy_cross_entropy + value_mse
#
# 启动方式:
#   python rl/train.py                    # 从头训练
#   python rl/train.py --resume path      # 从 checkpoint 继续
#   python rl/train.py --export-only path # 只导出权重
# =============================================================================
import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from collections import deque
from config import CONFIG
from model import QuoridorNet, export_weights
from self_play import (play_one_game, check_terminal, mcts_search,
                        action_to_index, min_distance_to_goal)
from quoridor_cpp import State, get_legal_actions


# =============================================================================
# Replay Buffer — 经验回放缓冲区
# =============================================================================

class ReplayBuffer:
    """存储自对弈样本，支持随机采样。"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, samples: list[dict]):
        """添加一批样本。"""
        for s in samples:
            self.buffer.append(s)

    def sample(self, batch_size: int) -> dict:
        """随机采样一个 mini-batch。"""
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
# 评估 — 新网络 vs 最佳网络
# =============================================================================

def play_eval_game(net_a: nn.Module, net_b: nn.Module,
                   config: dict, a_plays_as: int) -> tuple[int, float]:
    """net_a 与 net_b 对战一局，返回 (winner, score)，score 从 net_a 方视角 [-1, 1]。"""
    state = State()
    state.reset()
    step = 0
    MAX_MOVES = 120

    eval_config = dict(config)
    eval_config["dirichlet_weight"] = 0.0
    eval_config["temperature"] = 0.0

    while step < MAX_MOVES:
        winner = check_terminal(state)
        if winner:
            score = 1.0 if winner == a_plays_as else -1.0
            return winner, score

        cur_net = net_a if state.turn == a_plays_as else net_b
        pi, _ = mcts_search(state, cur_net, eval_config)
        action_idx = int(pi.argmax())

        for a in get_legal_actions(state):
            if action_to_index(a) == action_idx:
                if a.apply(state):
                    step += 1
                break

    # 未分胜负：用 BFS 最短距离算分（从 net_a 方视角）
    d1 = min_distance_to_goal(state, 1)
    d2 = min_distance_to_goal(state, 2)
    raw = (d2 - d1) / 8.0 if (d1 != float('inf') or d2 != float('inf')) else 0.0
    score = max(-1.0, min(1.0, raw))
    if a_plays_as == 2:
        score = -score
    winner = 1 if d1 < d2 else (2 if d2 < d1 else 0)
    return winner, score


def evaluate(net: nn.Module, best_net: nn.Module, config: dict) -> float:
    """评估新网络 vs 最佳网络，返回新网络平均得分 [-1, 1]。"""
    total_score = 0.0
    total = config["eval_games"]

    for i in range(total):
        a_plays_as = 1 if i < total // 2 else 2
        winner, score = play_eval_game(net, best_net, config, a_plays_as)
        total_score += score

        print(f"  评估 {i + 1:>2}/{total}  "
              f"score={score:+.3f}  "
              f"平均分 {total_score/(i+1):+.3f}", flush=True)

    return total_score / total


# =============================================================================
# 训练步骤
# =============================================================================

def train_step(net: nn.Module, optimizer: optim.Optimizer,
               batch: dict, entropy_weight: float = 0.01,
               grad_clip: float = 1.0) -> tuple[float, float, float]:
    """单步训练，返回 (policy_loss, value_loss, entropy)。

    损失函数:
        loss = policy_ce + value_mse - entropy_weight * entropy
    其中 entropy = -Σ p_pred · log(p_pred) 鼓励策略探索, 防止早熟坍缩。
    加入了梯度裁剪防止训练不稳定。
    """
    states          = batch["states"]
    target_policies = batch["policies"]
    target_values   = batch["values"].unsqueeze(1)

    pred_policies, pred_values = net(states)

    # 策略损失: 交叉熵  L_p = -Σ π_target · log(p_pred)
    policy_loss = -(target_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()

    # 价值损失: MSE  L_v = (v_target - v_pred)²
    value_loss = ((pred_values - target_values) ** 2).mean()

    # 熵正则: H = -Σ p_pred · log(p_pred), 鼓励策略不要过早坍缩到单一动作
    # 熵越大, 策略越均匀 → 防止非法动作处的概率被压低后网络失去探索能力
    entropy = -(pred_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()

    loss = policy_loss + value_loss - entropy_weight * entropy

    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪: 防止梯度爆炸 (深度网络 + AlphaZero 训练常见问题)
    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.item()


# =============================================================================
# Checkpoint 读写
# =============================================================================

def save_checkpoint(net: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, best_state_dict: dict, path: str):
    """保存训练状态到文件。"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_state_dict": best_state_dict,
    }, path)
    print(f"  [保存] checkpoint -> {path}")


def load_checkpoint(path: str, net: nn.Module,
                    optimizer: optim.Optimizer) -> tuple[int, dict]:
    """从文件恢复训练状态，返回 (epoch, best_state_dict)。"""
    checkpoint = torch.load(path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("best_state_dict")


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quoridor AlphaZero 训练")
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
    best_net = QuoridorNet(config["input_channels"]).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"])
    scheduler = ExponentialLR(optimizer, gamma=config["lr_decay"])

    start_epoch = 0
    best_state_dict = None

    # ── 从 checkpoint 恢复 ──
    if args.resume:
        epoch, best_sd = load_checkpoint(args.resume, net, optimizer)
        start_epoch = epoch + 1
        print(f"恢复训练: 从 epoch {epoch + 1} 继续")
        if best_sd is not None:
            best_state_dict = best_sd
            best_net.load_state_dict(best_state_dict)

    # ── 仅导出权重 ──
    if args.export_only:
        ckpt = torch.load(args.export_only, map_location="cpu")
        net.load_state_dict(ckpt["model_state_dict"])
        os.makedirs(config["export_dir"], exist_ok=True)
        path = os.path.join(config["export_dir"], config["export_name"])
        export_weights(net, path)
        print(f"权重导出 -> {path}")
        return

    # ── 初始化缓冲区 ──
    buffer = ReplayBuffer(config["buffer_capacity"])
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(net.state_dict())
        best_net.load_state_dict(best_state_dict)

    min_buffer = config["batch_size"] * 4   # buffer 攒够才开始训练
    steps_per_epoch = max(1, config["samples_per_epoch"] // config["batch_size"])

    print(f"开始训练: {config['epochs']} epochs, "
          f"每轮 {config['games_per_iteration']} 局自对弈, "
          f"训练 {steps_per_epoch} 步, "
          f"batch_size={config['batch_size']}")

    # ── 课程学习: 计算当前 epoch 的放墙概率 ──
    def _calc_wall_prob(epoch, cfg):
        moves_only = cfg.get("curriculum_moves_only_epochs", 10)
        fade = cfg.get("curriculum_wall_fade_epochs", 5)
        if epoch < moves_only:
            return 0.0                          # 纯移动
        elif epoch < moves_only + fade:
            t = (epoch - moves_only) / fade      # 0 → 1
            return 0.2 + 0.8 * t                 # 线性: 20% → 100%
        else:
            return 1.0                          # 正常对局

    # ── 对手池 (C2): 保留最近 N 个最强模型 + 冻结基线 ──
    opponent_pool = []       # list of dict: {"state_dict": ..., "epoch": ...}
    frozen_baseline = None   # 冻结基线: 永远不会被淘汰的最早成功模型
    pool_size = config.get("opponent_pool_size", 10)
    pool_play_ratio = 0.2    # 20% 的对局使用池对手，增加训练数据多样性

    # ── 训练循环 ──
    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")

        # BFS 退火: 每轮衰减 goal_bonus_weight (v3)
        if "goal_bonus_decay" in config:
            decay = config["goal_bonus_decay"]
            cur = config.get("goal_bonus_weight", 1.0)
            new = max(config.get("goal_bonus_min", 0.2), cur * decay)
            config["goal_bonus_weight"] = new
            print(f"  BFS: {new:.3f}", flush=True)

        wall_p = _calc_wall_prob(epoch, config)
        if wall_p < 1.0:
            print(f"  课程学习: 放墙概率={wall_p:.0%}", flush=True)
        if opponent_pool:
            print(f"  对手池: {len(opponent_pool)} 个历史最佳模型", flush=True)
        print(f"{'='*50}", flush=True)

        # ── 阶段 1: 自对弈数据生成 (含对手池对战) ──
        print("[1/3] 自对弈...", flush=True)
        net.eval()
        total_samples = 0

        # 如果对手池不为空，准备一个池对手网络用于部分对局
        pool_net = None
        use_pool = opponent_pool and np.random.random() < pool_play_ratio
        if use_pool:
            pool_entry = np.random.choice(opponent_pool)
            pool_net = QuoridorNet(config["input_channels"]).to(device)
            pool_net.load_state_dict(pool_entry["state_dict"])
            pool_net.eval()
            print(f"  本局使用对手池 (epoch {pool_entry['epoch']})", flush=True)

        n_games = config["games_per_iteration"]
        for game_idx in range(n_games):
            # 每局随机决定是否用池对手 (20% vs 80%)
            if opponent_pool and np.random.random() < pool_play_ratio:
                # 池对战: net vs 历史对手
                entry = np.random.choice(opponent_pool)
                opp_net = QuoridorNet(config["input_channels"]).to(device)
                opp_net.load_state_dict(entry["state_dict"])
                opp_net.eval()
                samples = play_one_game(net, config, wall_prob=wall_p,
                                       opponent_net=opp_net)
                mode = f"vs#{entry['epoch']}"
            else:
                # 标准自对弈
                samples = play_one_game(net, config, wall_prob=wall_p)
                mode = "self"

            buffer.push(samples)
            total_samples += len(samples)

            avg_r = sum(abs(s["value_target"]) for s in samples) / len(samples)
            print(f"  对局 {game_idx + 1:>3}/{n_games}  "
                      f"样本 {total_samples:>4}  "
                      f"buffer {len(buffer):>6}  "
                      f"reward={avg_r:.3f}  [{mode}]", flush=True)

        print(f"  [OK] 本轮 {total_samples} 样本, buffer 总大小 {len(buffer)}")

        # ── 阶段 2: 网络训练 ──
        if len(buffer) >= min_buffer:
            net.train()
            sum_p_loss = 0.0
            sum_v_loss = 0.0
            sum_entropy = 0.0
            sum_abs_reward = 0.0
            ent_w = config.get("entropy_weight", 0.01)
            grad_clip = config.get("grad_clip", 1.0)

            print(f"[2/3] 训练 {steps_per_epoch} 步...")
            for step in range(steps_per_epoch):
                batch = buffer.sample(config["batch_size"])
                batch = {k: v.to(device) for k, v in batch.items()}
                pl, vl, ent = train_step(net, optimizer, batch,
                                         entropy_weight=ent_w,
                                         grad_clip=grad_clip)
                sum_p_loss += pl
                sum_v_loss += vl
                sum_entropy += ent
                sum_abs_reward += batch["values"].abs().mean().item()

                if (step + 1) % max(1, steps_per_epoch // 5) == 0:
                    print(f"  step {step + 1:>3}/{steps_per_epoch}  "
                          f"policy_loss={pl:.4f}  value_loss={vl:.4f}  "
                          f"entropy={ent:.4f}")

            print(f"  [OK] 平均 policy_loss={sum_p_loss / steps_per_epoch:.4f}, "
                  f"value_loss={sum_v_loss / steps_per_epoch:.4f}, "
                  f"entropy={sum_entropy / steps_per_epoch:.4f}, "
                  f"avg_reward={sum_abs_reward / steps_per_epoch:.4f}")

            # 定期清理 buffer 前半部分 (防止早期低质量数据长期污染)
            buffer_cleanup_interval = config.get("buffer_cleanup_epochs", 5)
            if buffer_cleanup_interval > 0 and epoch > 0 and epoch % buffer_cleanup_interval == 0 and len(buffer) > min_buffer * 3:
                # 确保清理后至少保留 min_buffer * 2 条样本
                remove_n = min(len(buffer) // 4, len(buffer) - min_buffer * 2)
                for _ in range(remove_n):
                    buffer.buffer.popleft()
                print(f"  -> buffer 清理: 移除 {remove_n} 条最旧样本, 保留 {len(buffer)}")
        else:
            print(f"[2/3] 跳过训练 (buffer {len(buffer)} < {min_buffer})")

        # ── 阶段 3: 评估 ──
        if epoch < config.get("eval_start_epoch", 3):
            # 前几轮不评估，直接采用新模型
            print("[3/3] 跳过评估（前 N 轮直接采用新模型）")
            best_state_dict = copy.deepcopy(net.state_dict())
            best_net.load_state_dict(best_state_dict)

            os.makedirs(config["export_dir"], exist_ok=True)
            export_path = os.path.join(config["export_dir"], config["export_name"])
            export_weights(net, export_path)
            print(f"  -> 直接采用新模型，权重已导出: {export_path}")
        else:
            print("[3/3] 评估...")
            net.eval()
            best_net.eval()
            avg_score = evaluate(net, best_net, config)
            print(f"  [OK] 新网络 vs 最佳网络 平均分: {avg_score:.3f}")

            if avg_score >= config["eval_threshold"]:
                print("  -> 新网络胜出，更新最佳模型 + 导出权重")
                config["_last_best_update_epoch"] = epoch  # 记录最近一次更新

                # C2: 将旧最佳模型加入对手池 (含 epoch 信息)
                if best_state_dict is not None:
                    opponent_pool.append({
                        "state_dict": copy.deepcopy(best_state_dict),
                        "epoch": epoch
                    })
                    # 保留>2*pool_size 防止意外清空
                    while len(opponent_pool) > pool_size * 2:
                        # 不淘汰冻结基线 (按 epoch 标识而非 state_dict 比较,
                        # 避免 Tensor == 比较触发 RuntimeError)
                        if frozen_baseline is not None and \
                           opponent_pool[0].get("is_frozen", False):
                            opponent_pool.pop(1)  # 跳过冻结基线
                        else:
                            opponent_pool.pop(0)
                    # 最终裁剪为 pool_size
                    while len(opponent_pool) > pool_size:
                        if frozen_baseline is not None and \
                           opponent_pool[0].get("is_frozen", False):
                            opponent_pool.pop(1)
                        else:
                            opponent_pool.pop(0)
                    print(f"  -> 对手池: {len(opponent_pool)} 个模型 (含冻结基线)")

                # 冻结基线: 首个进入池的模型永久保留
                # epoch >= eval_start_epoch 且首次评估成功后创建
                if frozen_baseline is None and epoch >= config.get("eval_start_epoch", 10):
                    frozen_baseline = {
                        "state_dict": copy.deepcopy(net.state_dict()),
                        "epoch": epoch,
                        "is_frozen": True,
                    }
                    opponent_pool.insert(0, frozen_baseline)
                    print(f"  -> 冻结基线已创建 (epoch {epoch}), 永久保留")

                best_state_dict = copy.deepcopy(net.state_dict())
                best_net.load_state_dict(best_state_dict)

                os.makedirs(config["export_dir"], exist_ok=True)
                export_path = os.path.join(config["export_dir"], config["export_name"])
                export_weights(net, export_path)
                print(f"  -> 权重已导出: {export_path}")
            else:
                # 强制更新: 每 N 轮至少更新一次 best_net, 防止永远停滞
                force_every = config.get("eval_force_update_epochs", 0)
                last_update = config.get("_last_best_update_epoch", config["eval_start_epoch"])
                if force_every > 0 and (epoch - last_update) >= force_every:
                    print(f"  -> 未达到阈值但强制更新 (距上次更新 {epoch - last_update} 轮)")
                    best_state_dict = copy.deepcopy(net.state_dict())
                    best_net.load_state_dict(best_state_dict)
                    config["_last_best_update_epoch"] = epoch
                    os.makedirs(config["export_dir"], exist_ok=True)
                    export_path = os.path.join(config["export_dir"], config["export_name"])
                    export_weights(net, export_path)
                    print(f"  -> 权重已导出 (强制): {export_path}")
                else:
                    print(f"  -> 未达到阈值 ({config['eval_threshold']:.3f})，保持当前最佳模型")

            # C2: 每 5 轮额外评估新网络 vs 对手池中随机对手 (多样性检验)
            if opponent_pool and (epoch + 1) % 5 == 0:
                import random
                pool_opponent_sd = random.choice(opponent_pool)
                pool_net = QuoridorNet(config["input_channels"]).to(device)
                pool_net.load_state_dict(pool_opponent_sd["state_dict"])
                pool_net.eval()
                pool_score = evaluate(net, pool_net, config)
                print(f"  [对手池评估] 新网络 vs 历史对手 平均分: {pool_score:.3f}")

        # ── 保存 checkpoint ──
        ckpt_dir = os.path.join(config["export_dir"], "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
        save_checkpoint(net, optimizer, epoch, best_state_dict, ckpt_path)

        # 学习率衰减
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  lr -> {current_lr:.6f}")


if __name__ == "__main__":
    main()
