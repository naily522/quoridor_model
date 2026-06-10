# =============================================================================
# 超参数配置 — config.py (v3: 深度训练版)
#
# 网络: 64通道, 8残差块, 128隐藏层, 120万参数
# 核心: BFS退火 — 训练早期强引导方向, 后期放手让网络自主
# 训练时间: ~40-50h (CPU, 60 epochs)
# =============================================================================
import os

CONFIG = {
    # ─── 神经网络 (v3: 更大容量) ───
    "input_channels":      7,
    "num_actions":         225,
    "conv_channels":       64,
    "policy_channels":     32,
    "value_hidden":        128,
    "dropout_rate":        0.3,
    "res_blocks":          8,

    # ─── 训练 ───
    "learning_rate":       3e-4,
    "lr_decay":            0.98,
    "batch_size":          256,
    "buffer_capacity":     80000,
    "epochs":              60,
    "samples_per_epoch":   3000,
    "entropy_weight":      0.20,
    "grad_clip":           0.5,
    "buffer_cleanup_epochs": 0,

    # ─── MCTS ───
    "mcts_simulations":    400,
    "c_puct":              1.5,
    "dirichlet_alpha":     0.6,
    "dirichlet_weight":    0.15,
    "goal_bonus_weight":   2.0,        # BFS起始引导
    "goal_bonus_decay":    0.95,       # 每轮衰减系数
    "goal_bonus_min":      0.2,        # 最低保留引导
    "wall_penalty":        0.0,
    "terminal_value_weight": 0.9,
    "shape_value_weight":  0.3,
    "distance_scale":      2.0,

    # ─── 自对弈 ───
    "games_per_iteration": 30,
    "temperature":         1.0,
    "temperature_steps":   15,
    "temperature_min":     0.2,

    # ─── MC 回报 ───
    "mc_gamma":            0.97,
    "reward_scale":        1.0,
    "policy_label_smoothing": 0.05,

    # ─── 评估 ───
    "eval_games":          20,
    "eval_threshold":      0.10,
    "eval_start_epoch":    15,
    "eval_force_update_epochs": 5,

    # ─── 权重导出 ───
    "export_dir":          os.path.join(os.path.dirname(__file__), "weights"),
    "export_name":         "quoridor_v3.weights",

    # ─── 课程学习 ───
    "curriculum_moves_only_epochs": 5,
    "curriculum_wall_fade_epochs":  5,

    # ─── 对手池 ───
    "opponent_pool_size":   10,
}
