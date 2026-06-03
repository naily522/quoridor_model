# =============================================================================
# 超参数配置 — config.py
#
# 功能:
#   集中管理强化学习训练的所有超参数。
#   避免将魔数散落在各个文件中，方便调参。
#
# 使用方式:
#   from config import CONFIG
#   lr = CONFIG["learning_rate"]
# =============================================================================
import os

CONFIG = {
    # ─── 神经网络 ───
    "input_channels":      6,        # 输入特征图通道数
    "num_actions":         225,       # 动作空间大小 (81 移动 + 144 墙)
    "conv_channels":       32,        # 卷积层通道数
    "policy_channels":     32,        # 策略头 3×3 卷积降维通道数
    "value_hidden":        64,        # 价值头全连接隐藏层维度
    "dropout_rate":        0.3,       # Dropout 比例
    "res_blocks":          5,         # 残差块数量

    # ─── 训练 ───
    "learning_rate":       1e-3,      # 初始学习率
    "lr_decay":            0.9,       # 学习率衰减系数
    "batch_size":          256,       # 训练批次大小
    "buffer_capacity":     100000,    # Replay buffer 容量
    "epochs":              50,        # 训练轮数
    "samples_per_epoch":   2500,      # 每轮新生成的数据量

    # ─── MCTS ───
    "mcts_simulations":    60,        # 每次决策的 MCTS 模拟次数
    "c_puct":              1.5,       # MCTS 探索常数
    "dirichlet_alpha":     0.3,       # 根节点 Dirichlet 噪声参数
    "dirichlet_weight":    0.25,      # 根节点噪声混合权重
    "goal_bonus_weight":   0.3,       # MCTS leaf_value 中 BFS 距离差引导权重
    "wall_penalty":        0.0,       # 不再固定惩罚放墙，避免先天压制墙策略
    "terminal_value_weight": 0.7,     # 终局结果在 value_target 中的权重
    "shape_value_weight":  0.3,       # 过程中的距离差 shaping 权重

    # ─── 自对弈 ───
    "games_per_iteration": 50,        # 每次迭代的自对弈局数
    "temperature":         1.0,       # 前若干步的探索温度
    "temperature_steps":   10,        # 使用温度 > 0 的步数
    "temperature_min":     0.1,       # 后续步骤的固定温度

    # ─── 评估 ───
    "eval_games":          20,        # 评估时的对局数
    "eval_threshold":      0.55,      # 胜率超过此值才替换最佳模型
    "eval_start_epoch":    10,        # 前 N 轮不比较，直接采用新模型

    # ─── 权重导出 ───
    "export_dir":          os.path.join(os.path.dirname(__file__), "weights"),
    "export_name":         "quoridor_v1.weights",  # C++ 端加载的文件名
}
