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

CONFIG = {
    # ─── 神经网络 ───
    "input_channels":      22,        # 输入特征图通道数
    "num_actions":         164,       # 动作空间大小
    "conv_channels":       128,       # 卷积层通道数
    "dropout_rate":        0.3,       # Dropout 比例

    # ─── 训练 ───
    "learning_rate":       1e-3,      # 初始学习率
    "lr_decay":            0.9,       # 学习率衰减系数
    "batch_size":          256,       # 训练批次大小
    "buffer_capacity":     100000,    # Replay buffer 容量
    "epochs":              100,       # 训练轮数
    "samples_per_epoch":   5000,      # 每轮新生成的数据量

    # ─── MCTS ───
    "mcts_simulations":    800,       # 每次决策的 MCTS 模拟次数
    "c_puct":              1.5,       # MCTS 探索常数
    "dirichlet_alpha":     0.3,       # 根节点 Dirichlet 噪声参数
    "dirichlet_weight":    0.25,      # 根节点噪声混合权重

    # ─── 自对弈 ───
    "games_per_iteration": 100,       # 每次迭代的自对弈局数
    "temperature":         1.0,       # 前若干步的探索温度
    "temperature_steps":   10,        # 使用温度 > 0 的步数
    "temperature_min":     0.1,       # 后续步骤的固定温度

    # ─── 评估 ───
    "eval_games":          50,        # 评估时的对局数
    "eval_threshold":      0.55,      # 胜率超过此值才替换最佳模型

    # ─── 权重导出 ───
    "export_dir":          "rl/weights",           # 权重文件输出目录
    "export_name":         "quoridor_v1.weights",  # C++ 端加载的文件名
}
