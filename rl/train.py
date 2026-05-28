# =============================================================================
# 训练主入口 — train.py
#
# 功能:
#   强化学习训练循环的主脚本，协调自对弈、训练、评估、导出等环节。
#
# 主循环（伪代码）:
#
#   while 未达到停止条件:
#
#       阶段 1 — 自对弈数据生成
#       ├─ 当前网络 + MCTS 产生若干局游戏数据
#       └─ 存入 replay buffer
#
#       阶段 2 — 网络训练
#       ├─ 从 replay buffer 采样 mini-batch
#       ├─ 计算损失:
#       │    loss = L_policy + L_value + L2_regularization
#       │    L_policy = -π_target · log(p_pred)     # 交叉熵
#       │    L_value  = (v_target - v_pred)²         # MSE
#       ├─ 反向传播更新网络参数
#       └─ 定期评估: 与旧版网络或随机对手对弈
#
#       阶段 3 — 权重导出
#       ├─ 保存 checkpoint（PyTorch 格式）
#       └─ export_weights() → 二进制 .weights 文件
#           供 C++ RLPlayer::load_weights() 加载
#
# 启动方式:
#   python rl/train.py                    # 从头训练
#   python rl/train.py --resume path      # 从 checkpoint 继续训练
#   python rl/train.py --export-only path # 只导出权重
#
# 超参数见 config.py
# =============================================================================
