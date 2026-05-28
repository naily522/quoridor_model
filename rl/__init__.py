# Quoridor 强化学习训练框架
#
# 本包包含用 Python 训练 Quoridor AI 所需的全部模块。
# 训练完成后导出权重文件供 C++ RLPlayer 加载推理。
#
# 模块概览:
#   model.py      — 神经网络定义（策略-价值网络）
#   encode.py     — 状态编码（将 C++ 棋盘转为张量）
#   self_play.py  — 自对弈数据生成
#   train.py      — 训练主循环入口
#   config.py     — 超参数配置
