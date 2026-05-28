# =============================================================================
# 状态编码 — encode.py
#
# 功能:
#   将 Quoridor::State（C++ 棋盘结构）编码为神经网络能处理的张量。
#   编码方式必须与 C++ RLPlayer::encode_state() 完全一致，
#   否则训练和推理之间会存在"编码鸿沟"导致模型无效。
#
# 编码方案 — 多层特征图（适合 CNN）:
#     将棋盘编码为多个 9×9 的特征通道:
#       ch 0:  己方位置（=1 的位置）
#       ch 1:  对方位置
#       ch 2:  己方剩余墙数（整层填充）
#       ch 3:  对方剩余墙数（整层填充）
#       ch 4~: 棋盘上横墙位置（=1 的位置）
#       ch 5~: 棋盘上竖墙位置（=1 的位置）
#     输入形状: (batch, C, 9, 9)
#
# 注意:
#   编码是训练和推理共用的契约，修改此处时必须同步修改
#   player.hpp 中 RLPlayer::encode_state()。
#
# 使用方法:
#   from encode import encode_state
#   tensor = encode_state(state_dict)
# =============================================================================
import torch
from quoridor_cpp import State, ROW_SIZE, COLUMN_SIZE
from config import CONFIG

def encode_state(state: State) -> torch.Tensor:
    x = torch.zeros(CONFIG["input_channels"], ROW_SIZE, COLUMN_SIZE)

    # ch 0: current player position
    my_pos = state.get_pos(state.turn)
    x[0, my_pos[0] // 2, my_pos[1] // 2] = 1.0

    # ch 1: opponent position
    opp_pos = state.get_pos(3 - state.turn)
    x[1, opp_pos[0] // 2, opp_pos[1] // 2] = 1.0

    # ch 2: current player's remaining walls (normalized)
    x[2, :, :] = state.get_wall_num(state.turn) / 10.0

    # ch 3: opponent's remaining walls (normalized)
    x[3, :, :] = state.get_wall_num(3 - state.turn) / 10.0

    # ch 4: horizontal walls (read from C++ tracking)
    for wr in range(9):
        for wc in range(9):
            if state.get_h_wall(wr, wc):
                x[4, wr, wc] = 1.0

    # ch 5: vertical walls (read from C++ tracking)
    for wr in range(9):
        for wc in range(9):
            if state.get_v_wall(wr, wc):
                x[5, wr, wc] = 1.0

    return x
