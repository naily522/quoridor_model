# =============================================================================
# 状态编码 — encode.py
#
# 功能:
#   将 Quoridor::State（C++ 棋盘结构）编码为神经网络能处理的张量。
#   编码方式必须与 C++ RLPlayer::encode_state() 完全一致，
#   否则训练和推理之间会存在"编码鸿沟"导致模型无效。
#
# 编码方案 — 多层特征图（适合 CNN）:
#     将棋盘编码为 7 个 9×9 的特征通道 (v2: 绝对编码, 打破对称性):
#       ch 0:  Player 1 位置（绝对, =1 的位置）
#       ch 1:  Player 2 位置（绝对, =1 的位置）
#       ch 2:  Player 1 剩余墙数（整层填充, /10 归一化）
#       ch 3:  Player 2 剩余墙数（整层填充, /10 归一化）
#       ch 4:  横墙位置（绝对, =1 的位置）
#       ch 5:  竖墙位置（绝对, =1 的位置）
#       ch 6:  回合标识（整层填充: 1.0=P1回合, 0.0=P2回合）
#     输入形状: (batch, C=7, 9, 9)
#
# 使用方法:
#   from encode import encode_state
#   tensor = encode_state(state)
# =============================================================================
import torch
from quoridor_cpp import State, ROW_SIZE, COLUMN_SIZE, encode_state_fast
from config import CONFIG


def encode_state(state: State) -> torch.Tensor:
    """将 C++ State 编码为 [C, 9, 9] PyTorch 张量。
    使用 C++ 端 encode_state_fast() 单次跨语言调用完成全部编码。"""
    arr = encode_state_fast(state)  # numpy float32 [6, 9, 9]
    # 转为 PyTorch 张量（复制一份确保可写）
    return torch.from_numpy(arr.copy())
