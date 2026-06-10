# =============================================================================
# 模型定义 — model.py
#
# 功能:
#   定义 Quoridor 的神经网络结构（策略-价值网络）。
#   这是 AlphaZero 风格的双头网络:
#     策略头 (policy head):  输出 164 个动作的概率分布
#     价值头 (value head):   输出局面评分（-1 ~ 1）
#
# 输入:
#   经过 encode.py 编码后的 7 通道绝对编码状态张量 [B, 7, 9, 9]
#
# 输出:
#   policy: 长度为 225 的概率向量（对应 225 个动作）
#   value:  标量，当前玩家视角的胜率估计 [-1, 1]
#
# 架构说明:
#   主体为若干卷积层提取空间特征，然后分叉为策略头和价值头。
#   训练完成后通过 export_weights() 将参数导出为二进制文件，
#   供 C++ RLPlayer::load_weights() 读取。
#
# 使用方法（训练阶段）:
#   from model import QuoridorNet
#   net = QuoridorNet()
#   policy, value = net(encoded_state)
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F
from quoridor_cpp import State, ROW_SIZE, COLUMN_SIZE
from encode import encode_state
from config import CONFIG


# =============================================================================
# ResBlock — 残差块
#
# 数据流:
#
#      x ─────────────────────────────────┐
#      │                                   │
#      ▼                                   │
#  ┌──────┐    ┌───┐    ┌───┐    ┌──────┐   │
#  │ Conv │───▶│BN │───▶│ReLU│───▶│ Conv │───▶───┐
#  └──────┘    └───┘    └───┘    └──────┘   │   │
#      │                                     │   │
#      ▼                                     ▼   ▼
#      x'                                   x + residual
#                                                 │
#                                                 ▼
#                                               ReLU
#                                                 │
#                                                 ▼
#                                               输出
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        x = self.conv_block(x)
        x += residual
        x = F.relu(x)
        return x

class QuoridorNet(nn.Module):
    def __init__(self, channels):
        super(QuoridorNet, self).__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(channels, CONFIG["conv_channels"], kernel_size=3, padding=1),
            nn.BatchNorm2d(CONFIG["conv_channels"]),
            nn.ReLU(),
        )

        self.res_block = nn.Sequential(*[ResBlock(CONFIG["conv_channels"]) for _ in range(CONFIG["res_blocks"])])

        self.policy_head = nn.Sequential(
            nn.Conv2d(CONFIG["conv_channels"], CONFIG["policy_channels"], kernel_size=3, padding=1),
            nn.BatchNorm2d(CONFIG["policy_channels"]),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(CONFIG["policy_channels"] * ROW_SIZE * COLUMN_SIZE, CONFIG["num_actions"])

        self.value_head = nn.Sequential(
            nn.Conv2d(CONFIG["conv_channels"], 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(ROW_SIZE * COLUMN_SIZE, CONFIG["value_hidden"]),
            nn.ReLU(),
            nn.Linear(CONFIG["value_hidden"], 1),
        )

    def forward(self, x):
        x = self.conv_input(x)                              # [B,6,9,9] → [B,32,9,9]
        x = self.res_block(x)                               # [B,32,9,9] → [B,32,9,9]

        p = self.policy_head(x)                             # [B,32,9,9] → [B,32,9,9]
        p = p.view(p.size(0), -1)                           # → [B, 2592]
        p = self.policy_fc(p)                               # → [B, 225]
        p = F.softmax(p, dim=1)                             # → 概率分布

        v = self.value_head(x)                              # [B,32,9,9] → [B,1,9,9]
        v = v.view(v.size(0), -1)                           # → [B, 81]
        v = self.value_fc(v)                                # → [B, 1]
        v = F.tanh(v)                                       # → [-1, 1]

        return p, v
    
# =============================================================================
# export_weights — 权重导出 (含 BatchNorm 融合)
#
# 功能:
#   将 BatchNorm 层的参数融合进前一卷积层，导出为裸二进制 .weights 文件，
#   供 C++ network.hpp::NetworkWeights::load() 读取。
#
# BN 融合公式:
#   scale = gamma / sqrt(running_var + eps)
#   W' = W * scale                (out_c, 1, 1, 1 广播)
#   b' = (b - running_mean) * scale + beta
#
# 导出顺序 (与 network.hpp::load() 严格对应):
#   conv_input: w[NET_C,6,3,3] + b[NET_C]
#   NET_RES× ResBlock: w1[NET_C,NET_C,3,3]+b1[NET_C] + w2[NET_C,NET_C,3,3]+b2[NET_C]
#   policy_conv: w[NET_PC,NET_C,3,3] + b[NET_PC]
#   policy_fc:   w[225, NET_PC*81] + b[225]
#   value_conv:  w[1,NET_C,1,1] + b[1]
#   value_fc0:   w[NET_VH,81] + b[NET_VH]
#   value_fc2:   w[1,NET_VH] + b[1]
#
# 使用方式:
#   from model import QuoridorNet, export_weights
#   net = QuoridorNet(CONFIG["input_channels"])
#   export_weights(net, "rl/weights/quoridor_v1.weights")
# =============================================================================
def _fuse_conv_bn(sd, conv_key, bn_key):
    """从 state_dict 中取出 Conv+BN 并融合，返回 (fused_weight, fused_bias)."""
    import numpy as np
    w = sd[conv_key + ".weight"].cpu().numpy().astype("float32")
    b = sd[conv_key + ".bias"].cpu().numpy().astype("float32")
    gamma = sd[bn_key + ".weight"].cpu().numpy().astype("float32")
    beta = sd[bn_key + ".bias"].cpu().numpy().astype("float32")
    mean = sd[bn_key + ".running_mean"].cpu().numpy().astype("float32")
    var = sd[bn_key + ".running_var"].cpu().numpy().astype("float32")

    eps = 1e-5
    scale = gamma / np.sqrt(var + eps)  # [out_c]

    # W' = W * scale.reshape(out_c, 1, 1, 1)
    out_c = w.shape[0]
    fused_w = w * scale.reshape(out_c, *([1] * (w.ndim - 1)))
    # b' = (b - mean) * scale + beta
    fused_b = (b - mean) * scale + beta

    return fused_w, fused_b


def _write_tensor(f, arr):
    """将 numpy 数组或 torch tensor 写入文件 (float32 裸字节)."""
    import numpy as np
    import torch
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()
    f.write(np.asarray(arr, dtype="float32").tobytes())


def export_weights(net: QuoridorNet, path: str) -> None:
    """导出融合 BN 后的权重供 C++ 端加载。"""
    import numpy as np
    sd = net.state_dict()

    with open(path, "wb") as f:
        # ── conv_input: Conv2d(6→NET_C,3×3) + BN(NET_C) 融合 ──
        fw, fb = _fuse_conv_bn(sd, "conv_input.0", "conv_input.1")
        _write_tensor(f, fw)   # [NET_C, 6, 3, 3]
        _write_tensor(f, fb)   # [NET_C]

        # ── N× ResBlock (数量由 CONFIG["res_blocks"] 决定) ──
        for i in range(CONFIG["res_blocks"]):
            prefix = f"res_block.{i}.conv_block"
            # Conv1 + BN1 融合
            fw1, fb1 = _fuse_conv_bn(sd, f"{prefix}.0", f"{prefix}.1")
            _write_tensor(f, fw1)  # [32,32,3,3]
            _write_tensor(f, fb1)  # [32]
            # Conv2 + BN2 融合
            fw2, fb2 = _fuse_conv_bn(sd, f"{prefix}.3", f"{prefix}.4")
            _write_tensor(f, fw2)  # [32,32,3,3]
            _write_tensor(f, fb2)  # [32]

        # ── policy_head: Conv2d(NET_C→NET_PC,3×3) + BN(NET_PC) 融合 ──
        fw, fb = _fuse_conv_bn(sd, "policy_head.0", "policy_head.1")
        _write_tensor(f, fw)   # [NET_PC, NET_C, 3, 3]
        _write_tensor(f, fb)   # [NET_PC]

        # ── policy_fc: Linear(NET_PC*81→225) (无 BN) ──
        _write_tensor(f, sd["policy_fc.weight"])  # [225, NET_PC*81]
        _write_tensor(f, sd["policy_fc.bias"])    # [225]

        # ── value_head: Conv2d(NET_C→1,1×1) + BN(1) 融合 ──
        fw, fb = _fuse_conv_bn(sd, "value_head.0", "value_head.1")
        _write_tensor(f, fw)   # [1, NET_C, 1, 1]
        _write_tensor(f, fb)   # [1]

        # ── value_fc: Linear(81→NET_VH) + Linear(NET_VH→1) (无 BN) ──
        _write_tensor(f, sd["value_fc.0.weight"])  # [NET_VH, 81]
        _write_tensor(f, sd["value_fc.0.bias"])    # [NET_VH]
        _write_tensor(f, sd["value_fc.2.weight"])  # [1, NET_VH]
        _write_tensor(f, sd["value_fc.2.bias"])    # [1]