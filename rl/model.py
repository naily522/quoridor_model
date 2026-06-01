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
#   经过 encode.py 编码后的状态张量
#
# 输出:
#   policy: 长度为 164 的概率向量（对应 164 个合法动作）
#   value:  标量，当前玩家视角的胜率估计
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
# export_weights — 权重导出
#
# 功能:
#   将 PyTorch 模型的所有参数导出为裸二进制 .weights 文件，
#   供 C++ RLPlayer::load_weights() 读取。
#
# 导出格式:
#   参数按 net.parameters() 的顺序依次写入（参看 __init__ 中各层定义顺序）,
#   每个参数展平为连续的一维 float32 字节流:
#
#     conv_input.0.weight   [128, 6, 3, 3]  → 128×6×3×3 个 float32
#     conv_input.0.bias     [128]            → 128 个 float32
#     conv_input.1.weight   [128]            → BN gamma, 128 个 float32
#     conv_input.1.bias     [128]            → BN beta,  128 个 float32
#     res_blocks.0.conv_block.0.weight  [128, 128, 3, 3]
#     ...（后续所有参数依次排列）
#
#   最终文件 = 所有 float32 首尾相连的二进制流，
#   不含任何元数据（网络结构由 C++ 端硬编码保证）。
#
# C++ 端读取方式:
#   FILE* f = fopen(path, "rb");
#   fread(layer_weights, sizeof(float), layer_size, f);  // 按层依次读取
#   fclose(f);
#
# 使用方式:
#   from model import QuoridorNet, export_weights
#   net = QuoridorNet(CONFIG["input_channels"])
#   export_weights(net, "rl/weights/quoridor_v1.weights")
# =============================================================================
def export_weights(net: QuoridorNet, path: str) -> None:
    """将网络参数导出为二进制文件（供 C++ RLPlayer 加载）"""
    with open(path, "wb") as f:
        for param in net.parameters():
            f.write(param.data.cpu().numpy().astype("float32").tobytes())