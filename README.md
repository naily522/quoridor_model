# Quoridor — 步步为营

C++ 控制台版 Quoridor 棋盘游戏 + Python 强化学习 AI 训练框架。

---

## 项目结构

```
Quoridor/
│
├── main.cpp              # 入口，游戏主循环
├── quoridor.hpp           # 核心逻辑：棋盘状态、行动规则、连通性检查（BFS）
├── player.hpp             # 玩家体系：HumanPlayer（人类）+ RLPlayer（AI 推理）
├── view.hpp               # 控制台渲染（Windows Console API）
│
├── rl/                    # ★ Python 强化学习训练部分
│   ├── __init__.py        #   包入口，模块概览
│   ├── config.py          #   超参数配置（学习率、MCTS 模拟次数等）
│   ├── model.py           #   神经网络定义（策略-价值双头网络）
│   ├── encode.py          #   状态编码（C++ 棋盘 → PyTorch 张量）
│   ├── self_play.py       #   自对弈数据生成
│   ├── train.py           #   训练主循环（自对弈 → 训练 → 导出权重）
│   ├── requirements.txt   #   Python 依赖
│   ├── quoridor_bind.cpp  #   pybind11 绑定（C++ 游戏逻辑 → Python）
│   ├── quoridor_cpp.cp313-win_amd64.pyd  #   编译产物（Python 扩展）
│   ├── libwinpthread-1.dll               #   MinGW 运行时依赖
│   ├── build_pyd.sh       #   .pyd 编译脚本
│   └── weights/           #   训练导出的权重文件（供 C++ 加载）
│       └── .gitkeep
│
├── quoridor.exe           # 编译产物
└── README.md              # 本文件
```

---

## 整体架构

### 数据流

```
┌──────────────────────────────────────────────────┐
│  Python 训练阶段                                  │
│                                                    │
│  self_play.py                                      │
│  ┌──────────────────────┐                         │
│  │ 当前网络 + MCTS 自对弈 │──→  replay buffer      │
│  └──────┬───────────────┘                         │
│         │                                         │
│         ▼                                         │
│  model.py  ←── train.py  ←── config.py            │
│  (策略-价值网络)    (训练循环)    (超参数)           │
│         │                                         │
│         ▼                                         │
│  export_weights() →  .weights 二进制文件           │
└──────────────────┬───────────────────────────────┘
                   │ 权重文件
                   ▼
┌──────────────────────────────────────────────────┐
│  C++ 推理阶段 (RLPlayer)                           │
│                                                    │
│  player.hpp                                        │
│  ┌──────────────────────┐                         │
│  │ load_weights()        │  ← 加载 .weights        │
│  │ encode_state()        │  ← 棋盘 → 张量          │
│  │ forward()             │  ← 神经网络推理          │
│  │ softmax + sample()    │  ← 按概率采样动作        │
│  │ Action::apply()       │  ← 执行并校验合法性      │
│  └──────────────────────┘                         │
└──────────────────────────────────────────────────┘
```

### pybind11 跨语言桥接

```
┌─ Python 端 ────────────────────────────┐
│                                         │
│  from quoridor_cpp import State, Action │
│                                         │
│  s = State()         ← C++ 构造函数     │
│  s.reset()           ← C++ reset()     │
│  a = Action((r,c), ...)                │
│  a.apply(s)          ← C++ apply()     │
│  get_legal_actions(s)  ← 合法动作枚举    │
│  isconnect(s, ...)   ← BFS 连通性检查    │
└──────────┬────────────────────────────┘
           │ pybind11 绑定
           ▼
┌─ C++ 端 (quoridor.hpp) ────────────────┐
│                                         │
│  Quoridor::State, Quoridor::Action      │
│  isconnect(), 规则校验                   │
└─────────────────────────────────────────┘
```

### 两大阶段

| 阶段 | 位置 | 工具 | 产出 |
|------|------|------|------|
| **训练** | `rl/` 目录，Python | PyTorch + 自对弈 | `.weights` 权重文件 |
| **推理** | C++ `RLPlayer` | 加载权重 + 前向传播 | 每步决策 |

训练结束后，C++ 端完全独立运行，不依赖 Python 环境。

---

## 强化学习训练流程

### 算法选择（推荐：AlphaZero 风格）

1. **自对弈**：当前网络与自身对弈，每步用 **MCTS（蒙特卡洛树搜索）** 增强决策质量
2. **数据**：每局产生 `(state, π_target, z)` 三元组
   - `state`：编码后的局面
   - `π_target`：MCTS 搜索后的动作概率分布
   - `z`：最终胜负结果（当前玩家视角，+1 / -1）
3. **训练**：从 replay buffer 采样，最小化损失
   ```
   L = (π_target - π_pred)² 策略损失
     + (z - v_pred)²        价值损失
     + λ·‖θ‖²               L2 正则化
   ```
4. **迭代**：新网络与最佳网络比赛，胜率超过阈值则替换
5. **导出**：训练完成后导出权重供 C++ 加载

### 动作空间（225 个动作）

| 动作 ID | 类型 | 说明 |
|---------|------|------|
| 0 ~ 3 | 移动 | 上下左右 |
| 4 ~ 83 | 水平墙 | 8 行 × 10 列 |
| 84 ~ 163 | 垂直墙 | 10 行 × 8 列 |

> 初始状态下实际可用动作为 147 个（3 移动 + 144 放墙），
> 随棋盘局势动态变化。

---

## 各文件详细说明

### C++ 部分

| 文件 | 核心类/函数 | 职责 |
|------|------------|------|
| `quoridor.hpp` | `Quoridor::State`, `Quoridor::Action`, `isconnect()` | 棋盘状态、行动合法性、BFS 连通性检测 |
| `player.hpp` | `Player`（基类）, `HumanPlayer`, `RLPlayer` | 玩家抽象接口 + 人类交互 + AI 推理骨架 |
| `view.hpp` | `display_board()` | Windows 控制台棋盘渲染 |
| `main.cpp` | `main()` | 游戏主循环，回合制交替落子 |

### Python 部分

| 文件 | 职责 | 关键函数 |
|------|------|---------|
| `config.py` | 所有超参数集中管理 | `CONFIG` 字典 |
| `model.py` | 神经网络定义（卷积 + 策略头 + 价值头） | `QuoridorNet`, `export_weights()` |
| `encode.py` | 棋盘状态 → 张量 | `encode_state()` |
| `self_play.py` | MCTS 自对弈生成训练数据 | `self_play_game()`, `MCTS` 类 |
| `train.py` | 训练主循环，协调各模块 | `train()`, `evaluate()`, `export()` |
| `quoridor_bind.cpp` | pybind11 绑定 | 将 C++ 游戏逻辑暴露给 Python |

---

## 如何开始

### 编译 C++ 版本

```bash
g++ main.cpp -o quoridor.exe -std=c++17
```

运行后即双人轮流操作的控制台游戏。

### 编译 pybind11 扩展（Python 调用 C++ 游戏逻辑）

```bash
bash rl/build_pyd.sh
```

产出 `rl/quoridor_cpp.cp313-win_amd64.pyd`，编译后在 Python 中 import 即可调用 C++ 函数。

### 在 Python 中使用 C++ 函数

```python
import sys; sys.path.insert(0, 'rl')
from quoridor_cpp import (
    # ── 类 ──
    State,       # 棋盘状态
    Action,      # 动作（移动/放墙）

    # ── 常量 ──
    ROW_SIZE,    # 9
    COLUMN_SIZE, # 9
    WALL_NUM,    # 10
    WALL_LENGTH, # 2
    BOARD_SIZE,  # 19
    ACTION_NUM,  # 225

    # ── 函数 ──
    isconnect,            # BFS 连通性检查
    get_legal_moves,      # 合法移动枚举
    get_legal_walls,      # 合法放墙枚举
    get_legal_actions,    # 全部合法动作（融合移动+放墙）
)

# 创建棋盘
s = State()
s.reset()
print(f"当前轮到玩家 {s.turn}")
print(f"玩家1 位置: {s.get_pos(1)}")
print(f"玩家2 位置: {s.get_pos(2)}")

# 枚举所有合法动作
actions = get_legal_actions(s)
print(f"合法动作数: {len(actions)}")

# 执行一个动作
a = Action((3, 9), False, 0)   # 向下移动一步
ok = a.apply(s)                  # 应用动作
print(f"移动 {'成功' if ok else '失败'}, 轮到玩家 {s.turn}")

# 放墙（必须放在偶数坐标）
wall = Action((2, 2), True, 1)  # 水平墙
ok = wall.apply(s)

# 深拷贝（MCTS 树搜索时使用）
s2 = s.copy()
```

### 设置 Python 训练环境

```bash
pip install -r rl/requirements.txt
```

### 训练流程（待实现后）

```bash
# 从头训练
python rl/train.py

# 从 checkpoint 继续
python rl/train.py --resume rl/checkpoints/latest.pt

# 只导出权重（已有 checkpoint 时）
python rl/train.py --export-only rl/checkpoints/best.pt
```

---

## 关键设计决策

1. **纯头文件 C++**：所有逻辑写在 `.hpp` 中，编译仅需 `g++ main.cpp -std=c++17`，零依赖
2. **棋盘编码**：`19×19` 网格，奇/偶坐标编码格子和墙的关系，避免额外数据结构
3. **放墙校验**：双重检查——物理重叠 + BFS 路径连通性
4. **训练推理分离**：Python 训练，C++ 推理，通过二进制权重文件通信
5. **pybind11 桥接**：Python 训练期间直接调用 C++ 游戏规则（合法动作枚举、走棋校验），避免 Python 端重复实现一套规则逻辑
