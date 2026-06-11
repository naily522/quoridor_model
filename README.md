# Quoridor — 步步为营

Python 强化学习 AI 训练框架（游戏核心逻辑基于 C++ pybind11 扩展）。

---

## 项目结构

```
Quoridor/
│
├── quoridor.hpp                # 核心逻辑：棋盘状态、行动规则、连通性检查（BFS）
│
├── rl/                         # Python 强化学习训练 + 人机对战
│   │
│   ├── __init__.py             # 包入口
│   ├── config.py               # 超参数配置
│   ├── encode.py               # 状态编码（C++ 棋盘 → PyTorch 张量）
│   ├── model.py                # 神经网络定义（策略-价值双头网络）
│   ├── self_play.py            # MCTS 自对弈数据生成
│   ├── train.py                # 训练主循环
│   ├── requirements.txt        # Python 依赖
│   │
│   ├── cpp_build/              # ★ C++ 扩展构建源码
│   │   ├── quoridor_bind.cpp   #   pybind11 绑定（C++ 游戏逻辑 → Python）
│   │   └── setup.py            #   编译配置
│   │
│   ├── quoridor_cpp.cp313-win_amd64.pyd  # 编译好的 Python 扩展（游戏接口）
│   ├── quoridor_cpp.pyi                  # 类型桩（IDE 提示用）
│   │
│   └── weights/                # 训练产出的权重
│       ├── quoridor_v3.weights
│       └── checkpoints/
│
├── gui/                         # 图形界面（Pygame）
│   ├── __init__.py
│   └── gui.py                   # Pygame 前端，支持人机对战 & AI 自对弈
│
├── play_ai.py                  # 人机对战（Python 版，MCTS + 神经网络）
├── run_train.py                # 训练启动脚本
├── export_weights.py           # 独立权重导出工具
├── build_gui.spec              # PyInstaller 打包配置
│
└── README.md
```

---

## 整体架构

### 数据流

```
                          ┌──────────────────┐
                          │   quoridor.hpp    │
                          │  C++ 游戏核心逻辑  │
                          └────────┬─────────┘
                                   │ pybind11 绑定 (quoridor_bind.cpp)
                                   ▼
                   ┌───────────────────────────┐
                   │  quoridor_cpp.pyd         │
                   │  State / Action / 合法动作  │
                   └──────────┬────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌────────────┐     ┌──────────────┐     ┌──────────────┐
   │ encode.py  │     │ self_play.py │     │  play_ai.py  │
   │ 状态编码    │     │ MCTS + 自对弈 │     │ 人机对战      │
   └─────┬──────┘     └──────┬───────┘     └──────────────┘
         │                   │
         ▼                   ▼
   ┌─────────────────────────────────┐
   │  model.py (PyTorch 神经网络)     │
   │  train.py (训练主循环)            │
   └─────────────────────────────────┘
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

---

## 强化学习训练流程

### 算法（AlphaZero 风格）

1. **自对弈**：当前网络与自身对弈，每步用 **MCTS（蒙特卡洛树搜索）** 增强决策质量
2. **数据**：每局产生 `(state, π_target, z)` 三元组
   - `state`：编码后的局面
   - `π_target`：MCTS 搜索后的动作概率分布
   - `z`：最终胜负结果（当前玩家视角，+1 / -1）
3. **训练**：从 replay buffer 采样，最小化损失
   ```
   L = (π_target - π_pred)²  策略损失
     + (z - v_pred)²         价值损失
   ```
4. **迭代**：新网络与最佳网络比赛，胜率超过阈值则替换
5. **导出**：训练完成后导出 `.weights` 文件

### 动作空间（225 个动作）

| 动作 ID | 类型 | 说明 |
|---------|------|------|
| 0 ~ 80 | 移动 | 9×9 目标格 |
| 81 ~ 152 | 垂直墙 | 8×9 = 72 个位置 |
| 153 ~ 224 | 水平墙 | 9×8 = 72 个位置 |

> 初始状态下实际可用动作为 147 个（3 移动 + 144 放墙），
> 随棋盘局势动态变化。

---

## 各文件详细说明

### Python 训练源码（`rl/`）

| 文件 | 职责 | 关键函数 |
|------|------|---------|
| `config.py` | 所有超参数集中管理 | `CONFIG` 字典 |
| `model.py` | 神经网络定义（卷积 + 策略头 + 价值头） | `QuoridorNet`, `export_weights()` |
| `encode.py` | 棋盘状态 → 张量 | `encode_state()` |
| `self_play.py` | MCTS 自对弈生成训练数据 | `play_one_game()`, `mcts_search()` |
| `train.py` | 训练主循环，协调各模块 | `main()` |

### 人机对战 / 图形界面

| 文件 | 说明 |
|------|------|
| `play_ai.py` | 控制台人机对战，Python 版 MCTS + 神经网络 |
| `gui/gui.py` | Pygame 图形界面，支持人机对战 & AI 自对弈 |
| `build_gui.spec` | PyInstaller 打包配置，将 GUI 打包为独立 `.exe` |
| `export_weights.py` | 从 checkpoint 导出 `.weights` 权重文件的独立工具 |

### C++ 扩展（`rl/cpp_build/`）

| 文件 | 说明 |
|------|------|
| `quoridor_bind.cpp` | pybind11 绑定源码，将 `quoridor.hpp` 暴露给 Python |
| `setup.py` | 扩展编译配置 |

---

## 如何开始

### 安装依赖

```bash
pip install -r rl/requirements.txt
```

### 运行人机对战

```bash
python play_ai.py
```

AI 会自动加载 `rl/weights/` 下最新的 checkpoint 进行 MCTS 搜索。

### 运行图形界面

```bash
pip install pygame
python gui/gui.py
```

支持两种模式：
- **人机对战**：P1 人类（鼠标点击落子/放墙），P2 AI
- **AI 自对弈**：观看两个 AI 对局

快捷键：`N` = 新对局（人机）、`A` = 新对局（AI自对弈）、`ESC` = 退出

### 打包为独立 EXE（方便分发给无 Python 环境的用户）

```bash
pip install pyinstaller
pyinstaller build_gui.spec
```

产物在 `dist/Quoridor.exe`（单文件）和 `dist/Quoridor/`（目录模式）。

**后续重新打包只需**：
```bash
pyinstaller build_gui.spec
```
（修改了 `gui/gui.py` 或换用了新的 checkpoint 后，重新运行此命令即可。）

### 训练

```bash
# 从头训练
python run_train.py

# 从 checkpoint 继续
python run_train.py --resume rl/weights/checkpoints/epoch_010.pt

# 只导出权重
python run_train.py --export-only rl/weights/checkpoints/epoch_020.pt
```

---

## 重新编译 C++ 扩展

如果你修改了 `quoridor.hpp` 或 `quoridor_bind.cpp`，需要重新编译 `.pyd`：

```bash
cd rl/cpp_build
python setup.py build_ext --build-lib ..
```

编译产物 `quoridor_cpp.cp313-win_amd64.pyd` 会输出到 `rl/` 目录（即 `--build-lib ..` 指定的上级目录），覆盖旧的扩展文件。

### 前置条件

- Python 3.8+
- pybind11（`pip install pybind11`）
- C++17 编译器（Windows 推荐 MSVC，由 Visual Studio Build Tools 或 `pip install setuptools` 自动配置）

---

## 常见问题

### ImportError: No module named 'quoridor_cpp'

`.pyd` 找不到或被删除了。确认 `rl/quoridor_cpp.cp313-win_amd64.pyd` 存在，且从项目根目录运行 Python（`run_train.py` 会自动处理路径）。

如果需要重新编译，见上方"重新编译 C++ 扩展"章节。

### OSError: ... not found (DLL 加载失败)

`.pyd` 依赖的运行时库缺失。通常只需要安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vc/redist) 即可解决。如果使用 MinGW 编译，可能需要保留 `libwinpthread-1.dll` 在同目录下。

### torch 相关错误

确认已安装 PyTorch：`pip install -r rl/requirements.txt`。
如果使用 CPU 训练，`pip install torch --index-url https://download.pytorch.org/whl/cpu` 可避免下载 CUDA 版。

---

## 关键设计决策

1. **Python 优先**：训练和人机对战全部用 Python，不再维护独立的 C++ 可执行文件
2. **pybind11 桥接**：游戏核心逻辑（合法动作枚举、走棋校验）用 C++ 实现，Python 通过编译好的 `.pyd` 直接调用，避免 Python 端重复实现一套规则，同时保持训练速度
3. **棋盘编码**：`19×19` 网格，奇/偶坐标编码格子和墙的关系，避免额外数据结构
4. **放墙校验**：物理重叠 + BFS 路径连通性双重检查
