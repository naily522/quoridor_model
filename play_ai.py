#!/usr/bin/env python
# =============================================================================
# 人机对战 — play_ai.py
#
# 你（人类） vs AI（训练好的神经网络 + MCTS）
# 控制台棋盘界面，键盘操作
# =============================================================================
import sys, os


def _setup_paths():
    """设置模块和数据文件路径，兼容开发和 PyInstaller 打包。"""
    # PyInstaller 打包后的临时目录
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        # 将 .pyd 所在目录加入 PATH/DLL 搜索
        rl_dir = os.path.join(base, 'rl')
        os.add_dll_directory(rl_dir)
        sys.path.insert(0, rl_dir)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        rl_dir = os.path.join(base, 'rl')
        sys.path.insert(0, rl_dir)

    return base


BASE_DIR = _setup_paths()

from quoridor_cpp import (State, Action, get_legal_actions, get_legal_moves,
                           get_legal_walls, ROW_SIZE, COLUMN_SIZE, WALL_NUM)
from rl.config import CONFIG
from rl.model import QuoridorNet
from rl.encode import encode_state
from rl.self_play import action_to_index, index_to_action, mcts_search, check_terminal
import torch
import numpy as np

# =============================================================================
# 棋盘绘制
# =============================================================================

def display_board(state: State):
    """在控制台绘制 9×9 棋盘 + 围墙 + 玩家位置"""
    N = 2 * ROW_SIZE + 1  # 19

    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')

    print()
    print("     步步为营 (Quoridor) — 人机对战")
    print("     你 = 1 (蓝) 向下走  |  AI = 2 (绿) 向上走")
    print()

    # 列标
    print("     ", end="")
    for c in range(COLUMN_SIZE):
        print(f"  {c} ", end="")
    print()

    for r in range(N):
        # 行标
        if r % 2 == 1:
            print(f"  {r // 2}  ", end="")
        else:
            print("     ", end="")

        for c in range(N):
            cell = state.get_cell(r, c)
            is_even_r = (r % 2 == 0)
            is_even_c = (c % 2 == 0)

            if is_even_r and is_even_c:
                # 交叉点
                if cell == 1:
                    print("+", end="")
                else:
                    print("+", end="")
            elif is_even_r and not is_even_c:
                # 水平通道 (可能放水平墙)
                if cell == 1:
                    print("---", end="")
                else:
                    print("   ", end="")
            elif not is_even_r and is_even_c:
                # 垂直通道 (可能放垂直墙)
                if cell == 1:
                    print("|", end="")
                else:
                    print(" ", end="")
            else:
                # 可落子格子 (奇数行 + 奇数列)
                p1_r, p1_c = state.get_pos(1)
                p2_r, p2_c = state.get_pos(2)
                if r == p1_r and c == p1_c:
                    print("\033[44;37m 1 \033[0m", end="")
                elif r == p2_r and c == p2_c:
                    print("\033[42;37m 2 \033[0m", end="")
                else:
                    print(" . ", end="")
        print()
    print()
    print(f"  墙: [1] x{state.get_wall_num(1):<2}                          "
          f"墙: [2] x{state.get_wall_num(2):<2}")
    print()


# =============================================================================
# 人类操作
# =============================================================================

def human_choose_move(state: State) -> Action:
    """让玩家键盘选择移动目标。"""
    moves = get_legal_moves(state, state.turn)
    if not moves:
        return None

    print("  选择一个移动目标:")
    for i, (r, c) in enumerate(moves):
        label = chr(ord('a') + i)
        print(f"    [{label}] row={r//2} col={c//2}")

    while True:
        key = input("  选择 (a-x): ").strip().lower()
        if key and key[0] >= 'a' and key[0] <= 'z':
            idx = ord(key[0]) - ord('a')
            if 0 <= idx < len(moves):
                r, c = moves[idx]
                return Action((r, c), False, 0)
        print("  无效选择，重试。")


def human_choose_wall(state: State) -> Action:
    """让玩家键盘选择放墙位置。"""
    walls = get_legal_walls(state, state.turn)
    if not walls:
        print("  没有合法放墙位置！")
        return None

    print("  合法放墙位置（前 26 个）:")
    n_show = min(len(walls), 26)
    for i in range(n_show):
        r, c, d = walls[i]
        direction = "水平" if d == 1 else "垂直"
        label = chr(ord('a') + i)
        print(f"    [{label}] row={r//2} col={c//2} {direction}")

    while True:
        key = input("  选择 (a-z, 回车放弃): ").strip().lower()
        if key == "":
            return None
        if key and key[0] >= 'a' and key[0] <= 'z':
            idx = ord(key[0]) - ord('a')
            if 0 <= idx < n_show:
                r, c, d = walls[idx]
                return Action((r, c), True, d)
        print("  无效选择，重试。")


def human_play(state: State):
    """人类玩家交互。返回 True 表示成功走了一步。"""
    print(f"\n  === 你的回合 (Player {state.turn}) ===")

    # 选择移动还是放墙
    can_wall = state.get_wall_num(state.turn) > 0
    if can_wall:
        choice = input("  [M]移动  [W]放墙: ").strip().lower()
    else:
        choice = 'm'
        print("  没有墙了，只能移动。")

    if choice == 'w' and can_wall:
        action = human_choose_wall(state)
        if action is None:
            return False  # 放弃放墙，回到选择
    else:
        action = human_choose_move(state)

    if action is None:
        return False

    if action.apply(state):
        return True
    else:
        print("  非法动作！")
        return False


# =============================================================================
# AI 操作
# =============================================================================

def ai_play(state: State, net, config: dict):
    """AI 用 MCTS + 神经网络决策。"""
    print(f"\n  === AI 思考中 (Player {state.turn}) ===", flush=True)

    # MCTS 搜索
    pi, _ = mcts_search(state, net, config)

    # 按 pi 采样动作
    action_idx = np.random.choice(config["num_actions"], p=pi)
    action = index_to_action(action_idx)

    # 确保合法
    legal = get_legal_actions(state)
    legal_indices = [action_to_index(a) for a in legal]
    if action_idx not in legal_indices:
        # 回退到最高概率合法动作
        best_idx = max(legal_indices, key=lambda i: pi[i])
        action = index_to_action(best_idx)

    action_type = "放墙" if action.is_wall else "移动"
    print(f"  AI 选择: {action}  ({action_type})", flush=True)

    if action.apply(state):
        return True
    else:
        # 极端情况回退到第一个合法动作
        for a in legal:
            if a.apply(state):
                print(f"  回退到: {a}")
                return True
        return False


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 55)
    print("  步步为营 (Quoridor) — 人机对战")
    print("  你 = Player 1 (蓝, 向下走)")
    print("  AI = Player 2 (绿, 向上走)")
    print("=" * 55)

    # ── 加载 AI 模型 ──
    weights_dir = os.path.join(BASE_DIR, 'rl', 'weights')
    ckpt_dir = os.path.join(weights_dir, 'checkpoints')

    # 自动找最新 checkpoint
    checkpoint_path = None
    if os.path.isdir(ckpt_dir):
        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')],
            reverse=True
        )
        if ckpt_files:
            checkpoint_path = os.path.join(ckpt_dir, ckpt_files[0])

    device = torch.device('cpu')
    net = QuoridorNet(CONFIG["input_channels"]).to(device)
    net.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(ckpt["model_state_dict"])
        print(f"  模型: {os.path.basename(checkpoint_path)} (epoch {ckpt.get('epoch', '?')}) 加载成功")
    else:
        print(f"  警告: 未找到 checkpoint，AI 将随机走子")
        print(f"  期望目录: {ckpt_dir}")

    # AI 搜索配置
    ai_config = dict(CONFIG)
    ai_config["mcts_simulations"] = 200   # 对战用更多模拟，AI 更强
    ai_config["temperature"] = 0.0        # 确定性策略
    ai_config["dirichlet_weight"] = 0.0   # 不加噪声

    # ── 游戏循环 ──
    state = State()
    state.reset()
    step = 0
    MAX_STEPS = 200

    while step < MAX_STEPS:
        display_board(state)

        winner = check_terminal(state)
        if winner:
            display_board(state)
            if winner == 1:
                print("\n  *** 恭喜！你赢了！ ***")
            else:
                print("\n  *** AI 获胜！再接再厉！ ***")
            break

        if state.turn == 1:
            # 人类回合
            ok = human_play(state)
            if ok:
                step += 1
        else:
            # AI 回合
            ok = ai_play(state, net, ai_config)
            if ok:
                step += 1
            else:
                print("  AI 出错，跳过...")
                break

    else:
        display_board(state)
        print("\n  *** 达到最大步数，平局！ ***")

    print(f"\n  总步数: {step}")
    input("\n  按回车退出...")


if __name__ == "__main__":
    main()
