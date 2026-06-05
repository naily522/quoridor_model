# =============================================================================
# 自对弈 — self_play.py
#
# 功能:
#   让当前神经网络与自身对弈，生成训练数据。
#   这是强化学习中"数据生产"的环节。
#
# 核心函数:
#   mcts_search(state, net, config) → π
#       在给定局面下运行 MCTS 搜索，返回动作概率分布。
#
#   play_one_game(net, config) → list[samples]
#       完成一局自对弈，返回训练样本列表。
#       每条样本: {encoded_state, policy_target, value_target, turn}
#
# MCTS 搜索流程:
#       选择 (Select) → 扩展 (Expand) → 评估 (Evaluate) → 回传 (Backup)
#
#     选择: 从根节点出发，PUCT 分数逐层选择子节点，直到叶子
#     扩展: 叶子节点 → 查询网络得到 (policy, value) → 创建子节点
#     评估: 网络输出的 value 或终局结果
#     回传: 沿路径反向传播 value，交替正负号
#
# 使用方法:
#   from self_play import play_one_game
#   samples = play_one_game(net, CONFIG)
# =============================================================================
import math
from collections import deque
import numpy as np
import torch
from quoridor_cpp import State, get_legal_actions, ROW_SIZE
from encode import encode_state
from config import CONFIG


# =============================================================================
# 动作索引映射
#
# 网络输出 225 个动作的概率:
#   0 ~ 80:     移动 (9×9 = 81 个目标格)
#   81 ~ 152:   垂直墙 (8×9 = 72)
#   153 ~ 224:  水平墙 (9×8 = 72)
#
# 棋盘坐标 (board_r, board_c) ∈ {1,3,5,...,17} (奇数 = 可落子格)
# 墙起始坐标使用与 get_legal_actions 返回一致的值 (奇数坐标)
# =============================================================================

def action_to_index(action) -> int:
    """将 Action 对象映射为网络输出索引 (0~224)。"""
    r_idx = (action.pos[0] - 1) // 2
    c_idx = (action.pos[1] - 1) // 2
    if action.is_wall:
        if action.wall_dir == 0:          # 垂直墙
            return 81 + r_idx * 9 + c_idx
        else:                             # 水平墙
            return 153 + r_idx * 8 + c_idx
    else:
        return r_idx * 9 + c_idx


def index_to_action(index: int):
    """将网络输出索引 (0~224) 还原为 Action 对象（用于 C++ 端推理）。

    墙坐标为偶数（墙中点坐标），移动为奇数（玩家位置）。
    """
    from quoridor_cpp import Action
    if index < 81:                         # 移动 (奇数坐标)
        r = (index // 9) * 2 + 1
        c = (index % 9) * 2 + 1
        return Action((r, c), False, 0)
    elif index < 153:                      # 垂直墙 (偶数中点坐标)
        wall_idx = index - 81
        r = (wall_idx // 9) * 2 + 2
        c = (wall_idx % 9) * 2 + 2
        return Action((r, c), True, 0)
    else:                                  # 水平墙 (偶数中点坐标)
        wall_idx = index - 153
        r = (wall_idx // 8) * 2 + 2
        c = (wall_idx % 8) * 2 + 2
        return Action((r, c), True, 1)


# =============================================================================
# 终局判断
# =============================================================================

def check_terminal(state: State) -> int:
    """检测是否终局，返回胜方 (1/2)，未终局返回 0。"""
    if state.get_pos(1)[0] == 2 * ROW_SIZE - 1:   # Player 1 到达底部
        return 1
    if state.get_pos(2)[0] == 1:                   # Player 2 到达顶部
        return 2
    return 0


# =============================================================================
# BFS 最短距离 — 玩家到目标行的最小步数
# =============================================================================

def min_distance_to_goal(state: State, player: int) -> float:
    """BFS 计算玩家到目标行的最短步数（绕开墙），不可达返回 inf。"""
    start = state.get_pos(player)
    target_row = 2 * ROW_SIZE - 1 if player == 1 else 1

    if start[0] == target_row:
        return 0.0

    visited = [[False] * (2 * ROW_SIZE + 1) for _ in range(2 * ROW_SIZE + 1)]
    q = deque()
    q.append((start[0], start[1], 0))
    visited[start[0]][start[1]] = True

    while q:
        r, c, d = q.popleft()

        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc

            if nr < 0 or nr > 2 * ROW_SIZE or nc < 0 or nc > 2 * ROW_SIZE:
                continue
            if nr % 2 == 0 or nc % 2 == 0:          # 必须落在格子（奇数坐标）
                continue
            if state.get_cell(r + dr // 2, c + dc // 2):  # 路径上有墙
                continue
            if visited[nr][nc]:
                continue

            if nr == target_row:
                return d + 1

            visited[nr][nc] = True
            q.append((nr, nc, d + 1))

    return float('inf')


# =============================================================================
# MCTS 节点
# =============================================================================

class MCTSNode:
    """MCTS 树节点。

    每个节点对应一个棋盘局面，存储该局面的统计数据。
    价值始终从**当前轮到玩家的视角**存储。

    属性:
        state:        Quoridor::State (C++ 棋盘)
        prior_p:      网络输出的先验概率 (策略头)
        visit_count:  访问次数
        total_value:  累计价值总和 (用于计算平均价值 Q)
        children:     dict{action_idx → MCTSNode}
    """
    __slots__ = ('state', 'prior_p', 'visit_count', 'total_value', 'children')

    def __init__(self, state: State, prior_p: float):
        self.state = state
        self.prior_p = prior_p
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}

    @property
    def value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


# =============================================================================
# MCTS 搜索
# =============================================================================

def _filter_legal(actions, state, no_wall_player):
    """如果当前轮到 no_wall_player，过滤掉放墙动作。"""
    if no_wall_player is not None and state.turn == no_wall_player:
        return [a for a in actions if not a.is_wall]
    return actions


def mcts_search(state: State, net: torch.nn.Module,
                config: dict | None = None,
                no_wall_player: int | None = None) -> np.ndarray:
    """在当前局面运行 MCTS 搜索。

    参数:
        state:          当前棋盘状态
        net:            策略-价值网络 (QuoridorNet)
        config:         超参数字典 (默认使用 CONFIG)
        no_wall_player: 被禁止放墙的玩家编号 (1 或 2), None=双方均可放墙

    返回:
        π: 长度为 225 的动作概率分布 (visit count 归一化)
    """
    if config is None:
        config = CONFIG

    num_actions     = config["num_actions"]
    sims            = config["mcts_simulations"]
    c_puct          = config["c_puct"]
    dirichlet_alpha = config["dirichlet_alpha"]
    dirichlet_w     = config["dirichlet_weight"]

    # ── 推断网络所在设备 ──
    device = next(net.parameters()).device

    # ── 获取合法动作 ──
    legal_actions = _filter_legal(get_legal_actions(state), state, no_wall_player)
    if len(legal_actions) == 1:
        pi = np.zeros(num_actions)
        pi[action_to_index(legal_actions[0])] = 1.0
        return pi

    # 动态调整搜索次数: 动作少时无需过多模拟
    sims = min(config["mcts_simulations"], max(80, len(legal_actions) * 15))

    # ── 根节点网络评估 ──
    state_tensor = encode_state(state).unsqueeze(0).to(device)  # [1, 6, 9, 9]
    with torch.no_grad():
        policy, _ = net(state_tensor)
    policy = policy.squeeze(0).cpu().numpy()            # [225]

    # 掩码非法动作 + 重归一化
    masked = np.zeros(num_actions)
    for a in legal_actions:
        masked[action_to_index(a)] = policy[action_to_index(a)]
    p_sum = masked.sum()
    if p_sum > 1e-12:
        masked /= p_sum
    else:
        for a in legal_actions:
            masked[action_to_index(a)] = 1.0 / len(legal_actions)

    # ── 创建根节点 ──
    root_state = state.copy()
    root = MCTSNode(root_state, 1.0)
    for a in legal_actions:
        idx = action_to_index(a)
        child_state = root_state.copy()
        if a.apply(child_state):
            root.children[idx] = MCTSNode(child_state, float(masked[idx]))

    # 根节点 Dirichlet 噪声
    dirichlet = np.random.dirichlet([dirichlet_alpha] * len(root.children))
    for i, (_, child) in enumerate(root.children.items()):
        child.prior_p = (1.0 - dirichlet_w) * child.prior_p + dirichlet_w * dirichlet[i]

    # ── 搜索循环 ──
    for _ in range(sims):
        node = root
        path = []

        # 选择 (Selection)
        while node.is_expanded():
            # 检查所有子节点是否到达终局
            best_score = -math.inf
            best_idx = None
            best_child = None
            sqrt_n = math.sqrt(node.visit_count + 1)

            for idx, child in node.children.items():
                # Q(parent, action) = -V(child)  子节点是对手方视角，需翻转
                q = -child.value
                u = c_puct * child.prior_p * sqrt_n / (1 + child.visit_count)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_child = child

            path.append((node, best_idx))
            node = best_child

            if check_terminal(node.state):
                break

        # 评估 (Evaluate)
        winner = check_terminal(node.state)
        if winner:
            # 终局: 当前轮到玩家输了 (轮到谁即是对方刚赢)
            leaf_value = -1.0
        else:
            # 叶子节点 → 网络评估 → 扩展
            leaf_tensor = encode_state(node.state).unsqueeze(0).to(device)
            with torch.no_grad():
                leaf_policy, leaf_value_t = net(leaf_tensor)
            leaf_value = float(leaf_value_t.item())

            # 扩展叶子
            legal = _filter_legal(get_legal_actions(node.state),
                                 node.state, no_wall_player)
            if legal:
                lp = leaf_policy.squeeze(0).cpu().numpy()
                masked_lp = np.zeros(num_actions)
                for a in legal:
                    masked_lp[action_to_index(a)] = lp[action_to_index(a)]
                lp_sum = masked_lp.sum()
                if lp_sum > 1e-12:
                    masked_lp /= lp_sum
                else:
                    for a in legal:
                        masked_lp[action_to_index(a)] = 1.0 / len(legal)

                for a in legal:
                    idx = action_to_index(a)
                    cs = node.state.copy()
                    if a.apply(cs):
                        node.children[idx] = MCTSNode(cs, float(masked_lp[idx]))

        # 回传 (Backup) — 沿路径交替翻转价值
        v = leaf_value
        node.visit_count += 1
        node.total_value += v
        for parent_node, _ in reversed(path):
            v = -v                  # 翻转视角
            parent_node.visit_count += 1
            parent_node.total_value += v

    # ── 计算 π (visit count 归一化) ──
    temp = config["temperature"]
    if temp > 1e-6:
        visits = np.array([c.visit_count for c in root.children.values()])
        visits = visits ** (1.0 / temp)
        total = visits.sum()
        if total > 1e-12:
            visits /= total
        else:
            visits = np.ones(len(visits)) / len(visits)
    else:
        # 温度 = 0: argmax
        visits = np.zeros(len(root.children))
        best = max(root.children.values(), key=lambda c: c.visit_count)
        visits[list(root.children.values()).index(best)] = 1.0

    pi = np.zeros(num_actions)
    for i, (idx, _) in enumerate(root.children.items()):
        pi[idx] = visits[i]

    return pi


# =============================================================================
# 自对弈一局
# =============================================================================

def play_one_game(net: torch.nn.Module,
                  config: dict | None = None,
                  no_wall_player: int | None = None) -> list[dict]:
    """用当前网络完成一局自对弈，返回训练样本。

    每步记录 (state_encoding, π, player)，终局后填入 value_target。
    样本中的 value_target 从**当前玩家视角**出发 (+1 赢 / -1 输)。

    参数:
        net:            策略-价值网络，双方共用
        config:         超参数字典
        no_wall_player: 被禁止放墙的玩家 (1 或 2), None=双方均可

    返回:
        [ {encoded_state, policy_target, value_target, turn}, ... ]
    """
    if config is None:
        config = CONFIG

    game_data = []
    state = State()
    state.reset()
    step = 0
    MAX_MOVES = 120

    while step < MAX_MOVES:
        winner = check_terminal(state)
        if winner:
            break

        # 温度调度
        temp_override = config["temperature"]
        if step >= config["temperature_steps"]:
            temp_override = config["temperature_min"]

        search_config = dict(config)
        search_config["temperature"] = temp_override

        cur = state.turn
        pi = mcts_search(state, net, search_config, no_wall_player=no_wall_player)

        # 采样并落子
        legal = _filter_legal(get_legal_actions(state), state, no_wall_player)
        action_idx = np.random.choice(config["num_actions"], p=pi)
        action = None
        for a in legal:
            if action_to_index(a) == action_idx:
                action = a
                break

        # 编码走子前局面
        encoded = encode_state(state).cpu().numpy()

        ok = action.apply(state) if action else False
        mover = cur

        sample = {
            "encoded_state":  encoded,
            "policy_target":  pi.astype(np.float32),
            "value_target":   0.0,
            "turn":           mover,
        }
        game_data.append(sample)

        if ok:
            step += 1

    # ── 纯终局 value_target: 所有局面统一使用终局结果 ±1 ──
    if not winner:
        d1 = min_distance_to_goal(state, 1)
        d2 = min_distance_to_goal(state, 2)
        if d1 != float('inf') and d2 != float('inf') and d1 < d2:
            winner = 1
        elif d1 != float('inf') and d2 != float('inf') and d2 < d1:
            winner = 2
    if winner:
        for s in game_data:
            s["value_target"] = 1.0 if winner == s["turn"] else -1.0
    else:
        for s in game_data:
            s["value_target"] = -1.0

    return game_data


# =============================================================================
# 对手池对弈 — play_vs_opponent_game
# =============================================================================

def play_vs_opponent_game(net: torch.nn.Module,
                          opponent_net: torch.nn.Module,
                          config: dict,
                          net_plays_as: int = 1,
                          no_wall_net: bool = False,
                          no_wall_opp: bool = False) -> tuple[list[dict], float]:
    """新网络 vs 对手池模型一局，只收集新网络的训练样本。

    参数:
        net:           当前训练的网络
        opponent_net:  对手池中的历史模型
        config:        超参数字典
        net_plays_as:  新网络扮演哪一方 (1 或 2)
        no_wall_net:   True=新网络禁止放墙
        no_wall_opp:   True=对手禁止放墙

    返回:
        (samples, score): 训练样本列表 + 从新网络视角的终局得分 [-1, 1]
    """
    game_data = []
    state = State()
    state.reset()
    step = 0
    MAX_MOVES = 120  # vs-pool 对局需要足够步数走到底

    eval_config = dict(config)
    eval_config["dirichlet_weight"] = 0.0       # 对战不打噪声
    eval_config["temperature"] = 0.5           # vs-pool 温度，保证局面多样性

    while step < MAX_MOVES:
        winner = check_terminal(state)
        if winner:
            break

        is_net_turn = (state.turn == net_plays_as)

        # 选择网络和墙限制
        cur_net = net if is_net_turn else opponent_net
        nw = None
        if no_wall_net and is_net_turn:
            nw = state.turn
        elif no_wall_opp and not is_net_turn:
            nw = state.turn

        pi = mcts_search(state, cur_net, eval_config, no_wall_player=nw)

        # 采样并落子（概率采样，保证多样性）
        legal = _filter_legal(get_legal_actions(state), state, nw)
        action_idx = np.random.choice(config["num_actions"], p=pi)
        action = None
        for a in legal:
            if action_to_index(a) == action_idx:
                action = a
                break

        # 编码走子前局面（只收集新网络样本时）
        if is_net_turn:
            encoded = encode_state(state).cpu().numpy()

        ok = action.apply(state) if action else False

        # 只收集新网络的样本
        if is_net_turn:
            mover = state.turn
            # 注意: state.turn 在 action.apply 后已翻转，mover 是 apply 前的当前玩家
            mover = 3 - mover if ok else mover
            sample = {
                "encoded_state":  encoded,
                "policy_target":  pi.astype(np.float32),
                "value_target":   0.0,
                "turn":           mover,
            }
            game_data.append(sample)

        if ok:
            step += 1

    # ── 得分与纯终局 value_target ──
    if not winner:
        d1 = min_distance_to_goal(state, 1)
        d2 = min_distance_to_goal(state, 2)
        if d1 != float('inf') and d2 != float('inf') and d1 < d2:
            winner = 1
        elif d1 != float('inf') and d2 != float('inf') and d2 < d1:
            winner = 2
    if winner:
        for s in game_data:
            s["value_target"] = 1.0 if winner == s["turn"] else -1.0
        score = 1.0 if net_plays_as == winner else -1.0
    else:
        for s in game_data:
            s["value_target"] = -1.0
        score = -1.0

    return game_data, score
