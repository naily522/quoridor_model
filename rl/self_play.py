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

def mcts_search(state: State, net: torch.nn.Module,
                config: dict | None = None) -> tuple[np.ndarray, float]:
    """在当前局面运行 MCTS 搜索。

    参数:
        state:  当前棋盘状态
        net:    策略-价值网络 (QuoridorNet)
        config: 超参数字典 (默认使用 CONFIG)

    返回:
        (π, root_value): 动作概率分布 (visit count 归一化) 和根节点 Q 值
        root_value 从当前玩家视角，综合了 MCTS 搜索结果 + 距离 shaping
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
    legal_actions = get_legal_actions(state)

    # 课程学习: 如 _force_moves_only 标记为 True，只保留移动动作
    if config.get("_force_moves_only", False):
        legal_actions = [a for a in legal_actions if not a.is_wall]

    if len(legal_actions) == 0:
        # 极端情况: 无合法移动 → 允许放墙
        legal_actions = get_legal_actions(state)
    if len(legal_actions) == 0:
        # 极端情况: 无任何合法动作 → 返回均匀分布 + 0 价值
        pi = np.ones(num_actions) / num_actions
        return pi, 0.0
    if len(legal_actions) == 1:
        pi = np.zeros(num_actions)
        pi[action_to_index(legal_actions[0])] = 1.0
        # 唯一合法动作: 用网络快速评估该后继状态的价值
        only_a = legal_actions[0]
        temp_state = state.copy()
        if only_a.apply(temp_state):
            winner_temp = check_terminal(temp_state)
            if winner_temp:
                root_val = 1.0 if winner_temp == state.turn else -1.0
            else:
                st = encode_state(temp_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, vt = net(st)
                root_val = float(vt.item())
        else:
            root_val = 0.0
        return pi, root_val

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
                q = -child.value  # 取反: 子节点是对手视角, 父节点选对己方最有利的
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

            # 距离引导奖励: 用双方最短距离差做 shaping
            # (d2 - d1) / 8 始终是 P1 视角 (正值=P1优势, 负值=P2优势)
            # leaf_value 是当前轮到玩家的视角，需要为 P2 翻转符号
            bonus_w = config.get("goal_bonus_weight", 0.0)
            if bonus_w > 0:
                d1 = min_distance_to_goal(node.state, 1)
                d2 = min_distance_to_goal(node.state, 2)
                if d1 != float('inf') or d2 != float('inf'):
                    if d1 == float('inf') or d2 == float('inf'):
                        distance_score = 0.0
                    else:
                        # P1 视角的得分
                        score_p1 = max(-1.0, min(1.0, (d2 - d1) / 8.0))
                        # 转换为当前玩家视角
                        distance_score = score_p1 if node.state.turn == 1 else -score_p1
                    leaf_value += bonus_w * distance_score

            # 扩展叶子
            legal = get_legal_actions(node.state)
            if config.get("_force_moves_only", False):
                legal = [a for a in legal if not a.is_wall]
                if not legal:
                    legal = get_legal_actions(node.state)
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

    return pi, root.value


# =============================================================================
# 自对弈一局
# =============================================================================

def play_one_game(net: torch.nn.Module,
                  config: dict | None = None,
                  wall_prob: float = 1.0,
                  opponent_net: torch.nn.Module | None = None) -> list[dict]:
    """用当前网络完成一局对弈（自对弈或 vs 历史对手），返回训练样本。

    当 opponent_net=None 时为标准自对弈（双方用同一网络）;
    当 opponent_net 不为 None 时，对手方用 opponent_net 的 MCTS 决策，
    这提供了训练数据的多样性，防止策略坍缩到单一自我风格。

    参数:
        net:          当前策略-价值网络
        config:       超参数字典
        wall_prob:    课程学习: 每步允许放墙的概率
        opponent_net: 对手网络 (None=自对弈, 否则 vs 历史对手)

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

        temp_override = config["temperature"]
        if step >= config["temperature_steps"]:
            temp_override = config["temperature_min"]

        search_config = dict(config)
        search_config["temperature"] = temp_override

        if wall_prob < 1.0:
            allow_walls = np.random.random() < wall_prob
            if not allow_walls:
                search_config["_force_moves_only"] = True

        cur_turn = state.turn
        cur_net = opponent_net if (opponent_net is not None and cur_turn == 2) else net

        dist_before = min_distance_to_goal(state, cur_turn)

        pi, mcts_root_q = mcts_search(state, cur_net, search_config)

        # 策略标签平滑 — 防止策略坍缩
        ls = config.get("policy_label_smoothing", 0.0)
        if ls > 0:
            pi = (1.0 - ls) * pi + ls / config["num_actions"]

        sample = {
            "encoded_state":  encode_state(state).cpu().numpy(),
            "policy_target":  pi.astype(np.float32),
            "value_target":   0.0,           # 终局后填入
            "turn":           cur_turn,
            "progress":       0.0,           # 单步距离改善 (当前玩家视角)
            "mcts_root_q":    mcts_root_q,   # MCTS 根节点 Q 值 (备用)
        }
        game_data.append(sample)

        action_idx = np.random.choice(config["num_actions"], p=pi)
        action = None
        for a in get_legal_actions(state):
            if action_to_index(a) == action_idx:
                action = a
                break
        if action is not None and action.apply(state):
            step += 1
            dist_after = min_distance_to_goal(state, cur_turn)
            if dist_before != float('inf') and dist_after != float('inf'):
                sample["progress"] = (dist_before - dist_after) / 4.0
            elif dist_after != float('inf'):
                sample["progress"] = 0.5
            elif dist_before != float('inf'):
                sample["progress"] = -0.5

    # =========================================================================
    # Monte Carlo 折扣回报 — 替换原来的弱 shaping 信号
    #
    # 核心思路:
    #   将每一步的 BFS 距离改善作为即时奖励,
    #   终局信号作为终端价值,
    #   然后用 γ 折扣因子反向累积,
    #   使得早期"向目标前进"的步骤获得更大的累计信用。
    #
    # 零和博弈的处理:
    #   所有奖励先转为 P1 视角 (P2 改善 = 对 P1 不利, 取负),
    #   计算完 MC 回报后再翻回各自的玩家视角。
    # =========================================================================
    gamma = config.get("mc_gamma", 0.95)
    reward_scale = config.get("reward_scale", 0.5)

    # 终端价值 (P1 视角)
    d1 = min_distance_to_goal(state, 1)
    d2 = min_distance_to_goal(state, 2)
    shape_scale = config.get("distance_scale", 4.0)
    if d1 == float('inf') and d2 == float('inf'):
        shape = 0.0
    else:
        shape = max(-1.0, min(1.0, (d2 - d1) / shape_scale))

    if winner == 1:
        v_terminal_p1 = 1.0
    elif winner == 2:
        v_terminal_p1 = -1.0
    else:
        # 平局: 放大距离 shaping 信号
        tw = config.get("terminal_value_weight", 0.7)
        sw = config.get("shape_value_weight", 0.3)
        v_terminal_p1 = tw * shape * 2.0 + sw * shape  # 加强 draw 信号

    # 构建 P1 视角的每步奖励序列
    N = len(game_data)
    if N == 0:
        return game_data

    rewards_p1 = []
    for s in game_data:
        r = s.get("progress", 0.0) * reward_scale
        if s["turn"] == 1:
            rewards_p1.append(r)
        else:
            rewards_p1.append(-r)   # P2 的进步 = P1 的损失

    # 反向累积折扣回报
    G = v_terminal_p1
    for i in range(N - 1, -1, -1):
        G = rewards_p1[i] + gamma * G
        # G 现在是 P1 视角的 MC 回报
        if game_data[i]["turn"] == 1:
            game_data[i]["value_target"] = G
        else:
            game_data[i]["value_target"] = -G
        game_data[i]["value_target"] = max(-1.0, min(1.0,
                                            game_data[i]["value_target"]))

    return game_data
