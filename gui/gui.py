#!/usr/bin/env python
# =============================================================================
# Quoridor 图形界面 — gui.py
#
# Pygame 前端，支持人机对战和 AI 自对弈模式。
# 不修改任何现有代码，所有新增内容在此文件。
# =============================================================================
import sys
import os
import pygame
import numpy as np
import torch

# ── 路径设置（同时支持 python 直接运行和 PyInstaller 打包后运行） ──
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR  = os.path.join(BASE_DIR, 'rl')
if getattr(sys, 'frozen', False):
    os.add_dll_directory(RL_DIR)
sys.path.insert(0, RL_DIR)   # 使 quoridor_cpp.pyd 可被直接 import
sys.path.insert(0, BASE_DIR) # 使 from rl.xxx 能正确解析

# ── PyInstaller 打包后：注册 rl.xxx 的顶层别名 ──
# rl/model.py 等内部使用 "import encode"、"from config import CONFIG" 等裸名导入，
# 而 PyInstaller PYZ 中模块按全限定名 rl.encode 存储，裸名找不到。
# 这里预加载并注册别名，使得裸名导入在 frozen 环境下也能工作。
if getattr(sys, 'frozen', False):
    import importlib
    for _name in ('encode', 'config'):
        sys.modules[_name] = importlib.import_module(f'rl.{_name}')

from quoridor_cpp import (State, Action, get_legal_actions, get_legal_moves,
                           get_legal_walls, ROW_SIZE, COLUMN_SIZE, WALL_NUM)
from rl.config import CONFIG
from rl.model import QuoridorNet
from rl.self_play import action_to_index, index_to_action, mcts_search, check_terminal

# ═════════════════════════════════════════════════════════════════════════
#  常量
# ═════════════════════════════════════════════════════════════════════════

# 渲染尺寸
CELL_SIZE = 54          # 格子像素宽高
WALL_SIZE = 10          # 墙通道厚度
GRID_STEP = CELL_SIZE + WALL_SIZE  # 64 — 一个格子+一条通道的重复单元
BOARD_PX  = 9 * CELL_SIZE + 10 * WALL_SIZE  # 586 — 棋盘区域总宽高

MARGIN_TOP = 70
MARGIN_LEFT = 40
SIDE_PANEL_X = MARGIN_LEFT + BOARD_PX + 30  # 侧栏起始 x

WIN_W = SIDE_PANEL_X + 240
WIN_H = MARGIN_TOP + BOARD_PX + 40

# 颜色 — 简约现代风格
COLOR_BG        = (30,  30,  46)     # 窗口背景 — 深蓝黑
COLOR_BOARD_BG  = (195, 195, 205)    # 棋盘底 — 浅灰
COLOR_CELL      = (255, 255, 255)    # 格子 — 白
COLOR_CELL_BDR  = (210, 210, 218)    # 格子边框 — 浅灰
COLOR_WALL      = (80,  75,  85)     # 墙 — 深灰紫
COLOR_WALL_BD   = (55,  50,  60)     # 墙边框
COLOR_P1        = (74,  130, 255)    # 玩家1 蓝
COLOR_P2        = (255, 107, 107)    # 玩家2 珊瑚红
COLOR_P1_LIGHT  = (130, 175, 255)
COLOR_P2_LIGHT  = (255, 165, 165)
COLOR_HL_MOVE    = (80,  220, 80,  180)  # 合法移动（半透明）
COLOR_HL_WALL    = (255, 190, 50,  200)  # 合法放墙（半透明）
COLOR_HL_HOVER_M = (100, 255, 100, 220)  # 悬停移动
COLOR_HL_HOVER_W = (255, 230, 80,  230)  # 悬停放墙
COLOR_LAST_MOVE  = (255, 215, 0)         # 上一步高亮
COLOR_TEXT       = (230, 230, 240)
COLOR_TEXT_DIM   = (155, 155, 165)
COLOR_BTN_BG     = (55,  55,  70)
COLOR_BTN_HOVER  = (75,  75,  90)
COLOR_BTN_ACTIVE = (74,  130, 255)       # 模式激活按钮

CLICK_THRESHOLD = 32  # 点击匹配的最大像素距离


# ═════════════════════════════════════════════════════════════════════════
#  坐标工具函数
# ═════════════════════════════════════════════════════════════════════════

def board_pos_to_screen(r: int, c: int) -> tuple:
    """棋盘坐标 (r,c) ∈ 0..18 → 屏幕像素坐标（矩形左上角）"""
    gx = (c // 2) * GRID_STEP
    gy = (r // 2) * GRID_STEP
    if r % 2 == 1 and c % 2 == 1:          # 格子
        return (MARGIN_LEFT + gx + WALL_SIZE,
                MARGIN_TOP  + gy + WALL_SIZE)
    elif r % 2 == 1 and c % 2 == 0:        # 垂直通道
        return (MARGIN_LEFT + gx,
                MARGIN_TOP  + gy + WALL_SIZE)
    elif r % 2 == 0 and c % 2 == 1:        # 水平通道
        return (MARGIN_LEFT + gx + WALL_SIZE,
                MARGIN_TOP  + gy)
    else:                                   # 交叉点
        return (MARGIN_LEFT + gx,
                MARGIN_TOP  + gy)


def board_rect(r: int, c: int) -> pygame.Rect:
    """棋盘坐标 (r,c) → 像素矩形"""
    x, y = board_pos_to_screen(r, c)
    if r % 2 == 1 and c % 2 == 1:
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    elif r % 2 == 1 and c % 2 == 0:
        return pygame.Rect(x, y, WALL_SIZE, CELL_SIZE)
    elif r % 2 == 0 and c % 2 == 1:
        return pygame.Rect(x, y, CELL_SIZE, WALL_SIZE)
    else:
        return pygame.Rect(x, y, WALL_SIZE, WALL_SIZE)


def cell_center(r: int, c: int) -> tuple:
    """格子中心像素坐标（r,c 必须为奇数）"""
    rect = board_rect(r, c)
    return rect.centerx, rect.centery


def wall_center(r: int, c: int, d: int) -> tuple:
    """墙中心像素坐标（墙条覆盖 3 个段位，中心在交叉点）"""
    x = MARGIN_LEFT + (c // 2) * GRID_STEP + WALL_SIZE // 2
    y = MARGIN_TOP  + (r // 2) * GRID_STEP + WALL_SIZE // 2
    return (x, y)


def wall_rect(r: int, c: int, d: int) -> pygame.Rect:
    """墙动作的完整包围矩形 (r,c 为偶数交叉点, d=0垂直/d=1水平)"""
    if d == 0:  # 垂直墙: 跨 (r-1,c)竖通道 + (r,c)交叉点 + (r+1,c)竖通道
        x = MARGIN_LEFT + (c // 2) * GRID_STEP
        y = MARGIN_TOP + (r // 2) * GRID_STEP - CELL_SIZE
        return pygame.Rect(x, y, WALL_SIZE, 2 * CELL_SIZE + WALL_SIZE)
    else:  # 水平墙: 跨 (r,c-1)横通道 + (r,c)交叉点 + (r,c+1)横通道
        x = MARGIN_LEFT + (c // 2) * GRID_STEP - CELL_SIZE
        y = MARGIN_TOP + (r // 2) * GRID_STEP
        return pygame.Rect(x, y, 2 * CELL_SIZE + WALL_SIZE, WALL_SIZE)


def action_center(action) -> tuple:
    """Action 对应的屏幕中心像素"""
    r, c = action.pos
    if not action.is_wall:
        return cell_center(r, c)
    return wall_center(r, c, action.wall_dir)


def find_action_at(px: int, py: int, actions: list) -> object | None:
    """在 actions 中找匹配点击的合法动作（移动用圆形距离，墙用矩形碰撞）"""
    best = None
    best_d2 = CLICK_THRESHOLD * CLICK_THRESHOLD
    for a in actions:
        if not a.is_wall:
            ax, ay = action_center(a)
            d2 = (px - ax) * (px - ax) + (py - ay) * (py - ay)
            if d2 < best_d2:
                best_d2 = d2
                best = a
        else:
            # 墙使用矩形碰撞检测（覆盖 3 个墙段）
            rect = wall_rect(a.pos[0], a.pos[1], a.wall_dir)
            if rect.collidepoint(px, py):
                ax, ay = action_center(a)
                d2 = (px - ax) * (px - ax) + (py - ay) * (py - ay)
                if d2 < best_d2:
                    best_d2 = d2
                    best = a
    return best


def _draw_action_surf(surf: pygame.Surface, screen, cx, cy):
    """将半透明 surf 居中绘制到屏幕"""
    screen.blit(surf, surf.get_rect(center=(cx, cy)))


# ═════════════════════════════════════════════════════════════════════════
#  Gui 类
# ═════════════════════════════════════════════════════════════════════════

class QuoridorGUI:
    """Pygame 图形化 Quoridor 人机对战 / AI 自对弈"""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("步步为营  Quoridor")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        # 中文字体回退链
        _fonts = ['microsoftyahei', 'simhei', 'notosanscjk', 'arial']
        self.font = pygame.font.SysFont(_fonts, 16)
        self.font_big = pygame.font.SysFont(_fonts, 22)

        # ── 游戏状态 ──
        self.state = State()
        self.state.reset()
        self.game_step = 0
        self.last_action = None          # Action 对象（用于高亮）
        self.last_action_highlight = 0   # 剩余高亮时间 (ms)
        self.game_over = False
        self.winner = 0
        self.thinking = False            # AI 正在计算
        self.human_turn_prompt = ""
        self.human_last_error = ""

        # ── 模式: 'pvai' 或 'aivai' ──
        self.mode = 'pvai'
        self.ai_move_delay = 400  # AI vs AI 每步间隔 ms
        self.last_ai_move_time = 0

        # ── 操作模式: 移动 / 放墙 ──
        self.wall_mode = False       # True=放墙模式, False=移动模式
        self.hover_action = None     # 鼠标悬停的合法动作

        # ── 加载 AI ──
        self.net = self._load_net()
        self.ai_config = dict(CONFIG)
        self.ai_config["mcts_simulations"] = 200
        self.ai_config["temperature"] = 0.0
        self.ai_config["dirichlet_weight"] = 0.0

        # ── 按钮区域 ──
        self.btn_move_mode = pygame.Rect(SIDE_PANEL_X, 310, 100, 32)
        self.btn_wall_mode = pygame.Rect(SIDE_PANEL_X + 110, 310, 100, 32)
        self.btn_new_pvai = pygame.Rect(SIDE_PANEL_X, WIN_H - 90, 210, 34)
        self.btn_new_aivai = pygame.Rect(SIDE_PANEL_X, WIN_H - 50, 210, 34)

    # ── 模型加载 ────────────────────────────────────────────────────

    def _load_net(self) -> torch.nn.Module:
        ckpt_dir = os.path.join(BASE_DIR, 'rl', 'weights', 'checkpoints')
        device = torch.device('cpu')
        net = QuoridorNet(CONFIG["input_channels"]).to(device)
        net.eval()

        # 找最新 checkpoint
        path = None
        if os.path.isdir(ckpt_dir):
            files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')],
                           reverse=True)
            if files:
                path = os.path.join(ckpt_dir, files[0])

        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            net.load_state_dict(ckpt["model_state_dict"])
            print(f"[GUI] 加载模型: {os.path.basename(path)}")
        else:
            print(f"[GUI] 未找到 checkpoint，AI 使用随机走子")
        return net

    # ── 棋盘绘制 ────────────────────────────────────────────────────

    def draw_board(self):
        """绘制棋盘 — 从 19×19 网格读取 state.get_cell"""
        # 棋盘底（空通道和交叉点显示为灰色）
        board_rect_all = pygame.Rect(
            MARGIN_LEFT - 4, MARGIN_TOP - 4,
            BOARD_PX + 8, BOARD_PX + 8)
        pygame.draw.rect(self.screen, COLOR_BOARD_BG, board_rect_all,
                         border_radius=4)

        # 模式指示边框（绿=移动, 橙=放墙）
        mode_color = (55, 170, 55) if not self.wall_mode else (200, 150, 40)
        pygame.draw.rect(self.screen, mode_color, board_rect_all, 3,
                         border_radius=4)

        # 逐格绘制
        for r in range(1, 2 * ROW_SIZE):       # 1..17
            for c in range(1, 2 * COLUMN_SIZE): # 1..17
                rect = board_rect(r, c)
                is_cell = (r % 2 == 1 and c % 2 == 1)

                if self.state.get_cell(r, c):
                    # 墙段 → 用墙色填充
                    pygame.draw.rect(self.screen, COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, COLOR_WALL_BD, rect, 1)
                elif is_cell:
                    # 空格子
                    pygame.draw.rect(self.screen, COLOR_CELL, rect)
                    pygame.draw.rect(self.screen, COLOR_CELL_BDR, rect, 1)
                # else: 通道/交叉点留空（显示 COLOR_BOARD_BG）

    def draw_pieces(self):
        """绘制两个玩家的棋子"""
        for player in (1, 2):
            pr, pc = self.state.get_pos(player)
            cx, cy = cell_center(pr, pc)
            color = COLOR_P1 if player == 1 else COLOR_P2
            radius = CELL_SIZE // 2 - 4

            # 阴影
            pygame.draw.circle(self.screen, (0, 0, 0, 60),
                               (cx + 2, cy + 2), radius)
            # 主体
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            # 高光
            light = COLOR_P1_LIGHT if player == 1 else COLOR_P2_LIGHT
            pygame.draw.circle(self.screen, light, (cx - 4, cy - 4), radius // 3)
            # 棋子编号
            label = self.font.render(str(player), True, (255, 255, 255))
            lr = label.get_rect(center=(cx, cy))
            self.screen.blit(label, lr)

    # ── 高亮绘制 ---------------------------------------------------

    def _make_move_surf(self, color, size=CELL_SIZE, radius_offset=6):
        """创建移动高亮半透明圆 Surface"""
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(surf, color,
                           (size // 2, size // 2),
                           size // 2 - radius_offset)
        return surf

    def _make_wall_surf(self, d, color, extra=0):
        """创建墙高亮半透明条 Surface"""
        if d == 0:  # 垂直墙
            w, h = WALL_SIZE + 2 + extra, CELL_SIZE * 2 + WALL_SIZE + extra
        else:       # 水平墙
            w, h = CELL_SIZE * 2 + WALL_SIZE + extra, WALL_SIZE + 2 + extra
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(color)
        return surf

    def draw_highlights(self):
        """高亮合法动作、悬停动作和上一步"""
        # ── 上一步高亮（金色渐消） ──
        if self.last_action and self.last_action_highlight > 0:
            t = self.last_action_highlight / 1200.0
            alpha = int(180 * t)
            if self.last_action.is_wall:
                cx, cy = wall_center(*self.last_action.pos,
                                     self.last_action.wall_dir)
                surf = self._make_wall_surf(self.last_action.wall_dir,
                                            (*COLOR_LAST_MOVE, alpha), extra=4)
                _draw_action_surf(surf, self.screen, cx, cy)
            else:
                cx, cy = cell_center(*self.last_action.pos)
                surf = self._make_move_surf((*COLOR_LAST_MOVE, alpha),
                                            CELL_SIZE + 6, 2)
                _draw_action_surf(surf, self.screen, cx, cy)

        # ── 轮到人类时才显示可操作高亮 ──
        if not self._is_human_turn() or self.game_over:
            return

        legal = get_legal_actions(self.state)
        # 按当前模式筛选高亮
        show_walls = self.wall_mode
        show_moves = not self.wall_mode

        for a in legal:
            if not a.is_wall and not show_moves:
                continue
            if a.is_wall and not show_walls:
                continue

            r, c = a.pos
            if not a.is_wall:
                cx, cy = cell_center(r, c)
                surf = self._make_move_surf(COLOR_HL_MOVE)
                _draw_action_surf(surf, self.screen, cx, cy)
            else:
                cx, cy = wall_center(r, c, a.wall_dir)
                surf = self._make_wall_surf(a.wall_dir, COLOR_HL_WALL)
                _draw_action_surf(surf, self.screen, cx, cy)

        # ── 悬停高亮（覆盖在普通高亮之上） ──
        if self.hover_action:
            a = self.hover_action
            if not a.is_wall:
                cx, cy = cell_center(*a.pos)
                surf = self._make_move_surf(COLOR_HL_HOVER_M,
                                            CELL_SIZE + 4, 4)
                _draw_action_surf(surf, self.screen, cx, cy)
            else:
                cx, cy = wall_center(*a.pos, a.wall_dir)
                surf = self._make_wall_surf(a.wall_dir, COLOR_HL_HOVER_W,
                                            extra=2)
                _draw_action_surf(surf, self.screen, cx, cy)

    # ── 信息面板 ────────────────────────────────────────────────────

    def draw_info(self):
        """绘制右侧信息面板"""
        x = SIDE_PANEL_X
        y = MARGIN_TOP

        # 边框
        info_rect = pygame.Rect(x - 10, y - 10, 230, 280)
        pygame.draw.rect(self.screen, (55, 55, 62), info_rect, border_radius=6)

        # 标题
        title = self.font_big.render("步步为营", True, COLOR_TEXT)
        self.screen.blit(title, (x, y))
        y += 30

        # 步数
        text = self.font.render(f"步数  {self.game_step}", True, COLOR_TEXT)
        self.screen.blit(text, (x, y))
        y += 26

        # 回合
        turn_str = f"回合  Player {self.state.turn}"
        turn_color = COLOR_P1 if self.state.turn == 1 else COLOR_P2
        text = self.font.render(turn_str, True, turn_color)
        self.screen.blit(text, (x, y))
        y += 26

        # 对战模式
        mode_str = "人机对战" if self.mode == 'pvai' else "AI 自对弈"
        text = self.font.render(f"模式  {mode_str}", True, COLOR_TEXT_DIM)
        self.screen.blit(text, (x, y))
        y += 30

        # 墙数
        for p in (1, 2):
            color = COLOR_P1 if p == 1 else COLOR_P2
            num = self.state.get_wall_num(p)
            bar_w = 120
            fill_w = int(bar_w * num / WALL_NUM)
            pygame.draw.rect(self.screen, (40, 40, 45),
                             (x, y, bar_w, 14), border_radius=3)
            if fill_w > 0:
                bar_color = (color[0] // 2, color[1] // 2, color[2] // 2)
                pygame.draw.rect(self.screen, bar_color,
                                 (x, y, fill_w, 14), border_radius=3)
            label = self.font.render(f"P{p}  {num}/10", True, color)
            self.screen.blit(label, (x + bar_w + 10, y - 2))
            y += 22

        y += 6

        # AI 思考提示
        if self.thinking:
            text = self.font.render("AI 思考中...", True, (255, 200, 80))
            self.screen.blit(text, (x, y))
            y += 22

        # 人类提示
        if self.human_turn_prompt:
            prompt_color = (180, 220, 255) if not self.wall_mode else (255, 220, 150)
            text = self.font.render(self.human_turn_prompt, True, prompt_color)
            self.screen.blit(text, (x, y))

    # ── 操作模式切换 ───────────────────────────────────────────────

    def draw_mode_control(self):
        """绘制操作模式切换（分段控件 + 模式指示标签）"""
        mouse = pygame.mouse.get_pos()

        # 模式标签
        label = self.font.render("操作模式", True, COLOR_TEXT_DIM)
        self.screen.blit(label, (self.btn_move_mode.x, self.btn_move_mode.y - 20))

        # ── 左: 移动模式 ──
        btn_m = self.btn_move_mode
        hover_m = btn_m.collidepoint(mouse)
        active_m = not self.wall_mode
        if active_m:
            bg = (55, 170, 55) if hover_m else (45, 140, 45)
        else:
            bg = COLOR_BTN_HOVER if hover_m else COLOR_BTN_BG
        pygame.draw.rect(self.screen, bg, btn_m, border_radius=4)
        tc = (255, 255, 255) if active_m else (140, 140, 150)
        text = self.font.render("● 移动", True, tc)
        self.screen.blit(text, text.get_rect(center=btn_m.center))

        # ── 右: 放墙模式 ──
        btn_w = self.btn_wall_mode
        hover_w = btn_w.collidepoint(mouse)
        active_w = self.wall_mode
        if active_w:
            bg = (200, 150, 40) if hover_w else (170, 125, 30)
        else:
            bg = COLOR_BTN_HOVER if hover_w else COLOR_BTN_BG
        pygame.draw.rect(self.screen, bg, btn_w, border_radius=4)
        tc = (255, 255, 255) if active_w else (140, 140, 150)
        text = self.font.render("▦ 放墙", True, tc)
        self.screen.blit(text, text.get_rect(center=btn_w.center))

    # ── 按钮 ────────────────────────────────────────────────────────

    def draw_buttons(self):
        mouse = pygame.mouse.get_pos()
        for btn, label in [(self.btn_new_pvai, "新游戏 人机对战"),
                           (self.btn_new_aivai, "新游戏 AI 自对弈")]:
            hover = btn.collidepoint(mouse)
            color = COLOR_BTN_HOVER if hover else COLOR_BTN_BG
            pygame.draw.rect(self.screen, color, btn, border_radius=4)
            pygame.draw.rect(self.screen, (100, 100, 110), btn, 1, border_radius=4)
            text = self.font.render(label, True, COLOR_TEXT)
            tr = text.get_rect(center=btn.center)
            self.screen.blit(text, tr)

    def handle_button_click(self, px, py):
        if self.btn_move_mode.collidepoint(px, py):
            self.wall_mode = False
            return True
        if self.btn_wall_mode.collidepoint(px, py):
            self.wall_mode = True
            return True
        if self.btn_new_pvai.collidepoint(px, py):
            self._new_game('pvai')
            return True
        if self.btn_new_aivai.collidepoint(px, py):
            self._new_game('aivai')
            return True
        return False

    def _new_game(self, mode):
        self.state = State()
        self.state.reset()
        self.game_step = 0
        self.last_action = None
        self.last_action_highlight = 0
        self.game_over = False
        self.winner = 0
        self.thinking = False
        self.human_turn_prompt = ""
        self.human_last_error = ""
        self.mode = mode
        self.wall_mode = False
        self.hover_action = None
        self.last_ai_move_time = pygame.time.get_ticks()

    # ── 游戏逻辑 ────────────────────────────────────────────────────

    def _is_human_turn(self) -> bool:
        if self.game_over or self.thinking:
            return False
        if self.mode == 'aivai':
            return False
        return self.state.turn == 1

    def _update_hover(self, mx: int, my: int):
        """根据鼠标位置和当前模式更新悬停高亮"""
        self.hover_action = None
        if not self._is_human_turn() or self.game_over:
            return
        legal = get_legal_actions(self.state)
        if self.wall_mode:
            legal = [a for a in legal if a.is_wall]
        else:
            legal = [a for a in legal if not a.is_wall]
        if not legal:
            return
        self.hover_action = find_action_at(mx, my, legal)

    def _do_ai_move(self):
        """AI 走一步（MCTS 搜索）"""
        self.thinking = True
        winner = check_terminal(self.state)
        if winner:
            self.game_over = True
            self.winner = winner
            self.thinking = False
            return

        pi, _ = mcts_search(self.state, self.net, self.ai_config)
        # 采样
        action_idx = np.random.choice(CONFIG["num_actions"], p=pi)
        action = index_to_action(action_idx)

        # 合法性回退
        legal = get_legal_actions(self.state)
        legal_idx = [action_to_index(a) for a in legal]
        if action_idx not in legal_idx:
            best = max(legal_idx, key=lambda i: pi[i])
            action = index_to_action(best)

        action.apply(self.state)
        self.last_action = action
        self.last_action_highlight = 1200  # ms
        self.game_step += 1

        winner = check_terminal(self.state)
        if winner:
            self.game_over = True
            self.winner = winner
        self.thinking = False

    def _do_human_move(self, action) -> bool:
        """人类玩家执行动作"""
        if action.apply(self.state):
            self.last_action = action
            self.last_action_highlight = 1200
            self.game_step += 1
            self.human_last_error = ""

            winner = check_terminal(self.state)
            if winner:
                self.game_over = True
                self.winner = winner
            return True
        return False

    # ── 主循环 ──────────────────────────────────────────────────────

    def run(self):
        running = True
        self.last_ai_move_time = pygame.time.get_ticks()

        while running:
            dt = self.clock.tick(60)  # ~60 FPS
            now = pygame.time.get_ticks()

            # ── 事件 ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_n:
                        self._new_game('pvai')
                    if event.key == pygame.K_a:
                        self._new_game('aivai')
                    if event.key == pygame.K_TAB:
                        if self._is_human_turn():
                            self.wall_mode = not self.wall_mode
                    if event.key == pygame.K_1:
                        if self._is_human_turn():
                            self.wall_mode = False
                    if event.key == pygame.K_2:
                        if self._is_human_turn():
                            self.wall_mode = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    px, py = event.pos
                    if self.handle_button_click(px, py):
                        continue
                    if self._is_human_turn():
                        legal = get_legal_actions(self.state)
                        # 按当前模式筛选可点击的动作
                        if self.wall_mode:
                            legal = [a for a in legal if a.is_wall]
                        else:
                            legal = [a for a in legal if not a.is_wall]
                        action = find_action_at(px, py, legal)
                        if action:
                            self._do_human_move(action)

            # ── 高亮计时递减 ──
            if self.last_action_highlight > 0:
                self.last_action_highlight = max(0,
                    self.last_action_highlight - dt)

            # ── 鼠标悬停 ──
            mx, my = pygame.mouse.get_pos()
            self._update_hover(mx, my)

            # ── 人类回合提示 ──
            if self._is_human_turn():
                if self.wall_mode:
                    self.human_turn_prompt = "点击橙条放墙 · [TAB/2]切换"
                else:
                    self.human_turn_prompt = "点击绿点移动 · [TAB/1]切换"
            elif not self.thinking:
                self.human_turn_prompt = ""

            # ── AI 回合触发 ──
            ai_should = False
            if not self.game_over and not self.thinking:
                if self.mode == 'aivai':
                    if now - self.last_ai_move_time >= self.ai_move_delay:
                        ai_should = True
                elif self.mode == 'pvai' and self.state.turn == 2:
                    ai_should = True

            if ai_should:
                self.thinking = True
                # 立即刷新让 "AI 思考中..." 显示出来（后面是阻塞计算）
                self.screen.fill(COLOR_BG)
                self.draw_board()
                self.draw_highlights()
                self.draw_pieces()
                if self.game_over:
                    self._draw_game_over()
                self.draw_info()
                self.draw_mode_control()
                self.draw_buttons()
                pygame.display.flip()
                # 阻塞式 AI 计算
                self._do_ai_move()
                if self.mode == 'aivai':
                    self.last_ai_move_time = now

            # ── 绘制 ──
            self.screen.fill(COLOR_BG)
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()

            # 终局文字
            if self.game_over:
                self._draw_game_over()

            self.draw_info()
            self.draw_mode_control()
            self.draw_buttons()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def _draw_game_over(self):
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        if self.winner == 1:
            msg = "Player 1 获胜!"
            color = COLOR_P1
        elif self.winner == 2:
            msg = "Player 2 获胜!"
            color = COLOR_P2
        else:
            msg = "平局!"
            color = COLOR_TEXT

        text = self.font_big.render(msg, True, color)
        tr = text.get_rect(center=(WIN_W // 2, WIN_H // 2 - 20))
        # 背景
        bg = pygame.Rect(tr.left - 30, tr.top - 10,
                         tr.width + 60, tr.height + 50)
        pygame.draw.rect(self.screen, (40, 40, 48), bg, border_radius=8)
        pygame.draw.rect(self.screen, (80, 80, 90), bg, 2, border_radius=8)

        self.screen.blit(text, tr)
        tip = self.font.render("N 新游戏 · ESC 退出", True, COLOR_TEXT_DIM)
        tr2 = tip.get_rect(center=(WIN_W // 2, WIN_H // 2 + 25))
        self.screen.blit(tip, tr2)


# ═════════════════════════════════════════════════════════════════════════

def main():
    gui = QuoridorGUI()
    gui.run()


if __name__ == "__main__":
    main()
