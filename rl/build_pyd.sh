#!/bin/bash
# =============================================================================
# 编译 pybind11 扩展模块
#
# 用法:
#   cd Quoridor
#   bash rl/build_pyd.sh
#
# 产出:
#   rl/quoridor_cpp.cp313-win_amd64.pyd   ← Python 可直接 import 的扩展
#   rl/libwinpthread-1.dll                 ← MinGW 运行时（.pyd 依赖）
#
# Python 端使用:
#   import sys; sys.path.insert(0, 'rl')
#   import os; os.add_dll_directory(os.path.dirname(sys.executable))
#   from quoridor_cpp import State, Action, ...
# =============================================================================

set -e

cd "$(dirname "$0")/.."
PROJ_ROOT="$PWD"

echo "=== 1/4 检查编译工具 ==="
command -v g++
g++ --version | head -1
python --version

echo ""
echo "=== 2/4 清理旧文件 ==="
rm -f rl/quoridor_cpp.cp313-win_amd64.pyd
rm -f rl/libwinpthread-1.dll

echo ""
echo "=== 3/4 编译 quoridor_bind.cpp → .pyd ==="

PY_INCLUDE="D:/python/Include"
PY_LIB="D:/python/libs"
PYBIND11_INCLUDE="D:/python/Lib/site-packages/pybind11/include"
MINGW_LIB="D:/mingw64/x86_64-w64-mingw32/lib"

g++ -std=c++17 -O2 \
    -I"$PY_INCLUDE" \
    -I"$PYBIND11_INCLUDE" \
    rl/quoridor_bind.cpp \
    -shared \
    -L"$PY_LIB" -l"python313" \
    -static-libgcc -static-libstdc++ \
    "$MINGW_LIB/libwinpthread.a" \
    -o rl/quoridor_cpp.cp313-win_amd64.pyd

# 复制 MinGW 运行时 DLL（.pyd 的隐式依赖）
cp "D:/mingw64/bin/libwinpthread-1.dll" rl/

ls -lh rl/quoridor_cpp.cp313-win_amd64.pyd
echo "编译成功"

echo ""
echo "=== 4/4 功能验证 ==="

python -c "
import sys; sys.path.insert(0, 'rl')
import os
os.add_dll_directory(os.path.dirname(sys.executable))

from quoridor_cpp import (
    State, Action, isconnect,
    get_legal_moves, get_legal_actions,
    ROW_SIZE, COLUMN_SIZE, WALL_NUM, BOARD_SIZE
)

# ---- 1. 常量和初始状态 ----
s = State()
s.reset()
assert s.turn == 1
assert s.get_pos(1) == (1, 9)
assert s.get_pos(2) == (17, 9)
assert s.get_wall_num(1) == WALL_NUM
assert s.get_wall_num(2) == WALL_NUM
print('[PASS] 常量和初始状态')

# ---- 2. 棋盘边界 ----
assert s.get_cell(0, 0) == True   # 外墙
assert s.get_cell(1, 9) == False  # 玩家1格子不是墙
print('[PASS] 棋盘边界')

# ---- 3. 深拷贝 ----
s2 = s.copy()
s2.set_pos(1, 3, 5)
assert s.get_pos(1) == (1, 9)    # 原状态不受影响
assert s2.get_pos(1) == (3, 5)   # 拷贝独立
print('[PASS] 深拷贝独立')

# ---- 4. 合法移动 ----
moves = get_legal_moves(s, 1)
assert len(moves) == 3
assert (3, 9) in moves  # 下
assert (1, 7) in moves  # 左
assert (1, 11) in moves # 右
print('[PASS] 合法移动枚举')

# ---- 5. 执行移动 ----
a = Action((3, 9), False, 0)  # 向下走一步
ok = a.apply(s)
assert ok
assert s.get_pos(1) == (3, 9)
assert s.turn == 2
print('[PASS] 执行移动')

# ---- 6. 注意: Action::apply 对移动不做校验（信任调用方已枚举合法动作） ----
#      只有放墙路径才有合法性检查（坐标奇偶、重叠、阻挡通路）

# ---- 7. 放墙（偶数坐标才是合法墙位置） ----
s.reset()
wall = Action((2, 2), True, 1)  # 水平墙
ok = wall.apply(s)
assert ok
assert s.get_wall_num(1) == WALL_NUM - 1
assert s.turn == 2
print('[PASS] 合法放墙')

# ---- 8. 非法放墙（奇数坐标） ----
s.reset()
bad_wall = Action((3, 3), True, 0)  # 墙必须放在偶数坐标 (odd,odd)=方格
ok = bad_wall.apply(s)
assert not ok
print('[PASS] 非法放墙被拒绝')

# ---- 9. 非法放墙（重叠） ----
s.reset()
# 先在 (2,2) 放一堵水平墙
wall = Action((2, 2), True, 1)  # 水平墙占 (2,2)-(2,3)
wall.apply(s)
# 在同一位置再放一堵
dup = Action((2, 2), True, 0)  # 垂直墙占 (2,2)-(3,2)，与水平墙重叠在(2,2)
ok = dup.apply(s)
assert not ok, '重复放墙应该被拒绝'
print('[PASS] 重叠放墙被拒绝')

# ---- 10. Action 字符串表示 ----
a1 = Action((3, 5), False, 0)
a2 = Action((2, 2), True, 0)
assert 'move' in repr(a1).lower()
assert 'wall' in repr(a2).lower()
print('[PASS] Action 字符串表示')

# ---- 11. 总合法动作数 ----
s.reset()
actions = get_legal_actions(s)
assert len(actions) == 147  # 3 移动 + 144 放墙
print('[PASS] 总合法动作数')

# ---- 12. isconnect 连通性 ----
s.reset()
# 玩家1 从 (1,9) 出发, 目标行 = 2*ROW_SIZE-1 = 17 (最后一行可走格子)
connected = isconnect(s, s.get_pos(1), 2*ROW_SIZE - 1)
assert connected, '初始棋盘玩家1应该有连通路径'
# 玩家2 从 (17,9) 出发, 目标行 = 1 (第一行可走格子)
connected = isconnect(s, s.get_pos(2), 1)
assert connected, '初始棋盘玩家2应该有连通路径'
print('[PASS] isconnect 连通性')

print()
print('全部 10 项测试通过!')
"
