#!/bin/bash
# =============================================================================
# 在 Linux 上编译 pybind11 扩展模块 (适用于 AutoDL / 腾讯云等)
#
# 用法:
#   cd Quoridor/rl
#   bash build_linux.sh
#
# 产出:
#   rl/quoridor_cpp.cpython-3xx-x86_64-linux-gnu.so
#
# Python 端使用:
#   from quoridor_cpp import State, Action, get_legal_actions
# =============================================================================
set -e

cd "$(dirname "$0")"

echo "=== 1/4 检查环境 ==="
command -v python3
python3 --version
echo ""

echo "=== 2/4 安装/更新 pybind11 ==="
python3 -m pip install --upgrade pip
python3 -m pip install pybind11

echo ""
echo "=== 3/4 编译 .so ==="
# cleanup
rm -f quoridor_cpp*.so
rm -rf build

python3 setup.py build_ext --inplace
echo ""

echo "=== 4/4 验证 ==="
ls -lh quoridor_cpp*.so

python3 -c "
import sys
sys.path.insert(0, '.')
from quoridor_cpp import (
    State, Action, isconnect,
    get_legal_moves, get_legal_actions,
    ROW_SIZE, COLUMN_SIZE, WALL_NUM, BOARD_SIZE
)

s = State()
s.reset()
assert s.turn == 1
assert s.get_pos(1) == (1, 9)
assert s.get_pos(2) == (17, 9)
actions = get_legal_actions(s)
print(f'合法动作数: {len(actions)} (预期 147)')
print('[PASS] 全部验证通过!')
"

echo ""
echo "编译成功: quoridor_cpp 模块已就绪"
