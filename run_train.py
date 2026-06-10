# =============================================================================
# 训练启动脚本 — run_train.py
#
# 放在项目根目录，负责:
#   1. 将 rl/ 加入 sys.path (解决 rl/ 内部模块间的绝对导入问题)
#   2. 确保 quoridor_cpp.pyd 可被导入
#   3. 启动训练主循环
#
# 用法:
#   venv\Scripts\python.exe run_train.py              # 从头训练
#   venv\Scripts\python.exe run_train.py --resume xxx  # 从 checkpoint 恢复
#   venv\Scripts\python.exe run_train.py --export-only xxx  # 只导出权重
# =============================================================================
import sys
import os

# 将项目根目录加入 sys.path (保证 quoridor_cpp.pyd 能被找到)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 将 rl/ 加入 sys.path (解决 rl/ 内部模块之间的 "from config import ..." 等导入)
RL_DIR = os.path.join(ROOT, "rl")
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

# 将命令行参数转发给 train.main()
if __name__ == "__main__":
    # 重要: 把 argv[0] 替换为 train.py 的路径，让 argparse 正常工作
    sys.argv[0] = os.path.join(RL_DIR, "train.py")
    from train import main
    main()
