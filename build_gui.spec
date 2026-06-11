# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 — Quoridor GUI

用法:
    cd /d D:\GitHub\Quoridor
    pyinstaller build_gui.spec

产物:
    dist/Quoridor/          目录模式（启动快，调试方便）
    dist/Quoridor.exe       单文件模式（方便分发）

注意:
    1. 先装依赖: pip install pygame torch numpy pyinstaller
    2. 体积优化: 建议用 CPU-only PyTorch:
       pip install torch --index-url https://download.pytorch.org/whl/cpu
    3. 如需图标: 准备 quoridor.ico 放在项目根目录, 取消下面 Icon 行的注释
"""

import os

block_cipher = None

# 用户运行 pyinstaller 的目录即为项目根目录
ROOT = os.getcwd()


# ═══════════════════════════════════════════════════════════════════════
#  1. 收集 rl/ 中的二进制 (quoridor_cpp.pyd + 依赖 DLL)
# ═══════════════════════════════════════════════════════════════════════

binaries = []
rl_dir = os.path.join(ROOT, 'rl')
for f in os.listdir(rl_dir):
    if f.endswith('.pyd') or f.endswith('.dll'):
        binaries.append((os.path.join(rl_dir, f), 'rl'))


# ═══════════════════════════════════════════════════════════════════════
#  2. 收集权重文件 (递归)
# ═══════════════════════════════════════════════════════════════════════

datas = []
weights_root = os.path.join(rl_dir, 'weights')
if os.path.isdir(weights_root):
    for root, dirs, files in os.walk(weights_root):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.relpath(root, ROOT)       # 保留 rl/weights/... 结构
            datas.append((src, dst))


# ═══════════════════════════════════════════════════════════════════════
#  3. 分析依赖
# ═══════════════════════════════════════════════════════════════════════

a = Analysis(
    ['gui/gui.py'],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,

    hiddenimports=[
        'torch',
        'numpy',
        'pygame',
        'rl.encode',        # rl/ 内部用 import encode / from config import ...
        'rl.config',         # 打包后需要显式收集，否则裸名导入在 PYZ 里找不到
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # 排除不需要的 GUI/科学计算库（注意: 不要动 torch 相关模块）
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'scipy',
        'pandas',
    ],

    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# ═══════════════════════════════════════════════════════════════════════
#  4. 打包
# ═══════════════════════════════════════════════════════════════════════

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Quoridor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime*.dll'],
    runtime_tmpdir=None,
    console=False,               # 无控制台窗口（纯 GUI）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='quoridor.ico',       # ← 有图标文件后取消注释
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime*.dll'],
    name='Quoridor',
)
