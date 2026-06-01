# AutoDL 云端训练部署指南

## 一、创建实例

1. 登录 [autodl.com](https://autodl.com) → 控制台 → 创建实例
2. **推荐镜像**：`PyTorch 2.x + Python 3.10(ubuntu22.04)`
3. **推荐机器**：
   - 快速实验：**RTX 3090 (24G)** ≈ ¥1.5/小时
   - 预算充足：**RTX 4090 (24G)** ≈ ¥2.5/小时
   - 最低配：**RTX 2080 Ti (11G)** ≈ ¥1.0/小时
4. **系统盘**：50GB 够用
5. **数据盘**：如果需要存多个 checkpoint，加 50GB（¥0.1/天）

> 选好后**关机再开机**才真正分配资源，开机后 SSH 连接。

## 二、上传代码

### 方式 A：git（推荐）
```bash
# 在 AutoDL 终端执行
git clone <你的仓库地址>
cd Quoridor
```

### 方式 B：scp 上传本地代码
```bash
# 在本地终端执行
tar czf Quoridor.tar.gz Quoridor/
scp Quoridor.tar.gz root@<autodl实例IP>:/root/autodl-tmp/
# 在 AutoDL 终端执行
cd /root/autodl-tmp/
tar xzf Quoridor.tar.gz
cd Quoridor/rl
```

### 方式 C：AutoDL 网盘
- 在网页端上传压缩包到「网盘」，然后在实例内 `cp /autodl-pub/<你的网盘路径> .`

## 三、安装依赖 & 编译 C++ 模块

```bash
cd /root/autodl-tmp/Quoridor/rl

# 安装依赖（镜像通常自带 torch，只需补 pybind11）
pip install -r requirements.txt

# 编译 C++ 扩展
bash build_linux.sh
```

编译成功后应看到 `quoridor_cpp.cpython-*-x86_64-linux-gnu.so`。

## 四、运行训练

```bash
# 从头训练
python train.py

# 从 checkpoint 恢复训练
python train.py --resume weights/checkpoints/epoch_010.pt

# 后台运行（避免 SSH 断开导致中断  →  推荐用 tmux）
tmux new -s train
python train.py  # 在 tmux 窗口中运行
# Ctrl+B, D 脱离会话，回来用: tmux attach -t train
```

## 五、取回训练结果

### Checkpoint / 权重文件
```bash
# 在 AutoDL 终端打包
cd /root/autodl-tmp/Quoridor/rl
tar czf weights_backup.tar.gz weights/

# 在本机下载
scp root@<实例IP>:/root/autodl-tmp/Quoridor/rl/weights_backup.tar.gz .
```

## 六、省钱技巧

| 操作 | 说明 |
|------|------|
| **不用就关机** | AutoDL 按小时计费，关机只收很低的基础费用 |
| **用 tmux 跑训练** | 断开 SSH 训练不中断 |
| **竞价实例（便宜 30%-50%）** | 在创建实例时选择「竞价模式」，每小时更便宜 |
| **数据放 /root/autodl-tmp/** | 这是数据盘，关机后数据保留；系统盘会丢 |

## 七、常见问题

**Q: 提示 ModuleNotFoundError: No module named 'quoridor_cpp'**
A: 确认已运行 `bash build_linux.sh`，且从 `rl/` 目录运行 Python：
```bash
cd Quoridor/rl
python train.py
```

**Q: 编译时报 pybind11 找不到**
A: 手动安装：`pip install pybind11`

**Q: GPU 显存不够**
A: 你的模型很小（32通道, 5 ResBlock），batch_size=256 只需约 500MB 显存，3090 完全够用。

**Q: 磁盘空间不足**
A: AutoDL 系统盘一般只有 20GB，建议把代码放到 `/root/autodl-tmp/`（数据盘），需要的话在控制台扩容数据盘。
