# SFT 训练环境(uv 项目模式 · H20 · CUDA 12.4)

单人项目推荐用 uv 项目模式:一个 `pyproject.toml` 管所有,`uv.lock` 锁死版本,换机器 `uv sync` 一条命令复现。无需手动建 venv 或 activate。

## 首次搭建

把 `pyproject.toml` 放进项目目录,然后:

```bash
chmod +x setup.sh && ./setup.sh
```

`setup.sh` 里其实就一句核心命令 `uv sync`——它会自动下载 Python 3.11、创建 `.venv`、按 lock 装齐所有依赖。首次运行会生成 `uv.lock`。

预期验证输出:`cuda avail : True`,device 为 `NVIDIA H20-3e`,vram≈143。

## 日常三条命令

```bash
uv add <包名>              # 加依赖(自动更新 pyproject.toml + uv.lock + 安装)
uv sync                    # 对齐环境到 pyproject.toml/uv.lock
uv run python train.py     # 跑脚本(自动用项目环境,不用 activate)
```

## 关键配置说明

`pyproject.toml` 里这段决定了 torch 装 GPU 版:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
```

驱动显示 CUDA 12.4 → 用 cu124 轮子精确匹配。`explicit = true` 保证只有 torch 从这个源装,其余包走清华镜像。一次配好,以后不用碰。

## Flash Attention 2(可选)

H20 是 Hopper 架构,支持 FA2,长序列训练提速明显。它必须在 torch 装好之后单独装(构建时要 import torch):

```bash
uv pip install flash-attn --no-build-isolation
```

编译较久;失败可忽略,transformers 自动回退到 sdpa,功能不受影响。setup.sh 里已留注释行,取消注释即可。

## 模型缓存路径(可选)

ModelScope 权重较大,默认缓存 `~/.cache/modelscope`,建议指到数据盘:

```bash
export MODELSCOPE_CACHE=/data/modelscope   # 写入 ~/.bashrc 持久化
```

## 换机器 / 三个月后回来复现

`pyproject.toml` 和 `uv.lock` 提交进 git,新环境:

```bash
uv sync
```

即完全还原。

## 如果哪天要给别人一份 requirements.txt

```bash
uv export --format requirements-txt > requirements.txt
```

---

## 常见问题

| 问题 | 处理 |
|---|---|
| torch 下载慢 | 轮子 ~2.5GB,cu124 源在国外,耐心等;或临时挂代理 |
| flash-attn 编译报错 | 跳过用 sdpa;或先 `uv add ninja` 加速编译 |
| ModelScope 下载中断 | 支持断点续传,重跑即可 |
| 显存已被占(smi 显示 ~19GB) | 训练前 `nvidia-smi` 查残留进程,必要时 kill |
| 想临时进环境调试 | `uv run python` 或 `uv run ipython`(先 `uv add --dev ipython`) |
