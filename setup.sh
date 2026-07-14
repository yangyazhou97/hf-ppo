#!/usr/bin/env bash
set -e

# 按 pyproject.toml + uv.lock 装齐一切(含 Python 3.11 本身与 .venv)
uv sync

# 可选:Flash Attention 2(H20/Hopper 支持,长序列提速)
# 必须在 torch 装好之后单独装,失败可忽略,回退 sdpa
# uv pip install flash-attn --no-build-isolation

# 验证
uv run python - <<'PY'
import torch, transformers, trl, modelscope
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
print("device      :", torch.cuda.get_device_name(0))
print("vram (GB)   :", round(torch.cuda.get_device_properties(0).total_memory / 1e9))
print("transformers:", transformers.__version__)
print("trl         :", trl.__version__)
PY

echo "✅ Environment ready"
