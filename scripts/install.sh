#!/usr/bin/env bash
# 装 vLLM 0.32.0 + HyperQwen(170hx 分支) 补丁 + 170hx 覆盖文件。
#   HQ_DIR=/opt/hyperqwen PY=python3.12 ./scripts/install.sh
# 说明：本脚本只装软件，不下载任何权重。
set -euo pipefail

HQ_DIR=${HQ_DIR:-/opt/hyperqwen}
PY=${PY:-python3.12}
UPSTREAM=${UPSTREAM:-https://github.com/aiultrayang/HyperQwen-CMP170hx.git}
BRANCH=${BRANCH:-feat/170hx-decode-tuning}
PIN=${PIN:-b2e687e3a58f2b30d623c49905f41e8113e659d8}

# 1) 源码（含 patches/ 与 opt-170hx/）
if [ ! -d "$HQ_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$UPSTREAM" "$HQ_DIR"
  git -C "$HQ_DIR" checkout "$PIN"
fi

# 2) venv + vLLM 0.32.0
#    国内建议加 -i https://pypi.tuna.tsinghua.edu.cn/simple
#    flashinfer-python==0.7.0.post1 的 wheel 走 GitHub release，国内常 15s 超时：
#    先在本机 `pip download flashinfer-python==0.7.0.post1 --no-deps -d wheels`
#    传过去，再用 --find-links wheels 安装。
[ -d "$HQ_DIR/venv" ] || "$PY" -m venv "$HQ_DIR/venv"
"$HQ_DIR/venv/bin/pip" install -U pip
"$HQ_DIR/venv/bin/pip" install vllm==0.32.0 huggingface_hub hf_transfer ninja pandas \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88 "${@}"

# 3) 补丁系列（约 50 个，cut against vLLM 0.32.0）
SITE="$HQ_DIR/venv/lib/python3.12/site-packages/vllm"
bash "$HQ_DIR/patches/apply.sh" "$SITE"

# 4) 170hx 覆盖文件：4 个文件替换进 site-packages
#    envs.py                      → 注册 VLLM_SPEC_ATTN_NSEG / _MIN_BATCH
#    spec_decode_attn.py          → split-KV verify kernel（新文件）
#    flash_attn.py, triton_attn.py→ MIN_BATCH 闸门
for f in envs.py spec_decode_attn.py flash_attn.py triton_attn.py; do
  [ -f "$SITE/$f.pre170hx" ] || true
done
cp -f "$HQ_DIR/opt-170hx/envs.py"                  "$SITE/envs.py"
cp -f "$HQ_DIR/opt-170hx/spec_decode_attn.py"      "$SITE/v1/attention/ops/spec_decode_attn.py"
cp -f "$HQ_DIR/opt-170hx/flash_attn.py"            "$SITE/v1/attention/backends/flash_attn.py"
cp -f "$HQ_DIR/opt-170hx/triton_attn.py"           "$SITE/v1/attention/backends/triton_attn.py"

"$HQ_DIR/venv/bin/python" - <<'EOF'
import vllm, vllm.envs as e
print("vllm", vllm.__version__, "| nseg", e.VLLM_SPEC_ATTN_NSEG, "| min_batch", e.VLLM_SPEC_ATTN_MIN_BATCH)
EOF
echo "OK: $HQ_DIR"
