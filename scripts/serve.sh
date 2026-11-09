#!/usr/bin/env bash
# 启动 Qwen3.8-27B（W8A16 + MTP）在 CMP 170HX 上，带 split-KV verify attention。
#   MODEL=/models/model ./scripts/serve.sh            # 真启动
#   MODEL=/models/model DRY=1 ./scripts/serve.sh      # 只打印将用的环境与命令，不启动
# 依赖：已按 docs/install.md 装好 HyperQwen(170hx 分支) 的 venv 与补丁覆盖文件。
set -euo pipefail

HQ_DIR=${HQ_DIR:-/opt/hyperqwen}
PORT=${PORT:-18020}
MODEL=${MODEL:?set MODEL to the checkpoint directory}

export SPEC=${SPEC:-mtp}
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export CTX=${CTX:-fast}
export MAX_LEN=${MAX_LEN:-262144}
export MAX_SEQS=${MAX_SEQS:-8}
export GPU_UTIL=${GPU_UTIL:-0.95}
export SPEC_ATTN=${SPEC_ATTN:-1}
export VLLM_SPEC_ATTN_NSEG=${VLLM_SPEC_ATTN_NSEG:-16}
export VLLM_SPEC_ATTN_MIN_BATCH=${VLLM_SPEC_ATTN_MIN_BATCH:-0}
export MTP_DRAFT_VOCAB=${MTP_DRAFT_VOCAB:-0}
export CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-PIECEWISE}
export VLLM_MARLIN_REPACK_STAGED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
export EXTRA_ARGS="--max-num-batched-tokens ${BATCH_TOKENS:-16384}"

if [ "${DRY:-0}" = "1" ]; then
  echo "HQ_DIR=$HQ_DIR MODEL=$MODEL PORT=$PORT"
  echo "SPEC=$SPEC k=$DRAFT_TOKENS CTX=$CTX MAX_LEN=$MAX_LEN MAX_SEQS=$MAX_SEQS GPU_UTIL=$GPU_UTIL"
  echo "SPEC_ATTN=$SPEC_ATTN NSEG=$VLLM_SPEC_ATTN_NSEG MIN_BATCH=$VLLM_SPEC_ATTN_MIN_BATCH CUDAGRAPH=$CUDAGRAPH_MODE"
  echo "EXTRA_ARGS=$EXTRA_ARGS"
  echo "exec: bash $HQ_DIR/single-user/start_qwen.sh"
  exit 0
fi

cd "$HQ_DIR"
exec bash single-user/start_qwen.sh
