# 安装

前提：Ubuntu 22.04/24.04、x86-64、**Python 3.12**、约 40 GB 空闲磁盘（venv ≈ 8.4 GB + 源码缓存）、
一张已解锁的 170HX、能 `nvidia-smi` 看到 65536 MiB。

## 一步装完

```bash
HQ_DIR=/opt/hyperqwen ./scripts/install.sh
```

它做四件事，每件都可以单独重跑：

1. `git clone` 上游分支并 `checkout` 到 pin `b2e687e`（含 `patches/` 与 `opt-170hx/`）。
2. 建 venv，装 `vllm==0.32.0` 与 CUDA 13 编译器 pin。
3. `bash patches/apply.sh <site-packages>/vllm` —— 约 50 个补丁，**必须 p1 全中**（offset 正常，fuzz 会失败）。
4. 把 `opt-170hx/` 四个文件覆盖进 site-packages。

## 两个会卡住人的点

**flashinfer wheel 拉不下来。** `vllm==0.32.0` 精确 pin 了 `flashinfer-python==0.7.0.post1`，
它在 `flashinfer.ai/whl` 上只是一个指向 GitHub release 的 302，国内经常 15 s 超时导致整单失败。
在能访问的机器上取 wheel，再传过去：

```bash
# 能访问的机器
pip download flashinfer-python==0.7.0.post1 --no-deps -d wheels
# 目标机
pip install vllm==0.32.0 ... --find-links wheels
```

`flashinfer-cubin` 可以不要：launcher 已经 `VLLM_USE_FLASHINFER_SAMPLER=0`，用 MTP 时不会触发 FlashInfer 采样。

**launcher 硬编码 `--max-num-batched-tokens 2048`。** 投机解码会在此基础上再压
`max_num_scheduled_tokens`，长上下文预填明显变慢。`scripts/serve.sh` 用 `EXTRA_ARGS` 追加
`--max-num-batched-tokens 16384`，vLLM 会打一行 `Found duplicate keys` 然后取最后一个值（实测生效）。

## 装完自检

```bash
$HQ_DIR/venv/bin/python -c "import vllm; print(vllm.__version__)"          # 0.32.0
$HQ_DIR/venv/bin/python -c "import vllm.envs as e; print(e.VLLM_SPEC_ATTN_NSEG, e.VLLM_SPEC_ATTN_MIN_BATCH)"  # 0 4（默认；运行期覆盖）
ls $HQ_DIR/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/spec_decode_attn.py
nvidia-smi --query-gpu=memory.total,driver_version --format=csv
```

## 首次启动会慢

- `torch.compile` + CUDA graph 捕获约 3–5 分钟（缓存落在 `$HOME/.cache/vllm`，第二次起 1 分钟左右）。
- 首次遇到未捕获形状会 JIT 一个 Triton kernel（日志出现 `Triton kernel JIT compilation during inference:
  _spec_attn_partial`），随后消失。这不是错误。
