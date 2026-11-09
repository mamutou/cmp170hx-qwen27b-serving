# CMP 170HX 上跑 Qwen3.8-27B：vLLM 0.32.0 + split-KV verify attention

单张 **NVIDIA CMP 170HX（GA100 / sm80，显存解锁到 64 GB）** 上，用 **不修改模型权重** 的方式把
27B 稠密模型（W8A16 + 自带 MTP 头）的长上下文解码加速 **2.3–4.8×**，32K 并发吞吐 **2×**。

> 这是一份**部署配方 + 实测数据**仓库：上游代码一行没改（HyperQwen 的 `opt-170hx/` 分支 + vLLM 0.32.0 补丁系列），
> 本仓库交付的是**能跑通的配置矩阵、逐项实测、失败路线和踩坑记录**。
> 不是 fork，也不包含任何模型权重。

## 结果一览（同一份 W8A16 权重，四个配置对打）

单流解码，`step` = 相邻 SSE chunk 间隔中位数（一次 verify 步），单位 ms：

| 上下文 | 现役 vLLM 0.27.1（FA2） | HQ 0.32.0，不开 split-KV | **HQ 0.32.0 + split-KV** | split-KV + 关 cudagraph |
|---|---|---|---|---|
| 4K | 48.8 ms / 73 tok/s | 35.5 / 112 | **33.2 / 121（+65%）** | 57.7 / 69 |
| 32K | 59.1 / 67 | 57.0 / 70 | **36.4 / 110（+65%）** | 57.9 / 70 |
| 128K | 134.5 / 18 | 132.3 / 31 | **47.5 / 87（+382%）** | 57.6 / 72 |
| 262K | 237.8 / 17.4 | 234.8 / 17.4 | **62.1 / 69（+296%）** | 63.4 / 71 |

32K 输入、8 条流、128 输出：聚合解码 **C4 107 → 220 tok/s（2.1×）**，**C8 128 → 218 tok/s（1.7×）**。
262K 冷预填 TTFT **335 s → 218 s**（782 → 1201 tok/s）。

硬件：[CMP 170HX 64 GB](docs/hardware.md) · 190 W 功率墙 · driver 610.43.02 · PCIe Gen2 x16 · 单卡。

## 为什么有效

decoder-only 模型一次 verify 步要读完整份权重（W8A16 ≈ 9.5 GB / token @ k≈3.8），本来是带宽活；
但 27B 有 16 层全注意力，**长上下文时 verify 的 attention 反而成为大头**：4K 时它贡献 0，
128K 时贡献约 86 ms/step（占 64%）。上游 FlashAttention-2 只有在 query 数 = 1 时才切分 KV，
而投机验证一次有 k+1 个 query（本配置 k=3 → 4 个），于是整段 KV 只按 head 并行。

`patches/spec-decode-attn.patch` 用 Triton 写了一个 split-KV verify 内核：每个
(request, kv-head, query-tile) 拿 `NSEG` 个 block，online-softmax 出 partial 再 combine。
在 170HX 上还有两个专属调优点：**每 block 163 KB 共享内存**（sm86 只有 99 KB）允许
`D=256` 时用 64-token 的 KV tile，且 `D>128` 需要 8 warps 才放得下 `[BLOCK_M, D]` 累加器。

## 快速开始

```bash
git clone https://github.com/mamutou/cmp170hx-qwen27b-serving && cd cmp170hx-qwen27b-serving

# 1) 装软件（不下载任何权重）
HQ_DIR=/opt/hyperqwen ./scripts/install.sh

# 2) 看一眼将要用什么参数启动（不启动）
HQ_DIR=/opt/hyperqwen MODEL=/models/Qwen3.8-27B-W8A16 DRY=1 ./scripts/serve.sh

# 3) 真启动
HQ_DIR=/opt/hyperqwen MODEL=/models/Qwen3.8-27B-W8A16 ./scripts/serve.sh

# 4) 打点（IN OUT CONC SEED）
PORT=18020 MODEL=qwen3.8-27b python3 scripts/probe.py 32768 128 1 9111
```

细节：[安装](docs/install.md) · [硬件前提](docs/hardware.md) · [测量方法与全部数字](docs/benchmarks.md) · [踩坑](docs/gotchas.md)。

## 配置矩阵（决定成败的就这几个）

| 变量 | 值 | 为什么 |
|---|---|---|
| `SPEC` / `DRAFT_TOKENS` | `mtp` / `3` | 用检查点自带的 MTP 头，不需要额外 drafter 权重 |
| `CTX` | `fast` | bf16 KV + FLASH_ATTN；split-KV 内核只吃 bf16 KV |
| `SPEC_ATTN` | `1` | 打开 split-KV verify |
| `VLLM_SPEC_ATTN_NSEG` | `16` | 固定段数；自适应在 C8 只切 2 段，74 个 SM 里空 44 个 |
| `VLLM_SPEC_ATTN_MIN_BATCH` | `0` | 0 = 单流也走 split-KV（长上下文单流收益全在这里） |
| `CUDAGRAPH_MODE` | `PIECEWISE` | 短上下文必须开；关掉比现役还慢 18% |
| `--max-num-batched-tokens` | `16384` | 上游 launcher 硬编码 2048，长上下文预填会慢；用 `EXTRA_ARGS` 覆盖 |
| `VLLM_MARLIN_REPACK_STAGED` | `1` | GA100 上 Marlin repack 会打爆 VMM 映射（Xid 31），必须走 staged |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | 这张卡上 `True` 是独立的 Xid 31 触发源 |

完整清单见 [`.env.example`](.env.example)。

## 权重：什么都没动

两侧加载的是**同一份目录**：第三方 Qwen3.8-27B 的 **W8A16（int8, group 128, compressed-tensors）+ 随包 MTP 头**，
28.6–29.7 GiB。没有重新量化，没有 `prepare/`，没有另训/另下 drafter，`MTP_DRAFT_VOCAB=0`。

这也解释了本仓库数字和上游头条数字的差距：上游那组 155 tok/s 是 **W4A16（约 15 GB）+ DFlash2 k=7**，
字节数是 W8A16 的 1/2.1。拆开就是：

```
9.47 GB/token ÷ 4.44 GB/token = 2.13×（量化档）
688 GB/s ÷ 482 GB/s            = 1.43×（引擎效率）
3.38 ÷ 3.02                    = 1.12×（接受长度）
                              ≈ 3.05×  ← 与实测 155 / 50.9 吻合
```

**不改权重时，可动的只有 1.43× 那一项**，本仓库把其中长上下文的部分吃到了 2.3–4.8×。

## 已知边界

- 只验证了 **单卡**、`k=3`、`MAX_SEQS=8`、≤262144 上下文；>8 并发、64K+ 的 C8 未测。
- 品质只做了冒烟（中文问答输出连贯、thinking 正常分离），**没跑 GSM8K / PPL**。
- 未做 ≥1 h 长稳压测。sm80 上有两起投机解码故障报告（上游 issue #72 / #98），
  上报方都是 driver 610.43.02 且手工打补丁的 wheel；本机同为 610.43.02，本轮约 40 分钟含 262K 长上下文 0 Xid，
  **但上生产前仍建议自己跑一轮并发长稳压测**。
- W8A16 在这张卡上已接近带宽上限（约 450 GB/s / 1355–1605 GB/s），再往上要么改权重位宽，要么换卡。
- INT8 张量核在这张卡上只有 FP16 的约 1/8（≈46 TOPS），**不要开 W8A8 / `INT8_ACT` 一类计算路径**。

## 隐私与安全

本仓库**不包含**：模型权重与 tokenizer、任何 API key / token、内网地址与主机名、本机部署路径、
生产服务单元名与定时任务、GPU UUID / VBIOS 序列号、日志原文。
所有脚本用 `/opt/hyperqwen`、`/models/…`、`<GPU_HOST>` 这类中性路径，复制即可改名。
打点脚本只发随机 token id，不回传任何数据。

## 来源与许可

- [HyperQwen](https://github.com/syv-ai/HyperQwen)（Apache-2.0）：vLLM 补丁系列、KVarN、launcher。
- [HyperQwen-CMP170hx @ feat/170hx-decode-tuning](https://github.com/aiultrayang/HyperQwen-CMP170hx/tree/feat/170hx-decode-tuning)
  （Apache-2.0，pin `b2e687e`）：本仓库使用的 `opt-170hx/` 四个覆盖文件与 170HX 调参。
- [vLLM](https://github.com/vllm-project/vllm) 0.32.0（Apache-2.0）。
- 本仓库自身的配置矩阵、测量脚本与数据：Apache-2.0，见 [LICENSE](LICENSE)。

> 上游没有测过 W8A16 + MTP k=3 这个组合；本仓库的全部数字是在单张 170HX 上、按
> [docs/benchmarks.md](docs/benchmarks.md) 的协议跑出来的，可复现。
