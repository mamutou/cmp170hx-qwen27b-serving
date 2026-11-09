# 测量方法与全部数字

## 协议

- **探针**：`scripts/probe.py`，直接发**精确 token id 列表**（不经过 tokenizer，避免不同 chat template 影响预填长度），
  `ignore_eos=true`、`temperature=0`，SSE 流式打点。指标：
  - `step_med_ms`：相邻 chunk 间隔的**中位数**（一次 verify 步）；这是唯一能直接反映内核的指标。
  - `tok_per_chunk`：`completion_tokens ÷ chunk 数`，即每次 verify 被接受的平均 token 数（≈ 接受长度）。
  - `batch_tok_s`：并发下 `总生成 token ÷ 首末 chunk 时间窗`。
- **热身**：除 128K/262K 外，每个 prompt 先发一次 `max_tokens=1` 写热前缀缓存，
  否则冷预填会与解码交错，step 间隔没有意义（会把差异放大到 2 倍）。
- **样本**：每个点 3 次取中位；同配置重复运行的 step 波动 <0.5%（4K: 49.24 与 49.2 ms）。
- **口径**：所有数字都是**受控 bench**，不是生产日志分桶；单机单卡，其他负载全部停掉。

## 四臂

| 臂 | 引擎 | verify attention | cudagraph |
|---|---|---|---|
| base | vLLM 0.27.1（现役） | FA2 | NONE |
| A | HyperQwen 0.32.0 + 补丁 | FA2（`SPEC_ATTN=0`） | PIECEWISE |
| **B** | 同上 + `opt-170hx/` | split-KV，`NSEG=16 MIN_BATCH=0` | PIECEWISE |
| C | 同 B | split-KV | NONE |

公共参数：`MAX_LEN=262144 MAX_SEQS=8 GPU_UTIL=0.95 KV=bf16 BATCH_TOKENS=16384`，MTP `k=3`，prefix caching on。
权重：同一份 W8A16（int8 g128）+ MTP 头，28.6–29.7 GiB。

### 单流

| 上下文 | base | A | **B** | C |
|---|---|---|---|---|
| 4K | 48.8 ms / 73 tok/s | 35.5 / 112 | **33.2 / 121** | 57.7 / 69 |
| 32K | 59.1 / 67 | 57.0 / 70 | **36.4 / 110** | 57.9 / 70 |
| 128K | 134.5 / 18 | 132.3 / 31 | **47.5 / 87** | 57.6 / 72 |
| 262K | 237.8 / 17.4 | 234.8 / 17.4 | **62.1 / 69** | 63.4 / 71 |

### 并发（32K 输入 / 128 输出）

| | base | A | **B** | C |
|---|---|---|---|---|
| C4 聚合 | 107 tok/s | 119 | **220** | 174 |
| C8 聚合 | 128 tok/s | 135 | **218** | 194 |

### 预填

262K 冷预填：base 335.1 s（782 tok/s）→ B 218.2 s（1201 tok/s），**+54%**；A/C 与 B 相同。
预填不吃 split-KV（它走 FLASH_ATTN 预填路径），差异来自 0.32.0 与 `max-num-batched-tokens`。

## 归因

| 区间 | 主要来源 | 证据 |
|---|---|---|
| 4K（−32%） | 引擎 + PIECEWISE cudagraph | A 已经 −27%；B 只再 −6%；C 关掉 cudagraph 后比 base 还慢 18% |
| 32K（−38%） | split-KV | A ≈ base（−4%），B−A = −36% |
| 128K（−65%） | split-KV | A ≈ base（−2%），B−A = −64% |
| 262K（−74%） | split-KV | B−A = −74% |
| C8（1.7×） | 固定 16 段 | A 在 C8 只比 base 快 6% |

## 负结果（别重复）

| 试过 | 结果 | 机制 |
|---|---|---|
| 关 cudagraph（`cudagraph_mode=NONE`）求稳 | 4K 57.7 ms，比现役 48.8 慢 **18%** | MTP verify 每步的 launch/调度开销被放大；长上下文仍有 2.3× |
| 自适应 `NSEG=0`（按 batch 推段数） | C8 掉到 ~135 tok/s（固定 16 段是 218） | C8 时自适应算 2 段 → 74 个 SM 只用到 30 个 |
| 只用 0.32.0 不开 split-KV | 32K/128K 基本持平（−2~4%） | 补丁系列的收益不在这些形状上 |
| 换 W4A16 之外的省字节手段（KV 量化） | 未测 | split-KV 内核只吃 bf16 KV，先不混变量 |
| `--max-num-batched-tokens 2048`（launcher 默认） | 长预填变慢 | 投机解码会进一步压 `max_num_scheduled_tokens` |

## 天花板

单流每步要读一遍权重：W8A16 27B ≈ 28.6 GB，k≈3.8 → **7.5 GB/token**；69 tok/s 对应约
450 GB/s，而卡的 D2D 是 1355–1605 GB/s —— **还没到带宽墙**，剩下的是 attention 与发射开销。
再要成倍提升只有两条：把权重变窄（W4A16，实测字节 −2.1×）或换卡。前者会改权重，不在本仓库范围内。
