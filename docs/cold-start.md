# 冷启动：195 s → 26 s

同一个部署、同一张卡，只有"编译缓存在不在"这一个变量：

| 场景 | engine init | 其中 compile | /health 200 | 首个请求 |
|---|---|---|---|---|
| **缓存被清空** | **195.0 s** | 53.8 s | ~230 s | 首请求还要再等 ~75 s |
| **缓存完整** | **26.5 s** | 1.2 s | ~75 s | **0.77 s** |

差的 168 s 几乎全在两类工作：`torch.compile`（Dynamo + inductor + AOT 产物生成，53.8 s），
以及编译完成后的 profiling/warmup run —— 那一段会顺手 JIT 一堆还没编过的 kernel（~115 s）。

## 结论：别让它冷

编译缓存放在 `$HOME/.cache/vllm`（约 600 MB，其中 `torch_aot_compile` 占 599 MB）和 `$HOME/.triton`（17 MB）。
它在**重启后依然存在**，只在下面几种情况失效：

- 升级 vLLM / torch / CUDA 驱动
- **改任何影响编译的启动参数**：`--max-num-seqs`（决定 CUDA graph 桶数）、
  `--max-num-batched-tokens`（决定 compile range，日志里的 `Compiling a graph for compile range (1, 16384)`）、
  `--kv-cache-dtype`、attention backend、speculative 配置
- 手工清缓存 / 换模型目录

所以第一原则是**冻结启动参数**，第二原则是**留一份校验过的快照**。

## 快照（scripts/cache_snapshot.sh）

```bash
systemctl stop vllm-27b-hq        # 必须在停止状态做，否则抓到写了一半的产物
./scripts/cache_snapshot.sh save  # rsync 到 ~/cache-snapshots，校验文件数与字节数
./scripts/cache_snapshot.sh restore
```

`save` 会逐个目录对比"源文件数 / 副本文件数 / 总字节"，不一致直接报错退出（不会给你一个看着成功的坏快照）。
启动脚本在检测到 `~/.cache/vllm/torch_compile_cache` 缺失时会自动 `restore`，恢复本身约 2–3 s。

### 用 tar 会静默丢文件（实测）

最初的实现是 `tar --zstd`：**6567 个文件只打进去 5056 个，599 MB 只恢复出 480 MB，tar 退出码却是 0**。
结果恢复出一个"看着有 key 目录、实际缺产物"的缓存，引擎判定不可用 → 重新编译 **187 s**，
跟冷启动一样慢，还多花了恢复时间。所以现在的实现用 `rsync -a --delete` + 计数校验。

## 另外两个已验证的小杠杆

1. **权重预热进 page cache**：29.7 GiB 权重从 NVMe 读是 13.9 s，命中 page cache 后是 **5.4 s**。
   机器内存够的话，开机后 `cat` 一遍 safetensors（或 `vmtouch`）能省 8 s；内存紧张就别做。
2. **预热请求别走网关**：仓库里的 warmup 脚本原先把请求打到网关端口，结果 75 s 才返回，
   而且这些请求会参与网关的"模型切换"判定。改成直连模型端口（`PORT=18020`）+ 后端真实 served-name，
   首次请求从 75 s 降到 **0.77 s**。

## 不建议的两种做法

- **`--enforce-eager`**：确实不用编译（冷启也快），但短上下文解码慢 ~40%（实测 4K 33.2 ms → 57.7 ms）。
- **缩减 compile range**（把 `--max-num-batched-tokens` 从 16384 降到 8192）：能缩短 compile 时间，
  但会改变调度粒度，且**每次改都要重新付一次编译成本**，得重新测吞吐。
