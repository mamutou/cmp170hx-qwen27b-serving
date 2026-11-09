# 硬件前提：CMP 170HX 为什么能吃下 27B + 256K

| 项 | 实测值 | 对部署的意义 |
|---|---|---|
| GPU | NVIDIA CMP 170HX（GA100, sm_80） | 同 die 于 A100，binned 版；**没有 FP8/FP4 张量核** |
| 显存 | 65536 MiB（社区解锁，出厂 8 GB） | 27B W8A16(≈29 GiB) + 262144 ctx 的 bf16 KV 刚好装下 |
| SM | 74（解锁 v0.5 后；`multiProcessorCount`） | 每 SM 64 个 FP32 核 = 4736 核 |
| 共享内存 | 163 KB / block（sm86 是 99 KB） | `D=256` 时能放 64-token KV tile，这是长上下文加速的物理来源 |
| 带宽 | D2D 1355–1605 GB/s（按批次 strap 档） | 解码仍带宽受限，但 27B 在 32K 以上主要受 attention 限制 |
| PCIe | Gen2 x16（H2D ≈ 6.8 GB/s） | 只影响加载与卸载；decode 全在卡内 |
| 功率 | 190 W 墙（默认 250 W，被动散热建议 190 W） | 推理负载下 SM 已贴本卡上限，加功率换不到速度 |
| 驱动 | 610.43.02（解锁补丁模块） | 换内核/换驱动都要重编补丁模块，见下 |

## 解锁与驱动

CMP 170HX 出厂 8 GB / 70 SM，靠社区驱动补丁在 SEC2 引导期重写配置：
[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)（v0.5 起含 SM 重配置与 ECC）。
关键约束：

- 补丁内核模块**不在 DKMS 里**，换内核必须重跑 `install.sh` 再冷启动，否则 `nvidia-smi` 掉回 8 GB。
- 必须 `apt-mark hold` 全部 `nvidia-*` 包，否则 `unattended-upgrades` 会用官方驱动换掉补丁模块。
- 解锁后 **INT8 张量核仍是 1/8 速率**（≈46 TOPS）：本仓库只用 W8A16（反量化到 FP16 计算），
  不要开 W8A8、`INT8_ACT` 这类真 int8 GEMM 路径。

## 几个容易误判的物理事实

- **80 GB 不存在**：活 FBPA 由熔丝定死 16 个、档位已顶到 4096 MiB，64 GB 就是这张卡的物理上限。
- **PCIe 只有 Gen2**：Gen3/Gen4 在熔丝层被烧死，强行改 PL 层会让卡在枚举阶段掉总线。
- **GPC 时钟偏置可以超**（NVML `nvmlDeviceSetGpcClkVfOffset`），实测 +150 MHz 预填 +5~6%、
  解码在噪声内，+250 会直接 Xid 13；不值得为 6% 承担静默损坏风险。

## 若换卡

同价位下：A100 40 GB 带宽相当（1555 vs 1605 GB/s）、算力约 2×、容量 40 GB 装不下本配置；
A100 80 GB 是全面升级但价格另说。本仓库的配方与卡无关的部分（补丁系列、探针、配置矩阵）
在 sm80 上通用；`opt-170hx/` 的 tile 调参只对 163 KB smem 的 GA100 成立。
