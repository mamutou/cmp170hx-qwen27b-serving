# 踩坑记录

按“会浪费你几个小时”的顺序排。

1. **测试窗口会被“模型按需唤醒”的网关抢走 GPU。**
   机器上若有按请求唤醒模型的网关（或 `Restart=always` 的服务），你的基准跑到一半会出现
   `stopping all vLLM backends` 然后显存被别的模型占满。窗口开始前先停掉这些单元与相关定时任务，
   并给它们加一个 `ConditionPathExists=!/tmp/gpu-test.lock` 之类的隔离锁；结束再撤锁恢复。

2. **`pkill -f <模式>` 写在 ssh 一行命令里会杀掉自己。**
   你这条远程命令行本身包含该模式，`pkill` 匹配到自己 → 连接 255 断开，后面的命令全没执行。
   用 `pgrep -af` 取 PID 再 `kill -TERM <pid>`，或把模式写成 `[p]attern` 并单独一次调用。

3. **`systemd-run --user --unit=X` 停止后单元仍在，重名报 `already loaded`。**
   `systemctl --user reset-failed X` 或换个单元名。

4. **SSE 在 256K 上下文可能既不发 `data: [DONE]` 也不关流。**
   按 `[DONE]` 退出的客户端会永久挂住，而服务端此时早已生成完（`/metrics` 里
   `generation_tokens_total` 不再增长）。`scripts/probe.py` 改成按 `finish_reason` 退出，
   并在拿不到 `usage` 时用 `OUT` 兜底算 tokens/step。

5. **`--max-num-batched-tokens` 有两条来源。**
   HyperQwen 的 launcher 硬编码 2048，`EXTRA_ARGS` 追加 16384 后 vLLM 打
   `Found duplicate keys` 并取**最后一个**值。不覆盖的话长上下文预填会明显慢。

6. **`max_model_len` 与投机 token 的账。**
   `MAX_LEN=262144` 时，可以安全请求的输入是 `262144 - (k+1)` = 262140；探针用 262000 留余量。
   预填 + 生成超过上限时 vLLM 直接 400，不会自动截断。

7. **`flashinfer-python` 的 wheel 走 GitHub release。**
   国内 15 s 超时会让整个 `pip install vllm==0.32.0` 失败。见 [install.md](install.md) 的
   `pip download` + `--find-links` 做法。`flashinfer-cubin` 可省。

8. **GA100 上 Marlin repack 与 `expandable_segments:True` 都会触发 Xid 31。**
   前者用 `VLLM_MARLIN_REPACK_STAGED=1`（staged 到一块只增不减的缓冲），后者直接关掉。
   两个都是**必需项，不是调优**。

9. **不要用 INT8 计算路径。** 这张卡的 INT8 张量核约 46 TOPS（FP16 的 1/8），
   `INT8_ACT` / W8A8 在这种卡上只会更慢；W8A16 是“反量化成 FP16 再算”，不受影响。

10. **期望值别按上游头条数字对齐。** 上游 155 tok/s 是 W4A16 + DFlash2 k=7；
    同一张卡上 W8A16 + MTP k=3 的物理上限就在 70 tok/s 量级（每 token 多读 2.1× 字节）。
