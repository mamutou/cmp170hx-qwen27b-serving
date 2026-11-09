# 作为常驻服务跑（systemd + 网关按需唤醒）

把 27B 做成一个**独立 systemd 单元**，由网关按请求唤醒，和同机的其他大模型互斥。这套做法的价值是：
没人问 27B 时它不占显存，一旦有人用，网关停掉别的模型、把它拉起来，模型名对客户端保持不变。

## 单元

```ini
# /etc/systemd/system/vllm-27b-hq.service
[Unit]
Description=27B vLLM 0.32.0 + split-KV verify (Qwen3.8-27B W8A16+MTP, :18020)
After=network-online.target
Conflicts=other-llm.service            # 与本机其他大模型单元互斥
StartLimitIntervalSec=1800
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory=/opt/hyperqwen
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/local/bin/hq_prod.sh
Restart=on-failure
RestartSec=30
TimeoutStartSec=1800        # 首次 torch.compile + CUDA graph 捕获要 3–5 分钟
TimeoutStopSec=180
KillMode=mixed
StandardOutput=append:/var/log/vllm-27b-hq.log
StandardError=append:/var/log/vllm-27b-hq.log
```

`ExecStart` 就是本仓库的 [`scripts/serve.sh`](../scripts/serve.sh)（生产档位：`PORT=18020 SPEC_ATTN=1
NSEG=16 MIN_BATCH=0 CUDAGRAPH_MODE=PIECEWISE`）。**不要 `enable`**：它是按需唤醒的，
开机自启会抢走同机常驻模型的显存。

两个必须加的细节：

- **验收锁**：`ConditionPathExists=!<锁文件>`。做压测/烧机时先建锁，任何会话（包括网关）都拉不起来它。
- **网关侧**：把网关的“27B 槽位”指向这个单元与 18020，并保留旧单元做回退（`systemctl disable --now <新>` +
  `enable --now <旧>` 两条命令即回滚）。

## 网关要改两处，否则一定 404

1. **后端地址**：27B 从共享端口挪到 18020 后，网关探测“谁在线”和转发都必须用新地址，
   而其它模型（如 gptoss）仍留在原端口 —— 常见错误是把全局变量一改，另一个模型跟着指错。
2. **模型名改写**：上游 launcher 把 `--served-model-name` 硬编码成 `qwen3.8-27b`，
   客户端却可能用旧的公开名（`Qwen3.8-27B-…-MTP`）发请求，于是后端回
   `The model ... does not exist. Valid aliases: qwen3.8-27b`（404）。
   在网关转发前把 body 的 `model` 字段改写一次即可，**对外暴露的名字不变**。

```python
def rewrite_backend_model(body: bytes, profile: str | None) -> bytes:
    if profile != "qwen" or not body:
        return body
    try:
        d = json.loads(body.decode())
    except Exception:
        return body
    if isinstance(d, dict) and d.get("model") and d["model"] != "qwen3.8-27b":
        d["model"] = "qwen3.8-27b"
        return json.dumps(d, ensure_ascii=False).encode()
    return body
```

唤醒路径和常驻路径**两条都要改写** —— 只改唤醒那条，第一次请求成功、第二次 404。

## 三个必踩的网关 bug（都是把 27B 挪到独立端口后才暴露的）

1. **全局后端变量会连坐**：把"共享后端地址"改成 18020 后，其它模型（同机的常驻模型、gpt-oss 等）如果没有**显式**
   写自己的 `back`，就会一起去连 18020。表现是：那个模型明明在跑，网关却在错误的端口上等它 ready，
   一直等到 1200 s 超时，客户端拿到 503 —— 看起来像"目标模型没反应"。
2. **在线探测也只认一个端口**：探测函数若只探全局后端，另一个端口上的模型就永远"看不见"，
   每个请求都被判定为"需要切换"，于是白白停一次、起一次（1–2 分钟），并发下还会把正在加载的模型杀掉。
   正确做法：**按 profile 逐个探各自的后端**，先用精确 id 匹配。
3. **改投后仍按原 profile 转发**：防抖（cooldown）会把请求改投到当前 active 的模型，但如果转发目标
   还按"请求里的 profile"算，就会去连一个没起来的端口 → `ClientConnectorError` → 500。
   改投后必须用**实际生效的 profile** 取后端地址。

另外两条实测结论：

- **冷启时间决定防抖窗口**：27B 的 CUDA graph/compile 缓存热时 83 s 起来，冷缓存要 4–5 分钟。
  防抖窗口必须大于这个时间（本项目用 300 s），否则任何并发进来的另一个模型请求都会把正在加载的引擎杀掉，
  用户看到的就是"唤醒没反应"。代价是：切换后 300 s 内请求另一个模型，会被当前模型代答。
- **失败要快速失败**：`systemctl start` 之后除了轮询 HTTP，还要每轮查一次 `systemctl is-failed`，
  已失败就立刻抛错，别让客户端陪着你等满 20 分钟。

## 上线检查单

```bash
systemctl is-active vllm-27b-hq                 # 期望 active（由网关拉起后）
curl -s :18020/v1/models  | jq -r '.data[].id'  # qwen3.8-27b
curl -s :8080/v1/models   | jq -r '.data[]|[.id,.status]'   # 27B 应为 loaded
# 用旧别名经网关发一次，再发第二次（验证稳态路径也改写）
for i in 1 2; do curl -s :8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<旧别名>","messages":[{"role":"user","content":"回复OK"}],"max_tokens":16}' | head -c 120; echo; done
# 切到别的模型再切回来，确认互斥与唤醒都正常
```

## 品质与长稳

见 [benchmarks.md](benchmarks.md#品质抽测) 与 [benchmarks.md](benchmarks.md#长稳压测)。
