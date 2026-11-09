"""单流 / 并发解码探针（精确 token id，SSE 打点）。

用法: PORT=18020 MODEL=qwen3.8-27b python3 probe.py IN OUT CONC SEED
  WARM=1 先对每个 prompt 各打一次 max_tokens=1（写热前缀缓存），WARM=0 直接测。
输出一行 JSON: ttft_s / step_med_ms / chunks_per_stream / tok_per_chunk / batch_tok_s。

为什么不用 vLLM 自带的 bench：本栈在 256K 上下文下服务端可能既不发 SSE 的
`data: [DONE]`、也不关流，按 [DONE] 退出的客户端会永久挂住；这里按
`finish_reason` 退出，并在拿不到 usage 时用 OUT 兜底算 tokens/step。
"""
import json, os, random, statistics, sys, threading, time, urllib.request

PORT = int(os.environ.get("PORT", "18020"))
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
API_KEY = os.environ.get("API_KEY", "")
IN, OUT, CONC, SEED = (int(x) for x in sys.argv[1:5])
WARM = os.environ.get("WARM", "1") == "1"


def body(ids, n, stream):
    d = {"model": MODEL, "prompt": ids, "max_tokens": n, "temperature": 0.0,
         "ignore_eos": True, "stream": stream}
    if stream:
        d["stream_options"] = {"include_usage": True}
    return json.dumps(d).encode()


def post(ids, n, stream, timeout):
    hdr = {"Content-Type": "application/json"}
    if API_KEY:
        hdr["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=body(ids, n, stream), headers=hdr)
    return urllib.request.urlopen(req, timeout=timeout)


prompts = [[random.Random(SEED + i).randrange(1000, 200000) for _ in range(IN)]
           for i in range(CONC)]
if WARM:
    for ids in prompts:
        with post(ids, 1, False, 3600) as r:
            r.read()

res = [None] * CONC
barrier = threading.Barrier(CONC + 1)


def one(i):
    first = None; chunks = []; usage = None; err = None; t0 = None
    try:
        barrier.wait()
        t0 = time.time()
        with post(prompts[i], OUT, True, 3600) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                pay = line[6:]
                if pay == "[DONE]":
                    break
                d = json.loads(pay)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch:
                    now = time.time()
                    if first is None:
                        first = now
                    chunks.append(now)
                    if ch[0].get("finish_reason"):
                        break
    except Exception as e:
        err = repr(e)
    t1 = time.time()
    gaps = [(chunks[j] - chunks[j - 1]) * 1000 for j in range(1, len(chunks))]
    res[i] = dict(ttft_ms=(first - t0) * 1000 if first else float("nan"),
                  chunks=len(chunks),
                  gen=(usage or {}).get("completion_tokens") or OUT,
                  _c=chunks, _t0=t0, _t1=t1, err=err)


threads = [threading.Thread(target=one, args=(i,)) for i in range(CONC)]
for t in threads:
    t.start()
barrier.wait()
for t in threads:
    t.join()

ok = [r for r in res if r and not r["err"]]
if not ok:
    print(json.dumps({"in": IN, "out": OUT, "conc": CONC,
                      "errs": [r and r["err"] for r in res]}), flush=True)
    sys.exit(1)

gaps = [g for r in ok for g in
        [(c[i] - c[i - 1]) * 1000 for i in range(1, len(c))] for c in [r["_c"]]]
gen_tot = sum(r["gen"] for r in ok)
dec = (max(r["_c"][-1] for r in ok) - min(r["_c"][0] for r in ok)
       if all(r["_c"] for r in ok) else 0)
print(json.dumps(dict(
    in_=IN, out=OUT, conc=CONC, ok=len(ok),
    ttft_s=round(statistics.median([r["ttft_ms"] for r in ok]) / 1000, 1),
    step_med_ms=round(statistics.median(gaps), 2) if gaps else None,
    chunks_per_stream=round(statistics.median(r["chunks"] for r in ok), 1),
    tok_per_chunk=round(gen_tot / sum(r["chunks"] for r in ok), 2),
    batch_tok_s=round(gen_tot / dec, 1) if dec > 0 else None,
    errs=[r["err"] for r in res if r and r["err"]],
)), flush=True)
