#!/usr/bin/env python3
"""27B 长稳压测：C 路并发流持续打满 N 分钟，逐轮记录吞吐/延迟/显存温度，失败即记。

用法: PORT=18020 MODEL=qwen3.8-27b MINUTES=60 CONC=4 IN=8192 OUT=128 python3 soak.py
输出: JSONL 写到 $OUT（默认 ~/soak_27b.jsonl），每轮一行；结束打印汇总 JSON。
"""
import json, os, random, statistics, threading, time, urllib.request, subprocess

PORT = int(os.environ.get("PORT", "18020"))
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
MINUTES = float(os.environ.get("MINUTES", "60"))
CONC = int(os.environ.get("CONC", "4"))
IN = int(os.environ.get("IN", "8192"))
OUT = int(os.environ.get("OUT", "128"))
OUT_PATH = os.path.expanduser(os.environ.get("SOAK_OUT", "~/soak_27b.jsonl"))
DEAD_AFTER = 3


def post(ids, timeout=1800):
    body = json.dumps({"model": MODEL, "prompt": ids, "max_tokens": OUT,
                       "temperature": 0.0, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def one(prompt, slot):
    t0 = time.time(); first = None; chunks = []; gen = None; err = None
    try:
        with post(prompt) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                pay = line[6:]
                if pay == "[DONE]":
                    break
                d = json.loads(pay)
                if d.get("usage"):
                    gen = d["usage"]["completion_tokens"]
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
    res[slot] = dict(ttft=(first - t0) if first else None, chunks=len(chunks),
                     gen=gen or OUT, err=err, t0=t0, _c=chunks, _f=first,
                     end=chunks[-1] if chunks else None)


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                            "temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        u, m, t, p = [x.strip() for x in o.split(",")]
        return {"util": int(u), "mem": int(m), "temp": int(t), "power": int(float(p))}
    except Exception:
        return {}


f = open(OUT_PATH, "a", buffering=1)
t_start = time.time()
deadline = t_start + MINUTES * 60
it = 0
dead = 0
fails = 0
print(f"soak start {MINUTES}min conc={CONC} in={IN} out={OUT} -> {OUT_PATH}", flush=True)

while time.time() < deadline:
    it += 1
    res = [None] * CONC
    prompts = [[random.Random(1_000_000 + it * 100 + i).randrange(1000, 200000)
                for _ in range(IN)] for i in range(CONC)]
    th = [threading.Thread(target=one, args=(prompts[i], i)) for i in range(CONC)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    ok = [r for r in res if r and not r["err"]]
    errs = [r["err"] for r in res if r and r["err"]]
    if errs:
        fails += len(errs)
    if not ok:
        dead += 1
        rec = {"it": it, "ts": round(time.time() - t_start, 1), "ok": 0, "errs": errs}
        f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if dead >= DEAD_AFTER:
            print("ABORT: server not answering %d rounds in a row" % dead, flush=True)
            break
        time.sleep(5)
        continue
    dead = 0
    gaps = [c[i] - c[i - 1] for r in ok for c in [r["_c"]] for i in range(1, len(c))]
    gen_tot = sum(r["gen"] for r in ok)
    dec = (max(r["end"] for r in ok) - min(r["_f"] for r in ok)
           if all(r["end"] and r["_f"] for r in ok) else 0)
    ttft_med = statistics.median([r["ttft"] for r in ok if r["ttft"]])
    rec = {"it": it, "ts": round(time.time() - t_start, 1), "ok": len(ok),
           "ttft_med": round(ttft_med, 2),
           "prefill_tok_s": round(IN * len(ok) / ttft_med, 0) if ttft_med else None,
           "step_med_ms": round(statistics.median(gaps) * 1000, 1) if gaps else None,
           "batch_tok_s": round(gen_tot / dec, 1) if dec > 0 else None,
           "errs": errs}
    if it % 5 == 0:
        rec["gpu"] = gpu()
    f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

f.close()
rows = [json.loads(l) for l in open(OUT_PATH) if l.strip()]
rows_t = [r for r in rows if r.get("batch_tok_s")]
if rows_t:
    tps = [r["batch_tok_s"] for r in rows_t]
    n = max(1, len(tps) // 5)
    summary = {
        "minutes": round((time.time() - t_start) / 60, 1), "rounds": len(rows),
        "rounds_ok": len(rows_t), "request_errors": fails,
        "tok_s_median": round(statistics.median(tps), 1),
        "tok_s_p10": round(sorted(tps)[len(tps) // 10], 1),
        "tok_s_first20pct": round(statistics.median(tps[:n]), 1),
        "tok_s_last20pct": round(statistics.median(tps[-n:]), 1),
        "step_med_ms_median": round(statistics.median([r["step_med_ms"] for r in rows_t if r["step_med_ms"]]), 1),
        "temp_max": max([r["gpu"]["temp"] for r in rows if r.get("gpu")] or [0]),
        "mem_max": max([r["gpu"]["mem"] for r in rows if r.get("gpu")] or [0]),
        "power_max": max([r["gpu"]["power"] for r in rows if r.get("gpu")] or [0]),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
