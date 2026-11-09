#!/usr/bin/env python3
"""27B 品质抽测：短问答 + 代码可执行性 + 长上下文检索 + 复读检测。

用法: PORT=18020 MODEL=qwen3.8-27b python3 quality.py
输出: 每项 PASS/FAIL 与总分，最后一行 JSON 汇总。
"""
import json, os, random, re, statistics, subprocess, sys, tempfile, urllib.request

PORT = int(os.environ.get("PORT", "18020"))
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
API = f"http://127.0.0.1:{PORT}/v1"


def chat(prompt, max_tokens=900, temperature=0.0):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"{API}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return "\n".join(x for x in (m.get("content"), m.get("reasoning"),
                                 m.get("reasoning_content")) if x)


def completion(prompt, max_tokens=64):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"{API}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["text"]


# ---- 1. 短问答（可判分，答案唯一） ----
QA = [
    ("一个数的 3 倍加 7 等于 43，这个数是多少？只回答数字。", "12"),
    ("17 个球放进 4 个盒子，每盒最多 5 个，至少需要几个盒子？只回答数字。", "4"),
    ("从 1 加到 100 等于多少？只回答数字。", "5050"),
    ("把 2024 分解质因数，写成 a*b*c 的升序形式。", "2*2*2*11*23"),
    ("二进制 101101 的十进制值是多少？只回答数字。", "45"),
    ("字符串 'abcdef' 反转后是什么？", "fedcba"),
    ("如果今天是周三，100 天后是周几？只回答星期。", "周五"),
    ("9 个硬币里有一个偏轻，用天平最少称几次能找出？只回答数字。", "2"),
    ("IPv4 地址最大能表示的整数是多少？（十进制）", "4294967295"),
    ("一个立方体有 12 条棱，切成 27 个小立方体后，有 3 面涂色的小方块有几个？只回答数字。", "8"),
    ("写出 1 到 20 中所有质数，用逗号分隔。", "2,3,5,7,11,13,17,19"),
    ("sin(pi/2) 等于多少？只回答数字。", "1"),
]

# ---- 2. 代码：生成 + 真跑 ----
CODE = [
    ("写一个 Python 函数 dedup_keep_order(xs)，保持顺序去重，返回 list。只输出代码块。",
     "dedup_keep_order",
     "assert dedup_keep_order([1,2,1,3,2]) == [1,2,3]\nassert dedup_keep_order([]) == []"),
    ("写一个 Python 函数 is_balanced(s)，判断括号 ()[]{} 是否配对。只输出代码块。",
     "is_balanced",
     "assert is_balanced('([{}])') is True\nassert is_balanced('(]') is False\nassert is_balanced('') is True"),
    ("写一个 Python 函数 fib(n)，返回第 n 个斐波那契数（fib(0)=0）。只输出代码块。",
     "fib",
     "assert [fib(i) for i in range(8)] == [0,1,1,2,3,5,8,13]"),
]

# ---- 3. 长上下文检索 ----
NEEDLE = "ZQ7X-4429-KM"
FILLER = ("运维手册第%d节：检查电源冗余、风扇转速与温度曲线，记录日志编号并归档，"
          "确保下一次巡检可以复现同一批观测值。")

def needle_test(ctx_tokens):
    n_sections = max(4, int(ctx_tokens / 60))
    depth = n_sections // 3
    parts = []
    for i in range(n_sections):
        parts.append(FILLER % i + "\n")
        if i == depth:
            parts.append("第%d节：本次巡检的一次性校验码是 %s，请原样记住。\n" % (i, NEEDLE))
    parts.append("\n问题：文中出现的一次性校验码是什么？只回答校验码本身。\n")
    return completion("".join(parts), max_tokens=512)


results = []


def rec(name, ok, detail=""):
    results.append({"item": name, "pass": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))


print("== 引擎 ==", MODEL, "port", PORT)
def _norm(t):
    return (t.replace(" ", "").replace("×", "*").replace("·", "*").replace("^3", "**3")
             .replace("³", "**3").replace("^", "**"))


for q, want in QA:
    try:
        ans = chat(q, max_tokens=1200)
        n = _norm(ans)
        ok = want.replace(" ", "") in n or _norm(want) in n
        if q.startswith("把 2024"):
            ok = ok or ("2**3" in n and "11" in n and "23" in n)
        rec("QA: " + q[:26], ok, "" if ok else ans[-160:].replace("\n", " "))
    except Exception as e:
        rec("QA: " + q[:26], False, repr(e))

for prompt, fname, tests in CODE:
    try:
        out = chat(prompt, max_tokens=1200)
        m = re.search(r"```(?:python)?\s*(.*?)```", out, re.S)
        code = m.group(1) if m else out
        if fname not in code:
            rec("CODE: " + fname, False, "函数未出现")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n\n" + tests + "\nprint('OK')\n")
            path = f.name
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=60)
        rec("CODE: " + fname, p.returncode == 0 and "OK" in p.stdout,
            (p.stdout + p.stderr)[-160:].replace("\n", " "))
    except Exception as e:
        rec("CODE: " + fname, False, repr(e))

for ctx in (32768, 65536):
    try:
        got = needle_test(ctx)
        rec("NEEDLE %dK" % (ctx // 1024), NEEDLE in got, got[:80].replace("\n", " "))
    except Exception as e:
        rec("NEEDLE %dK" % (ctx // 1024), False, repr(e))

# ---- 4. 复读检测：长生成里的 20-gram 重复率 ----
try:
    out = chat("用中文写一段约 400 字的说明，解释为什么显存带宽决定大模型解码速度，"
               "要求行文连贯、不要重复句子。", max_tokens=900, temperature=0.6)
    words = re.findall(r"[\u4e00-\u9fff]{1}", out)
    grams = ["".join(words[i:i + 20]) for i in range(0, max(0, len(words) - 20))]
    dup = 1 - (len(set(grams)) / len(grams)) if grams else 1.0
    rec("REPEAT", dup < 0.10, "20-gram 重复率 %.1f%% (字符 %d)" % (dup * 100, len(words)))
except Exception as e:
    rec("REPEAT", False, repr(e))

npass = sum(r["pass"] for r in results)
print("\n== 汇总: %d/%d ==" % (npass, len(results)))
print(json.dumps({"model": MODEL, "port": PORT, "passed": npass, "total": len(results),
                  "items": results}, ensure_ascii=False))
