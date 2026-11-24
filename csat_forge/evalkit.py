"""LLM evaluation with fresh problems and misconception-tagged distractors.

Two ideas:
1. Contamination-free: every run draws new problems from the generators (seed printed
   for reproducibility), so there is no fixed test set to leak.
2. Misconception profile: a wrong choice is not just "wrong" — most tagged distractors
   come from a named mistake. We measure how strongly a model is *attracted* to each
   mistake, corrected for chance.

   For every wrongly-answered tagged item with k trap options among 4 wrong options,
   a uniformly random wrong pick would hit a trap with probability k/4. Summing these
   gives the expected number of trap picks; lift = observed / expected.
   lift > 1  → the model falls for that mistake more than chance (a real misconception)
   lift ~ 1  → its errors look random with respect to this mistake
"""
from __future__ import annotations
import json
import math
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict

from csat_forge import GENERATORS, generate

CIRC = "①②③④⑤"
_SYS = "당신은 수능 수학 풀이 전문가입니다."


def build_prompt(rec, cot=False) -> str:
    c = rec["completion"]
    ch = "\n".join(f"{CIRC[i]} {x}" for i, x in enumerate(c["choices"]))
    if cot:
        tail = "단계별로 풀이한 뒤, 마지막 줄에 '정답: N' (N은 보기 번호 1~5) 형식으로 답하세요."
    else:
        tail = "풀이 없이 정답 보기의 번호(1~5)만 '정답: N' 형식으로 답하세요."
    return f"{c['question']}\n\n{ch}\n\n{tail}"


def parse_choice(text: str, n: int = 5):
    if not text:
        return None
    ms = re.findall(r"정답\s*[:：]?\s*\**\s*([1-5])", text)
    if ms:
        v = int(ms[-1])
        return v if v <= n else None
    cm = [i + 1 for ch in text for i in [CIRC.find(ch)] if i >= 0]
    if cm:
        return cm[-1]
    ds = re.findall(r"(?<![\d.])([1-5])(?![\d.])", text)
    return int(ds[-1]) if ds else None


def correct_index(rec) -> int:
    return int(rec["completion"]["answer"].split("번")[0])


# ---------------- backends ----------------
class RandomBackend:
    name = "random"

    def __init__(self, seed=0):
        self.r = random.Random(seed)

    def complete(self, prompt):
        return f"정답: {self.r.randint(1, 5)}"


class OpenAICompat:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, LM Studio, Ollama...)."""

    def __init__(self, model, base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY",
                 max_tokens=1024):
        self.name, self.model, self.url = model, model, base_url.rstrip("/") + "/chat/completions"
        self.key, self.max_tokens = os.environ.get(key_env, ""), max_tokens

    def complete(self, prompt):
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                           "messages": [{"role": "system", "content": _SYS},
                                        {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["choices"][0]["message"]["content"]


class AnthropicBackend:
    def __init__(self, model, max_tokens=1024, key_env="ANTHROPIC_API_KEY"):
        self.name, self.model, self.max_tokens = model, model, max_tokens
        self.key = os.environ.get(key_env, "")

    def complete(self, prompt):
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "system": _SYS,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
            "Content-Type": "application/json", "x-api-key": self.key,
            "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return "".join(b.get("text", "") for b in json.load(r)["content"])


class MLXBackend:
    """Local Apple-silicon model through mlx-lm (pip install mlx-lm)."""

    def __init__(self, model, max_tokens=512):
        from mlx_lm import load
        self.name, self.max_tokens = model, max_tokens
        self.model, self.tok = load(model)

    def complete(self, prompt):
        from mlx_lm import generate as g
        msgs = [{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}]
        p = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return g(self.model, self.tok, prompt=p, max_tokens=self.max_tokens, verbose=False)


def make_backend(kind, model=None, base_url=None, max_tokens=None, key_env=None):
    if kind == "random":
        return RandomBackend()
    kw = {} if max_tokens is None else {"max_tokens": max_tokens}
    if kind == "mlx":
        return MLXBackend(model, **kw)
    if kind == "anthropic":
        return AnthropicBackend(model, **kw, **({"key_env": key_env} if key_env else {}))
    if kind == "openai":
        extra = {"base_url": base_url} if base_url else {}
        if key_env:
            extra["key_env"] = key_env
        return OpenAICompat(model, **extra, **kw)
    raise ValueError(kind)


# ---------------- run ----------------
def draw_items(n, seed, tagged_only=True, topics=None):
    """n fresh multiple-choice problems. tagged_only keeps problems with misconception tags."""
    rnd = random.Random(seed)
    names = topics or list(GENERATORS)
    out, tries = [], 0
    while len(out) < n and tries < n * 80:
        tries += 1
        nm = rnd.choice(names)
        random.seed(rnd.randrange(1 << 30))
        try:
            rec = generate(nm)
        except Exception:
            continue
        c = rec["completion"]
        if not c.get("choices"):
            continue
        if tagged_only and not c.get("trap_map"):
            continue
        out.append(rec)
    return out


def run_eval(backend, items, cot=False, progress=None):
    res = []
    for i, rec in enumerate(items):
        txt = backend.complete(build_prompt(rec, cot))
        pick = parse_choice(txt, len(rec["completion"]["choices"]))
        cidx = correct_index(rec)
        tm = rec["completion"].get("trap_map") or {}
        res.append({"generator": rec.get("generator"), "topic": rec["spec"].get("topic"),
                    "picked": pick, "correct": cidx, "ok": pick == cidx,
                    "trap_options": tm, "n_choices": len(rec["completion"]["choices"]),
                    "trap_picked": tm.get(str(pick)) if pick and pick != cidx else None,
                    "raw": (txt or "")[-600:]})
        if progress:
            progress(i + 1, len(items))
    return res


# ---------------- analysis ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def analyze(results):
    n = len(results)
    ok = sum(r["ok"] for r in results)
    unparsed = sum(r["picked"] is None for r in results)
    lo, hi = wilson(ok, n)
    per_topic = defaultdict(lambda: [0, 0])
    for r in results:
        per_topic[r["topic"]][0] += r["ok"]
        per_topic[r["topic"]][1] += 1
    # misconception lift
    exp, obs, ntr = Counter(), Counter(), Counter()
    wrong_tagged = 0
    for r in results:
        if r["ok"] or r["picked"] is None or not r["trap_options"]:
            continue
        wrong_tagged += 1
        nwrong = r["n_choices"] - 1
        for lab in r["trap_options"].values():
            exp[lab] += 1 / nwrong
        if r["trap_picked"]:
            obs[r["trap_picked"]] += 1
    total_trap = sum(obs.values())
    total_exp = sum(len(r["trap_options"]) / (r["n_choices"] - 1)
                    for r in results if not r["ok"] and r["picked"] and r["trap_options"])
    labels = [{"label": l, "picked": obs[l], "expected": round(exp[l], 2),
               "lift": round(obs[l] / exp[l], 2) if exp[l] else None} for l in exp]
    labels.sort(key=lambda d: (-(d["lift"] or 0), -d["expected"]))
    return {"n": n, "accuracy": ok / n if n else 0, "ci95": [lo, hi], "unparsed": unparsed,
            "per_topic": {t: v[0] / v[1] for t, v in sorted(per_topic.items())},
            "per_topic_n": {t: v[1] for t, v in sorted(per_topic.items())},
            "wrong_tagged": wrong_tagged, "trap_picks": total_trap,
            "trap_expected": round(total_exp, 2),
            "overall_lift": round(total_trap / total_exp, 2) if total_exp else None,
            "labels": labels}


def render_report(name, a, seed, cot, min_expected=3):
    L = [f"# csat-forge eval — {name}", "",
         f"- problems: **{a['n']}** (fresh, seed `{seed}`, {'CoT' if cot else 'direct answer'})",
         f"- accuracy: **{a['accuracy']*100:.1f}%** (95% CI {a['ci95'][0]*100:.1f}–{a['ci95'][1]*100:.1f}%)"
         f"  · chance = 20%", f"- unparsed answers: {a['unparsed']}", ""]
    if a["overall_lift"] is not None:
        L += [f"**Misconception attraction (overall)**: {a['trap_picks']} trap picks among "
              f"{a['wrong_tagged']} wrong tagged answers; chance would give {a['trap_expected']} "
              f"→ lift **{a['overall_lift']}×**", ""]
    rows = [d for d in a["labels"] if d["expected"] >= min_expected]
    if rows:
        L += ["| lift | picked / expected | misconception |", "|---:|:---:|---|"]
        for d in rows:
            L.append(f"| {d['lift']}× | {d['picked']} / {d['expected']} | {d['label']} |")
        L.append("")
    L += ["_lift > 1: the model is drawn to this mistake more than chance. Only labels with "
          f"expected ≥ {min_expected} are shown; small counts are noisy._"]
    return "\n".join(L)
