"""csat-forge CLI:  csat-forge gen | killer | list | diagnose"""
from __future__ import annotations
import argparse
import json
import random
import sys

from csat_forge import GENERATORS, generate, generate_meta, verify
from csat_forge.diagnosis import diagnose


def _show(rec, with_answer=True):
    c = rec["completion"]
    s = rec["spec"]
    print(f"[{s.get('topic')}] {s.get('score')}점 · {s.get('difficulty')} · {s.get('problem_type')}\n")
    print(c["question"], "\n")
    for i, ch in enumerate(c.get("choices") or [], 1):
        print(f"  {'①②③④⑤'[i-1]} {ch}")
    if with_answer:
        print("\n정답:", c["answer"])
        print("\n" + c["explanation"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="csat-forge", description="정답이 보장되는 수능 수학 문제 생성기")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="기본 유형 문제 생성")
    g.add_argument("--topic", help="생성기 이름 (csat-forge list 참고). 생략 시 무작위")
    g.add_argument("-n", type=int, default=1)
    g.add_argument("--seed", type=int)
    g.add_argument("--subjective", action="store_true", help="단답형(0~999)으로 변환")
    g.add_argument("--json", action="store_true", help="JSON Lines 출력")
    g.add_argument("--no-answer", action="store_true")

    k = sub.add_parser("killer", help="(가)(나)(다) 조건 합성형 킬러 문제 생성")
    k.add_argument("--domain")
    k.add_argument("-n", type=int, default=1)
    k.add_argument("--seed", type=int)
    k.add_argument("--json", action="store_true")
    k.add_argument("--no-answer", action="store_true")

    w = sub.add_parser("worksheet", help="인쇄용 문제지(HTML, 정답지 포함) 생성")
    w.add_argument("-n", type=int, default=10, help="기본 유형 문항 수")
    w.add_argument("--killer", type=int, default=0, help="킬러 문항 수")
    w.add_argument("--topics", help="쉼표로 구분한 생성기 이름. 생략 시 무작위")
    w.add_argument("--seed", type=int)
    w.add_argument("--title", default="수학 문제지")
    w.add_argument("--out", default="worksheet.html")

    sub.add_parser("list", help="생성기 목록")

    d = sub.add_parser("diagnose", help="학생이 고른 보기로 실수 유형 진단")
    d.add_argument("--topic", required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--chosen", type=int, required=True, help="고른 보기 번호(1~5)")

    a = ap.parse_args(argv)
    if a.cmd == "list":
        for name in GENERATORS:
            print(name)
        print(f"\n{len(GENERATORS)} generators", file=sys.stderr)
        return 0
    if a.cmd == "diagnose":
        rec = generate(a.topic, seed=a.seed)
        print(json.dumps(diagnose(rec, a.chosen), ensure_ascii=False, indent=2))
        return 0
    if a.seed is not None:
        random.seed(a.seed)
    if a.cmd == "worksheet":
        from csat_forge.worksheet import build_worksheet
        topics = a.topics.split(",") if a.topics else None
        recs = []
        for i in range(a.n):
            recs.append(generate(random.choice(topics) if topics else None))
        for _ in range(a.killer):
            r = generate_meta()
            if r:
                recs.append(r)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(build_worksheet(recs, a.title))
        print(f"{len(recs)}문항 → {a.out}")
        return 0
    for i in range(a.n):
        if a.cmd == "gen":
            rec = generate(a.topic, subjective=a.subjective)
        else:
            rec = generate_meta(domain=a.domain)
            if rec is None:
                print("합성 실패", file=sys.stderr)
                return 1
        if a.json:
            rec = {k: v for k, v in rec.items() if k != "_meta"}
            rec["verified"] = bool(verify(rec)) if a.cmd == "gen" else True
            print(json.dumps(rec, ensure_ascii=False))
        else:
            if i:
                print("\n" + "─" * 60 + "\n")
            _show(rec, not a.no_answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
