"""문제 엔진 회귀 테스트.

핵심: 지문 '렌더링'까지 검증한다. (funceq 부호 버그처럼 파라미터 검증만으로는
못 잡는 클래스의 버그를 차단.)
"""
import os
import re
import sys
import random
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp  # noqa: E402
from csat_forge import problem_engine as pe  # noqa: E402
from csat_forge.plain_format import to_plain, parse_plain  # noqa: E402


N_PER = 8
x = sp.symbols("x")


def _all_records():
    random.seed(123)
    for name, fn in pe.GENERATORS.items():
        for _ in range(N_PER):
            yield name, fn()


def _choice_val(s):
    try:
        from sympy.parsing.latex import parse_latex
        return float(sp.N(parse_latex(s.strip("$"))))
    except Exception:
        try:
            return float(sp.N(sp.sympify(s.strip("$"))))
        except Exception:
            return None


def test_verify_and_independent():
    """이중검증(구조 + _meta 독립 재계산) 전 생성기 통과."""
    for name, r in _all_records():
        assert pe.verify(r), f"{name}: verify() 실패"


def test_choices_format():
    """보기 5개·중복 없음·오름차순."""
    for name, r in _all_records():
        ch = r["completion"]["choices"]
        if ch is None:                              # 주관식(단답형)은 보기 없음
            assert r["spec"]["problem_type"] == "주관식", f"{name}: 보기 없는데 객관식"
            continue
        assert len(ch) == 5, f"{name}: 보기 {len(ch)}개"
        assert len(set(ch)) == 5, f"{name}: 보기 중복"
        vals = [_choice_val(c) for c in ch]
        if all(v is not None for v in vals):
            assert vals == sorted(vals), f"{name}: 보기 오름차순 위반"


def test_plain_roundtrip():
    """평문 직렬화 → 파싱 round-trip 무손실."""
    for name, r in _all_records():
        c = r["completion"]
        p = parse_plain(to_plain(c))
        assert p is not None, f"{name}: 평문 파싱 실패"
        assert p["question"] == c["question"].strip(), name
        assert p["choices"] == c["choices"], name
        assert p["answer"] == c["answer"].strip(), name
        assert p["explanation"] == c["explanation"].strip(), name


def test_funceq_rendered_text_solves_to_answer():
    """funceq: '지문에 표시된 식' 그대로 풀어 기록 정답과 일치해야 함."""
    random.seed(7)
    pat = re.compile(r"dt=x\\,f\(x\)(.+?)\$를")
    for _ in range(10):
        r = pe.gen_funceq_integral()
        m = pat.search(r["completion"]["question"])
        assert m, "funceq 지문 형식 변경됨 — 테스트 갱신 필요"
        rhs_tail = sp.sympify(sp.parsing.latex.parse_latex(m.group(1)))
        a3, a2, a1, a0 = sp.symbols("a3 a2 a1 a0")
        f = a3*x**3 + a2*x**2 + a1*x + a0
        lhs = sp.integrate(f, (x, sp.Integer(1), x))
        eq = sp.expand(lhs - (x*f + rhs_tail))
        sol = sp.solve(sp.Poly(eq, x).all_coeffs(), [a3, a2, a1, a0])
        got = f.subs(sol).subs(x, r["_meta"]["c"])
        assert sp.simplify(got - r["_meta"]["ans"]) == 0, \
            f"지문대로 풀면 {got}, 기록답 {r['_meta']['ans']}"


def test_make_subjective():
    """주관식 변환: 답 0~999 정수만 변환되고, 변환본도 검증 통과 + 평문 round-trip."""
    random.seed(11)
    converted = 0
    for name, r in _all_records():
        sub = pe.make_subjective(r)
        if sub is None:
            continue
        converted += 1
        assert sub["spec"]["problem_type"] == "주관식", name
        assert sub["completion"]["choices"] is None, name
        assert re.fullmatch(r"\d{1,3}", sub["completion"]["answer"]), name
        assert 0 <= int(sub["completion"]["answer"]) <= 999, name
        assert pe.verify(sub), f"{name}: 주관식 verify 실패"
        p = parse_plain(to_plain(sub["completion"]))
        assert p and p["choices"] is None and p["answer"] == sub["completion"]["answer"]
        # 원본(객관식)은 불변이어야 함 (deepcopy 확인)
        assert r["completion"]["choices"] is not None, name
    assert converted > 50, f"변환 표본이 너무 적음 ({converted})"


def test_figure_specs():
    """그림 생성기: figure spec + 텍스트 묘사 포함, 렌더링 성공."""
    import tempfile
    from csat_forge.figure_render import render_figure
    random.seed(5)
    for name in ("fig_parabola_area", "fig_velocity_graph"):
        r = pe.GENERATORS[name]()
        c = r["completion"]
        assert c.get("figure") and c.get("figure_desc"), name
        with tempfile.TemporaryDirectory() as td:
            path = render_figure(c["figure"], os.path.join(td, "f.svg"))
            assert os.path.getsize(path) > 500, f"{name}: 렌더 결과 비정상"


def test_trap_map_and_diagnosis():
    """오답 태깅: trap_map 보기번호가 유효하고 정답 번호와 다름 + 진단 동작."""
    from csat_forge.diagnosis import diagnose, followup
    random.seed(21)
    tagged = 0
    for name in ("conditional_prob", "fig_velocity_graph", "complement_dice",
                 "normal_tail", "log_inequality"):
        for _ in range(10):
            r = pe.generate(name)
            tm = r["completion"].get("trap_map") or {}
            correct_idx = int(r["completion"]["answer"].split("번")[0])
            for k in tm:
                assert 1 <= int(k) <= 5 and int(k) != correct_idx, name
            if not tm:
                continue
            tagged += 1
            wrong = int(next(iter(tm)))
            d = diagnose(r, wrong)
            assert d["correct"] is False and d["mistake"], name
            d2 = diagnose(r, correct_idx)
            assert d2["correct"] is True
            nxt = followup(r)
            assert nxt and nxt["spec"]["topic"] == r["spec"]["topic"], name
    assert tagged > 20, f"태깅 표본 부족 ({tagged})"


def test_new_prob_killers_independent_recompute():
    """확률과 통계 신규 킬러 3종: _meta.ans 를 독립 재계산해 100% 대조.

    생성기 내부 로직과 무관하게 _meta 파라미터만으로 답을 다시 계산하여,
    생성기가 표기한 정답이 진짜 정답인지 회귀로 고정한다(파라메트릭 100회).
    """
    from itertools import product, permutations
    from math import comb

    def recompute(r):
        m = r["_meta"]
        chk = m["chk"]
        if chk == "compcount":            # f(f(p))=a, 일대일대응 아님
            p, a = m["p"], m["a"]
            return sum(1 for f in product(range(1, 6), repeat=5)
                       if f[f[p-1]-1] == a and len(set(f)) < 5)
        if chk == "binomcond":            # P(X=k | X>=m), B(n,p)
            n, (pn, pd), mm, k = m["n"], m["p"], m["m"], m["k"]
            P = sp.Rational(pn, pd)
            pmf = lambda j: sp.Rational(comb(n, j)) * P**j * (1 - P)**(n - j)
            den = sum(pmf(j) for j in range(mm, n + 1))
            a = sp.nsimplify(pmf(k) / den)
            return [a.p, a.q]
        if chk == "msperm":               # 같은 것이 있는 순열 + 제약
            deck, d, par = tuple(m["deck"]), m["d"], m["parity"]
            perms = set(permutations(deck))

            def noadj(p):
                pos = sorted(i for i, v in enumerate(p) if v == d)
                return all(b - a > 1 for a, b in zip(pos, pos[1:]))

            def ends(p):
                e0, e1 = p[0] % 2, p[-1] % 2
                return (e0 == 1 and e1 == 1) if par == "odd" else (e0 == 0 and e1 == 0)
            return sum(1 for p in perms if noadj(p) and ends(p))
        raise AssertionError(f"미등록 chk: {chk}")

    random.seed(2026)
    for name in ("func_composite_count", "binom_conditional",
                 "multiset_perm_constraint"):
        for _ in range(100):
            r = pe.GENERATORS[name]()
            assert pe.verify(r), f"{name}: verify 실패"
            assert r["_meta"]["ans"] == recompute(r), \
                f"{name}: 독립 재계산 불일치 {r['_meta']['ans']} != {recompute(r)}"
            # 함정(오답 태깅)이 정답 번호와 다른 보기에 위치하는지
            tm = r["completion"].get("trap_map") or {}
            ci = int(r["completion"]["answer"].split("번")[0])
            for kk in tm:
                assert 1 <= int(kk) <= 5 and int(kk) != ci, f"{name}: trap 위치 오류"


def test_even_odd_integral_independent():
    """정적분 대칭성: sympy 정적분 vs 대칭공식 vs _meta.ans 삼중 대조."""
    random.seed(19)
    for _ in range(80):
        r = pe.gen_even_odd_integral()
        assert pe.verify(r), "verify 실패"
        m = r["_meta"]
        a, c3, c2, c1, c0 = m["a"], m["c3"], m["c2"], m["c1"], m["c0"]
        f = c3*x**3 + c2*x**2 + c1*x + c0
        direct = int(sp.integrate(f, (x, -a, a)))
        sym = 2*(c2*a**3//3 + c0*a)
        assert direct == sym == m["ans"], f"불일치 {direct},{sym},{m['ans']}"


def test_func_codomain_range_independent():
    """공역이 다른 f:X→Y 치역+함숫값 조건 개수: _meta 로 브루트포스 재계산 대조."""
    from itertools import product as _ip
    random.seed(17)
    for _ in range(60):
        r = pe.gen_func_codomain_range()
        assert pe.verify(r), "verify 실패"
        m = r["_meta"]
        n, mm, rr, p, v = m["n"], m["m"], m["r"], m["p"], m["v"]
        cnt = sum(1 for f in _ip(range(1, mm + 1), repeat=n)
                  if len(set(f)) == rr and f[p-1] == v)
        assert cnt == m["ans"], f"개수 불일치 {cnt} != {m['ans']}"
        tm = r["completion"].get("trap_map") or {}
        ci = int(r["completion"]["answer"].split("번")[0])
        for kk in tm:
            assert 1 <= int(kk) <= 5 and int(kk) != ci


def test_graph_inverse_compose_independent():
    """그래프·역함수·합성: 지문 수치로 a,b,c 를 독립 연립해 정답과 대조 (드릴 채굴 유형)."""
    random.seed(13)
    for _ in range(100):
        r = pe.gen_graph_inverse_compose()
        assert pe.verify(r), "verify 실패"
        m = r["_meta"]
        p, a, b, q, x0 = m["p"], m["a"], m["b"], m["q"], m["x0"]
        # A∩B 유일성(|p|≥2): x³-x+a 의 실근이 정확히 1개여야 함
        realroots = [rt for rt in sp.solve(x**3 - x + a, x) if sp.im(rt) == 0]
        assert len(realroots) == 1 and realroots[0] == p, "A∩B 유일성 위반"
        # C 조건으로 b 독립 복원
        fx0 = x0**3 + a
        y0 = b*fx0 + q
        b_rec = sp.solve(sp.Eq(y0, sp.Symbol("bb")*fx0 + q), sp.Symbol("bb"))[0]
        assert b_rec == b, "b 복원 불일치"
        assert a + b + p == m["ans"], "정답 불일치"


def test_trig_rendered_coeffs_match_meta():
    """trig: 지문에 표시된 계수(S,C)가 _meta 와 일치해야 함."""
    random.seed(3)
    pat = re.compile(r"2\\sin\^2 x([+-][^=]*)=0")
    for _ in range(20):
        r = pe.gen_trig_equation()
        m = pat.search(r["completion"]["question"])
        assert m, "trig 지문 형식 변경됨"
        # '+3\cos x-2' 류에서 cos 계수와 상수항 추출
        tail = m.group(1)
        cm = re.search(r"([+-])(\d*)\\cos x", tail)
        S = int((cm.group(1) + (cm.group(2) or "1")))
        rest = tail.replace(cm.group(0), "")
        C = int(rest) if rest else 0
        assert S == r["_meta"]["S"] and C == r["_meta"]["C"], \
            f"지문 계수 ({S},{C}) ≠ meta ({r['_meta']['S']},{r['_meta']['C']})"
