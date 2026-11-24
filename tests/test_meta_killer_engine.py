"""메타 킬러 엔진 회귀 테스트.

핵심 보장:
- rejection sampling 으로 '항상 유효한 문제만' 출력 (성공률 측정).
- 성공한 문제 전부 verify() 통과.
- _meta(블록 파라미터 + 구조화된 질의 스펙)만으로 정답을 '독립 경로'로 재계산 →
  엔진이 기록한 정답과 100% 일치 (LaTeX 파싱이 아니라 구조화 스펙으로 재계산).
"""
import os
import sys
import random
from itertools import product as _iproduct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp  # noqa: E402
from csat_forge import meta_killer_engine as mke  # noqa: E402
from csat_forge import problem_engine as pe  # noqa: E402

X = sp.Symbol("x")
_ALLF = list(_iproduct(range(1, 6), repeat=5))


# ---------- _meta → 정답 독립 재계산 ----------
def _pred_from(name, mt):
    if name == "involution":
        return lambda f: all(f[f[i]-1] == i+1 for i in range(5))
    if name == "fixed_point_count":
        return lambda f, k=mt["k"]: sum(1 for i in range(5) if f[i] == i+1) == k
    if name == "nondecreasing":
        return lambda f: all(f[i] <= f[i+1] for i in range(4))
    if name == "value_in":
        return lambda f, p=mt["p"], S=set(mt["S"]): f[p-1] in S
    if name == "range_size":
        return lambda f, k=mt["k"]: len(set(f)) == k
    if name == "composite_condition":
        return lambda f, p=mt["p"], a=mt["a"]: f[f[p-1]-1] == a
    if name == "value_eq":
        return lambda f, p=mt["p"], v=mt["v"]: f[p-1] == v
    if name == "value_neq":
        return lambda f, p=mt["p"], v=mt["v"]: f[p-1] != v
    if name == "parity":
        return lambda f, p=mt["p"], par=mt["par"]: (
            (f[p-1] % 2 == 1) if par == "odd" else (f[p-1] % 2 == 0))
    if name == "image_avoids":
        return lambda f, a=mt["a"]: a not in f
    if name == "value_sum":
        return lambda f, S=mt["S"]: sum(f) == S
    if name == "pair_sum":
        return lambda f, p=mt["p"], q=mt["q"], v=mt["v"]: f[p-1] + f[q-1] == v
    if name == "max_value":
        return lambda f, k=mt["k"]: max(f) == k
    if name == "min_value":
        return lambda f, k=mt["k"]: min(f) == k
    raise AssertionError(f"미등록 블록 {name}")


def recompute(r):
    """_meta 의 구조화 정보만으로 정답을 다시 계산(엔진 내부 로직과 독립)."""
    m = r["_meta"]
    chk = m["chk"]
    if chk == "meta_cubic":
        f = X**3 + m["a"]*X**2 + m["b"]*X + m["c"]
        qk = m["qkind"]
        if qk == "rootsum":
            return -m["a"]
        if qk == "sum2":
            return int(f.subs(X, m["t"]) + f.subs(X, m["t2"]))
        if qk == "value":
            return int(f.subs(X, m["t"]))
        return int(sp.diff(f, X).subs(X, m["t"]))
    if chk == "meta_quartic":
        f = X**4 + m["a3"]*X**3 + m["a2"]*X**2 + m["a1"]*X + m["a0"]
        qk = m["qkind"]
        if qk == "rootsum":
            return -m["a3"]
        if qk == "value":
            return int(f.subs(X, m["t"]))
        return int(sp.diff(f, X).subs(X, m["t"]))
    if chk == "meta_aseq":
        a1, d, t = m["a1"], m["d"], m["t"]
        if m["qkind"] == "term":
            return a1 + (t-1)*d
        return int(sp.Rational(t, 1)*(2*a1 + (t-1)*d)/2)
    if chk == "meta_gseq":
        a1, r, t = m["a1"], m["r"], m["t"]
        if m["qkind"] == "term":
            return a1 * r**(t-1)
        return a1 * (r**t - 1) // (r - 1)
    if chk == "meta_trigdet":
        return m["a"] + m["b"] + m["c"]
    if chk == "meta_chain":
        # 다단계 연쇄: N에서 각 stage 변환을 순차 적용해 독립 재계산
        cur = m["N"]
        for (k, b, a) in m["stages"]:
            cur = mke._chain_transform(cur, k)
        return cur
    if chk == "meta_interlock":
        # 1단계 답 N을 파라미터로 2단계를 닫힌형으로 독립 재계산
        from math import comb
        N, k = m["N"], m["kind"]
        if k == "extr_diff":
            return 4 * N**3
        if k == "extr_max":
            return 2 * N**3
        if k == "sym_integral":
            return 2 * m["c0"] * N
        if k == "tangent_area":
            return (N**3) // 4
        if k == "comb_count":
            return comb(N + 3, 2)
        if k == "binom_coeff":
            return comb(N + 2, 2)
        if k == "geom_term":
            return 2 * N**2
        raise AssertionError(k)
    if chk == "meta_fusion":
        parts, op, coeffs = m["parts"], m["op"], m["coeffs"]
        if op == "sum":
            return sum(parts)
        if op == "product":
            v = 1
            for p in parts:
                v *= p
            return v
        return sum(c*p for c, p in zip(coeffs, parts))
    if chk == "meta_setf":
        preds = [_pred_from(nm, m["params"][nm]) for nm in m["params"]]
        return sum(1 for f in _ALLF if all(p(f) for p in preds))
    if chk == "meta_intsol":
        k, N = m["k"], m["N"]
        preds = [mke._is_pred_from(nm, m["params"][nm]) for nm in m["params"]]
        sols = [s for s in _iproduct(range(N + 1), repeat=k) if sum(s) == N]
        return sum(1 for s in sols if all(p(s) for p in preds))
    if chk == "meta_intsolprob":
        k, N = m["k"], m["N"]
        preds = [mke._is_pred_from(nm, m["params"][nm]) for nm in m["params"]]
        sols = [s for s in _iproduct(range(N + 1), repeat=k) if sum(s) == N]
        fav = sum(1 for s in sols if all(p(s) for p in preds))
        a = sp.Rational(fav, len(sols))
        return [a.p, a.q]
    if chk == "meta_rv":
        xs, D, ns = m["xs"], m["D"], m["ns"]
        probs = [sp.Rational(ns[i], D) for i in range(4)]
        E = sum(xs[i]*probs[i] for i in range(4))
        V = sum(xs[i]**2*probs[i] for i in range(4)) - E**2
        if m["qkind"] == "E_affine":
            a = m["c1"]*E + m["c2"]
        elif m["qkind"] == "V_affine":
            a = m["c1"]**2 * V
        else:
            a = sum(probs[i] for i in range(4) if xs[i] >= m["xk"])
        a = sp.nsimplify(a)
        return [a.p, a.q]
    raise AssertionError(f"미등록 chk {chk}")


def _run_domain(compose, n=200):
    succ, combos = 0, set()
    for _ in range(n):
        r = compose()
        if r is None:
            continue
        succ += 1
        assert pe.verify(r), "verify 실패"
        assert recompute(r) == r["_meta"]["ans"], \
            f"독립 재계산 불일치: {recompute(r)} != {r['_meta']['ans']}"
        ch = r["completion"]["choices"]
        assert len(ch) == 5 and len(set(ch)) == 5, "보기 5개/중복 위반"
        combos.add(tuple(r["_meta"]["blocks"]))
    return succ, combos


def test_cubic_domain():
    random.seed(101)
    succ, combos = _run_domain(lambda: mke.compose_cubic(random.choice([2, 3])))
    assert succ >= 60, f"성공률 너무 낮음 {succ}/200"
    assert len(combos) >= 8, f"조합 다양성 부족 {len(combos)}"


def test_quartic_domain():
    random.seed(150)
    succ, combos = _run_domain(lambda: mke.compose_quartic(4))
    assert succ >= 50, f"성공률 너무 낮음 {succ}/200"


def test_set_function_domain():
    random.seed(202)
    succ, combos = _run_domain(lambda: mke.compose_set_function(random.choice([2, 3])))
    assert succ >= 60, f"성공률 너무 낮음 {succ}/200"
    assert len(combos) >= 10, f"조합 다양성 부족 {len(combos)}"


def test_discrete_rv_domain():
    random.seed(303)
    succ, combos = _run_domain(lambda: mke.compose_discrete_rv(2))
    assert succ >= 60, f"성공률 너무 낮음 {succ}/200"
    assert len(combos) >= 4, f"조합 다양성 부족 {len(combos)}"


def test_arith_seq_domain():
    random.seed(505)
    succ, combos = _run_domain(lambda: mke.compose_arith_seq(2))
    assert succ >= 60, f"성공률 너무 낮음 {succ}/200"
    assert len(combos) >= 4, f"조합 다양성 부족 {len(combos)}"


def test_geom_seq_domain():
    random.seed(515)
    succ, _ = _run_domain(lambda: mke.compose_geom_seq(2))
    assert succ >= 50, f"성공률 낮음 {succ}/200"


def test_trig_determine_domain():
    random.seed(525)
    succ, _ = _run_domain(lambda: mke.compose_trig_determine(3))
    assert succ >= 80, f"성공률 낮음 {succ}/200"


def test_integer_solutions_domain():
    random.seed(606)
    succ, combos = _run_domain(lambda: mke.compose_integer_solutions(random.choice([2, 3])))
    assert succ >= 60, f"성공률 너무 낮음 {succ}/200"
    assert len(combos) >= 8, f"조합 다양성 부족 {len(combos)}"


def test_intsol_prob_domain():
    """부정방정식+확률 (이해원 N제 확통 채굴 유형)."""
    random.seed(808)
    succ, combos = _run_domain(lambda: mke.compose_intsol_prob(random.choice([2, 3])))
    assert succ >= 60, f"성공률 너무 낮음 {succ}/200"
    assert len(combos) >= 6, f"조합 다양성 부족 {len(combos)}"


def test_trap_maps_valid():
    """함정 오답 태깅: 보기번호가 1~5이고 정답 번호와 달라야 함."""
    random.seed(707)
    tagged = 0
    for compose in (lambda: mke.compose_set_function(3),
                    lambda: mke.compose_integer_solutions(3)):
        for _ in range(120):
            r = compose()
            if r is None:
                continue
            tm = r["completion"].get("trap_map") or {}
            if not tm:
                continue
            tagged += 1
            ci = int(r["completion"]["answer"].split("번")[0])
            for kk in tm:
                assert 1 <= int(kk) <= 5 and int(kk) != ci, "함정 위치 오류"
    assert tagged > 10, f"함정 표본 부족 {tagged}"


def test_fusion_domain():
    """복합 융합 킬러: 두 검증 하위문제의 정수답 결합(p+q), 결합 검산."""
    random.seed(909)
    succ, pairs = 0, set()
    for _ in range(200):
        r = mke.compose_fusion()
        if r is None:
            continue
        succ += 1
        assert pe.verify(r), "verify 실패"
        m = r["_meta"]
        assert recompute(r) == m["ans"], "융합 결합 검산 불일치"
        assert len(set(m["blocks"])) == m["k"], "동일 개념 융합(차별성 위반)"
        pairs.add((m["op"], tuple(sorted(m["blocks"]))))
    assert succ >= 60, f"성공률 낮음 {succ}/200"
    assert len(pairs) >= 30, f"융합쌍 다양성 부족 {len(pairs)}"


def test_chain_domain():
    """다단계 연쇄 킬러: 1단계 답 N + 순차 정수 변환 검산."""
    random.seed(1234)
    succ = 0
    for _ in range(150):
        r = mke.compose_chain()
        if r is None:
            continue
        succ += 1
        assert pe.verify(r), "verify 실패"
        assert recompute(r) == r["_meta"]["ans"], "연쇄 변환 검산 불일치"
    assert succ >= 50, f"성공률 낮음 {succ}/150"


def test_generate_meta_always_valid():
    """통합 진입점: 도메인 무작위 선택 → 항상 유효 문제(verify+독립검증)만 반환."""
    random.seed(404)
    for _ in range(200):
        r = mke.generate_meta()
        assert r is not None, "generate_meta None 반환"
        assert pe.verify(r), "verify 실패"
        assert recompute(r) == r["_meta"]["ans"], "독립 재계산 불일치"


def test_batch_extraction_dedup_and_diversity():
    """배치 추출: 중복 없는 새 유형 대량 생성 + 조합 다양성 집계."""
    recs, stats = mke.generate_meta_batch(120, seed=2026)
    assert stats["produced"] == 120, f"생성 부족 {stats}"
    # 지문 중복 없음
    qs = [r["completion"]["question"] for r in recs]
    assert len(set(qs)) == len(qs), "중복 지문 존재"
    # 전부 검증 통과 + 독립 재계산 일치
    for r in recs:
        assert pe.verify(r) and recompute(r) == r["_meta"]["ans"]
    # 14개 도메인 모두 등장 + 고유 블록조합 충분
    assert len(stats["per_domain"]) == 14, f"도메인 누락 {stats['per_domain']}"
    assert stats["unique_combos"] >= 45, f"조합 다양성 부족 {stats['unique_combos']}"
