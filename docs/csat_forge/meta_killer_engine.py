"""조건결합형 킬러 '메타 엔진' — 조건 블록을 조합해 (가)(나)(다) 킬러를 자동 합성.

설계: 기존 problem_engine.py 는 "사람이 새 조합을 함수 하나씩 작성"이라 유형이
선형으로만 늘었다. 이 모듈은 '조건 블록(ConditionBlock)'을 추상화하고, 한 도메인
(삼차함수·집합함수·확률변수)에 적용 가능한 블록들을 2~3개 무작위 조합해 새 문제를
'합성'한다. 조합 수가 크므로 사실상 무한히 많은 신규 유형이 생긴다.

정답 보장 (절대 원칙):
- 정답은 항상 sympy(연립) 또는 itertools 브루트포스 '계산'으로 결정한다.
- 핵심 트릭: 먼저 '정답이 되는 대상(secret)'을 구성하고, 각 블록은 그 secret 이
  만족하는 '참인 조건'을 진술한다 → 연립이 항상 무모순. 따라서 rejection 은
  (1) 해의 유일성 (2) 정답·보기의 적절성(정수/범위) 만 거른다.
- 모순/비유일/부적절 조합은 폐기(reject)하고 다른 파라미터·조합으로 재시도한다.

출력 스키마는 problem_engine 과 동일(_rec + _meta). GENERATORS 에는 등록하지
않는다(프로토타입). verify() 로 독립 재검증 가능.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from itertools import product as _iproduct
from typing import Callable

import sympy as sp

from csat_forge.problem_engine import (_rec, make_mc, make_mc_tagged, _ltx,  # noqa: F401
                                verify)  # (재사용)

X = sp.Symbol("x")
A, B, C = sp.symbols("A B C")        # 삼차함수 미지수 (최고차 계수 1 고정)
_G = X**3 + A*X**2 + B*X + C         # 일반형 g(x)=x³+Ax²+Bx+C


# ======================================================================
#  ConditionBlock 추상화
# ======================================================================
@dataclass
class ConditionBlock:
    """하나의 (가)/(나)/(다) 조건 단위.

    name        : 식별자 (조합 다양성 집계용)
    domain      : 적용 가능한 대상 타입
    n_eq        : 미지수에 부여하는 방정식 수 (브루트포스 도메인은 0)
    build       : 컨텍스트를 받아 (text, payload, step) 를 반환하는 함수
                  - 방정식형(삼차·확률변수): payload = [sympy 식 (==0)]
                  - 브루트포스형(집합함수): payload = 술어(predicate) 함수
    structural  : True 면 secret 구성 단계에서 특수 처리 필요(예: 극값)
    """
    name: str
    domain: str
    n_eq: int
    build: Callable
    structural: bool = False
    meta: dict = field(default_factory=dict)


# ======================================================================
#  도메인 A : cubic_function  f(x)=x³+ax²+bx+c  (미지수 a,b,c — 3개)
# ======================================================================
def _cubic_secret_with_extremum():
    """극값 블록 포함 시: f'=3(x-p)(x-r) (p+r 짝수 → 정수계수)로 secret 구성."""
    while True:
        p, r = random.sample([-2, -1, 0, 1, 2, 3], 2)
        if (p + r) % 2 == 0:
            break
    a = -3 * (p + r) // 2
    b = 3 * p * r
    c = random.randint(-6, 6)
    f = X**3 + a*X**2 + b*X + c
    return f, {"a": a, "b": b, "c": c, "p": p, "r": r}


def _cubic_secret_plain():
    """극값 블록 없을 때: 정수계수 a,b,c 무작위 secret."""
    a = random.randint(-4, 4)
    b = random.randint(-5, 5)
    c = random.randint(-6, 6)
    return X**3 + a*X**2 + b*X + c, {"a": a, "b": b, "c": c}


def _blk_extremum(f, info, used_pts):
    """(가) x=p에서 극값 m. secret 의 임계점 p 와 m=f(p) 를 진술."""
    p = info["p"]
    m = int(f.subs(X, p))
    used_pts.add(p)
    text = f"$f(x)$는 $x={p}$에서 극값 ${m}$을 갖는다."
    eqs = [sp.diff(_G, X).subs(X, p), _G.subs(X, p) - m]
    step = f"$f'({p})=0$, $f({p})={m}$ — 두 식을 얻는다."
    return text, eqs, step


def _blk_integral(f, info, used_pts):
    """∫_0^q f dx = I."""
    q = random.choice([1, 2, 3])
    I = sp.integrate(f, (X, 0, q))
    text = f"$\\displaystyle\\int_0^{{{q}}} f(x)\\,dx={_ltx(I)}$"
    eqs = [sp.integrate(_G, (X, 0, q)) - I]
    step = f"$\\int_0^{{{q}}} f(x)dx={_ltx(I)}$ 을 전개해 식을 얻는다."
    return text, eqs, step


def _pick_point(used_pts):
    cand = [v for v in [-3, -2, -1, 1, 2, 3] if v not in used_pts]
    if not cand:
        return None
    p = random.choice(cand)
    used_pts.add(p)
    return p


def _blk_passes(f, info, used_pts):
    """f(k)=v (한 점을 지남)."""
    kk = _pick_point(used_pts)
    if kk is None:
        return None
    v = int(f.subs(X, kk))
    text = f"$f({kk})={v}$"
    eqs = [_G.subs(X, kk) - v]
    step = f"$f({kk})={v}$ 임을 이용한다."
    return text, eqs, step


def _blk_tangent(f, info, used_pts):
    """x=k에서의 접선의 기울기 s (=f'(k))."""
    kk = _pick_point(used_pts)
    if kk is None:
        return None
    s = int(sp.diff(f, X).subs(X, kk))
    text = f"곡선 $y=f(x)$ 위의 점 $({kk},\\,f({kk}))$에서의 접선의 기울기는 ${s}$이다."
    eqs = [sp.diff(_G, X).subs(X, kk) - s]
    step = f"$f'({kk})={s}$ 임을 이용한다."
    return text, eqs, step


def _blk_integral_interval(f, info, used_pts):
    """∫_p^q f dx = I (0이 아닌 구간)."""
    p, q = sorted(random.sample([-2, -1, 1, 2, 3], 2))
    I = sp.integrate(f, (X, p, q))
    text = f"$\\displaystyle\\int_{{{p}}}^{{{q}}} f(x)\\,dx={_ltx(I)}$"
    eqs = [sp.integrate(_G, (X, p, q)) - I]
    step = f"$\\int_{{{p}}}^{{{q}}} f(x)dx={_ltx(I)}$ 을 전개해 식을 얻는다."
    return text, eqs, step


def _blk_avg_rate(f, info, used_pts):
    """두 점 사이 평균변화율 m = (f(q)-f(p))/(q-p). 정수일 때만 채택."""
    pts = [v for v in [-3, -2, -1, 1, 2, 3] if v not in used_pts]
    if len(pts) < 2:
        return None
    p, q = sorted(random.sample(pts, 2))
    m = sp.Rational(int(f.subs(X, q)) - int(f.subs(X, p)), q - p)
    if m.q != 1:
        return None
    used_pts.update({p, q})
    text = (f"닫힌구간 $[{p},\\,{q}]$에서 $f(x)$의 평균변화율은 ${_ltx(m)}$이다.")
    eqs = [(_G.subs(X, q) - _G.subs(X, p)) - m*(q - p)]
    step = f"$\\dfrac{{f({q})-f({p})}}{{{q}-({p})}}={_ltx(m)}$ 임을 이용한다."
    return text, eqs, step


def _blk_value_relation(f, info, used_pts):
    """두 점 함숫값의 합 f(p)+f(q)=v."""
    pts = [v for v in [-3, -2, -1, 1, 2, 3] if v not in used_pts]
    if len(pts) < 2:
        return None
    p, q = sorted(random.sample(pts, 2))
    used_pts.update({p, q})
    v = int(f.subs(X, p)) + int(f.subs(X, q))
    text = f"$f({p})+f({q})={v}$"
    eqs = [(_G.subs(X, p) + _G.subs(X, q)) - v]
    step = f"$f({p})+f({q})={v}$ 임을 이용한다."
    return text, eqs, step


def _blk_second_deriv(f, info, used_pts):
    """도함수 f'의 x=k에서의 미분계수(=f''(k))가 s."""
    kk = _pick_point(used_pts)
    if kk is None:
        return None
    s = int(sp.diff(f, X, 2).subs(X, kk))
    text = (f"함수 $f$의 도함수 $f'(x)$에 대하여 $f'(x)$의 $x={kk}$에서의 "
            f"미분계수는 ${s}$이다.")
    eqs = [sp.diff(_G, X, 2).subs(X, kk) - s]
    step = f"$f'(x)$의 $x={kk}$ 미분계수$={s}$ (즉 $f''({kk})={s}$) 를 이용한다."
    return text, eqs, step


def _blk_inflection(f, info, used_pts):
    """곡선 y=f(x)의 변곡점의 x좌표가 k (=-a/3). a가 3의 배수일 때만 채택."""
    a = info["a"]
    if a % 3 != 0:
        return None
    k = -a // 3
    text = f"곡선 $y=f(x)$의 변곡점의 $x$좌표는 ${k}$이다."
    eqs = [sp.diff(_G, X, 2).subs(X, k)]      # 6k+2A=0 ⇒ A=-3k
    step = f"변곡점에서 $f''({k})=0$ 임을 이용한다."
    return text, eqs, step


CUBIC_BLOCKS = {
    "extremum_at": ConditionBlock("extremum_at", "cubic_function", 2, _blk_extremum, structural=True),
    "inflection": ConditionBlock("inflection", "cubic_function", 1, _blk_inflection),
    "integral_value": ConditionBlock("integral_value", "cubic_function", 1, _blk_integral),
    "passes_through": ConditionBlock("passes_through", "cubic_function", 1, _blk_passes),
    "tangent_slope_at": ConditionBlock("tangent_slope_at", "cubic_function", 1, _blk_tangent),
    "integral_interval": ConditionBlock("integral_interval", "cubic_function", 1, _blk_integral_interval),
    "avg_rate": ConditionBlock("avg_rate", "cubic_function", 1, _blk_avg_rate),
    "value_relation": ConditionBlock("value_relation", "cubic_function", 1, _blk_value_relation),
    "second_deriv": ConditionBlock("second_deriv", "cubic_function", 1, _blk_second_deriv),
}
_CUBIC_SINGLES = ["integral_value", "passes_through", "tangent_slope_at",
                  "integral_interval", "avg_rate", "value_relation", "second_deriv",
                  "inflection"]


def _cubic_final_query(f, info, used_pts):
    """최종 질의를 무작위로 선택 → (질의문, 정답, 해설꼬리, 질의스펙)."""
    kind = random.choice(["value", "deriv", "sum2", "rootsum"])
    if kind == "rootsum":                       # 세 근의 합 = -a (비에타)
        ans = -info["a"]
        return ("삼차방정식 $f(x)=0$의 모든 근의 합은?", ans,
                f"근과 계수의 관계에서 세 근의 합 $=-({info['a']})={ans}$",
                {"qkind": kind})
    t = _pick_point(used_pts)
    if t is None:
        t = random.choice([4, -4, 5])
    if kind == "sum2":
        t2 = _pick_point(used_pts)
        if t2 is None:
            t2 = random.choice([v for v in [4, -4, 5, 6] if v != t])
        ans = int(f.subs(X, t) + f.subs(X, t2))
        lo2, hi2 = sorted([t, t2])
        return (f"$f({lo2})+f({hi2})$의 값은?", ans,
                f"$f({lo2})+f({hi2})={ans}$", {"qkind": kind, "t": lo2, "t2": hi2})
    qspec = {"qkind": kind, "t": t}
    if kind == "value":
        ans = int(f.subs(X, t))
        return f"$f({t})$의 값은?", ans, f"$f({t})={ans}$", qspec
    ans = int(sp.diff(f, X).subs(X, t))
    return f"$f'({t})$의 값은?", ans, f"$f'({t})={ans}$", qspec


def compose_cubic(n_blocks=2, max_retries=50):
    """삼차함수 도메인 합성. 성공 시 _rec, 실패 시 None."""
    for _ in range(max_retries):
        # 1) 블록 조합 결정 (총 방정식 수 = 3 또는 일관·과결정)
        if n_blocks == 2:
            names = ["extremum_at", random.choice(_CUBIC_SINGLES)]
        else:
            if random.random() < 0.5:
                names = ["extremum_at"] + random.sample(_CUBIC_SINGLES, 2)
            else:
                names = random.sample(_CUBIC_SINGLES, 3)
        # 2) secret 구성
        if "extremum_at" in names:
            f, info = _cubic_secret_with_extremum()
        else:
            f, info = _cubic_secret_plain()
        # 3) 각 블록의 조건·식 생성
        used_pts = set()
        texts, eqs, steps, ok = [], [], [], True
        for nm in names:
            res = CUBIC_BLOCKS[nm].build(f, info, used_pts)
            if res is None:
                ok = False
                break
            t_, e_, s_ = res
            texts.append(t_)
            eqs.extend(e_)
            steps.append(s_)
        if not ok:
            continue
        # 4) 연립 → 유일해 검사
        sol = sp.solve(eqs, [A, B, C], dict=True)
        if len(sol) != 1:
            continue
        s = sol[0]
        if set(s.keys()) != {A, B, C}:           # 자유변수 잔존 → 비유일
            continue
        if not (s[A] == info["a"] and s[B] == info["b"] and s[C] == info["c"]):
            continue
        if not all(v.is_Integer for v in s.values()):
            continue
        # 5) 최종 질의 → 정답
        query, ans, qtail, qspec = _cubic_final_query(f, info, used_pts)
        if abs(ans) > 300:
            continue
        # 6) 문제·해설 조립
        labels = ["가", "나", "다", "라"]
        cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(len(texts)))
        q = (f"최고차항의 계수가 $1$인 삼차함수 $f(x)$가 다음 조건을 만족시킬 때, "
             f"{query}\n\n{cond_lines}")
        step_lines = "\n".join(f"{i+1}단계: ({labels[i]})에서 {steps[i]}"
                               for i in range(len(steps)))
        e = (f"{step_lines}\n"
             f"{len(steps)+1}단계: 연립하면 "
             f"$f(x)={sp.latex(sp.expand(f))}$.\n"
             f"{len(steps)+2}단계: {qtail}.\n\n")
        choices, lab = make_mc(sp.Integer(ans),
                               [ans-3, ans+3, ans-6, ans+6, -ans])
        e += f"정답: {lab}"
        rr = _rec("미적분Ⅰ", "조건결합형 삼차함수(메타)", "상", 4, "객관식",
                  q, choices, lab, e)
        rr["_meta"] = {"chk": "meta_cubic", "blocks": sorted(names),
                       "a": info["a"], "b": info["b"], "c": info["c"],
                       "query": query, **qspec, "ans": ans}
        return rr
    return None


# ======================================================================
#  도메인 A' : quartic_function  f(x)=x⁴+ax³+bx²+cx+d (미지수 4개)
#  드릴 수2의 '최고차항 계수 1인 사차함수 조건결합형' 채굴 — cubic 기계 일반화.
# ======================================================================
_QA, _QB, _QC, _QD = sp.symbols("QA QB QC QD")
_GQ = X**4 + _QA*X**3 + _QB*X**2 + _QC*X + _QD     # 일반형 사차


def _quartic_obs_block(f, used_pts):
    """관측형 블록(각 1식, 미지수 4개에 선형). 반환 (text, eq, step, kind)."""
    kind = random.choice(["passes", "tangent", "integral0", "integral_iv",
                          "value_relation", "second_deriv"])
    if kind == "value_relation":
        pts = [v for v in [-2, -1, 1, 2, 3] if v not in used_pts]
        if len(pts) < 2:
            return None
        p, q = sorted(random.sample(pts, 2))
        used_pts.update({p, q})
        v = int(f.subs(X, p)) + int(f.subs(X, q))
        return (f"$f({p})+f({q})={v}$", (_GQ.subs(X, p) + _GQ.subs(X, q)) - v,
                f"$f({p})+f({q})={v}$ 임을 이용한다.", kind)
    if kind == "second_deriv":
        kk = _pick_point(used_pts)
        if kk is None:
            return None
        s = int(sp.diff(f, X, 2).subs(X, kk))
        return (f"함수 $f$의 도함수 $f'(x)$의 $x={kk}$에서의 미분계수는 ${s}$이다.",
                sp.diff(_GQ, X, 2).subs(X, kk) - s,
                f"$f''({kk})={s}$ 를 이용한다.", kind)
    if kind == "passes":
        kk = _pick_point(used_pts)
        if kk is None:
            return None
        v = int(f.subs(X, kk))
        return f"$f({kk})={v}$", _GQ.subs(X, kk) - v, f"$f({kk})={v}$ 임을 이용한다.", kind
    if kind == "tangent":
        kk = _pick_point(used_pts)
        if kk is None:
            return None
        s = int(sp.diff(f, X).subs(X, kk))
        return (f"곡선 $y=f(x)$ 위의 점 $({kk},\\,f({kk}))$에서의 접선의 기울기는 ${s}$이다.",
                sp.diff(_GQ, X).subs(X, kk) - s, f"$f'({kk})={s}$ 임을 이용한다.", kind)
    if kind == "integral0":
        qq = random.choice([1, 2])
        I = sp.integrate(f, (X, 0, qq))
        return (f"$\\displaystyle\\int_0^{{{qq}}} f(x)\\,dx={_ltx(I)}$",
                sp.integrate(_GQ, (X, 0, qq)) - I,
                f"$\\int_0^{{{qq}}} f(x)dx={_ltx(I)}$ 를 전개한다.", kind)
    p, qn = sorted(random.sample([-2, -1, 1, 2], 2))
    I = sp.integrate(f, (X, p, qn))
    return (f"$\\displaystyle\\int_{{{p}}}^{{{qn}}} f(x)\\,dx={_ltx(I)}$",
            sp.integrate(_GQ, (X, p, qn)) - I,
            f"$\\int_{{{p}}}^{{{qn}}} f(x)dx={_ltx(I)}$ 를 전개한다.", kind)


def compose_quartic(n_blocks=4, max_retries=60):
    """사차함수 도메인 합성: 관측형 블록 4개로 a,b,c,d 결정 → f(t)/근의 합."""
    for _ in range(max_retries):
        # secret: 정수계수 monic 사차
        a3 = random.randint(-3, 3)
        a2 = random.randint(-4, 4)
        a1 = random.randint(-5, 5)
        a0 = random.randint(-6, 6)
        f = X**4 + a3*X**3 + a2*X**2 + a1*X + a0
        used_pts = set()
        texts, eqs, steps, names, ok = [], [], [], [], True
        for _ in range(4):                         # 미지수 4개 → 4식
            res = _quartic_obs_block(f, used_pts)
            if res is None:
                ok = False
                break
            t_, e_, s_, kind_ = res
            texts.append(t_)
            eqs.append(e_)
            steps.append(s_)
            names.append(kind_)
        if not ok:
            continue
        sol = sp.solve(eqs, [_QA, _QB, _QC, _QD], dict=True)
        if len(sol) != 1 or set(sol[0].keys()) != {_QA, _QB, _QC, _QD}:
            continue
        s = sol[0]
        if not (s[_QA] == a3 and s[_QB] == a2 and s[_QC] == a1 and s[_QD] == a0):
            continue
        if not all(v.is_Integer for v in s.values()):
            continue
        # 최종 질의
        qk = random.choice(["value", "deriv", "rootsum"])
        tval = None
        if qk == "rootsum":
            ans = -a3
            query = "사차방정식 $f(x)=0$의 모든 근의 합은?"
            qtail = f"근과 계수의 관계에서 모든 근의 합 $=-({a3})={ans}$"
        else:
            tval = _pick_point(used_pts) or random.choice([3, -3, 4])
            if qk == "value":
                ans = int(f.subs(X, tval))
                query, qtail = f"$f({tval})$의 값은?", f"$f({tval})={ans}$"
            else:
                ans = int(sp.diff(f, X).subs(X, tval))
                query, qtail = f"$f'({tval})$의 값은?", f"$f'({tval})={ans}$"
        if abs(ans) > 600:
            continue
        labels = ["가", "나", "다", "라"]
        cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(4))
        q = (f"최고차항의 계수가 $1$인 사차함수 $f(x)$가 다음 조건을 만족시킬 때, "
             f"{query}\n\n{cond_lines}")
        step_lines = "\n".join(f"{i+1}단계: ({labels[i]})에서 {steps[i]}" for i in range(4))
        e = (f"{step_lines}\n"
             f"5단계: 네 식을 연립하면 $f(x)={sp.latex(sp.expand(f))}$.\n"
             f"6단계: {qtail}.\n\n")
        d = max(2, abs(ans)//12 or 2)
        choices, lab = make_mc(sp.Integer(ans), [ans-d, ans+d, ans-2*d, ans+2*d, -ans])
        e += f"정답: {lab}"
        rr = _rec("미적분Ⅰ", "조건결합형 사차함수(메타)", "상", 4, "객관식",
                  q, choices, lab, e)
        rr["_meta"] = {"chk": "meta_quartic", "blocks": sorted(names),
                       "a3": a3, "a2": a2, "a1": a1, "a0": a0,
                       "qkind": qk, "t": tval, "ans": ans}
        return rr
    return None


# ======================================================================
#  도메인 B : set_function  f:X→X, X={1..5}  (브루트포스 카운트)
# ======================================================================
_ALLF = list(_iproduct(range(1, 6), repeat=5))   # 5^5 = 3125개 함수 (미리 열거)


def _bb_involution():
    return ("모든 $x\\in X$에 대하여 $f(f(x))=x$이다.",
            lambda f: all(f[f[i]-1] == i+1 for i in range(5)),
            "$f$는 대합(involution): $f(f(x))=x$.", {"k": "inv"})


def _bb_fixed_count():
    k = random.choice([0, 1, 2, 3])
    return (f"$f(x)=x$를 만족시키는 $x$의 개수는 ${k}$이다.",
            lambda f, k=k: sum(1 for i in range(5) if f[i] == i+1) == k,
            f"고정점($f(x)=x$)의 개수가 ${k}$.", {"k": k})


def _bb_nondecreasing():
    return ("$x_1<x_2$이면 $f(x_1)\\le f(x_2)$이다.",
            lambda f: all(f[i] <= f[i+1] for i in range(4)),
            "$f$는 (넓은 의미) 증가함수.", {})


def _bb_value_in():
    p = random.randint(1, 5)
    S = tuple(sorted(random.sample([1, 2, 3, 4, 5], random.choice([2, 3]))))
    sset = "\\{" + ",".join(str(v) for v in S) + "\\}"
    return (f"$f({p})\\in {sset}$이다.",
            lambda f, p=p, S=S: f[p-1] in S,
            f"$f({p})\\in{sset}$.", {"p": p, "S": list(S)})


def _bb_range_size():
    k = random.choice([2, 3, 4])
    return (f"$f$의 치역의 원소의 개수는 ${k}$이다.",
            lambda f, k=k: len(set(f)) == k,
            f"치역의 크기가 ${k}$.", {"k": k})


def _bb_composite_at():
    p, a = random.randint(1, 5), random.randint(1, 5)
    return (f"$f(f({p}))={a}$이다.",
            lambda f, p=p, a=a: f[f[p-1]-1] == a,
            f"$f(f({p}))={a}$.", {"p": p, "a": a})


def _bb_value_eq():
    p, v = random.randint(1, 5), random.randint(1, 5)
    return (f"$f({p})={v}$이다.",
            lambda f, p=p, v=v: f[p-1] == v,
            f"$f({p})={v}$.", {"p": p, "v": v})


def _bb_value_neq():
    p, v = random.randint(1, 5), random.randint(1, 5)
    return (f"$f({p})\\ne {v}$이다.",
            lambda f, p=p, v=v: f[p-1] != v,
            f"$f({p})\\ne{v}$.", {"p": p, "v": v})


def _bb_parity():
    p = random.randint(1, 5)
    par = random.choice(["odd", "even"])
    word = "홀수" if par == "odd" else "짝수"
    return (f"$f({p})$은 {word}이다.",
            lambda f, p=p, par=par: (f[p-1] % 2 == 1) if par == "odd" else (f[p-1] % 2 == 0),
            f"$f({p})$이 {word}.", {"p": p, "par": par})


def _bb_image_avoids():
    a = random.randint(1, 5)
    return (f"함수 $f$의 치역에 ${a}$은 속하지 않는다.",
            lambda f, a=a: a not in f,
            f"${a}$은 치역에 없음.", {"a": a})


def _bb_pair_sum():
    p, q = sorted(random.sample([1, 2, 3, 4, 5], 2))
    v = random.randint(3, 9)
    return (f"$f({p})+f({q})={v}$이다.",
            lambda f, p=p, q=q, v=v: f[p-1] + f[q-1] == v,
            f"$f({p})+f({q})={v}$.", {"p": p, "q": q, "v": v})


def _bb_max_value():
    k = random.choice([3, 4, 5])
    return (f"함수 $f$의 최댓값은 ${k}$이다.",
            lambda f, k=k: max(f) == k,
            f"$\\max f={k}$.", {"k": k})


def _bb_min_value():
    k = random.choice([1, 2, 3])
    return (f"함수 $f$의 최솟값은 ${k}$이다.",
            lambda f, k=k: min(f) == k,
            f"$\\min f={k}$.", {"k": k})


def _bb_value_sum():
    S = random.randint(10, 20)
    return (f"$f(1)+f(2)+f(3)+f(4)+f(5)={S}$이다.",
            lambda f, S=S: sum(f) == S,
            f"함숫값의 총합이 ${S}$.", {"S": S})


SETF_BLOCKS = {
    "involution": _bb_involution,
    "fixed_point_count": _bb_fixed_count,
    "nondecreasing": _bb_nondecreasing,
    "value_in": _bb_value_in,
    "range_size": _bb_range_size,
    "composite_condition": _bb_composite_at,
    "value_eq": _bb_value_eq,
    "value_neq": _bb_value_neq,
    "parity": _bb_parity,
    "image_avoids": _bb_image_avoids,
    "value_sum": _bb_value_sum,
    "pair_sum": _bb_pair_sum,
    "max_value": _bb_max_value,
    "min_value": _bb_min_value,
}


def compose_set_function(n_blocks=2, max_retries=50, count_range=(1, 30)):
    """집합함수 도메인 합성. 2~3개 블록의 AND 조건으로 함수 개수를 브루트포스."""
    lo, hi = count_range
    names_all = list(SETF_BLOCKS)
    for _ in range(max_retries):
        names = random.sample(names_all, n_blocks)
        texts, preds, steps, metas = [], [], [], {}
        for nm in names:
            t_, pred, s_, mt = SETF_BLOCKS[nm]()
            texts.append(t_)
            preds.append(pred)
            steps.append(s_)
            metas[nm] = mt
        cnt = sum(1 for f in _ALLF if all(p(f) for p in preds))
        if not (lo <= cnt <= hi):
            continue
        labels = ["가", "나", "다", "라"]
        cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(len(texts)))
        q = (f"집합 $X=\\{{1,2,3,4,5\\}}$에 대하여 다음 조건을 모두 만족시키는 "
             f"함수 $f:X\\to X$의 개수는?\n\n{cond_lines}")
        step_lines = "\n".join(f"{i+1}단계: ({labels[i]}) {steps[i]}"
                               for i in range(len(steps)))
        e = (f"{step_lines}\n"
             f"{len(steps)+1}단계: 위 조건을 모두 만족하는 함수를 전수 조사로 세면 "
             f"${cnt}$개이다.\n\n")
        dd = [d for d in [cnt-2, cnt+2, cnt-4, cnt+4, cnt+6, cnt-1, cnt+1] if d > 0]
        # 함정: 마지막 조건을 빠뜨린 개수
        naive = sum(1 for f in _ALLF if all(p(f) for p in preds[:-1]))
        traps = {}
        if naive != cnt and naive > 0:
            choices, lab, traps = make_mc_tagged(
                sp.Integer(cnt),
                [(sp.Integer(naive), f"조건 ({labels[len(steps)-1]}) 하나를 빠뜨리고 셈")] + dd)
        else:
            choices, lab = make_mc(sp.Integer(cnt), dd)
        e += f"정답: {lab}"
        rr = _rec("확률과 통계", "조건결합형 함수의 개수(메타)", "상", 4, "객관식",
                  q, choices, lab, e)
        if traps:
            rr["completion"]["trap_map"] = traps
        rr["_meta"] = {"chk": "meta_setf", "blocks": sorted(names),
                       "params": metas, "ans": cnt}
        return rr
    return None


# ======================================================================
#  도메인 C : discrete_rv  확률변수 X 의 분포표 (미지수 a,b,c — 3개)
# ======================================================================
_RVU = sp.symbols("ua ub uc")        # 분포표의 미지수 자리 (확률 분자)


def _rv_secret():
    """값 xs(4개) + 분자 numer(합 D)로 secret 분포 구성. 3자리를 미지수로."""
    xs = sorted(random.sample([0, 1, 2, 3, 4, 5, 6], 4))
    D = random.choice([8, 10, 12])
    while True:
        cuts = sorted(random.sample(range(1, D), 3))
        ns = [cuts[0], cuts[1]-cuts[0], cuts[2]-cuts[1], D-cuts[2]]
        if min(ns) >= 1:
            break
    return xs, D, ns


def _compose_rv_record(xs, D, ns, unk_idx, names, texts, steps, eqs, metas):
    """연립 풀어 유일성 확인 후 최종 질의까지 조립."""
    probs = [sp.Rational(ns[i], D) for i in range(4)]
    # 미지수 자리 분자: ua,ub,uc
    sol = sp.solve(eqs, list(_RVU[:len(unk_idx)]), dict=True)
    if len(sol) != 1:
        return None
    s = sol[0]
    if set(s.keys()) != set(_RVU[:len(unk_idx)]):
        return None
    if any(s[_RVU[j]] != ns[unk_idx[j]] for j in range(len(unk_idx))):
        return None
    # 최종 질의
    E = sum(xs[i]*probs[i] for i in range(4))
    V = sum(xs[i]**2*probs[i] for i in range(4)) - E**2
    qkind = random.choice(["E_affine", "V_affine", "tail_ge"])
    if qkind == "E_affine":
        c1, c2 = random.choice([(2, 1), (3, -1), (2, 3)])
        ans = c1*E + c2
        query = f"$\\mathrm{{E}}({c1}X{'+' if c2>=0 else ''}{c2})$의 값은?"
        qtail = f"$\\mathrm{{E}}({c1}X{'+' if c2>=0 else ''}{c2})={c1}\\mathrm{{E}}(X){'+' if c2>=0 else ''}{c2}={_ltx(ans)}$"
        qspec = {"qkind": qkind, "c1": c1, "c2": c2}
    elif qkind == "V_affine":
        c1 = random.choice([2, 3])
        ans = c1**2 * V
        query = f"$\\mathrm{{V}}({c1}X+1)$의 값은?"
        qtail = f"$\\mathrm{{V}}({c1}X+1)={c1}^2\\mathrm{{V}}(X)={_ltx(ans)}$"
        qspec = {"qkind": qkind, "c1": c1}
    else:
        kk = random.choice([1, 2])      # P(X≥ xs[kk])
        ans = sum(probs[i] for i in range(kk, 4))
        query = f"$\\mathrm{{P}}(X\\ge {xs[kk]})$의 값은?"
        qtail = f"$\\mathrm{{P}}(X\\ge {xs[kk]})={_ltx(ans)}$"
        qspec = {"qkind": qkind, "xk": xs[kk]}
    # 분수 정답 보기 구성
    if ans.q == 1:
        distract = [ans-1, ans+1, ans-2, ans+2, -ans]
    else:
        distract = [ans+sp.Rational(1, D), ans-sp.Rational(1, D),
                    ans*sp.Rational(3, 2), ans*sp.Rational(1, 2),
                    ans+sp.Rational(2, D)]
    choices, lab = make_mc(ans, distract)
    labels = ["가", "나", "다", "라"]
    rows_x = " & ".join(str(v) for v in xs)
    syms = {unk_idx[j]: ["a", "b", "c"][j] for j in range(len(unk_idx))}
    rows_p = " & ".join(syms.get(i, _ltx(probs[i])) for i in range(4))
    cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(len(texts)))
    q = (f"확률변수 $X$의 확률분포를 표로 나타내면 다음과 같다.\n\n"
         f"$\\begin{{array}}{{c|cccc|c}} X & {rows_x} & \\text{{합계}} \\\\\\hline "
         f"\\mathrm{{P}}(X=x) & {rows_p} & 1 \\end{{array}}$\n\n"
         f"다음 조건이 성립할 때, {query} (단, "
         f"{', '.join('$'+syms[i]+'$' for i in unk_idx)}는 상수)\n\n{cond_lines}")
    step_lines = "\n".join(f"{i+2}단계: ({labels[i]}) {steps[i]}"
                           for i in range(len(steps)))
    sol_txt = ", ".join(f"${syms[unk_idx[j]]}={_ltx(probs[unk_idx[j]])}$"
                        for j in range(len(unk_idx)))
    e = (f"1단계: 확률의 총합이 $1$이다.\n{step_lines}\n"
         f"{len(steps)+2}단계: 연립하면 {sol_txt}.\n"
         f"{len(steps)+3}단계: {qtail}.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조건결합형 확률분포(메타)", "상", 4, "객관식",
              q, choices, lab, e)
    rr["_meta"] = {"chk": "meta_rv", "blocks": sorted(names),
                   "xs": xs, "D": D, "ns": ns, "unk_idx": unk_idx,
                   "query": query, **qspec,
                   "ans": [ans.p, ans.q]}
    return rr


def compose_discrete_rv(n_blocks=2, max_retries=60):
    """확률변수 도메인 합성. 미지수 3자리 → 합=1(가) + 무작위 블록 2개로 결정."""
    for _ in range(max_retries):
        xs, D, ns = _rv_secret()
        probs = [sp.Rational(ns[i], D) for i in range(4)]
        unk_idx = sorted(random.sample(range(4), 3))   # 미지수 3자리
        # 미지수 자리 분자 = ua,ub,uc (순서대로)
        numsym = {unk_idx[j]: _RVU[j] for j in range(3)}
        # 표시 확률(분자) 벡터: 미지수 자리는 심볼, 나머지는 상수
        pn = [numsym.get(i, sp.Integer(ns[i])) for i in range(4)]
        # (가) 합 = 1 (분자 합 = D)
        eqs = [sum(pn) - D]
        E_expr = sum(xs[i]*sp.Rational(1, D)*pn[i] for i in range(4))
        # 무작위 블록 풀 (각 1식, secret 으로부터 참값 진술)
        E_true = sum(xs[i]*probs[i] for i in range(4))
        pool = []
        # expectation
        pool.append(("expectation_value",
                     f"$\\mathrm{{E}}(X)={_ltx(E_true)}$",
                     f"$\\mathrm{{E}}(X)={_ltx(E_true)}$ 를 이용한다.",
                     E_expr - E_true, {}))
        # prob_constraint: 미지수 한 자리 값 공개
        j = random.choice(unk_idx)
        pool.append(("prob_constraint",
                     f"$\\mathrm{{P}}(X={xs[j]})={_ltx(probs[j])}$",
                     f"$\\mathrm{{P}}(X={xs[j]})={_ltx(probs[j])}$.",
                     sp.Rational(1, D)*pn[j] - probs[j], {"j": j}))
        # cumulative: P(X<=xs[t]) 공개
        tt = random.choice([1, 2])
        cumv = sum(probs[i] for i in range(tt+1))
        pool.append(("cumulative",
                     f"$\\mathrm{{P}}(X\\le {xs[tt]})={_ltx(cumv)}$",
                     f"$\\mathrm{{P}}(X\\le {xs[tt]})={_ltx(cumv)}$.",
                     sum(sp.Rational(1, D)*pn[i] for i in range(tt+1)) - cumv,
                     {"t": tt}))
        # prob_ge: P(X>=xs[g]) 공개 (꼬리확률)
        gg = random.choice([1, 2])
        gev = sum(probs[i] for i in range(gg, 4))
        pool.append(("prob_ge",
                     f"$\\mathrm{{P}}(X\\ge {xs[gg]})={_ltx(gev)}$",
                     f"$\\mathrm{{P}}(X\\ge {xs[gg]})={_ltx(gev)}$.",
                     sum(sp.Rational(1, D)*pn[i] for i in range(gg, 4)) - gev,
                     {"g": gg}))
        # second_moment: E(X²) 공개 (분산 연결, 미지수에 선형)
        e2 = sum(xs[i]**2 * probs[i] for i in range(4))
        pool.append(("second_moment",
                     f"$\\mathrm{{E}}(X^2)={_ltx(e2)}$",
                     f"$\\mathrm{{E}}(X^2)=\\sum x_i^2 p_i={_ltx(e2)}$.",
                     sum(xs[i]**2 * sp.Rational(1, D)*pn[i] for i in range(4)) - e2,
                     {}))
        # prob_relation: 두 미지수 자리 비 (정수배일 때만)
        rel_added = False
        random.shuffle(unk_idx)
        for i1 in unk_idx:
            for i2 in unk_idx:
                if i1 != i2 and ns[i1] % ns[i2] == 0 and ns[i1] // ns[i2] in (2, 3):
                    cmul = ns[i1] // ns[i2]
                    pool.append(("prob_relation",
                                 f"$\\mathrm{{P}}(X={xs[i1]})={cmul}\\,\\mathrm{{P}}(X={xs[i2]})$",
                                 f"$\\mathrm{{P}}(X={xs[i1]})={cmul}\\mathrm{{P}}(X={xs[i2]})$.",
                                 pn[i1] - cmul*pn[i2], {"i1": i1, "i2": i2, "c": cmul}))
                    rel_added = True
                    break
            if rel_added:
                break
        unk_idx = sorted(unk_idx)
        # 블록 2개 무작위 선택 (서로 다른 이름)
        chosen = random.sample(pool, 2)
        names = [c[0] for c in chosen]
        texts = [c[1] for c in chosen]
        steps = [c[2] for c in chosen]
        for c in chosen:
            eqs.append(c[3])
        rec = _compose_rv_record(xs, D, ns, unk_idx, names, texts, steps, eqs, {})
        if rec is not None:
            return rec
    return None


# ======================================================================
#  도메인 D : arith_seq  등차수열 a_n=a_1+(n-1)d  (미지수 a_1, d — 2개)
# ======================================================================
_A1, _DD = sp.symbols("A1 DD")


def _aseq(a1, d, nn):
    return a1 + (nn - 1)*d


def _asum(a1, d, nn):
    return sp.Rational(nn, 1) * (2*a1 + (nn - 1)*d) / 2


def _arith_blocks(a1, d):
    """secret(a1,d)이 만족하는 참인 조건 풀 생성 (각 1식, a1·d 에 선형)."""
    pool = []
    # term_value: a_k = v
    kk = random.randint(2, 8)
    v = _aseq(a1, d, kk)
    pool.append(("term_value", f"$a_{{{kk}}}={v}$",
                 f"$a_{{{kk}}}=a_1+{kk-1}d={v}$ 임을 이용한다.",
                 _aseq(_A1, _DD, kk) - v, {"k": kk}))
    # partial_sum: S_k = v
    sk = random.randint(3, 8)
    sv = _asum(a1, d, sk)
    pool.append(("partial_sum", f"$\\displaystyle\\sum_{{i=1}}^{{{sk}}} a_i={_ltx(sv)}$",
                 f"$S_{{{sk}}}=\\dfrac{{{sk}(2a_1+{sk-1}d)}}{{2}}={_ltx(sv)}$ 임을 이용한다.",
                 _asum(_A1, _DD, sk) - sv, {"k": sk}))
    # term_relation: a_p + a_q = v
    p, qn = sorted(random.sample(range(1, 9), 2))
    rv = _aseq(a1, d, p) + _aseq(a1, d, qn)
    pool.append(("term_relation", f"$a_{{{p}}}+a_{{{qn}}}={rv}$",
                 f"$a_{{{p}}}+a_{{{qn}}}={rv}$ 임을 이용한다.",
                 _aseq(_A1, _DD, p) + _aseq(_A1, _DD, qn) - rv, {"p": p, "q": qn}))
    # term_multiple: a_p = c·a_q (정수배일 때만)
    pp, qq = random.sample(range(2, 9), 2)
    aq = _aseq(a1, d, qq)
    if aq != 0 and _aseq(a1, d, pp) % aq == 0 and _aseq(a1, d, pp)//aq in (2, 3, -2):
        cm = _aseq(a1, d, pp)//aq
        pool.append(("term_multiple", f"$a_{{{pp}}}={cm}\\,a_{{{qq}}}$",
                     f"$a_{{{pp}}}={cm}a_{{{qq}}}$ 임을 이용한다.",
                     _aseq(_A1, _DD, pp) - cm*_aseq(_A1, _DD, qq),
                     {"p": pp, "q": qq, "c": cm}))
    return pool


def compose_arith_seq(n_blocks=2, max_retries=50):
    """등차수열 도메인 합성. 블록 2개로 a_1, d 결정 → a_t 또는 S_t 질의."""
    for _ in range(max_retries):
        a1 = random.randint(-6, 8)
        d = random.choice([x for x in range(-5, 6) if x != 0])
        pool = _arith_blocks(a1, d)
        if len(pool) < 2:
            continue
        chosen = random.sample(pool, 2)
        names = [c[0] for c in chosen]
        texts = [c[1] for c in chosen]
        steps = [c[2] for c in chosen]
        eqs = [c[3] for c in chosen]
        sol = sp.solve(eqs, [_A1, _DD], dict=True)
        if len(sol) != 1 or set(sol[0].keys()) != {_A1, _DD}:
            continue
        if not (sol[0][_A1] == a1 and sol[0][_DD] == d):
            continue
        # 최종 질의
        qkind = random.choice(["term", "sum"])
        t = random.randint(5, 12)
        if qkind == "term":
            ans = int(_aseq(a1, d, t))
            query = f"$a_{{{t}}}$의 값은?"
            qtail = f"$a_{{{t}}}=a_1+{t-1}d={ans}$"
        else:
            ans = int(_asum(a1, d, t))
            query = f"$\\displaystyle\\sum_{{i=1}}^{{{t}}} a_i$의 값은?"
            qtail = f"$S_{{{t}}}=\\dfrac{{{t}(2a_1+{t-1}d)}}{{2}}={ans}$"
        if abs(ans) > 400:
            continue
        labels = ["가", "나", "다", "라"]
        cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(len(texts)))
        q = (f"공차가 $0$이 아닌 등차수열 $\\{{a_n\\}}$이 다음 조건을 만족시킬 때, "
             f"{query}\n\n{cond_lines}\n\n(단, $S_n=\\sum_{{i=1}}^{{n}} a_i$)")
        step_lines = "\n".join(f"{i+1}단계: ({labels[i]}) {steps[i]}"
                               for i in range(len(steps)))
        e = (f"{step_lines}\n"
             f"{len(steps)+1}단계: 연립하면 $a_1={a1}$, $d={d}$.\n"
             f"{len(steps)+2}단계: {qtail}.\n\n")
        choices, lab = make_mc(sp.Integer(ans),
                               [ans-d, ans+d, ans-2*d, ans+2*d, -ans])
        e += f"정답: {lab}"
        rr = _rec("대수", "조건결합형 등차수열(메타)", "상", 4, "객관식",
                  q, choices, lab, e)
        rr["_meta"] = {"chk": "meta_aseq", "blocks": sorted(names),
                       "a1": a1, "d": d, "qkind": qkind, "t": t, "ans": ans}
        return rr
    return None


# ======================================================================
#  도메인 D' : geom_seq  등비수열 a_n=a·r^(n-1) (공비 양수, 미지수 a, r)
# ======================================================================
def compose_geom_seq(n_blocks=2, max_retries=40):
    """공비가 양수인 등비수열: 두 항 값으로 a,r 결정 → a_t 또는 S_t."""
    for _ in range(max_retries):
        a1 = random.choice([x for x in range(-6, 9) if x != 0])
        r = random.choice([2, 3])
        p, qn = sorted(random.sample(range(1, 8), 2))
        v1 = a1 * r**(p - 1)
        v2 = a1 * r**(qn - 1)
        # 재구성(독립검증): 양수 공비 ⇒ r=(v2/v1)^(1/(q-p)) 유일
        ratio = sp.Rational(v2, v1)
        r_rec = sp.real_root(ratio, qn - p)
        if r_rec != r:
            continue
        a1_rec = sp.Rational(v1, r**(p - 1))
        if a1_rec != a1:
            continue
        qkind = random.choice(["term", "sum"])
        t = random.randint(4, 9)
        if qkind == "term":
            ans = a1 * r**(t - 1)
            query = f"$a_{{{t}}}$의 값은?"
            qtail = f"$a_{{{t}}}=a\\,r^{{{t-1}}}={ans}$"
        else:
            ans = a1 * (r**t - 1) // (r - 1)
            query = f"$\\displaystyle\\sum_{{i=1}}^{{{t}}} a_i$의 값은?"
            qtail = f"$S_{{{t}}}=\\dfrac{{a(r^{{{t}}}-1)}}{{r-1}}={ans}$"
        if abs(ans) > 9000:
            continue
        labels = ["가", "나"]
        cond = f"(가) $a_{{{p}}}={v1}$\n(나) $a_{{{qn}}}={v2}$"
        q = (f"공비가 양수인 등비수열 $\\{{a_n\\}}$이 다음 조건을 만족시킬 때, "
             f"{query}\n\n{cond}\n\n(단, $S_n=\\sum_{{i=1}}^{{n}} a_i$)")
        e = (f"1단계: (가)(나)에서 $\\dfrac{{a_{{{qn}}}}}{{a_{{{p}}}}}=r^{{{qn-p}}}"
             f"=\\dfrac{{{v2}}}{{{v1}}}$, 공비가 양수이므로 $r={r}$.\n"
             f"2단계: $a={a1}$.\n3단계: {qtail}.\n\n")
        choices, lab = make_mc(sp.Integer(ans),
                               [ans*r, ans//r if ans % r == 0 else ans+r,
                                ans+v1, ans-v1, -ans])
        e += f"정답: {lab}"
        rr = _rec("대수", "조건결합형 등비수열(메타)", "상", 4, "객관식",
                  q, choices, lab, e)
        rr["_meta"] = {"chk": "meta_gseq", "blocks": ["term", "term"],
                       "a1": a1, "r": r, "qkind": qkind, "t": t, "ans": int(ans)}
        return rr
    return None


# ======================================================================
#  도메인 F : trig_determine  f(x)=a sin(bx)+c (a>0,b>0), max·min·주기로 결정
# ======================================================================
def compose_trig_determine(n_blocks=3, max_retries=40):
    """삼각함수 f(x)=a sin(bx)+c 를 최댓값·최솟값·주기로 결정 → a+b+c (중상)."""
    for _ in range(max_retries):
        a = random.choice([1, 2, 3])
        b = random.choice([1, 2, 3])
        c = random.randint(-3, 3)
        M = a + c
        m = c - a
        ans = a + b + c
        period = 2*sp.pi / b
        labels = ["가", "나", "다", "라"]
        cond = (f"(가) $f(x)$의 최댓값은 ${M}$이다.\n"
                f"(나) $f(x)$의 최솟값은 ${m}$이다.\n"
                f"(다) $f(x)$의 주기는 ${_ltx(period)}$이다.")
        q = (f"함수 $f(x)=a\\sin(bx)+c$가 다음 조건을 만족시킬 때, $a+b+c$의 값은? "
             f"(단, $a>0$, $b$는 자연수)\n\n{cond}")
        e = (f"1단계: 최댓값 $a+c={M}$, 최솟값 $-a+c={m}$ 에서 $a={a}$, $c={c}$.\n"
             f"2단계: 주기 $\\dfrac{{2\\pi}}{{b}}={_ltx(period)}$ 에서 $b={b}$.\n"
             f"3단계: $a+b+c={a}+{b}+({c})={ans}$.\n\n")
        choices, lab = make_mc(sp.Integer(ans),
                               [ans-1, ans+1, ans-2, ans+2, -ans])
        e += f"정답: {lab}"
        rr = _rec("대수", "삼각함수 결정(최대·최소·주기)", "중", 3, "객관식",
                  q, choices, lab, e)
        rr["_meta"] = {"chk": "meta_trigdet", "blocks": ["max", "min", "period"],
                       "a": a, "b": b, "c": c, "ans": ans}
        return rr
    return None


# ======================================================================
#  도메인 E : integer_solutions  부정방정식 x1+..+xk=N 해의 개수 (브루트포스)
# ======================================================================
_VARNAMES = ["x", "y", "z", "w"]


def _is_blocks(k, N):
    """변수 k개·합 N 에 대한 제약 블록 풀 → [(name, text, pred, step, mt)]."""
    V = _VARNAMES[:k]
    pool = []
    i = random.randrange(k)
    U = random.randint(3, 6)
    pool.append(("upper_bound", f"${V[i]}\\le {U}$",
                 (lambda s, i=i, U=U: s[i] <= U),
                 f"${V[i]}\\le {U}$", {"i": i, "U": U}))
    i = random.randrange(k)
    L = random.randint(1, 3)
    pool.append(("lower_bound", f"${V[i]}\\ge {L}$",
                 (lambda s, i=i, L=L: s[i] >= L),
                 f"${V[i]}\\ge {L}$", {"i": i, "L": L}))
    i = random.randrange(k)
    par = random.choice(["even", "odd"])
    word = "짝수" if par == "even" else "홀수"
    pool.append(("parity", f"${V[i]}$는 {word}이다.",
                 (lambda s, i=i, par=par: (s[i] % 2 == 0) if par == "even" else (s[i] % 2 == 1)),
                 f"${V[i]}$가 {word}", {"i": i, "par": par}))
    a, b = random.sample(range(k), 2)
    pool.append(("order", f"${V[a]}\\le {V[b]}$",
                 (lambda s, a=a, b=b: s[a] <= s[b]),
                 f"${V[a]}\\le {V[b]}$", {"a": a, "b": b}))
    i = random.randrange(k)
    c = random.randint(1, min(5, N - 1))
    pool.append(("exact_value", f"${V[i]}={c}$",
                 (lambda s, i=i, c=c: s[i] == c),
                 f"${V[i]}={c}$", {"i": i, "c": c}))
    a2, b2 = random.sample(range(k), 2)
    cc = random.randint(2, max(2, N - 2))
    pool.append(("pair_sum", f"${V[a2]}+{V[b2]}\\le {cc}$",
                 (lambda s, a=a2, b=b2, cc=cc: s[a] + s[b] <= cc),
                 f"${V[a2]}+{V[b2]}\\le {cc}$", {"a": a2, "b": b2, "c": cc}))
    im = random.randrange(k)
    mm = random.choice([2, 3])
    pool.append(("multiple_of", f"${V[im]}$는 ${mm}$의 배수이다.",
                 (lambda s, i=im, m=mm: s[i] % m == 0),
                 f"${V[im]}$는 ${mm}$의 배수", {"i": im, "m": mm}))
    so, sb = random.sample(range(k), 2)
    pool.append(("strict_order", f"${V[so]}<{V[sb]}$",
                 (lambda s, a=so, b=sb: s[a] < s[b]),
                 f"${V[so]}<{V[sb]}$", {"a": so, "b": sb}))
    return pool


def _is_pred_from(name, mt):
    if name == "upper_bound":
        return lambda s, i=mt["i"], U=mt["U"]: s[i] <= U
    if name == "lower_bound":
        return lambda s, i=mt["i"], L=mt["L"]: s[i] >= L
    if name == "parity":
        return lambda s, i=mt["i"], par=mt["par"]: (
            (s[i] % 2 == 0) if par == "even" else (s[i] % 2 == 1))
    if name == "order":
        return lambda s, a=mt["a"], b=mt["b"]: s[a] <= s[b]
    if name == "exact_value":
        return lambda s, i=mt["i"], c=mt["c"]: s[i] == c
    if name == "pair_sum":
        return lambda s, a=mt["a"], b=mt["b"], c=mt["c"]: s[a] + s[b] <= c
    if name == "multiple_of":
        return lambda s, i=mt["i"], m=mt["m"]: s[i] % m == 0
    if name == "strict_order":
        return lambda s, a=mt["a"], b=mt["b"]: s[a] < s[b]
    raise AssertionError(name)


def compose_integer_solutions(n_blocks=2, max_retries=50, count_range=(4, 250)):
    """부정방정식 해의 개수 도메인. 합=N(기본) + 제약 2~3개의 AND 브루트포스."""
    lo, hi = count_range
    for _ in range(max_retries):
        k = random.choice([3, 4])
        N = random.randint(8, 15)
        V = _VARNAMES[:k]
        sols = [s for s in _iproduct(range(N + 1), repeat=k) if sum(s) == N]
        pool = _is_blocks(k, N)
        chosen = random.sample(pool, min(n_blocks, len(pool)))
        names = [c[0] for c in chosen]
        texts = [c[1] for c in chosen]
        preds = [c[2] for c in chosen]
        steps = [c[3] for c in chosen]
        metas = {c[0]: c[4] for c in chosen}
        cnt = sum(1 for s in sols if all(p(s) for p in preds))
        if not (lo <= cnt <= hi):
            continue
        # 함정: 제약 하나(마지막)를 빠뜨린 개수
        naive = sum(1 for s in sols if all(p(s) for p in preds[:-1]))
        labels = ["가", "나", "다", "라"]
        cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(len(texts)))
        vlist = ",\\,".join(V)
        q = (f"방정식 ${'+'.join(V)}={N}$을 만족시키는 음이 아닌 정수 "
             f"${vlist}$의 순서쌍 $({','.join(V)})$ 중 다음 조건을 모두 "
             f"만족시키는 것의 개수는?\n\n{cond_lines}")
        step_lines = "\n".join(f"{i+1}단계: ({labels[i]}) {steps[i]} 조건을 적용한다."
                               for i in range(len(steps)))
        e = (f"{step_lines}\n"
             f"{len(steps)+1}단계: 조건을 모두 만족하는 순서쌍을 전수 조사로 세면 "
             f"${cnt}$개이다.\n\n")
        dd = [d for d in [cnt-2, cnt+2, cnt-4, cnt+4, cnt+6] if d > 0]
        traps = {}
        if naive != cnt and naive > 0:
            choices, lab, traps = make_mc_tagged(
                sp.Integer(cnt),
                [(sp.Integer(naive), f"조건 ({labels[len(steps)-1]}) 하나를 빠뜨리고 셈")] + dd)
        else:
            choices, lab = make_mc(sp.Integer(cnt), dd)
        e += f"정답: {lab}"
        rr = _rec("확률과 통계", "조건결합형 부정방정식 해의 개수(메타)", "상", 4,
                  "객관식", q, choices, lab, e)
        if traps:
            rr["completion"]["trap_map"] = traps
        rr["_meta"] = {"chk": "meta_intsol", "blocks": sorted(names),
                       "k": k, "N": N, "params": metas, "ans": cnt}
        return rr
    return None


def compose_intsol_prob(n_blocks=2, max_retries=60, prob_range=(sp.Rational(1, 20),
                                                                 sp.Rational(9, 10))):
    """부정방정식 해 + 확률 (킬러, 상). 이해원 N제 확통 KEY 유형에서 채굴.

    'x1+..+xk=N 의 음이 아닌 정수해 중 임의로 하나를 택할 때 조건을 만족시킬 확률'.
    정답 = (조건 만족 해 수)/(전체 해 수) 를 브루트포스로 둘 다 세어 100% 보장.
    """
    from math import comb
    lo, hi = prob_range
    for _ in range(max_retries):
        k = random.choice([3, 4])
        N = random.randint(6, 14)
        V = _VARNAMES[:k]
        sols = [s for s in _iproduct(range(N + 1), repeat=k) if sum(s) == N]
        total = len(sols)
        assert total == comb(N + k - 1, k - 1)        # 자가검증(중복조합 공식)
        pool = _is_blocks(k, N)
        chosen = random.sample(pool, min(n_blocks, len(pool)))
        names = [c[0] for c in chosen]
        texts = [c[1] for c in chosen]
        preds = [c[2] for c in chosen]
        steps = [c[3] for c in chosen]
        metas = {c[0]: c[4] for c in chosen}
        fav = sum(1 for s in sols if all(p(s) for p in preds))
        ans = sp.Rational(fav, total)
        if not (lo <= ans <= hi):
            continue
        labels = ["가", "나", "다", "라"]
        cond_lines = "\n".join(f"({labels[i]}) {texts[i]}" for i in range(len(texts)))
        q = (f"방정식 ${'+'.join(V)}={N}$을 만족시키는 음이 아닌 정수 "
             f"${','.join(V)}$의 모든 순서쌍 $({','.join(V)})$ 중에서 임의로 한 개를 "
             f"택할 때, 택한 순서쌍이 다음 조건을 모두 만족시킬 확률은?\n\n{cond_lines}")
        step_lines = "\n".join(f"{i+2}단계: ({labels[i]}) {steps[i]} 조건을 적용한다."
                               for i in range(len(steps)))
        e = (f"1단계: 전체 순서쌍의 수는 ${{}}_{{{k}}}\\mathrm{{H}}_{{{N}}}"
             f"=\\binom{{{N + k - 1}}}{{{k - 1}}}={total}$이다.\n"
             f"{step_lines}\n"
             f"{len(steps)+2}단계: 조건을 모두 만족하는 순서쌍은 ${fav}$개이므로 "
             f"확률은 $\\dfrac{{{fav}}}{{{total}}}={_ltx(ans)}$이다.\n\n")
        dd = []
        for f2 in [fav - 1, fav + 1, fav - 2, fav + 2, fav + 3]:
            if 0 < f2 < total:
                dd.append(sp.Rational(f2, total))
        choices, lab = make_mc(ans, dd)
        e += f"정답: {lab}"
        rr = _rec("확률과 통계", "조건결합형 부정방정식과 확률(메타)", "상", 4,
                  "객관식", q, choices, lab, e)
        rr["_meta"] = {"chk": "meta_intsolprob", "blocks": sorted(names),
                       "k": k, "N": N, "params": metas,
                       "fav": fav, "total": total, "ans": [ans.p, ans.q]}
        return rr
    return None


# ======================================================================
#  도메인 G : fusion  서로 다른 두 개념을 융합한 복합 킬러 (차별화 핵심)
#  각 하위문제는 엔진/메타 검증 생성기가 산출 → 정수 정답 a1,a2 결합(p+q).
#  정답은 각 부분의 검증에서 상속, 결합은 산술. 유형=생성기쌍 → 조합 폭발.
# ======================================================================
import re as _re_fuse

_FUSE_META = ["cubic", "quartic", "set_function", "arith_seq", "geom_seq",
              "trig_determine", "integer_solutions"]
_FUSE_ENG_CACHE = None


def _int_answer(r):
    """레코드의 정답을 정수로 추출. 정수형이 아니면 None."""
    a = r["completion"]["answer"]
    m = _re_fuse.match(r"^\d+번 \(\$(-?\d+)\$\)$", a)
    if m:
        return int(m.group(1))
    if _re_fuse.fullmatch(r"-?\d+", a):
        return int(a)
    return None


def _fuse_eng_pool():
    """problem_engine 생성기 중 '정수답을 안정적으로 내는' 것을 자동 발견(캐시).

    하드코딩 대신 전 생성기를 표집해 정수답(|값|≤300)을 신뢰성 있게 내는 것만
    융합 풀로 채택 → 엔진이 커질수록 융합 다양성이 자동으로 따라 커진다.
    """
    global _FUSE_ENG_CACHE
    if _FUSE_ENG_CACHE is not None:
        return _FUSE_ENG_CACHE
    from csat_forge import problem_engine as _pe
    good = []
    for nm, fn in _pe.GENERATORS.items():
        oks = 0
        try:
            for _ in range(6):
                r = fn()
                if r["completion"].get("figure"):     # 그림형 제외
                    oks = 0
                    break
                a = _int_answer(r)
                if a is not None and abs(a) <= 300:
                    oks += 1
        except Exception:
            continue
        if oks >= 5:                                   # 6회 중 5회+ 정수답 → 채택
            good.append(nm)
    _FUSE_ENG_CACHE = good
    return good


def _fuse_combine(parts, op, coeffs):
    """결합 연산 적용 → 정답."""
    if op == "sum":
        return sum(parts)
    if op == "product":
        v = 1
        for p in parts:
            v *= p
        return v
    return sum(c*p for c, p in zip(coeffs, parts))   # weighted


def compose_fusion(n_blocks=2, max_retries=60, min_scopes=2, meta_only=False):
    """서로 다른 2~3개념의 하위문제를 융합 → 합·곱·가중 결합 (완전킬러).

    각 하위문제는 엔진/메타 검증 생성기가 산출(정수 정답) → 결합값이 최종 정답.
    정답은 각 부분 검증에서 상속, 결합은 산술 검산. 22·30번급 다개념 복합 킬러.
    min_scopes: 서로 다른 단원 수의 최소값(3이면 대수·미적분Ⅰ·확통 모두 강제).
    """
    from csat_forge import problem_engine as _pe
    eng_pool = _fuse_eng_pool()                       # 자동 발견된 정수답 생성기 전체
    PV = ["p", "q", "r", "s", "t"]
    LB = ["P", "Q", "R", "S", "T"]

    def gen_one():
        # 메타(전부 상·킬러) 비중을 높여 하위문제 난도를 끌어올림.
        # meta_only면 하위문제도 전부 조건결합형 킬러로 강제(극상위용).
        if meta_only or random.random() < 0.55:
            d = random.choice(_FUSE_META)
            r = generate_meta(domain=d)
            src = f"meta_{d}"
        else:
            nm = random.choice(eng_pool)
            try:
                r = _pe.GENERATORS[nm]()
            except Exception:
                return None
            r["generator"] = nm
            src = nm
            if r is not None and r["spec"].get("difficulty") == "하":
                return None                       # 시시한 하위문제 배제(변별력↑)
        if r is None or not _pe.verify(r):
            return None
        a = _int_answer(r)
        if a is None or abs(a) > 300:
            return None
        return r, a, src

    K = min(n_blocks, 5) if (n_blocks and n_blocks >= 3) else random.choice([2, 2, 3])
    if min_scopes >= 3:
        K = max(K, 3)                              # 3개 단원 모두 담으려면 ≥3
    for _ in range(max_retries):
        picks = [gen_one() for _ in range(K)]
        if any(p is None for p in picks):
            continue
        recs = [p[0] for p in picks]
        parts = [p[1] for p in picks]
        srcs = [p[2] for p in picks]
        if len(set(srcs)) != K:                   # 전부 서로 다른 생성기
            continue
        if len(set(r["spec"]["scope"] for r in recs)) < min_scopes:  # 서로 다른 단원 강제(교차개념)
            continue
        # 결합 연산 선택 (곱은 절댓값 작을 때만)
        if K == 2 and all(abs(a) <= 20 for a in parts) and 0 not in parts and random.random() < 0.3:
            op, coeffs = "product", [1, 1]
        elif random.random() < 0.4:
            op = "weighted"
            coeffs = [random.choice([2, 3]) for _ in range(K)]
            # 일부 계수를 음수로 → 부호 처리 실수 유발(변별력↑)
            if random.random() < 0.5:
                coeffs[random.randrange(K)] *= -1
        else:
            op, coeffs = "sum", [1]*K
        ans = _fuse_combine(parts, op, coeffs)
        if abs(ans) > 999:
            continue
        # 결합식 표기
        if op == "product":
            comb_txt = "\\times ".join(PV[:K])
        elif op == "weighted":
            comb_txt = "".join(
                (("+" if coeffs[i] > 0 else "-") if i else ("" if coeffs[i] > 0 else "-"))
                + f"{abs(coeffs[i])}{PV[i]}" for i in range(K))
        else:
            comb_txt = "+".join(PV[:K])
        defs = "\n\n".join(
            f"[{LB[i]}] {recs[i]['completion']['question'].strip()}\n"
            f"   (위 물음의 답을 ${PV[i]}$라 한다.)" for i in range(K))
        q = (f"{K}개의 정수 ${', '.join(PV[:K])}$를 다음과 같이 정할 때, "
             f"${comb_txt}$의 값은?\n\n{defs}")
        step_parts = "\n".join(f"{i+1}단계: [{LB[i]}]를 풀면 ${PV[i]}={parts[i]}$."
                               for i in range(K))
        e = (f"{step_parts}\n"
             f"{K+1}단계: ${comb_txt}={ans}$.\n\n")
        # 변별력: 흔한 결합 실수를 함정 오답으로
        def _pl(v):
            return abs(v) <= 999 and v != ans and abs(v - ans) <= max(12, abs(ans))
        traps_in, seen = [], set()
        drop = _fuse_combine(parts[:-1], op, coeffs[:-1])      # 마지막 항 누락
        if _pl(drop) and drop not in seen:
            traps_in.append((sp.Integer(drop), f"항 ${PV[K-1]}$를 빠뜨리고 결합"))
            seen.add(drop)
        if op != "sum":                                        # 가중·곱을 단순합으로 오인
            plain = sum(parts)
            if _pl(plain) and plain not in seen:
                traps_in.append((sp.Integer(plain), "결합 규칙을 무시하고 단순히 더함"))
                seen.add(plain)
        if op in ("sum", "weighted"):                          # 한 항의 부호 실수
            for i in range(K):
                flip = ans - 2*coeffs[i]*parts[i]
                if _pl(flip) and flip not in seen:
                    traps_in.append((sp.Integer(flip), f"항 ${PV[i]}$의 부호 실수"))
                    seen.add(flip)
                    break
        d = max(2, abs(ans) // 10 or 2)
        pad = [c for c in (ans-d, ans+d, ans-2*d, ans+2*d, -ans)
               if c != ans and c not in seen]
        choices, lab, traps = make_mc_tagged(sp.Integer(ans), traps_in + pad)
        e += f"정답: {lab}"
        scopes = sorted(set(r["spec"]["scope"] for r in recs))
        scope = "+".join(scopes)
        rr = _rec(scope, "복합 개념 융합 킬러(메타)", "상", 4, "객관식",
                  q, choices, lab, e)
        if traps:
            rr["completion"]["trap_map"] = traps
        rr["_meta"] = {"chk": "meta_fusion", "blocks": srcs, "parts": parts,
                       "op": op, "coeffs": coeffs, "k": K, "ans": ans}
        return rr
    return None


# ======================================================================
#  도메인 H : chain  검증 킬러를 풀어 답 N → N에 정수 변환 (2단계 연쇄 킬러)
#  30번급 다단계 구조. 1단계는 68개 융합풀의 검증 생성기, 2단계는 정수 변환.
# ======================================================================
def _chain_transform(p, kind, m=None):
    import math
    from math import comb
    if kind == "divisors":
        return sum(1 for i in range(1, p+1) if p % i == 0)
    if kind == "mod":
        return p % m
    if kind == "digitsum":
        return sum(int(c) for c in str(abs(p)))
    if kind == "units":
        return p % 10
    if kind == "floorsqrt":
        return math.isqrt(p)
    if kind == "popcount":
        return bin(p).count("1")
    if kind == "triangular":
        return p*(p+1)//2
    if kind == "comb_solutions":
        return comb(p+2, 2)
    if kind == "divisor_sum":
        return sum(i for i in range(1, p+1) if p % i == 0)
    if kind == "totient":
        return sum(1 for k in range(1, p+1) if math.gcd(p, k) == 1)
    if kind == "choose3":
        return comb(p, 3)
    if kind == "pair_count":            # 서로 다른 p개에서 2개 택하는 수
        return comb(p, 2)
    if kind == "sum_squares":           # 1²+2²+…+p²
        return p*(p+1)*(2*p+1)//6
    if kind == "digit_product":         # 각 자리 숫자의 곱
        v = 1
        for c in str(abs(p)):
            v *= int(c)
        return v
    if kind == "deriv_cubic":           # f(x)=x³의 x=p 미분계수 = 3p² (미적분)
        return 3 * p * p
    if kind == "integral_2x":           # ∫_0^p 2x dx = p² (미적분)
        return p * p
    if kind == "geom_pow":              # 2⁰+2¹+…+2^(p-1) = 2^p − 1 (대수·등비합)
        return 2**p - 1
    if kind == "prime_factors":
        v, ps = p, set()
        d = 2
        while d*d <= v:
            while v % d == 0:
                ps.add(d); v //= d
            d += 1
        if v > 1:
            ps.add(v)
        return len(ps)
    raise AssertionError(kind)


def _chain_phrase(kind, v):
    """변환 1개를 변수 v(LaTeX 토큰)에 대한 '명사구'로 — 다단계 연쇄 진술용."""
    return {
        "divisors": (f"${v}$의 양의 약수의 개수", "는?"),
        "digitsum": (f"${v}$의 모든 자리 숫자의 합", "은?"),
        "units": (f"${v}$의 일의 자리 숫자", "는?"),
        "floorsqrt": (f"$\\lfloor\\sqrt{{{v}}}\\rfloor$", "의 값은?"),
        "popcount": (f"${v}$을 $2$진법으로 나타낼 때 숫자 $1$의 개수", "는?"),
        "triangular": (f"$1+2+\\cdots+{v}$", "의 값은?"),
        "comb_solutions": (f"방정식 $x+y+z={v}$를 만족시키는 음이 아닌 "
                           f"정수해 $(x,y,z)$의 개수", "는?"),
        "divisor_sum": (f"${v}$의 모든 양의 약수의 합", "은?"),
        "totient": (f"${v}$ 이하의 자연수 중 ${v}$과 서로소인 것의 개수", "는?"),
        "choose3": (f"서로 다른 ${v}$개에서 $3$개를 택하는 경우의 수", "는?"),
        "pair_count": (f"서로 다른 ${v}$개에서 $2$개를 택하는 경우의 수", "는?"),
        "sum_squares": (f"$1^2+2^2+\\cdots+{v}^2$", "의 값은?"),
        "digit_product": (f"${v}$의 모든 자리 숫자의 곱", "은?"),
        "deriv_cubic": (f"함수 $f(x)=x^3$의 $x={v}$에서의 미분계수", "는?"),
        "integral_2x": (f"$\\displaystyle\\int_0^{{{v}}} 2x\\,dx$", "의 값은?"),
        "geom_pow": (f"$2^0+2^1+\\cdots+2^{{{v}-1}}$", "의 값은?"),
        "prime_factors": (f"${v}$의 서로 다른 소인수의 개수", "는?"),
    }[kind]


def _valid_transforms(v):
    """현재 값 v에서 의미 있는(정의역에 맞는) 변환 종류 목록."""
    valid = ["divisors", "digitsum", "units", "floorsqrt", "popcount",
             "prime_factors"]
    if v <= 30:
        valid.append("triangular")
    if v <= 25:
        valid.append("comb_solutions")
    if v <= 60:
        valid.append("divisor_sum")
    if v <= 80:
        valid.append("totient")
    if 3 <= v <= 14:
        valid.append("choose3")
    if 2 <= v <= 45:                 # comb(45,2)=990 ≤ 999
        valid.append("pair_count")
    if 2 <= v <= 13:                 # 1²+…+13²=819 ≤ 999
        valid.append("sum_squares")
    if v >= 10:                      # 자릿수 곱은 두 자리 이상일 때만 의미
        valid.append("digit_product")
    if 2 <= v <= 18:                 # 3·18²=972 ≤ 999
        valid.append("deriv_cubic")
    if 2 <= v <= 31:                 # 31²=961 ≤ 999
        valid.append("integral_2x")
    if 2 <= v <= 9:                  # 2⁹−1=511 ≤ 999
        valid.append("geom_pow")
    return valid


_CHAIN_VARS = ["N", "M", "L", "K", "J"]
# 비최종 단계 변환 우선순위 가중(클수록 값 보존/증가 → 연쇄가 시시하게 붕괴하지 않음)
_CHAIN_W = {
    "triangular": 5, "comb_solutions": 5, "choose3": 5, "sum_squares": 5,
    "geom_pow": 5, "deriv_cubic": 5, "integral_2x": 5,
    "pair_count": 4, "divisor_sum": 4,
    "totient": 3, "floorsqrt": 2, "digitsum": 2,
    "divisors": 1, "prime_factors": 1, "popcount": 1, "units": 1,
    "digit_product": 1,
}


def _chain_on(r, src, depth, topic="다단계 연쇄 킬러(메타)", lead_extra=None):
    """주어진 기반 레코드 r의 정수 답 N에 정수 변환을 depth회 '순차' 적용해
    새 연쇄 킬러 레코드를 만든다(공용 헬퍼). 실패 시 None."""
    from csat_forge import problem_engine as _pe
    if r is None or not _pe.verify(r):
        return None
    N0 = _int_answer(r)
    if N0 is None or not (2 <= N0 <= 400):
        return None
    stages, cur, ok = [], N0, True
    for st in range(depth):
        last = (st == depth - 1)
        # 비최종 단계는 값이 커지거나 유지되는 변환을 우선(붕괴 방지 → 변별력↑)
        cands = _valid_transforms(cur)
        # 곱셈형 가중 랜덤: 높은 가중을 선호하되 분포가 겹쳐 다양성 유지
        cands.sort(key=lambda k: -(_CHAIN_W.get(k, 1) * (0.2 + random.random())))
        picked = None
        # 1차: 비최종은 '시시하지 않은'(≥6) 중간값을 요구
        for strict in (True, False):
            for kind in cands:
                nxt = _chain_transform(cur, kind)
                if nxt == cur:
                    continue
                if last:
                    if 0 <= nxt <= 999:
                        picked = (kind, nxt)
                        break
                else:
                    lo = 6 if strict else 2
                    if lo <= nxt <= 400:
                        picked = (kind, nxt)
                        break
            if picked is not None or last:
                break
        if picked is None:
            ok = False
            break
        stages.append((picked[0], cur, picked[1]))
        cur = picked[1]
    if not ok:
        return None
    ans = cur
    q1 = r["completion"]["question"].strip()
    lead = lead_extra or "다음"
    lines = [f"{lead} [문제]의 답을 ${_CHAIN_VARS[0]}$이라 하자."]
    steps = [f"1단계: [문제]를 풀면 ${_CHAIN_VARS[0]}={N0}$이다."]

    def _eq(pv, aft):
        # pv가 수식($...$)으로 끝나면 닫는 $를 떼고 '=aft$'를 붙여 $$ 인접 방지
        return (pv[:-1] + f"={aft}$") if pv.endswith("$") else (pv + f"$={aft}$")
    for i, (kind, bef, aft) in enumerate(stages):
        invar = _CHAIN_VARS[i]
        phrase, qsuf = _chain_phrase(kind, invar)
        pv, _ = _chain_phrase(kind, str(bef))
        if i < len(stages) - 1:
            outvar = _CHAIN_VARS[i + 1]
            lines.append(f"{phrase}을(를) ${outvar}$(이)라 하자.")
            steps.append(f"{i+2}단계: ${invar}={bef}$이므로 {_eq(pv, aft)}, "
                         f"즉 ${outvar}={aft}$.")
        else:
            lines.append(f"이때 {phrase}{qsuf}")
            steps.append(f"{i+2}단계: ${invar}={bef}$이므로 {_eq(pv, aft)}.")
    q = " ".join(lines) + f"\n\n[문제] {q1}"
    e = "\n".join(steps) + "\n\n"
    # 변별력: 흔한 실수를 함정 오답으로(정답 근처의 타당한 값만)
    def _plausible(v):
        return 0 <= v <= 999 and v != ans and abs(v - ans) <= max(10, ans)
    traps_in, seen = [], set()
    last_kind, early = stages[-1][0], stages[-1][1]   # 마지막 변환 종류·직전 값
    # 마지막에 '다른 변환'을 적용하는 실수(킬러 최빈 오답)
    for alt in _valid_transforms(early):
        if alt == last_kind:
            continue
        av = _chain_transform(early, alt)
        if _plausible(av) and av not in seen:
            traps_in.append((sp.Integer(av), f"마지막에 다른 연산({alt})을 적용"))
            seen.add(av)
            break
    if _plausible(early) and early not in seen:       # 마지막 변환을 빠뜨림
        traps_in.append((sp.Integer(early), "마지막 변환을 빠뜨리고 멈춤"))
        seen.add(early)
    # 중간 단계값에서 멈추는 실수(다단계 최빈) — 정답에 가까운 중간값부터
    for si in range(len(stages) - 2, -1, -1):
        mid = stages[si][2]
        if _plausible(mid) and mid not in seen:
            traps_in.append((sp.Integer(mid),
                             f"{si+2}단계(${_CHAIN_VARS[si+1]}$)에서 멈춤"))
            seen.add(mid)
    if _plausible(N0) and N0 not in seen:             # 변환 전 N을 그대로 답
        traps_in.append((sp.Integer(N0), "변환 전 값 $N$을 그대로 답함"))
        seen.add(N0)
    d = max(1, abs(ans) // 8 or 1)
    pad = [c for c in (ans-d, ans+d, ans+2*d, ans+3*d, ans-2*d)
           if c >= 0 and c != ans and c not in seen]
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), traps_in + pad)
    e += f"정답: {lab}"
    rr = _rec(r["spec"]["scope"], topic, "상", 4, "객관식", q, choices, lab, e)
    if traps:
        rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "meta_chain", "blocks": [src], "N": N0,
                   "stages": [(k, b, a) for (k, b, a) in stages],
                   "depth": depth, "kind": stages[-1][0], "ans": ans}
    return rr


def compose_chain(n_blocks=2, max_retries=80, depth=None):
    """다단계 연쇄 킬러: 검증 킬러의 정수 답 N → 정수 변환을 depth회 '순차' 적용.

    1단계(검증 생성기, 또는 융합)에서 정답 $N$을 구하고, 이후 각 단계에서 직전
    값의 약수 개수·자릿수 합·삼각수·약수합 등을 차례로 구한다(값을 $M, L,\\dots$로
    명명). 모든 단계가 정수 산술이라 정답 100% 보장. depth가 길수록 진짜 킬러처럼
    추론 단계가 늘어난다(기본 2~3단계).
    """
    from csat_forge import problem_engine as _pe
    eng_pool = _fuse_eng_pool()
    D = depth if depth else random.choice([2, 2, 3])   # 기본 다단계
    for _ in range(max_retries):
        # 1단계: 기반 킬러(약 1/4 확률로 '융합'을 기반 → 다개념 + 다단계)
        roll = random.random()
        if roll < 0.22:
            r = compose_fusion()
            src = "meta_fusion"
        elif roll < 0.55:
            d = random.choice(_FUSE_META)
            r = generate_meta(domain=d)
            src = f"meta_{d}"
        else:
            nm = random.choice(eng_pool)
            try:
                r = _pe.GENERATORS[nm]()
            except Exception:
                continue
            r["generator"] = nm
            src = nm
        rr = _chain_on(r, src, D)
        if rr is not None:
            return rr
    return None


def compose_mega(n_blocks=3, max_retries=120, n_concepts=3, depth=None):
    """초통합 킬러: 한 문제에 서로 다른 단원 n_concepts개를 '융합'한 뒤,
    그 결과값에 다시 '다단계 연쇄'를 적용한다 (개념 다수 혼합 + 다단계).

    구조: [P]·[Q]·[R](서로 다른 단원의 검증 하위문제) 를 풀어 결합값 $V$를 얻고,
    $V$를 $N$으로 삼아 약수 개수·자릿수 합 등 변환을 순차로 적용. 모든 부분이
    검증/정수 산술이라 정답 100% 보장. 평가원의 '과목 간 통합' 방향의 최상위형.
    """
    D = depth if depth else random.choice([2, 3])
    for _ in range(max_retries):
        base = compose_fusion(n_blocks=max(3, n_concepts), min_scopes=n_concepts)
        if base is None:
            continue
        nsc = len(base["_meta"].get("blocks", []))
        rr = _chain_on(base, "meta_fusion", D,
                       topic="초통합 다개념 연쇄 킬러(메타)",
                       lead_extra="서로 다른 단원의 세 물음으로 이루어진")
        if rr is not None:
            rr["_meta"]["mega"] = True
            rr["_meta"]["fusion_scopes"] = base["spec"]["scope"]
            rr["_meta"]["fusion_k"] = nsc
            # scope를 융합 단원 전체로 표기(통합 가시화)
            rr["spec"]["scope"] = base["spec"]["scope"]
            return rr
    return None


# ======================================================================
#  도메인 I : interlock  1단계 답 N이 2단계 함수의 '파라미터'로 전달되는
#  진짜 순차 의존형 킬러(30번급). 2단계는 미적분(극값·대칭적분)으로 N에서 정확 계산.
# ======================================================================
def _interlock_stage2(N):
    """N을 파라미터로 받는 2단계 문제 1개(미적분·확통·대수).
    반환 (질의구, 질의접미, 정답, 해설, 메타{kind, unit, ...})."""
    from math import comb
    x = sp.Symbol("x")
    kind = random.choice(["extr_diff", "extr_max", "sym_integral", "tangent_area",
                          "comb_count", "binom_coeff", "geom_term"])
    # --- 확률과 통계: N개에서 2개 택하는 경우의 수 ---
    if kind == "comb_count":
        ans = comb(N + 3, 2)
        q2 = "서로 다른 $(N+3)$개에서 $2$개를 택하는 경우의 수"
        qsuf = "는?"
        e2 = (f"$_{{N+3}}\\mathrm{{C}}_2=\\dfrac{{(N+3)(N+2)}}{{2}}$. "
              f"$N={N}$이면 $_{{{N+3}}}\\mathrm{{C}}_2={ans}$.")
        return q2, qsuf, ans, e2, {"kind": kind, "unit": "확률과 통계"}
    # --- 확률과 통계: 이항정리 계수 ---
    if kind == "binom_coeff":
        ans = comb(N + 2, 2)
        q2 = "다항식 $(x+1)^{N+2}$의 전개식에서 $x^2$의 계수"
        qsuf = "는?"
        e2 = (f"$x^2$의 계수는 $_{{N+2}}\\mathrm{{C}}_2$. "
              f"$N={N}$이면 $_{{{N+2}}}\\mathrm{{C}}_2={ans}$.")
        return q2, qsuf, ans, e2, {"kind": kind, "unit": "확률과 통계"}
    # --- 대수: 등비수열 제3항 ---
    if kind == "geom_term":
        ans = 2 * N**2
        q2 = "첫째항이 $2$, 공비가 $N$인 등비수열 $\\{a_n\\}$의 제$3$항 $a_3$"
        qsuf = "은?"
        e2 = (f"$a_3=2\\cdot N^{{2}}$. $N={N}$이면 ${ans}$.")
        return q2, qsuf, ans, e2, {"kind": kind, "unit": "대수"}
    if kind == "extr_diff":
        g = x**3 - 3*N**2*x
        crit = sp.solve(sp.diff(g, x), x)
        vals = [g.subs(x, c) for c in crit]
        ans = int(max(vals) - min(vals))               # 4N³
        q2 = "함수 $g(x)=x^3-3N^2x$의 극댓값과 극솟값의 차"
        qsuf = "는?"
        e2 = ("$g'(x)=3x^2-3N^2=3(x-N)(x+N)$이므로 $x=\\pm N$에서 극값을 갖는다. "
              f"극댓값 $g(-N)=2N^3$, 극솟값 $g(N)=-2N^3$이므로 차는 $4N^3$. "
              f"$N={N}$이면 $4\\cdot{N}^3={ans}$.")
    elif kind == "extr_max":
        ans = 2*N**3
        q2 = "함수 $g(x)=x^3-3N^2x$의 극댓값"
        qsuf = "은?"
        e2 = (f"$x=\\pm N$에서 극값. 극댓값 $g(-N)=2N^3$. $N={N}$이면 ${ans}$.")
    elif kind == "sym_integral":
        c0 = random.choice([3, 4, 5, 6])
        ans = 2*c0*N
        q2 = (f"$\\displaystyle\\int_{{-N}}^{{N}} (x^3-3N^2x+{c0})\\,dx$의 값")
        qsuf = "은?"
        e2 = (f"$x^3-3N^2x$는 기함수이므로 대칭구간 적분이 $0$이다. "
              f"$\\int_{{-N}}^{{N}} {c0}\\,dx={c0}\\cdot 2N={ans}$ ($N={N}$).")
        return q2, qsuf, ans, e2, {"kind": kind, "c0": c0, "unit": "미적분Ⅰ"}
    else:  # tangent_area: y=x²의 x=N 접선과 x축·y축이 만드는 삼각형 넓이
        # 접선 y=2N x - N² , x절편 N/2, y절편 -N² → 넓이 = N³/4 (N 짝수일 때 정수)
        if N % 2 != 0:
            return None
        ans = (N**3)//4
        q2 = ("곡선 $y=x^2$ 위의 점 $(N,\\,N^2)$에서의 접선이 $x$축, $y$축과 "
              "만나는 점을 각각 $\\mathrm{P},\\mathrm{Q}$라 할 때, 삼각형 $\\mathrm{OPQ}$의 넓이")
        qsuf = "는?"
        e2 = (f"접선은 $y=2Nx-N^2$. $x$절편 $\\frac{{N}}{{2}}$, $y$절편 $-N^2$이므로 "
              f"넓이 $=\\frac12\\cdot\\frac{{N}}{{2}}\\cdot N^2=\\frac{{N^3}}{{4}}$. "
              f"$N={N}$이면 ${ans}$.")
    return q2, qsuf, ans, e2, {"kind": kind, "unit": "미적분Ⅰ"}


def compose_interlock(n_blocks=2, max_retries=160, depth=None):
    """인터로킹 킬러: 1단계 검증 킬러의 답 $N$이 2단계 함수의 '파라미터'가 된다.

    1단계를 풀어 $N$을 구하지 못하면 2단계(미적분: 극값·대칭적분·접선넓이)를
    시작조차 할 수 없는 진짜 순차 의존 구조(30번급). 2단계 정답은 $N$에서 sympy로
    정확 계산하므로 100% 보장. fusion/chain(병렬·후처리)과 달리 '값이 다음 문제의
    구조를 바꾸는' 결합이라 변별력이 가장 높다.
    """
    from csat_forge import problem_engine as _pe
    eng_pool = _fuse_eng_pool()
    for _ in range(max_retries):
        if random.random() < 0.5:
            d = random.choice(_FUSE_META)
            r = generate_meta(domain=d)
            src = f"meta_{d}"
        else:
            nm = random.choice(eng_pool)
            try:
                r = _pe.GENERATORS[nm]()
            except Exception:
                continue
            r["generator"] = nm
            src = nm
        if r is None or not _pe.verify(r):
            continue
        N = _int_answer(r)
        if N is None or not (2 <= N <= 6):             # 2단계 정답이 0~999 정수가 되도록
            continue
        st2 = _interlock_stage2(N)
        if st2 is None:
            continue
        q2, qsuf, ans, e2, mt = st2
        if not (0 <= ans <= 999):
            continue
        q1 = r["completion"]["question"].strip()
        q = (f"아래 [1단계]의 답을 $N$이라 할 때, {q2}{qsuf}\n\n[1단계] {q1}")
        e = (f"1단계: [1단계]를 풀면 $N={N}$이다.\n2단계: {e2}\n\n")
        # 변별력 함정: N 그대로 답 / 극댓값↔차 혼동(2N³↔4N³) / 부호
        def _pl(v):
            return 0 <= v <= 999 and v != ans and abs(v - ans) <= max(12, ans)
        from math import comb as _comb
        traps_in, seen = [], set()
        k2 = mt["kind"]
        _alts = []
        if k2 == "extr_diff":
            _alts = [(2*N**3, "극댓값과 '차'를 혼동")]
        elif k2 == "extr_max":
            _alts = [(4*N**3, "극댓값과 '차'를 혼동")]
        elif k2 == "geom_term":
            _alts = [(2*N, "공비를 한 번만 곱함(제2항)"), (N**2, "첫째항을 빠뜨림")]
        elif k2 == "comb_count":
            _alts = [(_comb(N+3, 1), "조합 대신 $_{n}\\mathrm{C}_1$로 셈"),
                     (_comb(N, 2), "$+3$을 빠뜨림")]
        elif k2 == "binom_coeff":
            _alts = [(N+2, "$x^1$의 계수로 혼동"), (_comb(N+2, 3), "$x^3$의 계수로 혼동")]
        elif k2 == "sym_integral":
            _alts = [(mt.get("c0", 0)*N, "구간 길이를 $N$으로(½ 누락)")]
        for av, why in _alts:
            if _pl(av) and av not in seen:
                traps_in.append((sp.Integer(av), why))
                seen.add(av)
        if _pl(N) and N not in seen:
            traps_in.append((sp.Integer(N), "1단계 값 $N$을 그대로 답함"))
            seen.add(N)
        d = max(2, abs(ans) // 9 or 2)
        pad = [c for c in (ans-d, ans+d, ans-2*d, ans+2*d, ans+3*d)
               if c >= 0 and c != ans and c not in seen]
        choices, lab, traps = make_mc_tagged(sp.Integer(ans), traps_in + pad)
        e += f"정답: {lab}"
        # 1단계 단원 + 2단계 단원을 결합해 표기(과목 간 통합 가시화)
        units = sorted({r["spec"]["scope"], mt.get("unit", "미적분Ⅰ")})
        scope = "+".join(units)
        rr = _rec(scope, "인터로킹(파라미터 전달) 킬러(메타)", "상", 4,
                  "객관식", q, choices, lab, e)
        if traps:
            rr["completion"]["trap_map"] = traps
        rr["_meta"] = {"chk": "meta_interlock", "blocks": [src], "N": N,
                       "ans": ans, **mt}
        return rr
    return None


# ======================================================================
#  도메인 J : ultra  '극상위' 정점 킬러.
#  하위문제도 전부 조건결합형 킬러인 4개념(3단원) 융합 → 그 값에 다단계(3) 연쇄.
#  추론 단계가 가장 길고(서브 4문제 + 결합 + 변환 3) 계산량이 크다.
# ======================================================================
def compose_ultra(n_blocks=4, max_retries=400, depth=3, concepts=4):
    """극상위 통합 킬러: 메타킬러 N개(3단원 이상) 융합 → 다단계 연쇄.

    mega(3개념+연쇄)를 한 단계 더: 하위문제를 전부 조건결합형 킬러로 강제하고
    개념 수(concepts=4~5)·연쇄 깊이(depth=3~4)를 늘려 결합·추론 부담을 최대화.
    모든 부분이 검증/정수 산술이라 정답 100% 보장. 30번 '최고난도' 자리용.
    """
    concepts = max(3, min(concepts, 5))
    for _ in range(max_retries):
        base = compose_fusion(n_blocks=concepts, min_scopes=3, meta_only=True)
        if base is None:
            continue
        rr = _chain_on(base, "meta_fusion", depth,
                       topic="극상위 통합 킬러(메타)",
                       lead_extra="서로 다른 단원의 네 물음으로 이루어진")
        if rr is not None:
            rr["_meta"]["mega"] = True
            rr["_meta"]["ultra"] = True
            rr["_meta"]["fusion_scopes"] = base["spec"]["scope"]
            rr["_meta"]["fusion_k"] = len(base["_meta"].get("blocks", []))
            rr["spec"]["scope"] = base["spec"]["scope"]
            return rr
    return None


# ======================================================================
#  통합 진입점
# ======================================================================
_DOMAINS = {
    "cubic": compose_cubic,
    "quartic": compose_quartic,
    "set_function": compose_set_function,
    "discrete_rv": compose_discrete_rv,
    "arith_seq": compose_arith_seq,
    "geom_seq": compose_geom_seq,
    "trig_determine": compose_trig_determine,
    "integer_solutions": compose_integer_solutions,
    "intsol_prob": compose_intsol_prob,
    "fusion": compose_fusion,
    "chain": compose_chain,
    "mega": compose_mega,
    "interlock": compose_interlock,
    "ultra": compose_ultra,
}


_FIXED2 = {"discrete_rv", "arith_seq", "geom_seq", "trig_determine", "fusion",
           "chain", "mega", "interlock", "ultra"}  # 고정 조건수
# 브루트포스 도메인은 4조건(가나다라)까지 → '완전킬러' 심화
_ALLOW4 = {"set_function", "integer_solutions", "intsol_prob"}


def generate_meta(domain=None, n_blocks=None, max_outer=20):
    """도메인 무작위 선택 → 합성. 실패 시 다른 조합으로 재시도."""
    for _ in range(max_outer):
        d = domain or random.choice(list(_DOMAINS))
        if n_blocks:
            nb = n_blocks
        elif d in _ALLOW4:
            nb = random.choice([2, 3, 3, 4])        # 4조건 = 완전킬러 심화
        else:
            nb = random.choice([2, 3])
        if d in _FIXED2:
            nb = 2
        rec = _DOMAINS[d](nb)
        if rec is not None:
            rec["generator"] = f"meta_{d}"
            return rec
    return None


def generate_meta_batch(n=50, seed=None, domains=None, dedup=True,
                        attempt_factor=60):
    """새 유형 문제를 n개 추출. 문제 지문 기준 중복 제거 + 조합 다양성 집계.

    반환: (records, stats). stats = {requested, produced, attempts,
    unique_combos, per_domain, per_combo}.
    """
    if seed is not None:
        random.seed(seed)
    doms = domains or list(_DOMAINS)
    out, seen, attempts = [], set(), 0
    per_domain, per_combo = {}, {}
    while len(out) < n and attempts < n*attempt_factor:
        attempts += 1
        d = random.choice(doms)
        r = generate_meta(domain=d)
        if r is None:
            continue
        key = r["completion"]["question"]
        if dedup and key in seen:
            continue
        seen.add(key)
        chk = r["_meta"]["chk"]
        combo = (chk, tuple(r["_meta"]["blocks"]))
        per_domain[d] = per_domain.get(d, 0) + 1
        per_combo[combo] = per_combo.get(combo, 0) + 1
        out.append(r)
    stats = {"requested": n, "produced": len(out), "attempts": attempts,
             "unique_combos": len(per_combo), "per_domain": per_domain,
             "per_combo": {f"{c[0]}:{'+'.join(c[1])}": v
                           for c, v in sorted(per_combo.items())}}
    return out, stats


def _main():
    import argparse
    import json
    ap = argparse.ArgumentParser(
        description="메타 킬러 엔진으로 새 유형 문제를 대량 추출(jsonl).")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="jsonl 출력 경로(미지정 시 stdout 통계만)")
    ap.add_argument("--domains", default=None,
                    help="쉼표구분 (cubic,set_function,discrete_rv,arith_seq)")
    args = ap.parse_args()
    doms = args.domains.split(",") if args.domains else None
    recs, stats = generate_meta_batch(args.n, seed=args.seed, domains=doms)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps({"spec": r["spec"],
                                     "completion": r["completion"],
                                     "generator": r.get("generator")},
                                    ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
