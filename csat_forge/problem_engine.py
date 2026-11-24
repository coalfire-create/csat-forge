"""파라메트릭 + 기호계산 검증형 수학 문제 생성 엔진 (2028 개편 범위).

설계 철학: 수학 문제 생성기의 핵심은 '정답 정확성'이다. LLM 단독 생성은
오답을 자주 내므로, 여기서는 토픽별 생성기가 파라미터를 무작위화하고
sympy 로 정답을 '계산'하여 100% 정답을 보장한다. 오답지·단계별 풀이까지
자동 생성하며, 모든 문제는 verify()로 독립 재검증된다.

범위(2028 수능): 대수(지수로그·삼각·수열) + 미적분Ⅰ(다항함수 극한·미분·적분)
              + 확률과 통계(경우의수·확률·통계). 미적분Ⅱ·기하 제외.

출력 스키마: generated/2028_set*.jsonl 과 동일
  {"spec":{subject,exam_system,scope,topic,difficulty,score,problem_type},
   "completion":{question,choices,answer,explanation}}
"""
from __future__ import annotations
import copy
import random
import re as _re
import sympy as sp

x, t, k, n = sp.symbols("x t k n")
CIRC = "①②③④⑤"


# ---------- 공통 유틸 ----------
def _ltx(e) -> str:
    """sympy 식 → LaTeX 문자열."""
    if isinstance(e, str):
        return e
    return sp.latex(sp.nsimplify(e) if isinstance(e, float) else e)


def _fmt(v) -> str:
    """정답 값을 $...$ 로."""
    return f"${_ltx(v)}$"


def make_mc(correct, distractors):
    """객관식 보기 구성 (기존 시그니처). 반환: (choices, answer_label)."""
    choices, lab, _ = make_mc_tagged(correct, distractors)
    return choices, lab


def make_mc_tagged(correct, distractors):
    """객관식 보기 구성 + 오답 실수 태깅.
    distractors 항목은 값 또는 (값, 실수라벨) 튜플. 라벨이 붙은 오답이 보기에
    포함되면 trap_map={'보기번호': 라벨} 로 반환 → 오답 진단에 사용.
    수능 형식: 보기는 오름차순 배열. 오답이 ans±k 대칭이면 정렬 시 정답이
    항상 중앙(3번)에 가므로, 정답 아래/위 오답 개수(j, 4-j)를 무작위 배분해
    정답 위치 편향을 제거한다.
    반환: (choices[str], answer_label, trap_map)."""
    # (값, 라벨) 분리
    labels = {}
    plain = []
    for d in distractors:
        if isinstance(d, tuple):
            v, lbl = d
            key = sp.nsimplify(v)
            labels[key] = lbl
            plain.append(v)
        else:
            plain.append(d)
    distractors = plain
    c0 = sp.nsimplify(correct)
    cf = float(sp.N(c0))
    seen = {c0}
    cands = []
    for v in distractors:
        key = sp.nsimplify(v)
        if key in seen:
            continue
        seen.add(key)
        cands.append(key)
    below = sorted((v for v in cands if float(sp.N(v)) < cf),
                   key=lambda v: float(sp.N(v)), reverse=True)   # 정답에 가까운 순
    above = sorted((v for v in cands if float(sp.N(v)) > cf),
                   key=lambda v: float(sp.N(v)))
    # 합성 오답 간격: 기존 오답 중 정답과 가장 가까운 간격
    step = min((abs(v - c0) for v in cands), key=lambda d: float(sp.N(d))) if cands else sp.Integer(1)
    vals = [float(sp.N(v)) for v in cands] + [cf]
    pos_only = all(x > 0 for x in vals)        # 개수/양수형이면 합성도 양수 유지
    unit = all(0 <= x <= 1 for x in vals)      # 확률형이면 (0,1) 유지

    def _ok(w):
        wf = float(sp.N(w))
        if w in seen:
            return False
        if pos_only and wf <= 0:
            return False
        if unit and not (0 < wf < 1):
            return False
        return True

    def _extend(side, direction):
        i = 1
        while len(side) < 4 and i <= 12:
            w = sp.nsimplify(c0 + direction * step * i)
            if _ok(w):
                seen.add(w)
                side.append(w)
            i += 1

    _extend(below, -1)
    _extend(above, +1)
    lo, hi = max(0, 4 - len(above)), min(4, len(below))
    if lo > hi:                                # 후보 부족 (이론상 드묾) → 가능한 만큼
        pool = ([c0] + below + above)[:5]
    else:
        j = random.randint(lo, hi)             # 정답보다 작은 보기 개수
        pool = below[:j] + [c0] + above[:4 - j]
    pool.sort(key=lambda v: float(sp.N(v)))
    idx = next(i for i, v in enumerate(pool) if sp.simplify(v - correct) == 0)
    choices = [_fmt(v) for v in pool]
    trap_map = {}
    for i, v in enumerate(pool):
        key = sp.nsimplify(v)
        if key in labels and i != idx:
            trap_map[str(i + 1)] = labels[key]
    return choices, f"{idx+1}번 ({_fmt(correct)})", trap_map


def _rec(scope, topic, diff, score, ptype, q, choices, ans, expl):
    return {
        "spec": {"subject": "수학", "exam_system": "2028학년도(개편)", "scope": scope,
                 "topic": topic, "difficulty": diff, "score": score, "problem_type": ptype},
        "completion": {"question": q, "choices": choices, "answer": ans, "explanation": expl},
    }


# ======================================================================
#  대수
# ======================================================================
def gen_log_equation():
    """log_a b + log_b a = c, ab=s 꼴 → a²+b² (상)."""
    base = random.choice([2, 3])
    p = random.randint(2, 4)               # b = a^p
    a = random.randint(2, 3)
    b = a ** p
    s = a * b
    c = sp.Rational(p) + sp.Rational(1, p)
    ans = a**2 + b**2
    distract = [a*b, a+b, (a+b)**2, a**2 + b**2 + a*b]
    choices, lab = make_mc(ans, distract)
    q = (f"$1$보다 큰 두 실수 $a$, $b$가 $\\log_a b+\\log_b a={_ltx(c)}$, $ab={s}$를 "
         f"만족시킨다. $a<b$일 때, $a^2+b^2$의 값은?")
    e = (f"1단계: $\\log_a b=u$로 놓으면 $u+\\dfrac1u={_ltx(c)}$에서 $u={p}$ 또는 $u=\\dfrac1{p}$. "
         f"$a<b$이므로 $u={p}$, 즉 $b=a^{{{p}}}$.\n"
         f"2단계: $ab=a^{{{p+1}}}={s}$에서 $a={a}$, $b={b}$.\n"
         f"3단계: $a^2+b^2={a**2}+{b**2}={ans}$.\n\n정답: {lab}")
    return _rec("대수", "지수와 로그", "상", 4, "객관식", q, choices, lab, e)


def gen_geom_seq():
    """등비수열: a_p+a_{p+2}=A, a_{p+2}+a_{p+4}=B → a_1 (상)."""
    r = random.choice([2, 3])
    a1 = random.randint(1, 5)
    p = random.choice([1, 2])
    f = lambda i: a1 * r**(i-1)
    A = f(p) + f(p+2)
    B = f(p+2) + f(p+4)
    ans = a1
    choices, lab = make_mc(ans, [a1+1, a1+2, a1*r, max(1, a1-1), a1*2])
    q = (f"첫째항이 양수인 등비수열 $\\{{a_n\\}}$이 $a_{p}+a_{p+2}={A}$, "
         f"$a_{p+2}+a_{p+4}={B}$를 만족시킨다. $a_1$의 값은?")
    e = (f"1단계: $a_{p+2}+a_{p+4}=r^2(a_{p}+a_{p+2})$이므로 ${B}={A}r^2$, $r^2={B//A}$. "
         f"첫째항이 양수이므로 $r={r}$.\n"
         f"2단계: $a_{p}+a_{p+2}=a_1 r^{{{p-1}}}(1+r^2)={A}$에서 $a_1={ans}$.\n\n정답: {lab}")
    return _rec("대수", "등비수열", "상", 4, "객관식", q, choices, lab, e)


def gen_sigma():
    """Σ(ak+b)² (중). 파라메트릭."""
    a = random.choice([2, 3])
    b = random.choice([-1, 1, -2])
    m = random.choice([8, 10, 12])
    ans = sum((a*i + b)**2 for i in range(1, m+1))
    choices, lab = make_mc(ans, [ans-20, ans+20, ans-40, ans+40])
    term = sp.latex((a*k + b)**2)
    q = f"$\\displaystyle\\sum_{{k=1}}^{{{m}}}{term}$의 값은?"
    S2 = m*(m+1)*(2*m+1)//6
    S1 = m*(m+1)//2
    e = (f"1단계: ${term}={sp.latex(sp.expand((a*k+b)**2))}$.\n"
         f"2단계: $\\sum k^2={S2}$, $\\sum k={S1}$, $\\sum 1={m}$.\n"
         f"3단계: ${a*a}\\cdot{S2}{'+' if 2*a*b >= 0 else '-'}{abs(2*a*b)}\\cdot{S1}"
         f"+{b*b}\\cdot{m}={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "시그마(거듭제곱의 합)", "중", 3, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "sigma", "a": a, "b": b, "m": m, "ans": int(ans)}
    return rr


def gen_trig_equation():
    """2sin²x + S·cosx + C = 0, [0,2π) 실근의 합 (상). 파라메트릭."""
    half = sp.Rational(1, 2)
    # (cos 근 a, cos 근 b): |근|>1 이면 버리는 함정 포함. 2(a+b), 2ab 가 정수인 쌍만.
    pairs = [(half, -2), (-half, 2), (-half, -3), (half, 3),
             (-1, half), (-1, -half), (1, half), (1, -half)]
    a, b = random.choice(pairs)
    # cos형 (참 방정식): 2cos²x - 2(a+b)cosx + 2ab = 0
    # sin형 변환: 2sin²x + S cosx + C = 0  (S = 2(a+b), C = -(2+P))
    S = int(2*(a + b))
    P = int(2*a*b)
    C = -(2 + P)

    def sols(v):
        if v == 1:
            return [sp.Integer(0)]
        if v == -1:
            return [sp.pi]
        if v == half:
            return [sp.pi/3, 5*sp.pi/3]
        if v == -half:
            return [2*sp.pi/3, 4*sp.pi/3]
        return []                       # |v|>1 → 해 없음
    roots = sols(a) + sols(b)
    ans = sum(roots)

    def _t(c, term):                    # 부호 포함 항 렌더링
        if c == 0:
            return ""
        sgn = "+" if c > 0 else "-"
        mag = "" if (abs(c) == 1 and term) else str(abs(c))
        return f"{sgn}{mag}{term}"
    cos_t = "\\cos x"
    eq = "2\\sin^2 x" + _t(S, cos_t) + _t(C, "") + "=0"
    coseq = "2\\cos^2 x" + _t(-S, cos_t) + _t(P, "") + "=0"
    choices, lab = make_mc(ans, [sp.pi, 2*sp.pi, 3*sp.pi, 4*sp.pi, sp.Rational(5, 2)*sp.pi])
    q = f"$0\\le x<2\\pi$일 때, 방정식 ${eq}$의 모든 실근의 합은?"
    invalid = [v for v in (a, b) if abs(v) > 1]
    note = (f" (단, $|\\cos x|\\le1$이므로 $\\cos x={sp.latex(invalid[0])}$는 제외)"
            if invalid else "")
    roots_l = ",\\ ".join(sp.latex(r) for r in roots)
    e = (f"1단계: $\\sin^2x=1-\\cos^2x$를 대입하면 ${coseq}$.\n"
         f"2단계: 인수분해하면 $\\cos x={sp.latex(a)}$ 또는 $\\cos x={sp.latex(b)}${note}.\n"
         f"3단계: $x={roots_l}$, 합은 ${sp.latex(ans)}$.\n\n정답: {lab}")
    rr = _rec("대수", "삼각함수 방정식", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "trigeq", "S": S, "C": C, "ans_pi": int(ans / sp.pi)}
    return rr


# ======================================================================
#  미적분Ⅰ (다항함수)
# ======================================================================
def gen_tangent_cubic():
    """y=x³+ax 위 접선이 곡선과 다시 만나는 점에서의 기울기 (상)."""
    a = random.choice([-2, -3, -1, -4])
    f = x**3 + a*x
    x0 = random.choice([1, 2])
    slope0 = sp.diff(f, x).subs(x, x0)
    y0 = f.subs(x, x0)
    line = slope0*(x - x0) + y0
    other = sp.solve(sp.expand(f - line), x)
    other = [r for r in other if sp.simplify(r - x0) != 0]
    xb = other[0]
    ans = sp.diff(f, x).subs(x, xb)
    choices, lab = make_mc(ans, [ans-3, ans+3, ans-6, ans+6, slope0])
    f_l = sp.latex(f)                       # 수동 부호 조립 금지 → sympy 렌더링
    fp_l = sp.latex(sp.diff(f, x))
    q = (f"곡선 $y={f_l}$ 위의 점 $({x0},\\,{int(y0)})$에서의 "
         f"접선이 이 곡선과 다시 만나는 점에서의 접선의 기울기는?")
    e = (f"1단계: $y'={fp_l}$, 점 $x={x0}$에서 기울기 ${int(slope0)}$, "
         f"접선 $y={sp.latex(sp.expand(line))}$.\n"
         f"2단계: ${f_l}={sp.latex(sp.expand(line))}$의 해는 "
         f"$x={x0}$(중근), $x={int(xb)}$.\n"
         f"3단계: 다시 만나는 점 $x={int(xb)}$에서 $y'={int(ans)}$.\n\n정답: {lab}")
    r = _rec("미적분Ⅰ", "미분·접선", "상", 4, "객관식", q, choices, lab, e)
    r["_meta"] = {"chk": "tangent", "a": int(a), "x0": int(x0), "ans": int(ans)}
    return r


def gen_extremum_diff():
    """f=2x³+bx²+cx+d 의 극대-극소 차 (상)."""
    # f' = 6(x-p)(x-q), p<q
    p, q = sorted(random.sample([-2, -1, 1, 2, 3], 2))
    d = random.randint(0, 5)
    fp = 6*(x-p)*(x-q)
    f = sp.integrate(fp, x) + d
    M = f.subs(x, p)   # 극대 (p<q, 6>0 → x=p 극대)
    mlow = f.subs(x, q)
    ans = sp.simplify(M - mlow)
    choices, lab = make_mc(ans, [ans-2, ans+2, ans-5, ans+5])
    fl = sp.latex(sp.expand(f))
    q_ = f"함수 $f(x)={fl}$의 극댓값을 $M$, 극솟값을 $m$이라 할 때, $M-m$의 값은?"
    e = (f"1단계: $f'(x)={sp.latex(sp.expand(fp))}=6(x-({p}))(x-({q}))$이므로 "
         f"$x={p}$에서 극대, $x={q}$에서 극소.\n"
         f"2단계: $M=f({p})={int(M)}$, $m=f({q})={int(mlow)}$.\n"
         f"3단계: $M-m={int(ans)}$.\n\n정답: {lab}")
    r = _rec("미적분Ⅰ", "삼차함수의 극값", "상", 4, "객관식", q_, choices, lab, e)
    r["_meta"] = {"chk": "extremum", "p": int(p), "q": int(q), "ans": int(ans)}
    return r


def gen_area_between():
    """포물선 y=-x²+ax 와 직선 y=mx 사이 넓이 → m (상)."""
    a = random.choice([4, 6])
    c = random.choice([2, 3])       # c = a-m = 교점 폭
    m = a - c
    area = sp.Rational(c**3, 6)
    ans = m
    choices, lab = make_mc(ans, [m+1, m-1 if m-1>0 else m+2, m+2, sp.Rational(2*m+1,2)])
    q = (f"곡선 $y=-x^2+{a}x$와 직선 $y=mx\\,(0<m<{a})$로 둘러싸인 부분의 넓이가 "
         f"${_ltx(area)}$일 때, 상수 $m$의 값은?")
    e = (f"1단계: 교점 $x=0,\\ {c}=({a}-m)$. 넓이 $=\\displaystyle\\int_0^{{{a}-m}}"
         f"\\{{(-x^2+{a}x)-mx\\}}dx=\\dfrac{{({a}-m)^3}}6$.\n"
         f"2단계: $\\dfrac{{({a}-m)^3}}6={_ltx(area)}$에서 $({a}-m)^3={c**3}$, ${a}-m={c}$.\n"
         f"3단계: $m={ans}$.\n\n정답: {lab}")
    r = _rec("미적분Ⅰ", "정적분·넓이", "상", 4, "객관식", q, choices, lab, e)
    r["_meta"] = {"chk": "area", "a": int(a), "c": int(c), "ans": int(ans)}
    return r


def gen_motion():
    """v(t)=3(t-p)(t-q): 두 번째 방향 전환까지 거리 (상). 파라메트릭."""
    # p+q 짝수 → 위치함수가 정수계수 → 거리 정수
    p, q = random.choice([(1, 3), (2, 4), (1, 5), (3, 5), (2, 6)])
    v = sp.expand(3*(t - p)*(t - q))
    F = sp.integrate(v, t)             # F(0)=0
    d1 = F.subs(t, p)                  # ∫0^p v dt (v>0 구간)
    d2 = F.subs(t, p) - F.subs(t, q)   # ∫p^q |v| dt (v<0 구간)
    ans = sp.Integer(d1 + d2)
    choices, lab = make_mc(ans, [ans-2, ans+2, ans-4, ans+4])
    q_ = (f"수직선 위를 움직이는 점 P의 시각 $t\\,(t\\ge0)$에서의 속도가 "
          f"$v(t)={sp.latex(v)}$이다. 점 P가 출발한 후 운동 방향을 두 번째로 "
          f"바꾸는 순간까지 움직인 거리는?")
    e = (f"1단계: $v(t)=3(t-{p})(t-{q})$, 방향 전환 $t={p},{q}$. 두 번째는 $t={q}$.\n"
         f"2단계: $\\int_0^{{{p}}}v\\,dt={int(d1)}$, "
         f"$\\int_{{{p}}}^{{{q}}}|v|\\,dt={int(d2)}$.\n"
         f"3단계: 거리 $={int(ans)}$.\n\n정답: " + lab)
    rr = _rec("미적분Ⅰ", "속도와 거리", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "motion", "p": p, "q": q, "ans": int(ans)}
    return rr


def gen_cubic_roots_count():
    """삼차함수가 x축과 세 점에서 만나는 정수 a 개수 (상). 두 패밀리 파라메트릭."""
    fam = random.choice(["sq", "lin"])
    kk = random.choice([1, 2])
    gap = 4 * kk**3                     # 두 패밀리 모두 정수 a 개수 = 4k³-1
    if fam == "sq":                     # f = x³ - 3k x² + a: 극대 f(0)=a, 극소 a-4k³
        fl = sp.latex(x**3 - 3*kk*x**2) + "+a"
        e12 = (f"1단계: $f'(x)=3x^2-{6*kk}x=3x(x-{2*kk})$, $x=0$ 극대 $f(0)=a$, "
               f"$x={2*kk}$ 극소 $f({2*kk})=a-{gap}$.\n"
               f"2단계: 세 실근 ⇔ 극대$>0>$극소 ⇔ $0<a<{gap}$.\n")
    else:                               # f = x³ - 3k²x + a: 극대 a+2k³, 극소 a-2k³
        fl = sp.latex(x**3 - 3*kk**2*x) + "+a"
        e12 = (f"1단계: $f'(x)=3x^2-{3*kk**2}=3(x+{kk})(x-{kk})$, "
               f"$x=-{kk}$ 극대 $a+{2*kk**3}$, $x={kk}$ 극소 $a-{2*kk**3}$.\n"
               f"2단계: 세 실근 ⇔ 극대$>0>$극소 ⇔ $-{2*kk**3}<a<{2*kk**3}$.\n")
    ans = gap - 1
    choices, lab = make_mc(ans, [ans-2, ans-1, ans+1, ans+2])
    q = (f"삼차함수 $f(x)={fl}$의 그래프가 $x$축과 서로 다른 세 점에서 "
         f"만나도록 하는 정수 $a$의 개수는?")
    e = e12 + f"3단계: 정수 $a$는 ${ans}$개.\n\n정답: " + lab
    rr = _rec("미적분Ⅰ", "삼차함수와 실근의 개수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "cubicroots", "fam": fam, "k": kk, "ans": ans}
    return rr


# ======================================================================
#  확률과 통계
# ======================================================================
def gen_complement_dice():
    """주사위 두 개 여사건 확률 (중). 파라메트릭+브루트포스."""
    variant = random.choice(["evenprod", "atleast", "distinct"])
    aparam = 0
    if variant == "evenprod":
        cond = lambda a, b: (a*b) % 2 == 0
        desc = "나온 두 눈의 수의 곱이 짝수일"
        comp_desc = "곱이 홀수, 즉 두 눈 모두 홀수"
    elif variant == "atleast":
        aparam = random.choice([4, 5, 6])
        cond = lambda a, b: a >= aparam or b >= aparam
        desc = f"나온 눈의 수 중 적어도 하나가 ${aparam}$ 이상일"
        comp_desc = f"두 눈 모두 ${aparam-1}$ 이하"
    else:
        cond = lambda a, b: a != b
        desc = "나온 두 눈의 수가 서로 다를"
        comp_desc = "두 눈이 서로 같음"
    cnt = sum(1 for a in range(1, 7) for b in range(1, 7) if cond(a, b))
    comp = 36 - cnt
    ans = sp.Rational(cnt, 36)
    choices, lab, traps = make_mc_tagged(ans, [
        (sp.Rational(comp, 36), "여사건의 확률을 그대로 답함 — 1에서 빼지 않음"),
        sp.Rational(cnt-3, 36), sp.Rational(min(cnt+3, 35), 36), sp.Rational(cnt-1, 36)])
    q = f"서로 다른 두 개의 주사위를 동시에 던질 때, {desc} 확률은?"
    e = (f"1단계: 여사건({comp_desc})의 경우의 수는 ${comp}$가지, "
         f"확률 $\\dfrac{{{comp}}}{{36}}$.\n"
         f"2단계: $1-\\dfrac{{{comp}}}{{36}}={_ltx(ans)}$.\n\n정답: " + lab)
    rr = _rec("확률과 통계", "여사건의 확률", "중", 3, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "compdice", "variant": variant, "a": aparam,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_conditional_prob():
    """남녀·안경 조건부확률 (상). 파라메트릭."""
    pm = random.choice([50, 60, 70])
    pf = 100 - pm
    rm = random.choice([20, 30, 40])
    rf = random.choice([30, 40, 45, 50])
    jm = sp.Rational(pm*rm, 10000)
    jf = sp.Rational(pf*rf, 10000)
    ans = sp.nsimplify(jf/(jm + jf))
    wrong_swap = sp.nsimplify(jm/(jm + jf))     # 흔한 실수: 남학생 확률
    naive = sp.Rational(rf, rm + rf)            # 흔한 실수: 비율 가중 무시
    choices, lab, traps = make_mc_tagged(ans, [
        (wrong_swap, "조건부확률 대상 혼동 — 여학생이 아니라 남학생일 확률을 구함"),
        (naive, "남녀 비율 가중 무시 — 안경 비율끼리만 비교함"),
        ans*sp.Rational(3, 4), ans*sp.Rational(5, 4)])

    def _d(v):  # 소수 표기 (해설용)
        s = f"{float(v):.4f}".rstrip("0")
        return s[:-1] if s.endswith(".") else s
    q = (f"어느 학교 학생의 ${pm}\\%$는 남학생, ${pf}\\%$는 여학생이다. "
         f"남학생의 ${rm}\\%$, 여학생의 ${rf}\\%$가 안경을 썼다. "
         f"이 학교에서 임의로 택한 한 학생이 안경을 썼을 때, 이 학생이 여학생일 확률은?")
    e = (f"1단계: $P(\\text{{여}}\\cap\\text{{안}})={_d(sp.Rational(pf,100))}\\times"
         f"{_d(sp.Rational(rf,100))}={_d(jf)}$, "
         f"$P(\\text{{남}}\\cap\\text{{안}})={_d(sp.Rational(pm,100))}\\times"
         f"{_d(sp.Rational(rm,100))}={_d(jm)}$.\n"
         f"2단계: $P(\\text{{안}})={_d(jm+jf)}$.\n"
         f"3단계: $P(\\text{{여}}\\mid\\text{{안}})=\\dfrac{{{_d(jf)}}}{{{_d(jm+jf)}}}"
         f"={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조건부확률", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "condprob", "pm": pm, "rm": rm, "rf": rf,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_binomial_param():
    """B(n,p): E=np, V=np(1-p) → n (상)."""
    p = sp.Rational(random.choice([3, 2, 4]), 5)
    nn = random.choice([10, 15, 20])
    E = nn*p
    V = nn*p*(1-p)
    ans = nn
    choices, lab = make_mc(ans, [nn-2, nn+2, nn-5, nn+5])
    q = (f"확률변수 $X$가 이항분포 $\\mathrm{{B}}(n,p)$를 따르고 $\\mathrm{{E}}(X)={_ltx(E)}$, "
         f"$\\mathrm{{V}}(X)={_ltx(V)}$일 때, $n$의 값은?")
    e = (f"1단계: $np={_ltx(E)}$, $np(1-p)={_ltx(V)}$. 나누면 $1-p={_ltx(1-p)}$, $p={_ltx(p)}$.\n"
         f"2단계: $n={_ltx(E)}/{_ltx(p)}={ans}$.\n\n정답: {lab}")
    return _rec("확률과 통계", "이항분포의 평균과 분산", "상", 4, "객관식", q, choices, lab, e)


def gen_normal_tail():
    """N(m,σ²)에서 P(X≥m+2σ) 류 (중)."""
    m = random.choice([50, 60, 100])
    s = random.choice([4, 8, 10])
    # z 값을 다양화 (m+zs 가 정수가 되도록 s 는 짝수 배수 포함)
    z, ztxt, table = random.choice([(1, "1", "0.3413"), (2, "2", "0.4772"),
                                    (sp.Rational(3, 2), "1.5", "0.4332")])
    ans_txt = f"{0.5 - float(table):.4f}"
    # 수능 표기(소수)로 보기 구성 — make_mc 의 분수 LaTeX 대신 직접 조립
    pool = list({ans_txt, table, f"{0.5 + float(table):.4f}",
                 "0.1587" if ans_txt != "0.1587" else "0.0228",
                 "0.0668" if ans_txt != "0.0668" else "0.0228"})
    pool.sort(key=float)        # 수능 형식: 오름차순
    idx = pool.index(ans_txt)
    choices = [f"${v}$" for v in pool]
    lab = f"{idx+1}번 (${ans_txt}$)"
    # 오답 실수 태깅 (표값 자체 / P(X≤) 혼동)
    traps = {}
    for i, v in enumerate(pool):
        if i == idx:
            continue
        if v == table:
            traps[str(i+1)] = "0.5에서 표값을 빼지 않고 표값을 그대로 답함"
        elif v == f"{0.5 + float(table):.4f}":
            traps[str(i+1)] = "P(X≥)와 P(X≤) 혼동 — 0.5+표값을 구함"
    val = int(m + z*s)
    q = (f"확률변수 $X$가 정규분포 $\\mathrm{{N}}({m},\\,{s}^2)$을 따를 때, $\\mathrm{{P}}(X\\ge{val})$의 값은? "
         f"(단, $\\mathrm{{P}}(0\\le Z\\le{ztxt})={table}$)")
    e = (f"1단계: $Z=\\dfrac{{X-{m}}}{{{s}}}$, $X={val}\\Rightarrow Z={ztxt}$.\n"
         f"2단계: $P(X\\ge{val})=P(Z\\ge{ztxt})=0.5-{table}={ans_txt}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "정규분포", "중", 3, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    return rr


def gen_combination_sum3():
    """1~N 중 서로 다른 3개의 합이 3의 배수인 경우의 수 (상). 브루트포스."""
    from itertools import combinations
    N = random.choice([7, 8, 9, 10])
    ans = sum(1 for c in combinations(range(1, N+1), 3) if sum(c) % 3 == 0)
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, ans-1])
    q = (f"$1$부터 ${N}$까지의 자연수 중에서 서로 다른 세 개의 수를 선택할 때, "
         f"선택한 세 수의 합이 $3$의 배수인 경우의 수는?")
    e = (f"1단계: $1\\sim{N}$을 $3$으로 나눈 나머지(0,1,2)별로 분류한다.\n"
         f"2단계: 세 수의 합이 $3$의 배수가 되는 나머지 조합은 (0,0,0),(1,1,1),(2,2,2),(0,1,2)이다.\n"
         f"3단계: 각 군의 원소 수로 경우의 수를 세면 총 ${ans}$가지.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "경우의 수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "comb3", "N": N, "ans": ans}
    return rr


def gen_dice_event_prob():
    """두 주사위 합이 s 이상일 확률 (중상). 브루트포스 분수."""
    s = random.choice([7, 8, 9, 10])
    cnt = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b >= s)
    ans = sp.Rational(cnt, 36)
    dd = [sp.Rational(cnt-2, 36), sp.Rational(cnt+2, 36), sp.Rational(cnt-1, 36), sp.Rational(cnt+3, 36)]
    choices, lab = make_mc(ans, dd)
    q = f"서로 다른 두 개의 주사위를 동시에 던질 때, 나온 두 눈의 수의 합이 ${s}$ 이상일 확률은?"
    e = (f"1단계: 전체 경우의 수는 $36$이다.\n"
         f"2단계: 두 눈의 합이 ${s}$ 이상인 경우는 ${cnt}$가지이다.\n"
         f"3단계: 확률 $=\\dfrac{{{cnt}}}{{36}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "확률의 계산", "중", 3, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "diceev", "s": s, "ans": [cnt, 36]}
    return rr


def gen_limit_poly():
    """lim f/x²=L, lim_{x→r0} f/(x-r0)=c → f(r) (상). 파라메트릭."""
    L = random.choice([1, 2, 3])
    r0 = random.choice([1, 2, -1])
    c = random.randint(2, 6)
    r = random.choice([2, 3, 4, -2])
    while r == r0:
        r = random.choice([2, 3, 4, -2])
    # f=Lx²+bx+d, f(r0)=0, f'(r0)=c
    b, d = sp.symbols("b d")
    sol = sp.solve([L*r0**2 + b*r0 + d, 2*L*r0 + b - c], [b, d])
    bb, dd = sol[b], sol[d]
    f = L*x**2 + bb*x + dd
    ans = int(f.subs(x, r))
    choices, lab = make_mc(ans, [ans-2, ans+2, ans-3, ans+3])
    denom = sp.latex(x - r0)                # 'x - 1' / 'x + 1' (sympy 렌더링)
    q = (f"다항함수 $f(x)$가 $\\displaystyle\\lim_{{x\\to\\infty}}\\dfrac{{f(x)}}{{x^2}}={L}$, "
         f"$\\displaystyle\\lim_{{x\\to{r0}}}\\dfrac{{f(x)}}{{{denom}}}={c}$를 만족시킨다. "
         f"$f({r})$의 값은?")
    e = (f"1단계: 첫 극한이 ${L}$로 수렴하므로 $f(x)={L}x^2+bx+d$.\n"
         f"2단계: 둘째 극한이 수렴하므로 $f({r0})=0$, 그 값은 $f'({r0})={c}$.\n"
         f"3단계: 연립하면 $f(x)={sp.latex(sp.expand(f))}$, $f({r})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "함수의 극한", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "limpoly", "L": L, "r0": r0, "c": c, "r": r, "ans": ans}
    return rr


def gen_arith_seq():
    """등차수열 a_p, a_q 주어짐 → Σ_{1}^{m} a_k (상). 파라메트릭."""
    d = random.choice([2, 3, -2, 4])
    a1 = random.randint(1, 6)
    p, q = sorted(random.sample([2, 3, 4, 5, 6, 7], 2))
    A = a1 + (p-1)*d
    B = a1 + (q-1)*d
    m = random.choice([8, 10, 12])
    ans = m*(2*a1 + (m-1)*d)//2
    choices, lab = make_mc(ans, [ans-d*m, ans+d*m, ans-2*m, ans+2*m])
    q_ = (f"등차수열 $\\{{a_n\\}}$에 대하여 $a_{p}={A}$, $a_{q}={B}$일 때, "
          f"$\\displaystyle\\sum_{{k=1}}^{{{m}}}a_k$의 값은?")
    e = (f"1단계: 공차 $d=\\dfrac{{a_{q}-a_{p}}}{{{q}-{p}}}=\\dfrac{{{B}-{A}}}{{{q-p}}}={d}$.\n"
         f"2단계: $a_1=a_{p}-({p}-1)d={a1}$.\n"
         f"3단계: $\\sum_{{k=1}}^{{{m}}}a_k=\\dfrac{{{m}}}2(2\\cdot{a1}+{m-1}\\cdot{d})={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "등차수열의 합", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "arith", "a1": a1, "d": d, "m": m, "ans": int(ans)}
    return rr


def gen_def_integral_func():
    """∫_1^x f(t)dt = x³+ax²+bx+const(F(1)=0) → f(r) (상)."""
    a = random.randint(-3, 3)
    b = random.randint(-3, 3)
    r = random.choice([2, 3, -1])
    f = 3*x**2 + 2*a*x + b
    ans = int(f.subs(x, r))
    choices, lab = make_mc(ans, [ans-2, ans+2, ans-4, ans+4])
    const = -(1 + a + b)   # F(1)=0
    F = x**3 + a*x**2 + b*x + const
    q = (f"다항함수 $f(x)$가 모든 실수 $x$에 대하여 "
         f"$\\displaystyle\\int_1^x f(t)\\,dt={sp.latex(sp.expand(F))}$를 만족시킬 때, "
         f"$f({r})$의 값은?")
    e = (f"1단계: 양변을 $x$로 미분하면 $f(x)={sp.latex(sp.expand(f))}$.\n"
         f"2단계: ($x=1$ 대입 시 좌변 $0$, 우변도 $0$으로 일관.)\n"
         f"3단계: $f({r})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분으로 정의된 함수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "defint", "a": a, "b": b, "r": r, "ans": ans}
    return rr


def gen_log_inequality():
    """log_2(x-p)+log_2(x+q) ≤ c → 정수해 개수 (상)."""
    # 정답이 1개 이상 나오는 파라미터만 채택 (정답 0개는 수능 스타일이 아님)
    while True:
        p = random.choice([1, 2])
        q = random.choice([2, 3, 4])
        c = random.choice([2, 3, 4])
        # (x-p)(x+q) ≤ 2^c, x>p
        bound = 2**c
        xs = [xi for xi in range(p+1, p+50) if (xi-p)*(xi+q) <= bound]
        if xs:
            break
    ans = len(xs)
    # 진수조건 누락 오답: x>p 제한 없이 (x-p)(x+q)≤bound 인 모든 정수를 셈
    cnt_all = sum(1 for xi in range(-60, 60) if (xi-p)*(xi+q) <= bound)
    dd = [ans+1, ans+2, ans+3, ans+4, max(1, ans-1)]
    if cnt_all != ans:
        dd = [(cnt_all, "진수조건 누락 — x>"+str(p)+" 제한 없이 부등식만 풂")] + dd
    choices, lab, traps = make_mc_tagged(ans, dd)
    q_ = (f"부등식 $\\log_2(x-{p})+\\log_2(x+{q})\\le {c}$를 만족시키는 정수 $x$의 개수는?")
    e = (f"1단계: 진수조건 $x>{p}$. $\\log_2\\{{(x-{p})(x+{q})\\}}\\le{c}$에서 "
         f"$(x-{p})(x+{q})\\le {bound}$.\n"
         f"2단계: 정수해는 ${{{', '.join(map(str,xs)) if xs else '없음'}}}$로 ${ans}$개.\n\n정답: {lab}")
    rr = _rec("대수", "로그부등식", "상", 4, "객관식", q_, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "loineq", "p": p, "q": q, "c": c, "ans": ans}
    return rr


def gen_abs_cubic_nondiff():
    """|f| 가 한 점에서만 미분불가 → f 결정 → f(c) (킬러, 상)."""
    r = random.choice([1, 2, 3, -1, -2])
    c = random.choice([v for v in [-2, -1, 2, 3, 4] if v not in (0, r)])
    # f(0)=f'(0)=0 → x=0 이중근, |f| 가 x=r 에서만 미분불가 → f=x²(x-r)
    ans = c**2 * (c - r)
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"최고차항의 계수가 $1$인 삼차함수 $f(x)$가 $f(0)=0$, $f'(0)=0$을 만족시킨다. "
         f"함수 $g(x)=|f(x)|$가 오직 $x={r}$에서만 미분가능하지 않을 때, $f({c})$의 값은?")
    e = (f"1단계: $f(0)=f'(0)=0$이므로 $x=0$은 이중근. 최고차계수 $1$이라 $f(x)=x^2(x-s)$.\n"
         f"2단계: $g=|f|=x^2|x-s|$는 $x=s$에서만 미분불가하므로 $s={r}$, 즉 $f(x)=x^2(x-({r}))$.\n"
         f"3단계: $f({c})={c}^2\\cdot({c}-({r}))={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "절댓값함수의 미분가능성", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "abscubic", "r": r, "c": c, "ans": int(ans)}
    return rr


def gen_tangent_count_cubic():
    """외부 점에서 y=x³ 에 세 접선 → 정수 k 개수 (킬러, 상)."""
    a = random.choice([2, 3])
    # (a,k) 지나는 접선: k=3a t²-2t³=g(t), 극소 g(0)=0, 극대 g(a)=a³ → 0<k<a³
    ans = a**3 - 1
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-1, ans+3, ans-3])
    q = (f"점 $({a},\\,k)$에서 곡선 $y=x^3$에 서로 다른 세 개의 접선을 그을 수 있도록 하는 "
         f"정수 $k$의 개수는?")
    e = (f"1단계: 접점 $(t,t^3)$에서의 접선 $y-t^3=3t^2(x-t)$가 $({a},k)$를 지나므로 "
         f"$k=3\\cdot{a}\\,t^2-2t^3=g(t)$.\n"
         f"2단계: $g'(t)=-6t(t-{a})$, 극솟값 $g(0)=0$, 극댓값 $g({a})={a}^3={a**3}$.\n"
         f"3단계: 세 접선 ⇔ $0<k<{a**3}$, 정수 $k$는 ${ans}$개.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "곡선 밖의 점에서의 접선", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "tancount", "a": a, "ans": int(ans)}
    return rr


def gen_binomial_prob():
    """B(n,p)에서 P(X=k) 정확 계산 (상). 파라메트릭."""
    from math import comb
    n = random.choice([4, 5, 6])
    _pn, _pd = random.choice([(1, 2), (1, 3), (2, 3), (1, 4)])
    p = sp.Rational(_pn, _pd)
    kk = random.randint(1, n-1)
    ans = sp.Rational(comb(n, kk)) * p**kk * (1-p)**(n-kk)
    dd = [ans*sp.Rational(3, 2), ans/2 if ans/2 != ans else ans*2,
          ans + sp.Rational(1, 10), sp.Rational(comb(n, kk)) * p**kk]
    choices, lab = make_mc(ans, dd)
    q = (f"확률변수 $X$가 이항분포 $\\mathrm{{B}}\\!\\left({n},\\,{_ltx(p)}\\right)$를 따를 때, "
         f"$\\mathrm{{P}}(X={kk})$의 값은?")
    e = (f"1단계: $\\mathrm{{P}}(X={kk})=\\binom{{{n}}}{{{kk}}}"
         f"\\left({_ltx(p)}\\right)^{{{kk}}}\\left({_ltx(1-p)}\\right)^{{{n-kk}}}$.\n"
         f"2단계: 계산하면 ${_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "이항분포의 확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "binprob", "n": n, "p": [p.p, p.q], "k": kk, "ans": [ans.p, ans.q]}
    return rr


def gen_root_count_cubic():
    """g(t)=#{f(x)=t} 불연속(=극값) + 극점·극솟값 → f(c) (킬러, 상)."""
    # p+q 가 짝수여야 (3/2)(p+q) 가 정수 → f가 정수계수 → 답이 정수
    while True:
        p, q = sorted(random.sample([-2, -1, 0, 1, 2, 3], 2))   # p<q: x=p 극대, x=q 극소
        if (p + q) % 2 == 0:
            break
    m = random.randint(-6, 6)                                # 극솟값
    fp = 3*(x - p)*(x - q)
    F0 = sp.integrate(fp, x)                                 # 상수항 0
    C = m - F0.subs(x, q)
    f = sp.expand(F0 + C)
    M = int(f.subs(x, p))                                    # 극댓값
    c = random.choice([v for v in [-3, 4, 5, -4] if v not in (p, q)])
    fc = f.subs(x, c)
    assert fc == int(fc), f"non-integer answer {fc}"
    ans = int(fc)
    choices, lab = make_mc(sp.Integer(ans), [ans-3, ans+3, ans-6, ans+6, M])
    q_ = (f"최고차항의 계수가 $1$인 삼차함수 $f(x)$에 대하여 방정식 $f(x)=t$의 서로 다른 "
          f"실근의 개수를 $g(t)$라 하자. 함수 $g(t)$는 $t={m}$과 $t={M}$에서만 불연속이고, "
          f"$f(x)$는 $x={q}$에서 극솟값 ${m}$을 가질 때, $f({c})$의 값은?")
    e = (f"1단계: $g(t)$의 불연속점은 극값이므로 극솟값 ${m}$, 극댓값 ${M}$. 극소가 $x={q}$, "
         f"극대가 $x={p}$이다.\n"
         f"2단계: $f'(x)=3(x-({p}))(x-({q}))$이므로 적분 후 $f({q})={m}$로 상수를 정하면 "
         f"$f(x)={sp.latex(f)}$.\n"
         f"3단계: $f({c})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "실근의 개수 함수 g(t)", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "rootcount", "p": p, "q": q, "m": m, "c": c, "ans": ans}
    return rr


def gen_quartic_symmetric():
    """대칭 사차함수 f(x)=f(2h-x), 두 점 값 → f(c) (킬러, 상)."""
    h = random.choice([0, 1, 2])
    A = random.choice([-2, -3, -4, 1, 2])      # (x-h)^2 계수
    B = random.randint(-5, 8)                   # 상수
    f = (x - h)**4 + A*(x - h)**2 + B
    # 두 점은 대칭축까지 거리가 서로 달라야 (그래야 A,B가 유일하게 결정됨)
    while True:
        pts = random.sample([h-2, h-1, h+1, h+2, h+3], 2)
        if abs(pts[0]-h) != abs(pts[1]-h):
            break
    p1, p2 = pts
    v1, v2 = int(f.subs(x, p1)), int(f.subs(x, p2))
    c = random.choice([v for v in [h-3, h+4, h-4, h+5] if v not in pts])
    ans = int(f.subs(x, c))
    choices, lab = make_mc(sp.Integer(ans), [ans-4, ans+4, ans-8, ans+8, ans-2])
    q_ = (f"최고차항의 계수가 $1$인 사차함수 $f(x)$가 모든 실수 $x$에 대하여 "
          f"$f(x)=f({2*h}-x)$를 만족시킨다. $f({p1})={v1}$, $f({p2})={v2}$일 때, "
          f"$f({c})$의 값은?")
    e = (f"1단계: $f(x)=f({2*h}-x)$이므로 $f$는 $x={h}$에 대칭, "
         f"$f(x)=(x-{h})^4+A(x-{h})^2+B$ 꼴.\n"
         f"2단계: $f({p1})={v1}$, $f({p2})={v2}$를 연립하면 $A={A}$, $B={B}$.\n"
         f"3단계: $f({c})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "대칭 사차함수", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "quartsym", "h": h, "A": A, "B": B, "c": c, "ans": ans}
    return rr


def gen_seq_recurrence_sum():
    """귀납적 수열의 부분합 (킬러, 상). 브루트포스."""
    a1 = random.choice([20, 30, 40, 25])
    p = random.choice([3, 4, 5])
    qn = random.choice([7, 9, 11])
    N = random.choice([10, 12, 15])
    seq, a = [], a1
    for _ in range(N):
        seq.append(a)
        a = a - p if a > 0 else a + qn
    ans = sum(seq)
    choices, lab = make_mc(sp.Integer(ans), [ans-p, ans+p, ans-qn, ans+qn, ans-2*p])
    q_ = (f"수열 $\\{{a_n\\}}$이 $a_1={a1}$이고, 모든 자연수 $n$에 대하여 "
          f"$a_{{n+1}}=\\begin{{cases}}a_n-{p}&(a_n>0)\\\\ a_n+{qn}&(a_n\\le0)\\end{{cases}}$를 "
          f"만족시킬 때, $\\displaystyle\\sum_{{n=1}}^{{{N}}}a_n$의 값은?")
    e = (f"1단계: 정의에 따라 차례로 계산하면 처음 몇 항은 "
         f"${', '.join(map(str, seq[:8]))},\\dots$ 이다.\n"
         f"2단계: 제${N}$항까지의 합 $\\displaystyle\\sum_{{n=1}}^{{{N}}}a_n={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "수열의 귀납적 정의", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "seqsum", "a1": a1, "p": p, "qn": qn, "N": N, "ans": int(ans)}
    return rr


def gen_funceq_integral():
    """∫_1^x f = x·f(x) - g(x) 꼴 함수방정식 → f 결정 → f(c) (킬러, 상)."""
    # ∫_1^x f dt = x f(x) - A x^4 - B x^2  형태. 미분: f=f+xf'-4Ax^3-2Bx → f'=4Ax^2+2B
    A = random.choice([1, 2, 3])
    B = random.choice([-2, -1, 1, 2])
    # f' = 4A x^2 + 2B → f = (4A/3) x^3 + 2B x + C ; 단 정수 위해 A는 3의 배수 아니어도 분수 → A를 3배수로
    A = random.choice([3, 6])
    fp = 4*A*x**2 + 2*B
    f0 = sp.integrate(fp, x)          # (4A/3)x^3 + 2Bx
    # x=1 대입: ∫_1^1=0 = 1·f(1) - A - B → f(1)=A+B
    C = (A + B) - f0.subs(x, 1)
    f = sp.expand(f0 + C)
    c = random.choice([2, 3, -1, -2])
    ans = int(f.subs(x, c))
    choices, lab = make_mc(sp.Integer(ans), [ans-3, ans+3, ans-6, ans+6, -ans])
    # 수동 부호 조립 금지: RHS = x·f(x) - A x⁴ - B x² 를 sympy 가 렌더링
    rhs_tail = sp.latex(sp.expand(-A*x**4 - B*x**2))      # 항상 '- A x^4 ± B x^2'
    q = (f"다항함수 $f(x)$가 모든 실수 $x$에 대하여 "
         f"$\\displaystyle\\int_1^x f(t)\\,dt=x\\,f(x){rhs_tail}$를 "
         f"만족시킬 때, $f({c})$의 값은?")
    e = (f"1단계: 양변을 $x$로 미분하면 $f(x)=f(x)+xf'(x)"
         f"{sp.latex(sp.expand(sp.diff(-A*x**4 - B*x**2, x)))}$... "
         f"정리하면 $xf'(x)={sp.latex(4*A*x**3 + 2*B*x)}$, "
         f"즉 $f'(x)={sp.latex(4*A*x**2 + 2*B)}$.\n"
         f"2단계: 적분하면 $f(x)={sp.latex(f0)}+C$. 원식에 $x=1$을 넣으면 $0=f(1)-{A}-({B})$에서 "
         f"$f(1)={A+B}$이므로 $C={C}$.\n"
         f"3단계: $f(x)={sp.latex(f)}$이므로 $f({c})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분 함수방정식", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "funceq", "A": A, "B": B, "c": c, "ans": ans}
    return rr


def gen_func_count_odd():
    """f:{1..5}→{1..5}, 곱 홀수·대소·치역 조건 함수 개수 (킬러, 상). 브루트포스."""
    from itertools import product as iproduct
    rcount = random.choice([2, 3])
    cnt = 0
    for f in iproduct(range(1, 6), repeat=5):
        if (f[0]*f[2]*f[4]) % 2 == 1 and f[1] < f[3] and len(set(f)) == rcount:
            cnt += 1
    choices, lab = make_mc(sp.Integer(cnt), [cnt-4, cnt+4, cnt-8, cnt+8, cnt-2])
    q = (f"집합 $X=\\{{1,2,3,4,5\\}}$에 대하여 다음 조건을 만족시키는 함수 $f:X\\to X$의 개수는?\n\n"
         f"(가) $f(1)\\times f(3)\\times f(5)$는 홀수이다.\n(나) $f(2)<f(4)$\n"
         f"(다) 함수 $f$의 치역의 원소의 개수는 ${rcount}$이다.")
    e = (f"1단계: (가)에서 $f(1),f(3),f(5)$는 모두 홀수$(\\{{1,3,5\\}})$.\n"
         f"2단계: (나)에서 $f(2)<f(4)$.\n"
         f"3단계: (다) 치역 원소 ${rcount}$개 조건까지 만족하는 경우를 세면 ${cnt}$이다.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조건부 함수의 개수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "funccount", "r": rcount, "ans": cnt}
    return rr


def gen_cond_prob_dice():
    """주사위 두 번: a*b가 m의 배수일 때 a+b 조건 확률 (킬러, 상). 브루트포스."""
    m = random.choice([3, 4, 6])
    s = random.choice([6, 7, 8])
    cond = [(a, b) for a in range(1, 7) for b in range(1, 7) if (a*b) % m == 0]
    fav = [p for p in cond if p[0]+p[1] <= s]
    ans = sp.Rational(len(fav), len(cond))
    dd = [sp.Rational(len(fav)-1, len(cond)), sp.Rational(len(fav)+1, len(cond)),
          sp.Rational(len(fav), len(cond))+sp.Rational(1, 9), sp.Rational(len(fav)-2, len(cond))]
    choices, lab = make_mc(ans, dd)
    q = (f"한 개의 주사위를 두 번 던질 때 나오는 눈의 수를 차례로 $a$, $b$라 하자. "
         f"$a\\times b$가 ${m}$의 배수일 때, $a+b\\le{s}$일 확률은?")
    e = (f"1단계: $a\\times b$가 ${m}$의 배수인 경우(조건)는 ${len(cond)}$가지이다.\n"
         f"2단계: 그 중 $a+b\\le{s}$인 경우는 ${len(fav)}$가지이다.\n"
         f"3단계: 조건부확률 $=\\dfrac{{{len(fav)}}}{{{len(cond)}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조건부확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "conddice", "m": m, "s": s, "ans": [ans.p, ans.q]}
    return rr


def gen_seq_count_target():
    """a_1=k 귀납수열에서 a_m=target 되는 k(1..N) 합 (킬러, 상). 브루트포스."""
    p = random.choice([3, 4])
    q_ = random.choice([5, 7])
    m = random.choice([4, 5])
    target = random.choice([1, 2, 4])
    def T(a): return a + q_ if a <= 0 else a - p
    lo, hi = 1, 40
    res = []
    for k in range(lo, hi+1):
        a = k
        for _ in range(m-1):
            a = T(a)
        if a == target:
            res.append(k)
    ans = sum(res)
    choices, lab = make_mc(sp.Integer(ans), [ans-p, ans+p, ans-q_, ans+q_, ans-1])
    qtext = (f"수열 $\\{{a_n\\}}$이 $a_1=k$이고, 모든 자연수 $n$에 대하여 "
             f"$a_{{n+1}}=\\begin{{cases}}a_n-{p}&(a_n>0)\\\\ a_n+{q_}&(a_n\\le0)\\end{{cases}}$를 "
             f"만족시킨다. $a_{m}={target}$이 되도록 하는 $40$ 이하의 모든 자연수 $k$의 값의 합은?")
    e = (f"1단계: 각 $k$에 대해 점화식으로 $a_{m}$을 계산한다.\n"
         f"2단계: $a_{m}={target}$이 되는 $k$는 ${res}$이다.\n"
         f"3단계: 합은 ${ans}$이다.\n\n정답: {lab}")
    rr = _rec("대수", "수열의 귀납적 정의", "상", 4, "객관식", qtext, choices, lab, e)
    rr["_meta"] = {"chk": "seqcount", "p": p, "q": q_, "m": m, "target": target, "ans": int(ans)}
    return rr


def gen_intdef_extremum():
    """g(x)=∫_0^x 6(t-p)(t-q)dt 의 극솟값 (킬러, 상). sympy 정적분."""
    p = random.choice([-3, -2, -1, 0, 1])
    q = random.choice([2, 3, 4, 5, 6])
    f = 6 * (t - p) * (t - q)
    g = sp.integrate(f, (t, 0, x))          # 2x³-3(p+q)x²+6pq x (정수계수)
    gmin = int(g.subs(x, q))                # f가 (p,q)에서 음 → x=q에서 극소
    gmax = int(g.subs(x, p))
    choices, lab = make_mc(sp.Integer(gmin),
                           [gmax, gmin + (q - p), gmin - (q - p), gmin + 2, -gmin])
    def _fac(v):  # (t - v) 를 부호에 맞춰 표기
        return "t" if v == 0 else (f"(t-{v})" if v > 0 else f"(t+{-v})")
    fac = f"6{_fac(p)}{_fac(q)}"
    facx = fac.replace("t", "x")
    q_ = f"함수 $g(x)=\\displaystyle\\int_0^x {fac}\\,dt$의 극솟값은?"
    e = (f"1단계: $g'(x)={facx}$이므로 $g$는 $x={p}$에서 극대, $x={q}$에서 극소이다.\n"
         f"2단계: $g(x)=\\int_0^x {fac}\\,dt={sp.latex(sp.expand(g))}$.\n"
         f"3단계: 극솟값 $g({q})={gmin}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분으로 정의된 함수의 극값", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "intdefext", "p": p, "q": q, "ans": gmin}
    return rr


def gen_comb_bounded():
    """x+y+z+w=N (음 아닌 정수, 각 ≤U) 해의 개수 (킬러, 상). 브루트포스."""
    from itertools import product as _ip
    U = random.choice([4, 5, 6, 7])
    N = random.choice([8, 9, 10, 11, 12, 13, 14])
    cnt = sum(1 for c in _ip(range(U + 1), repeat=4) if sum(c) == N)
    # 개수 문제 → 음수 오답 제외 (부족분은 make_mc 가 양수로 합성)
    dd = [d for d in [cnt + 6, cnt - 6, cnt + 12, cnt - 12, cnt + 3] if d > 0]
    choices, lab = make_mc(sp.Integer(cnt), dd)
    q_ = (f"방정식 $x+y+z+w={N}$을 만족시키는 음이 아닌 정수 $x,y,z,w$의 순서쌍 중 "
          f"$x,y,z,w$가 모두 ${U}$ 이하인 것의 개수는?")
    e = (f"1단계: 제한이 없으면 ${{}}_{{4}}H_{{{N}}}={sp.binomial(N+3,3)}$가지.\n"
         f"2단계: 각 변수가 ${U}$ 이하라는 상한을 포함배제(또는 직접 셈)로 처리한다.\n"
         f"3단계: 조건을 만족시키는 순서쌍의 개수는 ${cnt}$이다.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "중복조합의 경우의 수", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "combbnd", "N": N, "U": U, "ans": cnt}
    return rr


def gen_log_floor_sum():
    """∑_{n=1}^{N} ⌊log_2 n⌋ 의 값 (킬러, 상). 정수합."""
    import math
    N = random.randint(16, 48)
    vals = [int(math.floor(math.log2(nn))) for nn in range(1, N + 1)]
    ans = sum(vals)
    choices, lab = make_mc(sp.Integer(ans), [ans + 2, ans - 2, ans + 5, ans - 5, ans + 1])
    q_ = (f"$\\displaystyle\\sum_{{n=1}}^{{{N}}}\\left\\lfloor \\log_2 n \\right\\rfloor$의 값은? "
          f"(단, $\\lfloor x \\rfloor$는 $x$를 넘지 않는 최대 정수)")
    e = (f"1단계: $\\lfloor\\log_2 n\\rfloor=k \\iff 2^k\\le n<2^{{k+1}}$이다.\n"
         f"2단계: 구간별로 같은 값 $k$를 갖는 $n$의 개수를 세어 더한다.\n"
         f"3단계: $n=1$부터 ${N}$까지 합하면 ${ans}$이다.\n\n정답: {lab}")
    rr = _rec("대수", "로그와 가우스 기호", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "logfloor", "N": N, "ans": ans}
    return rr


# ======================================================================
#  하 난이도 (2점) — 2028 예시문항 체제의 1~3번용
# ======================================================================
def gen_exp_calc():
    """유리지수 계산 (하, 2점). 예: 4^(3/2)=8."""
    base, num, den = random.choice([(4, 3, 2), (8, 2, 3), (9, 3, 2), (27, 2, 3),
                                    (16, 3, 4), (32, 3, 5), (8, 4, 3), (4, 5, 2)])
    r = sp.Rational(num, den)
    ans = int(sp.Integer(base) ** r)
    choices, lab = make_mc(sp.Integer(ans), [ans//2, ans*2, ans-2, ans+2])
    q = f"${base}^{{{_ltx(r)}}}$의 값은?"
    p_ = sp.factorint(base)
    pb, pe_ = next(iter(p_.items()))
    e = (f"1단계: ${base}={pb}^{{{pe_}}}$이므로 ${base}^{{{_ltx(r)}}}"
         f"={pb}^{{{_ltx(pe_*r)}}}$.\n"
         f"2단계: $={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "지수의 계산", "하", 2, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "expcalc", "base": base, "num": num, "den": den, "ans": ans}
    return rr


def gen_diff_at_point():
    """다항함수 미분계수 (하, 2점)."""
    a = random.choice([-2, -1, 1, 2, 3])
    b = random.choice([-3, -1, 1, 2, 4])
    c = random.choice([1, 2, -1])
    f = x**3 + a*x**2 + b*x
    ans = int(sp.diff(f, x).subs(x, c))
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4])
    q = f"함수 $f(x)={sp.latex(f)}$에 대하여 $f'({c})$의 값은?"
    e = (f"1단계: $f'(x)={sp.latex(sp.diff(f, x))}$.\n"
         f"2단계: $f'({c})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "미분계수", "하", 2, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "diffat", "a": a, "b": b, "c": c, "ans": ans}
    return rr


def gen_arith_term():
    """등차수열 특정 항 (하, 2점)."""
    a1 = random.randint(1, 9)
    d = random.choice([2, 3, 4, -2, 5])
    kk = random.choice([5, 6, 7, 8])
    ans = a1 + (kk - 1) * d
    choices, lab = make_mc(sp.Integer(ans), [ans-d, ans+d, ans-2*d, ans+2*d])
    q = (f"첫째항이 ${a1}$이고 공차가 ${d}$인 등차수열 $\\{{a_n\\}}$에 대하여 "
         f"$a_{{{kk}}}$의 값은?")
    e = (f"1단계: $a_n=a_1+(n-1)d$.\n"
         f"2단계: $a_{{{kk}}}={a1}+{kk-1}\\cdot({d})={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "등차수열", "하", 2, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "arithterm", "a1": a1, "d": d, "k": kk, "ans": ans}
    return rr


def gen_log_calc():
    """로그 계산 (하, 2점). log_a(a^k·m) 꼴."""
    a = random.choice([2, 3, 5])
    kk = random.randint(2, 5)
    ans = kk
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2])
    val = a ** kk
    q = f"$\\log_{{{a}}}{val}$의 값은?"
    e = (f"1단계: ${val}={a}^{{{kk}}}$.\n"
         f"2단계: $\\log_{{{a}}}{a}^{{{kk}}}={kk}$.\n\n정답: {lab}")
    rr = _rec("대수", "로그의 계산", "하", 2, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "logcalc", "a": a, "k": kk, "ans": ans}
    return rr


def gen_limit_simple():
    """약분형 극한 (하, 2점). lim (x²-a²)/(x-a) = 2a."""
    a = random.choice([1, 2, 3, -2])
    ans = 2 * a
    choices, lab = make_mc(sp.Integer(ans), [a, ans-1, ans+1, a*a])
    sgn = "-" if a >= 0 else "+"
    q = (f"$\\displaystyle\\lim_{{x\\to{a}}}\\dfrac{{x^2-{a*a}}}{{x{sgn}{abs(a)}}}$의 값은?")
    e = (f"1단계: $x^2-{a*a}=(x-{a})(x+{a})$로 약분.\n"
         f"2단계: $\\lim_{{x\\to{a}}}(x+{a})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "함수의 극한(약분)", "하", 2, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "limsimple", "a": a, "ans": ans}
    return rr


# ======================================================================
#  (가)(나) 조건 결합형 킬러
# ======================================================================
def gen_cond_cubic_integral():
    """(가) 극값 조건 + (나) 정적분 조건 → 삼차함수 결정 → f(t) (킬러, 상)."""
    while True:
        p, r = random.sample([-2, -1, 0, 1, 2, 3], 2)   # f'=3(x-p)(x-r)
        if (p + r) % 2 == 0:                            # a 정수 조건
            break
    m = random.randint(-5, 5)                           # 극값 f(p)=m
    a = -3 * (p + r) // 2
    b = 3 * p * r
    c0 = m - (p**3 + a*p**2 + b*p)
    f = x**3 + a*x**2 + b*x + c0
    q_up = random.choice([1, 2])
    I = sp.integrate(f, (x, 0, q_up))                   # (나) 조건값 (유리수 허용)
    tgt = random.choice([v for v in [-3, -2, 4, 5] if v not in (p, r)])
    ans = int(f.subs(x, tgt))
    # 유일성 자가검증: 세 조건으로 연립해 풀었을 때 구성한 f 와 일치해야 함
    A, B, C = sp.symbols("A B C")
    g = x**3 + A*x**2 + B*x + C
    sol = sp.solve([sp.diff(g, x).subs(x, p),
                    g.subs(x, p) - m,
                    sp.integrate(g, (x, 0, q_up)) - I], [A, B, C], dict=True)
    assert len(sol) == 1 and sol[0][A] == a and sol[0][B] == b and sol[0][C] == c0
    choices, lab = make_mc(sp.Integer(ans), [ans-3, ans+3, ans-6, ans+6, -ans])
    q = (f"최고차항의 계수가 $1$인 삼차함수 $f(x)$가 다음 조건을 만족시킬 때, "
         f"$f({tgt})$의 값은?\n\n"
         f"(가) $f(x)$는 $x={p}$에서 극값 ${m}$을 갖는다.\n"
         f"(나) $\\displaystyle\\int_0^{{{q_up}}}f(x)\\,dx={_ltx(I)}$")
    e = (f"1단계: (가)에서 $f'({p})=0$, $f({p})={m}$. "
         f"$f(x)=x^3+Ax^2+Bx+C$로 놓으면 두 식이 나온다.\n"
         f"2단계: (나)의 정적분을 전개해 셋을 연립하면 $A={a}$, $B={b}$, $C={c0}$, "
         f"즉 $f(x)={sp.latex(sp.expand(f))}$.\n"
         f"3단계: $f({tgt})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "조건을 만족시키는 삼차함수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "condcubic", "p": p, "r": r, "m": m, "q": q_up,
                   "tgt": tgt, "ans": ans}
    return rr


def gen_piecewise_diff():
    """(가) 미분가능 + (나)(다) 함숫값 → 구간별 함수 결정 → g(t) (킬러, 상)."""
    kk = random.choice([-1, 0, 1, 2])
    a = random.choice([-3, -2, -1, 1, 2])
    b = random.randint(-4, 4)
    c = 3*kk**2 + a                       # 미분가능 ⇒ 기울기 일치
    d = (kk**3 + a*kk + b) - c*kk         # 연속
    u = kk - random.choice([1, 2])        # (나) 왼쪽 점
    s = kk + random.choice([1, 2])        # (다) 오른쪽 점
    w = u**3 + a*u + b
    v = c*s + d
    tgt = random.choice([t_ for t_ in [kk-3, kk+3, kk+4] if t_ not in (u, s)])
    ans = (tgt**3 + a*tgt + b) if tgt <= kk else (c*tgt + d)
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"함수 $g(x)=\\begin{{cases}}x^3+ax+b&(x\\le {kk})\\\\ cx+d&(x>{kk})\\end{{cases}}$가 "
         f"다음 조건을 만족시킬 때, $g({tgt})$의 값은? (단, $a,b,c,d$는 상수)\n\n"
         f"(가) $g(x)$는 실수 전체의 집합에서 미분가능하다.\n"
         f"(나) $g({u})={w}$\n(다) $g({s})={v}$")
    e = (f"1단계: (가)에서 $x={kk}$의 연속·미분계수 일치: "
         f"$c=3\\cdot{kk}^2+a$, ${kk}^3+a\\cdot{kk}+b=c\\cdot{kk}+d$.\n"
         f"2단계: (나) $g({u})={u}^3+a\\cdot{u}+b={w}$, (다) $g({s})=c\\cdot{s}+d={v}$를 "
         f"연립하면 $a={a}$, $b={b}$, $c={c}$, $d={d}$.\n"
         f"3단계: $g({tgt})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "구간별 함수의 미분가능성", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "pwdiff", "k": kk, "a": a, "b": b, "u": u, "s": s,
                   "tgt": tgt, "ans": int(ans)}
    return rr


def gen_involution_count():
    """f∘f=항등 (대합) + 추가 조건 함수 개수 (킬러, 상). 브루트포스."""
    from itertools import product as _ip
    aa = random.choice([1, 2, 3])
    variant = random.choice(["nonfix1", "nonfix2", "pair", "exact2"])
    if variant == "nonfix1":
        cond_txt = f"(나) $f({aa})\\ne {aa}$"
        cond = lambda f: f[aa-1] != aa
    elif variant == "nonfix2":
        bb = aa % 5 + 1
        cond_txt = f"(나) $f({aa})\\ne {aa}$이고 $f({bb})\\ne {bb}$"
        cond = lambda f: f[aa-1] != aa and f[bb-1] != bb
    elif variant == "pair":
        bb = aa % 5 + 1
        cond_txt = f"(나) $f({aa})={bb}$"
        cond = lambda f: f[aa-1] == bb
    else:
        cond_txt = "(나) $f(x)\\ne x$인 $x$의 개수는 $2$이다."
        cond = lambda f: sum(1 for i in range(5) if f[i] != i+1) == 2
    cnt = 0
    for f in _ip(range(1, 6), repeat=5):
        if all(f[f[i]-1] == i+1 for i in range(5)) and cond(f):
            cnt += 1
    choices, lab = make_mc(sp.Integer(cnt), [cnt-2, cnt+2, cnt+4, cnt+6, max(1, cnt-4)])
    q = (f"집합 $X=\\{{1,2,3,4,5\\}}$에 대하여 다음 조건을 만족시키는 함수 "
         f"$f:X\\to X$의 개수는?\n\n"
         f"(가) 모든 $x\\in X$에 대하여 $f(f(x))=x$이다.\n{cond_txt}")
    e = (f"1단계: (가)에서 $f$는 대합 — 치환 구조로 보면 고정점과 호환(2-순환)만 가능하다.\n"
         f"2단계: i) 고정점 개수별로 경우를 나누고, ii) 각 경우에 (나) 조건을 적용해 센다.\n"
         f"3단계: 합하면 ${cnt}$개.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "대합 함수의 개수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "involution", "variant": variant, "a": aa, "ans": cnt}
    return rr


def _trig_count_rule(r):
    """sin x = r (0≤x<2π) 의 해 개수: |r|=1→1, |r|<1→2, |r|>1 또는 허수→0."""
    r = sp.nsimplify(r)
    if sp.im(r) != 0:
        return 0
    a = abs(float(r))
    if a > 1 + 1e-9:
        return 0
    return 1 if abs(a - 1) < 1e-9 else 2


def gen_units_digit():
    """거듭제곱의 일의 자리 (킬러, 상). 일의 자리 주기, 정확.

    $n^k$의 일의 자리는 $n$의 일의 자리의 거듭제곱 주기로 결정된다.
    """
    base = random.choice([2, 3, 7, 8, 12, 13, 17, 23])
    k = random.choice([50, 99, 100, 2024, 2025, 77])
    d = base % 10
    cyc = []
    v = d
    for _ in range(4):
        cyc.append(v % 10)
        v = (v*d) % 10
    # 주기 = cyc의 실제 사이클 길이
    period = cyc.index(cyc[0], 1) if cyc[0] in cyc[1:] else 4
    seq = cyc[:period] if period else cyc
    period = len(seq)
    ans = seq[(k-1) % period]
    choices, lab = make_mc(sp.Integer(ans), [(ans+2) % 10, (ans+4) % 10,
                                             (ans+6) % 10, (ans+8) % 10, (ans+1) % 10])
    q = (f"${base}^{{{k}}}$의 일의 자리의 숫자는?")
    pos = (k-1) % period + 1
    e = (f"1단계: ${base}$(일의 자리 ${d}$)의 거듭제곱의 일의 자리는 "
         f"${', '.join(map(str, seq))}$가 주기 ${period}$로 반복된다.\n"
         f"2단계: ${k}=\\,$(주기 ${period}$의 배수)$+{pos if pos != period else 0}$ 위치이므로 "
         f"일의 자리는 ${ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "거듭제곱의 일의 자리", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "unitsdigit", "base": base, "k": k, "ans": ans}
    return rr


def gen_combination_symmetry():
    """조합의 대칭성 ₙCᵣ=ₙC_s (킬러, 상). r+s=n, 정확.

    $_nC_r=\\,_nC_s$ ($r\\ne s$)이면 $r+s=n$. 만족하는 미지수의 값(들)을 구한다.
    """
    from math import comb
    n = random.choice([8, 10, 12, 14])
    r = random.randint(1, n//2 - 1)
    s = n - r
    ask = random.choice(["sum", "other"])
    if ask == "sum":
        ans = r + s; query = "$x$의 값의 합은?"
    else:
        ans = s; query = "$x$의 값 중 큰 것은?"
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, n+1])
    q = (f"$_{{{n}}}\\mathrm{{C}}_{{x}}=\\,_{{{n}}}\\mathrm{{C}}_{{{r}}}$을 만족시키는 "
         f"자연수 $x$에 대하여, {query}")
    e = (f"1단계: $_n\\mathrm{{C}}_r=\\,_n\\mathrm{{C}}_{{n-r}}$이므로 $x={r}$ 또는 "
         f"$x={n}-{r}={s}$.\n"
         f"2단계: {query.replace('?','')} ${ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조합의 대칭성", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "combsym", "n": n, "r": r, "ask": ask, "ans": ans}
    return rr


def gen_definite_integral_param():
    """정적분 = 값 조건으로 상한 결정 (킬러, 상). 단조 → 유일근, 정확.

    $\\int_0^a (3x^2+2bx)\\,dx=a^3+ba^2=V$ ($b\\ge0$, $f\\ge0$)에서 양수 $a$ 유일.
    """
    a = random.choice([1, 2, 3, 4])
    b = random.choice([0, 1, 2, 3])
    V = a**3 + b*a**2
    fl = f"3x^2+{2*b}x" if b else "3x^2"
    choices, lab = make_mc(sp.Integer(a), [a+1, a-1 if a > 1 else a+2, a+2, a*2, a+3])
    q = (f"양수 $a$에 대하여 $\\displaystyle\\int_0^a ({fl})\\,dx={V}$일 때, $a$의 값은?")
    e = (f"1단계: $\\displaystyle\\int_0^a ({fl})\\,dx=\\Big[x^3+{b}x^2\\Big]_0^a=a^3+{b}a^2$.\n"
         f"2단계: $a^3+{b}a^2={V}$를 만족하는 양수 $a={a}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분과 미정 상한", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "intparam", "a": a, "b": b, "V": V, "ans": a}
    return rr


def gen_without_replacement_prob():
    """비복원 추출 조건부확률 (킬러, 상). 순서쌍 브루트포스, 정확.

    흰 $w$, 검 $b$개에서 차례로 비복원 추출. 첫 번째가 검은색일 때 두 번째가
    흰색일 확률 등.
    """
    from itertools import permutations
    w = random.randint(2, 5)
    b = random.randint(2, 5)
    balls = ['W']*w + ['B']*b
    idx = list(range(w+b))
    first_color, second_color = random.choice([("B", "W"), ("W", "B"), ("W", "W"), ("B", "B")])
    num = den = 0
    for i in idx:
        for j in idx:
            if i == j:
                continue
            if balls[i] == first_color:
                den += 1
                if balls[j] == second_color:
                    num += 1
    ans = sp.Rational(num, den)
    cmap = {"W": "흰색", "B": "검은색"}
    dd = [sp.Rational(num+1, den), sp.Rational(num-1, den) if num >= 2 else sp.Rational(num+2, den),
          sp.Rational(w if second_color == "W" else b, w+b), ans*sp.Rational(3, 2)]
    choices, lab = make_mc(ans, [d for d in dd if 0 < d < 1 and d != ans])
    q = (f"흰 공 ${w}$개와 검은 공 ${b}$개가 들어 있는 주머니에서 공을 한 개씩 "
         f"차례로 비복원으로 꺼낸다. 첫 번째 꺼낸 공이 {cmap[first_color]}이었을 때, "
         f"두 번째 꺼낸 공이 {cmap[second_color]}일 확률은?")
    e = (f"1단계: 첫 공이 {cmap[first_color]}인 조건에서 남은 공으로 두 번째를 따진다.\n"
         f"2단계: 조건부확률 $=\\dfrac{{{num}}}{{{den}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "비복원 추출의 조건부확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "noreplace", "w": w, "b": b, "fc": first_color,
                   "sc": second_color, "ans": [ans.p, ans.q]}
    return rr


def gen_factor_theorem():
    """인수정리 — 나누어떨어짐 조건 (킬러, 상). 연립, 정확.

    $f(x)=x^3+ax^2+bx+c$가 $(x-p)(x-q)$로 나누어떨어지면 $f(p)=f(q)=0$.
    두 조건으로 미정계수를 정한 뒤 합 등을 구한다.
    """
    p, qq, r3 = random.sample([-3, -2, -1, 1, 2, 3], 3)
    # f(x)=(x-p)(x-q)(x-r3) = x³ + A x² + B x + C
    A = -(p+qq+r3)
    B = p*qq + qq*r3 + r3*p
    C = -p*qq*r3
    # 미정계수: x³ + A x² + b x + c 가 (x-p)(x-q)로 나누어떨어짐 → b,c 미지수
    ask = random.choice(["sum", "third"])
    if ask == "sum":
        ans = B + C; query = "$b+c$의 값은?"
    else:
        ans = r3; query = "나머지 한 근의 값은?"
    Aterm = f"+{A}x^2" if A > 0 else (f"{A}x^2" if A < 0 else "")
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-1, ans+1, -ans])
    q_ = (f"다항식 $f(x)=x^3{Aterm}+bx+c$가 $(x-({p}))(x-({qq}))$로 나누어떨어질 때, "
          f"{query} (단, $b$, $c$는 상수)")
    e = (f"1단계: $f({p})=0$, $f({qq})=0$ (인수정리). 두 식을 연립하면 "
         f"$b={B}$, $c={C}$.\n"
         f"2단계: 이때 $f(x)=(x-({p}))(x-({qq}))(x-({r3}))$. "
         f"{query.replace('의 값은?','')}$={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "인수정리와 나누어떨어짐", "상", 4, "객관식", q_, choices, lab, e)
    rr["_meta"] = {"chk": "factorthm", "p": p, "q": qq, "r3": r3, "ask": ask, "ans": ans}
    return rr


def gen_geometric_three():
    """등비수열을 이루는 세 수 (킬러, 상). 곱·합 조건, 정확.

    세 수 $\\dfrac ar,\\,a,\\,ar$의 곱 $=a^3$, 합 $=a(1/r+1+r)$. 곱으로 $a$ 결정.
    """
    a = random.choice([2, 3, 4])
    r = random.choice([2, 3])
    nums = [a, a*r, a*r*r]                    # a, ar, ar² (양수 등비)
    prod = nums[0]*nums[1]*nums[2]
    Ssum = sum(nums)
    ask = random.choice(["mid", "max", "sqsum"])
    if ask == "mid":
        ans = nums[1]; query = "가운데 수는?"
    elif ask == "max":
        ans = nums[2]; query = "가장 큰 수는?"
    else:
        ans = sum(v*v for v in nums); query = "세 수의 제곱의 합은?"
    choices, lab = make_mc(sp.Integer(ans), [ans-a, ans+a, ans-2, ans+2, ans*2])
    q = (f"등비수열을 이루는 세 양수의 곱이 ${prod}$이고 합이 ${Ssum}$이다. "
         f"{query} (단, 세 수는 증가한다.)")
    e = (f"1단계: 세 수를 ${nums[0]},\\,{nums[1]},\\,{nums[2]}$ (공비 ${r}$)로 놓으면 "
         f"곱 $=({nums[1]})^3={prod}$에서 가운데 수 ${nums[1]}$.\n"
         f"2단계: 합 ${Ssum}$ 조건으로 공비 ${r}$ 확정. "
         f"{query.replace('?','')} ${ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "등비수열을 이루는 세 수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "geo3", "a": a, "r": r, "ask": ask, "ans": ans}
    return rr


def gen_quadratic_root_relation():
    """이차방정식 근과 계수의 관계 (킬러, 상). 대칭식, 정확.

    $x^2-px+q=0$의 두 근 $\\alpha,\\beta$에 대해 $\\alpha+\\beta=p$, $\\alpha\\beta=q$.
    대칭식의 값을 구한다.
    """
    p = random.randint(-6, 6)
    q = random.randint(-6, 6)
    if q == 0:
        q = 2
    kind = random.choice(["sq", "cube", "recip", "diffsq"])
    if kind == "sq":
        ans = sp.Integer(p*p - 2*q); tgt = "\\alpha^2+\\beta^2"
        rule = "(\\alpha+\\beta)^2-2\\alpha\\beta=p^2-2q"
    elif kind == "cube":
        ans = sp.Integer(p**3 - 3*p*q); tgt = "\\alpha^3+\\beta^3"
        rule = "(\\alpha+\\beta)^3-3\\alpha\\beta(\\alpha+\\beta)=p^3-3pq"
    elif kind == "recip":
        ans = sp.Rational(p, q); tgt = "\\dfrac1\\alpha+\\dfrac1\\beta"
        rule = "\\dfrac{\\alpha+\\beta}{\\alpha\\beta}=\\dfrac pq"
    else:
        ans = sp.Integer(p*p - 4*q); tgt = "(\\alpha-\\beta)^2"
        rule = "(\\alpha+\\beta)^2-4\\alpha\\beta=p^2-4q"
    pterm = f"-{p}x" if p > 0 else (f"+{-p}x" if p < 0 else "")
    qterm = f"+{q}" if q > 0 else f"{q}"
    traps = {}
    if isinstance(ans, sp.Rational) and ans.q != 1:
        choices, lab = make_mc(ans, [ans+sp.Rational(1, q if q else 1), ans-1, ans*2,
                                     ans*sp.Rational(1, 2), -ans])
    else:
        # 함정: 곱(교차)항을 빠뜨린 흔한 실수
        forgot = {"sq": p*p, "cube": p**3, "diffsq": p*p}.get(kind)
        dd = [ans-2, ans+2, ans-4, ans+4, -ans]
        if forgot is not None and forgot != int(ans):
            dd = [(sp.Integer(forgot),
                   "$(\\alpha+\\beta)$의 거듭제곱만 쓰고 곱(교차)항을 빠뜨림")] + dd
        choices, lab, traps = make_mc_tagged(ans, dd)
    q_ = (f"이차방정식 $x^2{pterm}{qterm}=0$의 두 근을 $\\alpha$, $\\beta$라 할 때, "
          f"${tgt}$의 값은?")
    e = (f"1단계: 근과 계수의 관계에서 $\\alpha+\\beta={p}$, $\\alpha\\beta={q}$.\n"
         f"2단계: ${tgt}={rule}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("대수", "이차방정식 근과 계수의 관계", "상", 4, "객관식", q_, choices, lab, e)
    if traps:
        rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "quadroot", "p": p, "q": q, "kind": kind,
                   "ans": [ans.p, ans.q] if isinstance(ans, sp.Rational) else [int(ans), 1]}
    return rr


def gen_abs_value_equation():
    """절댓값 방정식 |x-a|+|x-b|=c (킬러, 상). 정확.

    $a<b$, $c>b-a$이면 해는 $x=\\dfrac{a+b\\pm c}{2}$ 두 개. 두 해의 곱·합 등.
    """
    a, b = sorted(random.sample([-4, -2, 0, 1, 3, 5], 2))
    c = (b - a) + random.choice([2, 4, 6])     # > b-a 보장
    if (a + b + c) % 2:
        c += 1
    hi = (a + b + c)//2
    lo = (a + b - c)//2
    ask = random.choice(["sum", "prod", "max"])
    if ask == "sum":
        ans = a + b; query = "모든 실근의 합은?"
    elif ask == "prod":
        ans = lo*hi; query = "모든 실근의 곱은?"
    else:
        ans = hi; query = "가장 큰 실근은?"
    aterm = f"-({a})" if a != 0 else ""
    bterm = f"-({b})" if b != 0 else ""
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-1, ans+1, -ans])
    q = (f"방정식 $|x{aterm}|+|x{bterm}|={c}$의 {query}")
    e = (f"1단계: $|x-({a})|+|x-({b})|={c}$ ($c={c}>{b}-({a})={b-a}$)이므로 해는 "
         f"$x=\\dfrac{{{a}+{b}\\pm {c}}}{{2}}$, 즉 $x={lo}$ 또는 $x={hi}$.\n"
         f"2단계: {query.replace('?','')} ${ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "절댓값 방정식", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "abseq", "a": a, "b": b, "c": c, "ask": ask, "ans": ans}
    return rr


def gen_quadratic_inequality_intcount():
    """이차부등식의 정수해 개수 (킬러, 상). 근 사이 정수, 정확.

    $(x-r_1)(x-r_2)<0 \\iff r_1<x<r_2$. 정수해 개수 $=r_2-r_1-1$.
    """
    r1, r2 = sorted(random.sample([-5, -3, -2, -1, 1, 2, 3, 5, 6], 2))
    p = r1 + r2
    q = r1*r2
    ans = r2 - r1 - 1
    pterm = f"-{p}x" if p > 0 else (f"+{-p}x" if p < 0 else "")
    qterm = f"+{q}" if q > 0 else (f"{q}" if q < 0 else "")
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, r2-r1])
    qtxt = (f"부등식 $x^2{pterm}{qterm}<0$을 만족시키는 정수 $x$의 개수는?")
    e = (f"1단계: $x^2{pterm}{qterm}=(x-({r1}))(x-({r2}))<0$에서 ${r1}<x<{r2}$.\n"
         f"2단계: 이 범위의 정수는 ${r1+1}$부터 ${r2-1}$까지 ${ans}$개.\n\n정답: {lab}")
    rr = _rec("대수", "이차부등식의 정수해", "상", 4, "객관식", qtxt, choices, lab, e)
    rr["_meta"] = {"chk": "quadineqcnt", "r1": r1, "r2": r2, "ans": ans}
    return rr


def gen_synthetic_division():
    """조립제법 — 다항식의 몫 (킬러, 상). sympy div 검증.

    $f(x)=x^3+ax^2+bx+c$를 $x-k$로 나눈 몫 $Q(x)$에 대해 $Q(t)$ 등을 구한다.
    """
    a = random.randint(-4, 4)
    b = random.randint(-4, 4)
    c = random.randint(-4, 4)
    k = random.choice([-2, -1, 1, 2])
    f = x**3 + a*x**2 + b*x + c
    Q, R = sp.div(f, x - k, x)
    t = random.choice([v for v in [-2, -1, 0, 2, 3] if v != k])
    ans = int(Q.subs(x, t))
    fl = sp.latex(sp.expand(f))
    Ql = sp.latex(sp.expand(Q))
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-3, ans+3, -ans])
    q = (f"다항식 $f(x)={fl}$를 $x-({k})$로 나눈 몫을 $Q(x)$라 할 때, $Q({t})$의 값은?")
    e = (f"1단계: 조립제법(또는 직접 나눗셈)으로 $Q(x)={Ql}$ (나머지 ${int(R)}$).\n"
         f"2단계: $Q({t})={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "다항식의 나눗셈(조립제법)", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "synthdiv", "a": a, "b": b, "c": c, "k": k, "t": t, "ans": ans}
    return rr


def gen_variance_direct():
    """확률변수의 분산 V(X)·V(aX+b) (킬러, 상). V=E(X²)-E(X)², 정확.

    분포표(모든 확률 주어짐)에서 $\\mathrm{V}(X)=\\mathrm{E}(X^2)-\\{\\mathrm{E}(X)\\}^2$,
    $\\mathrm{V}(aX+b)=a^2\\mathrm{V}(X)$.
    """
    xs = sorted(random.sample([0, 1, 2, 3, 4], 4))
    D = random.choice([8, 10])
    while True:
        cuts = sorted(random.sample(range(1, D), 3))
        ps = [cuts[0], cuts[1]-cuts[0], cuts[2]-cuts[1], D-cuts[2]]
        if min(ps) >= 1:
            break
    probs = [sp.Rational(p, D) for p in ps]
    E = sum(xs[i]*probs[i] for i in range(4))
    V = sum(xs[i]**2*probs[i] for i in range(4)) - E**2
    a = random.choice([1, 2, 3])
    ans = a*a*V
    rows_x = " & ".join(str(v) for v in xs)
    rows_p = " & ".join(_ltx(p) for p in probs)
    target = "\\mathrm{V}(X)" if a == 1 else f"\\mathrm{{V}}({a}X+1)"
    choices, lab = make_mc(ans, [ans+sp.Rational(1, 4), ans-sp.Rational(1, 4),
                                 V if a != 1 else 2*V, a*V, ans+1])
    q = (f"확률변수 $X$의 확률분포를 표로 나타내면 다음과 같다.\n\n"
         f"$\\begin{{array}}{{c|cccc|c}} X & {rows_x} & \\text{{합계}} \\\\\\hline "
         f"\\mathrm{{P}}(X=x) & {rows_p} & 1 \\end{{array}}$\n\n"
         f"${target}$의 값은?")
    e = (f"1단계: $\\mathrm{{E}}(X)={_ltx(E)}$, $\\mathrm{{E}}(X^2)={_ltx(V+E**2)}$.\n"
         f"2단계: $\\mathrm{{V}}(X)=\\mathrm{{E}}(X^2)-\\{{\\mathrm{{E}}(X)\\}}^2={_ltx(V)}$.\n"
         + (f"3단계: ${target}={a}^2\\mathrm{{V}}(X)={_ltx(ans)}$.\n\n" if a != 1 else "\n")
         + f"정답: {lab}")
    rr = _rec("확률과 통계", "확률변수의 분산", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "vardirect", "xs": xs, "ps": ps, "D": D, "a": a,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_binomial_normal_approx():
    """이항분포의 정규근사 확률 (킬러, 상). B(n,p)≈N(np,npq), 표값.

    $X\\sim B(n,p)$, 큰 $n$에서 $X\\approx N(np,npq)$. 표준화 후 표값 이용.
    """
    n, p, sig = random.choice([
        (100, sp.Rational(1, 2), 5), (100, sp.Rational(1, 5), 4),
        (48, sp.Rational(1, 4), 3), (100, sp.Rational(9, 10), 3),
        (25, sp.Rational(1, 5), 2)])
    mu = int(n*p)
    TBL = {1: "0.3413", 2: "0.4772"}
    z, tv = random.choice(list(TBL.items()))
    a = mu + z*sig
    ans_txt = f"{0.5 - float(tv):.4f}".rstrip("0").rstrip(".")
    pool = sorted({ans_txt, tv, f"{0.5+float(tv):.4f}".rstrip('0').rstrip('.'),
                   "0.1587" if ans_txt != "0.1587" else "0.0228", "0.0228" if ans_txt != "0.0228" else "0.6915"}, key=float)
    while len(pool) < 5:
        pool.append(f"{0.5-float(tv)+0.07*len(pool):.4f}")
    pool = sorted(set(pool), key=float)[:5]
    if ans_txt not in pool:
        pool[-1] = ans_txt; pool = sorted(set(pool), key=float)
    idx = pool.index(ans_txt)
    choices = [f"${v}$" for v in pool]
    lab = f"{idx+1}번 (${ans_txt}$)"
    q = (f"확률변수 $X$가 이항분포 $\\mathrm{{B}}\\!\\left({n},\\,{_ltx(p)}\\right)$를 따를 때, "
         f"$X$가 정규분포를 따른다고 보고 $\\mathrm{{P}}(X\\ge {a})$를 구하면? (단, "
         f"$\\mathrm{{P}}(0\\le Z\\le {z})={tv}$)")
    e = (f"1단계: $\\mathrm{{E}}(X)={n}\\times{_ltx(p)}={mu}$, "
         f"$\\mathrm{{V}}(X)={n}\\times{_ltx(p)}\\times{_ltx(1-p)}={sig**2}$, 표준편차 ${sig}$.\n"
         f"2단계: $Z=\\dfrac{{X-{mu}}}{{{sig}}}$, $X={a}\\Rightarrow Z={z}$.\n"
         f"3단계: $\\mathrm{{P}}(X\\ge{a})=0.5-{tv}={ans_txt}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "이항분포의 정규근사", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "binnorm", "tv": tv, "ans": ans_txt}
    return rr


def gen_combination_repetition():
    """중복조합 ₙHᵣ (킬러, 상). C(n+r-1,r), 정확.

    서로 다른 $n$종류에서 중복을 허락하여 $r$개를 고르는 경우의 수 $={}_nH_r
    =\\binom{n+r-1}{r}$.
    """
    from math import comb
    n = random.choice([3, 4, 5])
    r = random.choice([3, 4, 5, 6])
    ans = comb(n+r-1, r)
    item = random.choice(["사탕", "공", "과일", "음료"])
    s = max(2, round(ans/8))
    choices, lab = make_mc(sp.Integer(ans), [ans-s, ans+s, comb(n+r-1, n-1)
                                             if comb(n+r-1, n-1) != ans else ans+2*s,
                                             ans*2, ans+2*s])
    q = (f"서로 다른 종류의 {item} ${n}$종류가 있다. 중복을 허락하여 ${r}$개를 "
         f"고르는 경우의 수는? (단, 각 종류는 ${r}$개 이상씩 있다.)")
    e = (f"1단계: 중복조합 ${{}}_{{{n}}}\\mathrm{{H}}_{{{r}}}"
         f"=\\binom{{{n}+{r}-1}}{{{r}}}=\\binom{{{n+r-1}}}{{{r}}}$.\n"
         f"2단계: 계산하면 ${ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "중복조합", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "combrep", "n": n, "r": r, "ans": ans}
    return rr


def gen_arith_three_condition():
    """등차수열을 이루는 세 수 (킬러, 상). 합·곱 조건으로 결정, 정확.

    세 수 $a-d,\\,a,\\,a+d$의 합 $=3a$, 곱 $=a(a^2-d^2)$. 합과 곱으로 $a,d$ 결정.
    """
    a = random.randint(3, 8)
    d = random.choice([1, 2, 3])
    nums = [a-d, a, a+d]
    Ssum = sum(nums)
    Pprod = nums[0]*nums[1]*nums[2]
    ask = random.choice(["max", "prod_term", "sq_sum"])
    if ask == "max":
        ans = a + d; query = "가장 큰 수는?"
    elif ask == "sq_sum":
        ans = sum(v*v for v in nums); query = "세 수의 제곱의 합은?"
    else:
        ans = (a-d)*(a+d); query = "가장 큰 수와 가장 작은 수의 곱은?"
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-1, ans+1, ans+4])
    q = (f"등차수열을 이루는 세 자연수의 합이 ${Ssum}$이고 곱이 ${Pprod}$이다. "
         f"{query}")
    e = (f"1단계: 세 수를 $a-d,\\,a,\\,a+d$로 놓으면 합 $=3a={Ssum}$에서 $a={a}$.\n"
         f"2단계: 곱 $={a}({a}^2-d^2)={Pprod}$에서 $d={d}$. 세 수는 ${a-d},\\,{a},\\,{a+d}$.\n"
         f"3단계: {query.replace('?','')} ${ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "등차수열을 이루는 세 수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "arith3", "a": a, "d": d, "ask": ask, "ans": ans}
    return rr


def gen_confidence_interval():
    """모평균 신뢰구간의 길이 (킬러, 상). x̄±zσ/√n, 정확.

    신뢰구간의 길이 $=2z\\dfrac{\\sigma}{\\sqrt n}$ ($z$는 신뢰수준의 임계값).
    """
    z, lvl = random.choice([(sp.Rational(196, 100), "95"), (sp.Rational(258, 100), "99")])
    s = random.choice([2, 3, 4, 5, 10])
    n = random.choice([4, 9, 16, 25, 100])
    rn = int(n**0.5)
    length = 2*z*sp.Rational(s, rn)
    lt = f"{float(length):.4f}".rstrip("0").rstrip(".")
    # 보기 직접 조립
    pool = list({lt, f"{float(z*sp.Rational(s,rn)):.4f}".rstrip('0').rstrip('.'),
                 f"{float(length)*2:.4f}".rstrip('0').rstrip('.'),
                 f"{float(length)+0.5:.4f}".rstrip('0').rstrip('.'),
                 f"{float(length)-0.5:.4f}".rstrip('0').rstrip('.')})
    while len(pool) < 5:
        pool.append(f"{float(length)+0.1*len(pool):.4f}")
    pool = sorted(set(pool), key=float)[:5]
    if lt not in pool:
        pool[-1] = lt; pool = sorted(set(pool), key=float)
    idx = pool.index(lt)
    choices = [f"${v}$" for v in pool]
    lab = f"{idx+1}번 (${lt}$)"
    q = (f"어떤 모집단에서 크기 ${n}$인 표본을 임의추출하여 모평균을 신뢰도 "
         f"${lvl}\\%$로 추정하려고 한다. 모표준편차가 ${s}$일 때, 이 신뢰구간의 "
         f"길이는? (단, 신뢰도 ${lvl}\\%$의 임계값은 ${_ltx(z)}$)")
    e = (f"1단계: 신뢰구간의 길이 $=2\\times{_ltx(z)}\\times\\dfrac{{{s}}}{{\\sqrt{{{n}}}}}"
         f"=2\\times{_ltx(z)}\\times\\dfrac{{{s}}}{{{rn}}}$.\n"
         f"2단계: 계산하면 ${lt}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "모평균의 신뢰구간", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "confint", "z": [z.p, z.q], "s": s, "n": n, "ans": lt}
    return rr


def gen_sample_mean_prob():
    """표본평균의 확률 (킬러, 상). X̄~N(m,σ²/n) 표준화, 정확.

    $\\bar X\\sim N(m,\\sigma^2/n)$, $P(\\bar X\\ge a)=0.5-P(0\\le Z\\le z)$,
    $z=\\dfrac{a-m}{\\sigma/\\sqrt n}$.
    """
    TBL = {1: "0.3413", sp.Rational(3, 2): "0.4332", 2: "0.4772"}
    z, tv = random.choice(list(TBL.items()))
    m = random.choice([50, 60, 100])
    n = random.choice([4, 9, 16, 25])
    rn = int(n**0.5)
    s = random.choice([rn*2, rn*3, rn*4])      # σ/√n = σ/rn 정수
    se = s // rn
    a = int(m + z*se)
    ans_txt = f"{0.5 - float(tv):.4f}".rstrip("0").rstrip(".")
    zt = "1" if z == 1 else ("1.5" if z == sp.Rational(3, 2) else "2")
    pool = sorted({ans_txt, tv, f"{0.5+float(tv):.4f}".rstrip('0').rstrip('.'),
                   "0.1587" if ans_txt != "0.1587" else "0.0228",
                   "0.0668" if ans_txt != "0.0668" else "0.0013"}, key=float)[:5]
    while len(pool) < 5:
        pool.append(f"{0.5-float(tv)+0.05*len(pool):.4f}")
    pool = sorted(set(pool), key=float)[:5]
    if ans_txt not in pool:
        pool[-1] = ans_txt; pool = sorted(set(pool), key=float)
    idx = pool.index(ans_txt)
    choices = [f"${v}$" for v in pool]
    lab = f"{idx+1}번 (${ans_txt}$)"
    q = (f"정규분포 $\\mathrm{{N}}({m},\\,{s}^2)$을 따르는 모집단에서 크기 ${n}$인 표본을 "
         f"임의추출할 때, 표본평균 $\\bar X$에 대하여 $\\mathrm{{P}}(\\bar X\\ge {a})$의 "
         f"값은? (단, $\\mathrm{{P}}(0\\le Z\\le {zt})={tv}$)")
    e = (f"1단계: $\\bar X\\sim\\mathrm{{N}}({m},\\,\\tfrac{{{s}^2}}{{{n}}})$, 표준편차 "
         f"$\\dfrac{{{s}}}{{\\sqrt{{{n}}}}}={se}$.\n"
         f"2단계: $Z=\\dfrac{{\\bar X-{m}}}{{{se}}}$, $\\bar X={a}\\Rightarrow Z={zt}$.\n"
         f"3단계: $\\mathrm{{P}}(\\bar X\\ge{a})=0.5-{tv}={ans_txt}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "표본평균의 확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "samplemean", "tv": tv, "ans": ans_txt}
    return rr


def gen_derivative_limit():
    """미분계수의 극한 정의 (킬러, 상). 정의 변형 인식, 정확.

    $\\lim_{h\\to0}\\dfrac{f(a+ph)-f(a+qh)}{h}=(p-q)f'(a)$.
    """
    aa = random.randint(-2, 3)
    bb = random.randint(-4, 4)
    cc = random.randint(-3, 3)
    f = x**3 + aa*x**2 + bb*x + cc
    fp = sp.diff(f, x)
    t = random.choice([-1, 1, 2])
    p, qd = random.choice([(2, 0), (3, 1), (1, -1), (3, 0)])
    fpv = int(fp.subs(x, t))
    ans = (p - qd)*fpv
    fl = sp.latex(sp.expand(f))
    pterm = f"{p}h" if p != 1 else "h"
    qterm = (f"{qd}h" if qd not in (0, 1, -1) else ("h" if qd == 1 else ("-h" if qd == -1 else "")))
    numer = f"f({t}+{pterm})-f({t}{'+' if qd>=0 else ''}{qterm})" if qd != 0 else f"f({t}+{pterm})-f({t})"
    choices, lab = make_mc(sp.Integer(ans), [ans-fpv, ans+fpv, fpv, ans-2, ans+2])
    q = (f"함수 $f(x)={fl}$에 대하여 "
         f"$\\displaystyle\\lim_{{h\\to0}}\\dfrac{{{numer}}}{{h}}$의 값은?")
    e = (f"1단계: 미분계수의 정의에서 이 극한은 $({p}-({qd}))f'({t})={p-qd}\\,f'({t})$.\n"
         f"2단계: $f'(x)={sp.latex(fp)}$, $f'({t})={fpv}$이므로 값은 ${p-qd}\\times{fpv}={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "미분계수의 극한 정의", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "derivlim", "aa": aa, "bb": bb, "cc": cc, "t": t,
                   "p": p, "qd": qd, "ans": ans}
    return rr


def gen_extreme_value_sum():
    """삼차함수 극댓값과 극솟값의 합 (킬러, 상). 정확.

    $f'=3(x-r_1)(x-r_2)$ ($r_1+r_2$ 짝수). 극댓값 $f(r_1)$ + 극솟값 $f(r_2)$.
    """
    while True:
        r1, r2 = sorted(random.sample([-2, -1, 0, 1, 2, 3], 2))
        if (r1 + r2) % 2 == 0:
            break
    a = -3*(r1 + r2)//2
    b = 3*r1*r2
    c = random.randint(-4, 4)
    f = x**3 + a*x**2 + b*x + c
    ans = int(f.subs(x, r1)) + int(f.subs(x, r2))
    fl = sp.latex(sp.expand(f))
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"함수 $f(x)={fl}$의 극댓값과 극솟값의 합은?")
    e = (f"1단계: $f'(x)=3(x-({r1}))(x-({r2}))$이므로 $x={r1}$에서 극대, $x={r2}$에서 극소.\n"
         f"2단계: 극댓값 $f({r1})={int(f.subs(x,r1))}$, 극솟값 $f({r2})={int(f.subs(x,r2))}$.\n"
         f"3단계: 합 $={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "극댓값과 극솟값의 합", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "extsum", "r1": r1, "r2": r2, "c": c, "ans": ans}
    return rr


def gen_normal_unknown_mean():
    """정규분포 평균 미지수 m 구하기 (킬러, 상). 표준화, 정확.

    $X\\sim N(m,\\sigma^2)$, $P(X\\le a)=0.5+P(0\\le Z\\le z)$ → $\\dfrac{a-m}{\\sigma}=z$
    → $m=a-z\\sigma$.
    """
    TBL = {1: "0.3413", sp.Rational(3, 2): "0.4332", 2: "0.4772"}
    z, tv = random.choice(list(TBL.items()))
    s = random.choice([2, 4, 6, 8])
    m = random.choice([10, 20, 30, 50])
    a = int(m + z*s)
    ans = m
    zt = "1" if z == 1 else ("1.5" if z == sp.Rational(3, 2) else "2")
    prob = f"{0.5+float(tv):.4f}".rstrip("0").rstrip(".")
    choices, lab = make_mc(sp.Integer(ans), [ans-s, ans+s, a, ans-2, ans+2])
    q = (f"확률변수 $X$가 정규분포 $\\mathrm{{N}}(m,\\,{s}^2)$을 따르고 "
         f"$\\mathrm{{P}}(X\\le {a})={prob}$일 때, $m$의 값은? (단, "
         f"$\\mathrm{{P}}(0\\le Z\\le {zt})={tv}$)")
    e = (f"1단계: $\\mathrm{{P}}(X\\le {a})=0.5+{tv}={prob}$이므로 "
         f"$\\dfrac{{{a}-m}}{{{s}}}={zt}$.\n"
         f"2단계: ${a}-m={zt}\\times{s}={int(z*s)}$이므로 $m={a}-{int(z*s)}={ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "정규분포의 평균 결정", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "normmean", "m": m, "s": s, "z": [int(z*2), 2], "a": a, "ans": ans}
    return rr


def gen_multiset_permutation():
    """같은 것이 있는 순열 (킬러, 상). n!/∏(c!), 브루트포스 검증.

    여러 글자(일부 중복)를 일렬로 배열하는 경우의 수.
    """
    from math import factorial as fac
    from collections import Counter
    deck = random.choice([
        list("abba c".replace(" ", "")), list("sense"), list("banana"),
        list("acccb"), list("aabbc"), list("aaabc"), list("mamma"),
    ])
    n = len(deck)
    cnt = fac(n)
    for c in Counter(deck).values():
        cnt //= fac(c)
    ans = cnt
    s = max(2, round(ans/8))
    choices, lab = make_mc(sp.Integer(ans), [ans-s, ans+s, fac(n), ans*2, ans+2*s])
    letters = ", ".join(f"${ch}$" for ch in deck)
    counts = ", ".join(f"${ch}$ {c}개" for ch, c in sorted(Counter(deck).items()))
    q = (f"{counts}의 ${n}$개의 문자를 모두 일렬로 나열하는 경우의 수는?")
    denom = "!\\,".join(str(c) for c in sorted(Counter(deck).values(), reverse=True) if c > 1)
    e = (f"1단계: 같은 것이 있는 순열 $=\\dfrac{{{n}!}}{{{denom}!}}$.\n"
         f"2단계: 계산하면 ${ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "같은 것이 있는 순열", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "msperm2", "deck": deck, "ans": ans}
    return rr


def gen_arithmetic_sum_max_n():
    """등차수열 부분합 S_n이 최대가 되는 n (킬러, 상). 직접 탐색, 정확.

    $a_1>0$, $d<0$인 등차수열에서 $a_n\\ge0$인 마지막 $n$에서 $S_n$이 최대.
    """
    d = random.choice([-2, -3, -4])
    a1 = random.choice([k for k in range(10, 41) if k % (-d) != 0])  # a_n=0 회피
    Smax = -10**9
    nbest = 1
    S = 0
    for n in range(1, 60):
        S += a1 + (n-1)*d
        if S > Smax:
            Smax, nbest = S, n
    ans = nbest
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, ans+3])
    q = (f"첫째항이 ${a1}$, 공차가 ${d}$인 등차수열 $\\{{a_n\\}}$의 첫째항부터 "
         f"제$n$항까지의 합 $S_n$이 최대가 되도록 하는 자연수 $n$의 값은?")
    e = (f"1단계: $a_n={a1}+(n-1)({d})\\ge0$인 마지막 $n$까지 더할 때 $S_n$이 최대.\n"
         f"2단계: $a_n\\ge0 \\iff n\\le{a1}/{-d}+1$. 계산하면 $n={ans}$에서 최대.\n\n정답: {lab}")
    rr = _rec("대수", "등차수열 부분합의 최대", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "arithmax", "a1": a1, "d": d, "ans": ans}
    return rr


def gen_parabola_line_area():
    """포물선과 직선 사이의 넓이 (킬러, 상). 1/6 공식, sympy 검증.

    $y=x^2$과 직선이 $x=\\alpha,\\beta$에서 만날 때 둘로 둘러싸인 넓이
    $=\\dfrac{(\\beta-\\alpha)^3}{6}$.
    """
    al, be = sorted(random.sample([-3, -2, -1, 0, 1, 2, 3], 2))
    mm = al + be                              # 직선 y=(α+β)x-αβ
    cc = -al*be
    ans = sp.Rational((be-al)**3, 6)
    # sympy 검증
    line = mm*x + cc
    area = sp.integrate(line - x**2, (x, al, be))
    assert area == ans, "넓이 불일치"
    lineexpr = sp.latex(sp.expand(line))
    choices, lab = make_mc(ans, [ans+sp.Rational(1, 6), ans-sp.Rational(1, 6),
                                 ans*2, ans*sp.Rational(1, 2), ans+1])
    q = (f"곡선 $y=x^2$과 직선 $y={lineexpr}$로 둘러싸인 부분의 넓이는?")
    e = (f"1단계: 교점의 $x$좌표는 $x^2={lineexpr}$, 즉 $x={al},\\,{be}$.\n"
         f"2단계: 넓이 $=\\displaystyle\\int_{{{al}}}^{{{be}}}\\big(({lineexpr})-x^2\\big)dx"
         f"=\\dfrac{{({be}-({al}))^3}}{{6}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "포물선과 직선 사이의 넓이", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "paraline", "al": al, "be": be, "ans": [ans.p, ans.q]}
    return rr


def gen_lattice_path():
    """격자 최단경로(경유점) 경우의 수 (킬러, 상). 조합 곱, 정확.

    $(0,0)\\to(a,b)$ 최단경로 중 점 $(p,q)$를 지나는 것의 수
    $=\\binom{p+q}{p}\\binom{(a-p)+(b-q)}{a-p}$.
    """
    from math import comb
    a = random.randint(3, 5)
    b = random.randint(3, 5)
    p = random.randint(1, a-1)
    q = random.randint(1, b-1)
    ans = comb(p+q, p) * comb((a-p)+(b-q), a-p)
    s = max(2, round(ans/8))
    choices, lab = make_mc(sp.Integer(ans), [ans-s, ans+s, comb(a+b, a),
                                             ans*2, ans+2*s])
    q_ = (f"그림과 같은 도로망에서 $\\mathrm{{A}}(0,0)$ 지점에서 출발하여 "
          f"$\\mathrm{{B}}({a},{b})$ 지점까지 최단 거리로 갈 때, 도중에 점 "
          f"$\\mathrm{{P}}({p},{q})$를 지나는 경우의 수는? (단, 오른쪽 또는 위로만 이동)")
    e = (f"1단계: $\\mathrm{{A}}\\to\\mathrm{{P}}$: $\\binom{{{p+q}}}{{{p}}}={comb(p+q,p)}$.\n"
         f"2단계: $\\mathrm{{P}}\\to\\mathrm{{B}}$: $\\binom{{{(a-p)+(b-q)}}}{{{a-p}}}={comb((a-p)+(b-q),a-p)}$.\n"
         f"3단계: 곱하면 ${ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "격자 최단경로의 경우의 수", "상", 4, "객관식", q_, choices, lab, e)
    rr["completion"]["figure"] = {"kind": "grid", "a": a, "b": b, "via": [p, q]}
    rr["completion"]["figure_desc"] = (
        f"가로 {a}칸 세로 {b}칸 격자 도로망. A(0,0)에서 B({a},{b})까지, "
        f"경유점 P({p},{q})가 표시됨.")
    rr["_meta"] = {"chk": "lattice", "a": a, "b": b, "p": p, "q": q, "ans": ans}
    return rr


def gen_circular_arrangement():
    """원순열 + 이웃 조건 (킬러, 상). 공식, 브루트포스로 검증.

    $n$명을 원탁에 앉힐 때 특정 두 사람이 이웃(또는 이웃하지 않게) 하는 경우의 수.
    이웃 $=2(n-2)!$, 비이웃 $=(n-1)!-2(n-2)!$.
    """
    from math import factorial as fac
    n = random.choice([5, 6, 7])
    adj = random.choice([True, False])
    ans = 2*fac(n-2) if adj else fac(n-1) - 2*fac(n-2)
    cond = "이웃하게" if adj else "이웃하지 않게"
    s = max(2, round(ans/8))
    choices, lab = make_mc(sp.Integer(ans), [ans-s, ans+s, fac(n-1), 2*fac(n-2) if not adj else fac(n-1)-2*fac(n-2), ans+2*s])
    q = (f"${n}$명이 원형의 탁자에 둘러앉을 때, 특정한 두 사람 $\\mathrm{{A}}$, "
         f"$\\mathrm{{B}}$가 서로 {cond} 앉는 경우의 수는? (단, 회전하여 일치하는 "
         f"것은 같은 것으로 본다.)")
    if adj:
        e = (f"1단계: $\\mathrm{{A}}$, $\\mathrm{{B}}$를 한 묶음으로 보면 ${n-1}$개의 "
             f"원순열 $({n-2})!$, 두 사람의 자리바꿈 $2$.\n"
             f"2단계: $2\\times({n-2})!={ans}$.\n\n정답: {lab}")
    else:
        e = (f"1단계: 전체 원순열 $({n-1})!={fac(n-1)}$에서 이웃하는 경우 "
             f"$2({n-2})!={2*fac(n-2)}$를 뺀다.\n"
             f"2단계: ${fac(n-1)}-{2*fac(n-2)}={ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "원순열과 이웃 조건", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "circular", "n": n, "adj": adj, "ans": ans}
    return rr


def gen_recurrence_telescope():
    """계차수열(망원) 점화식 → 특정 항 (킬러, 상). a_N=a_1+Σ계차, 정확.

    $a_{n+1}-a_n=dn+e$ (계차가 일차)일 때 $a_N=a_1+\\sum_{k=1}^{N-1}(dk+e)$.
    """
    a1 = random.randint(-3, 5)
    d = random.choice([2, 3, 4])
    e = random.randint(-3, 3)
    N = random.choice([6, 7, 8, 10])
    ans = a1 + sum(d*k + e for k in range(1, N))
    eterm = (f"+{e}" if e > 0 else (f"{e}" if e < 0 else ""))
    choices, lab = make_mc(sp.Integer(ans), [ans-d, ans+d, ans-2*d, ans+2*d, -ans])
    q = (f"수열 $\\{{a_n\\}}$이 $a_1={a1}$이고, 모든 자연수 $n$에 대하여 "
         f"$a_{{n+1}}-a_n={d}n{eterm}$을 만족시킬 때, $a_{{{N}}}$의 값은?")
    e_ = (f"1단계: $a_{{{N}}}=a_1+\\sum_{{k=1}}^{{{N-1}}}(a_{{k+1}}-a_k)"
          f"=a_1+\\sum_{{k=1}}^{{{N-1}}}({d}k{eterm})$.\n"
          f"2단계: $\\sum_{{k=1}}^{{{N-1}}}({d}k{eterm})={sum(d*k+e for k in range(1,N))}$이므로 "
          f"$a_{{{N}}}={a1}+{sum(d*k+e for k in range(1,N))}={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "계차수열(망원) 점화식", "상", 4, "객관식", q, choices, lab, e_)
    rr["_meta"] = {"chk": "rectele", "a1": a1, "d": d, "e": e, "N": N, "ans": ans}
    return rr


def gen_inverse_function_value():
    """역함수의 함숫값 (킬러, 상). 증가함수에서 유일근, 정확.

    $f(x)=x^3+px+q$ (증가, $p>0$)의 역함수 $g$에 대하여 $g(k)$는 $f(x)=k$의
    유일한 실근. $k=f(a)$로 구성하면 $g(k)=a$.
    """
    p = random.choice([1, 2, 3])             # p>0 → 증가함수 (f'=3x²+p>0)
    q = random.randint(-3, 3)
    a = random.choice([-2, -1, 1, 2, 3])
    k = a**3 + p*a + q
    ans = a
    qterm = (f"+{q}" if q > 0 else (f"{q}" if q < 0 else ""))
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, -ans])
    qtxt = (f"함수 $f(x)=x^3+{p}x{qterm}$의 역함수를 $g$라 할 때, $g({k})$의 값은?")
    e = (f"1단계: $f'(x)=3x^2+{p}>0$이므로 $f$는 증가함수, 역함수가 존재한다.\n"
         f"2단계: $g({k})=t$이면 $f(t)={k}$, 즉 $t^3+{p}t{qterm}={k}$. "
         f"$t={a}$을 대입하면 성립.\n"
         f"3단계: $g({k})={a}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "역함수의 함숫값", "상", 4, "객관식", qtxt, choices, lab, e)
    rr["_meta"] = {"chk": "invfunc", "p": p, "q": q, "k": k, "ans": a}
    return rr


def gen_tangent_yintercept():
    """곡선 위 점에서의 접선의 y절편 (킬러, 상). 정확.

    $f(x)=x^3+ax^2+bx+c$의 $x=p$에서의 접선 $y=f(p)+f'(p)(x-p)$의 $y$절편
    $=f(p)-p\\,f'(p)$.
    """
    a = random.randint(-3, 3)
    b = random.randint(-4, 4)
    c = random.randint(-4, 4)
    p = random.choice([-2, -1, 1, 2, 3])
    f = x**3 + a*x**2 + b*x + c
    fp = sp.diff(f, x)
    fpv = int(fp.subs(x, p))
    fv = int(f.subs(x, p))
    ans = fv - p*fpv
    fl = sp.latex(sp.expand(f))
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"곡선 $y={fl}$ 위의 점 $({p},\\,{fv})$에서의 접선의 $y$절편은?")
    e = (f"1단계: $f'(x)={sp.latex(fp)}$, $f'({p})={fpv}$, $f({p})={fv}$.\n"
         f"2단계: 접선 $y={fv}+{fpv}(x-({p}))$. $x=0$ 대입하면 "
         f"$y={fv}-({p})\\cdot{fpv}={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "접선의 y절편", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "tanyint", "a": a, "b": b, "c": c, "p": p, "ans": ans}
    return rr


def gen_increasing_cubic_param():
    """삼차함수가 증가함수가 되는 정수 a의 개수 (킬러, 상). 판별식, 정확.

    $f(x)=x^3+ax^2+bx+c$가 실수 전체에서 증가 $\\iff f'=3x^2+2ax+b\\ge0$ 항상
    $\\iff a^2\\le 3b$. 정수 $a$의 개수 $=2\\lfloor\\sqrt{3b}\\rfloor+1$.
    """
    import math
    b = random.choice([3, 4, 6, 8, 12, 16, 27])
    root = math.isqrt(3*b)
    ans = 2*root + 1
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-1, ans+1, ans+4])
    q = (f"함수 $f(x)=x^3+ax^2+{b}x+1$이 실수 전체의 집합에서 증가하도록 하는 "
         f"정수 $a$의 개수는?")
    e = (f"1단계: 증가함수 $\\iff f'(x)=3x^2+2ax+{b}\\ge0$ (항상). 판별식 "
         f"$(2a)^2-12\\cdot{b}\\le0$, 즉 $a^2\\le{3*b}$.\n"
         f"2단계: $-\\sqrt{{{3*b}}}\\le a\\le\\sqrt{{{3*b}}}$, $\\lfloor\\sqrt{{{3*b}}}\\rfloor={root}$ "
         f"이므로 정수 $a$는 $-{root}$부터 ${root}$까지 ${ans}$개.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "증가함수가 되는 조건", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "inccubic", "b": b, "ans": ans}
    return rr


def gen_three_dice_max():
    """주사위 3개의 최댓값 확률 (킬러, 상). 6³ 브루트포스, 정확.

    세 눈의 최댓값이 정확히 $k$일 확률 $=\\dfrac{k^3-(k-1)^3}{216}$ (전수로 검증).
    """
    from itertools import product as _ip
    k = random.choice([3, 4, 5, 6])
    fav = sum(1 for s in _ip(range(1, 7), repeat=3) if max(s) == k)
    ans = sp.Rational(fav, 216)
    dd = [sp.Rational(fav-6, 216), sp.Rational(fav+6, 216),
          sp.Rational(k**3, 216), sp.Rational(fav, 216)+sp.Rational(1, 6)]
    choices, lab = make_mc(ans, [d for d in dd if 0 < d < 1 and d != ans])
    q = (f"한 개의 주사위를 세 번 던질 때, 나오는 세 눈의 수의 최댓값이 ${k}$일 "
         f"확률은?")
    e = (f"1단계: 전체 경우 $6^3=216$.\n"
         f"2단계: 최댓값이 ${k}$인 경우는 (모두 ${k}$ 이하)$-$(모두 ${k-1}$ 이하)"
         f"$={k}^3-{k-1}^3={fav}$.\n"
         f"3단계: 확률 $=\\dfrac{{{fav}}}{{216}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "주사위 최댓값의 확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "threedicemax", "k": k, "ans": [ans.p, ans.q]}
    return rr


def gen_group_partition():
    """서로 다른 사람을 조로 나누는 경우의 수 (킬러, 상). 다항계수, 정확.

    서로 다른 $N$명을 크기 $s_1,s_2,\\dots$인 조로 나눈다(같은 크기 조는 순서 무시).
    """
    from math import factorial as fac, comb
    setup = random.choice([
        (6, [2, 2, 2], "$2$명씩 $3$개의 조"),
        (6, [3, 3], "$3$명씩 $2$개의 조"),
        (5, [2, 3], "$2$명, $3$명의 두 조"),
        (8, [4, 4], "$4$명씩 $2$개의 조"),
        (7, [2, 2, 3], "$2$명, $2$명, $3$명의 세 조"),
    ])
    N, sizes, desc = setup
    from collections import Counter
    cnt = fac(N)
    for sz in sizes:
        cnt //= fac(sz)
    for c in Counter(sizes).values():        # 같은 크기 조끼리 순서 무시
        cnt //= fac(c)
    ans = cnt
    s = max(2, round(ans/8))
    choices, lab = make_mc(sp.Integer(ans), [ans-s, ans+s, ans*2, ans//2 if ans % 2 == 0 else ans+3, ans+2*s])
    q = (f"서로 다른 ${N}$명을 {desc}로 나누는 경우의 수는?")
    sizestr = "!\\,".join(str(sz) for sz in sizes)
    e = (f"1단계: ${N}!$를 각 조 크기의 계승으로 나누고, 같은 크기 조의 개수의 "
         f"계승으로 한 번 더 나눈다.\n"
         f"2단계: 계산하면 ${ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조 나누기(분할)", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "grouppart", "N": N, "sizes": sizes, "ans": ans}
    return rr


def gen_factorial_equation():
    """순열·조합 방정식 → n (킬러, 상). 직접 탐색, 정확.

    $_nP_2=v$, $_nP_3=v$, $_nC_2=v$ 등에서 자연수 $n$을 구한다.
    """
    from math import comb, perm
    kind = random.choice(["P2", "P3", "C2", "C3"])
    n = random.randint(4, 9)
    if kind == "P2":
        v = perm(n, 2); expr = "_{n}\\mathrm{P}_{2}"; rule = "n(n-1)"
    elif kind == "P3":
        v = perm(n, 3); expr = "_{n}\\mathrm{P}_{3}"; rule = "n(n-1)(n-2)"
    elif kind == "C2":
        v = comb(n, 2); expr = "_{n}\\mathrm{C}_{2}"; rule = "\\dfrac{n(n-1)}{2}"
    else:
        v = comb(n, 3); expr = "_{n}\\mathrm{C}_{3}"; rule = "\\dfrac{n(n-1)(n-2)}{6}"
    ans = n
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, ans+3])
    q = (f"$1$보다 큰 자연수 $n$에 대하여 ${expr}={v}$일 때, $n$의 값은?")
    e = (f"1단계: ${expr}={rule}={v}$.\n"
         f"2단계: 식을 풀면 $n={n}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "순열·조합 방정식", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "facteq", "kind": kind, "v": v, "ans": ans}
    return rr


def gen_expected_value():
    """확률변수의 기댓값 E(aX+b) (킬러, 상). 분포표에서 직접, 정확.

    확률변수 $X$의 분포(모든 확률 주어짐)에서 $\\mathrm{E}(X)$를 구하고
    $\\mathrm{E}(aX+b)=a\\mathrm{E}(X)+b$.
    """
    xs = sorted(random.sample([0, 1, 2, 3, 4, 5], 4))
    D = random.choice([8, 10])
    while True:
        cuts = sorted(random.sample(range(1, D), 3))
        ps = [cuts[0], cuts[1]-cuts[0], cuts[2]-cuts[1], D-cuts[2]]
        if min(ps) >= 1:
            break
    probs = [sp.Rational(p, D) for p in ps]
    E = sum(xs[i]*probs[i] for i in range(4))
    a, b = random.choice([(2, 1), (3, -1), (2, 3), (4, 0)])
    ans = a*E + b
    rows_x = " & ".join(str(v) for v in xs)
    rows_p = " & ".join(_ltx(p) for p in probs)
    choices, lab = make_mc(ans, [ans+1, ans-1, a*E, ans+sp.Rational(1, 2), ans-sp.Rational(1, 2)])
    q = (f"확률변수 $X$의 확률분포를 표로 나타내면 다음과 같다.\n\n"
         f"$\\begin{{array}}{{c|cccc|c}} X & {rows_x} & \\text{{합계}} \\\\\\hline "
         f"\\mathrm{{P}}(X=x) & {rows_p} & 1 \\end{{array}}$\n\n"
         f"$\\mathrm{{E}}({a}X{'+' if b>=0 else ''}{b})$의 값은?")
    e = (f"1단계: $\\mathrm{{E}}(X)=\\sum x_i p_i={_ltx(E)}$.\n"
         f"2단계: $\\mathrm{{E}}({a}X{'+' if b>=0 else ''}{b})={a}\\mathrm{{E}}(X)"
         f"{'+' if b>=0 else ''}{b}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "확률변수의 기댓값", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "expval", "xs": xs, "ps": ps, "D": D, "a": a, "b": b,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_at_least_one_prob():
    """여사건 — 적어도 한 번 성공할 확률 (킬러, 상). 1-(1-p)ⁿ, 정확.

    매회 성공확률 $p$인 독립시행을 $n$번 할 때 적어도 한 번 성공할 확률
    $=1-(1-p)^n$.
    """
    pn, pd = random.choice([(1, 6), (1, 3), (1, 2), (2, 5), (1, 4)])
    p = sp.Rational(pn, pd)
    n = random.choice([2, 3, 4])
    ans = 1 - (1-p)**n
    naive = n*p                                # 함정: np (≤1일 때)
    dd = [(1-p)**n, ans-(1-p)**n if ans-(1-p)**n > 0 else ans*sp.Rational(1, 2),
          ans*sp.Rational(4, 5), ans*sp.Rational(6, 5) if ans*sp.Rational(6, 5) < 1 else ans*sp.Rational(9, 10)]
    if naive < 1:
        dd = [(naive, "여사건 미사용 — np 로 잘못 계산")] + dd
    choices, lab, traps = make_mc_tagged(ans, dd)
    ctx = random.choice([("한 개의 주사위를", f"${pd}$의 약수가 아닌 눈" if False else "특정 눈"),
                         ("", "")])
    q = (f"매회 성공할 확률이 ${_ltx(p)}$인 독립시행을 ${n}$번 반복할 때, "
         f"적어도 한 번 성공할 확률은?")
    e = (f"1단계: (적어도 한 번 성공) $=1-$ (모두 실패).\n"
         f"2단계: 모두 실패할 확률 $=(1-{_ltx(p)})^{{{n}}}={_ltx((1-p)**n)}$.\n"
         f"3단계: $1-{_ltx((1-p)**n)}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "여사건의 확률(적어도 하나)", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "atleastone", "p": [p.p, p.q], "n": n,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_log_equation_domain():
    """로그방정식 (진수조건) → 해 (킬러, 상). 이차방정식 + 진수>0 필터, 정확.

    $\\log_p(x-s)+\\log_p(x-t)=m$ → $(x-s)(x-t)=p^m$, 진수조건 $x>s,\\,x>t$인 근만.
    """
    p = 2                                    # 밑 2 (정수 거듭제곱을 서로 다른 두 인수로 표현 가능)

    def _plog(v):                            # v가 p의 거듭제곱이면 지수, 아니면 None
        e = 0
        while v % p == 0:
            v //= p
            e += 1
        return e if v == 1 and e >= 1 else None
    sol = None
    for _ in range(200):
        x0 = random.randint(5, 9)            # 정답이 될 정수해
        s = random.randint(1, x0-2)
        t = random.randint(1, x0-2)
        if s == t:
            continue
        m = _plog((x0-s)*(x0-t))
        if m is None:
            continue
        other = s + t - x0                   # 근의 합 = s+t → 다른 근
        if other > max(s, t):                # 다른 근도 진수조건 통과 → 해 2개라 폐기
            continue
        sol = (x0, s, t, m, other)
        break
    if sol is None:                          # 폴백 (항상 유효): (x-3)(x-1)=8
        x0, s, t, m, other = 5, 3, 1, 3, -1
    else:
        x0, s, t, m, other = sol
    ans = x0
    roots = sorted([x0, other])
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, -ans])
    q = (f"방정식 $\\log_{{{p}}}(x-{s})+\\log_{{{p}}}(x-{t})={m}$의 해는?")
    e = (f"1단계: 진수조건 $x>{max(s,t)}$. 로그의 합을 정리하면 "
         f"$(x-{s})(x-{t})={p}^{{{m}}}={p**m}$.\n"
         f"2단계: $x^2-{s+t}x+{s*t-p**m}=0$을 풀면 $x={x0}$ 또는 $x={other}$.\n"
         f"3단계: 진수조건에서 $x={x0}$.\n\n정답: {lab}")
    rr = _rec("대수", "로그방정식과 진수조건", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "logeqdom", "p": p, "s": s, "t": t, "m": m, "ans": x0}
    return rr


def gen_trig_identity_sincos():
    """sinθ+cosθ=k → sin2θ (킬러, 상). 항등식 정확.

    $(\\sin\\theta+\\cos\\theta)^2=1+2\\sin\\theta\\cos\\theta=1+\\sin2\\theta=k^2$
    → $\\sin2\\theta=k^2-1$.
    """
    kn, kd = random.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5)])
    k = sp.Rational(kn, kd)
    ans = k*k - 1                              # sin2θ ∈ [-1,0]
    dd = [k*k, 1 - k*k, ans + sp.Rational(1, 4), ans - sp.Rational(1, 4), -ans]
    choices, lab = make_mc(ans, [d for d in dd if -1 <= d <= 1 and d != ans])
    q = (f"$\\sin\\theta+\\cos\\theta={_ltx(k)}$일 때, $\\sin 2\\theta$의 값은?")
    e = (f"1단계: $(\\sin\\theta+\\cos\\theta)^2=1+2\\sin\\theta\\cos\\theta=1+\\sin2\\theta$.\n"
         f"2단계: $\\left({_ltx(k)}\\right)^2=1+\\sin2\\theta$이므로 "
         f"$\\sin2\\theta={_ltx(k)}^2-1={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("대수", "삼각함수의 항등식(sin+cos)", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "trigid", "kn": kn, "kd": kd, "ans": [ans.p, ans.q]}
    return rr


def gen_closed_interval_extreme():
    """닫힌구간에서 삼차함수의 최대·최소 (킬러, 상). 끝점+극점 비교, 정확.

    $f'=3(x-r_1)(x-r_2)$ ($r_1+r_2$ 짝수→정수계수). 구간 양 끝과 극점에서의
    함숫값을 비교해 최댓값·최솟값 결정.
    """
    while True:
        r1, r2 = sorted(random.sample([-2, -1, 0, 1, 2, 3], 2))
        if (r1 + r2) % 2 == 0:
            break
    a = -3*(r1 + r2)//2
    b = 3*r1*r2
    c = random.randint(-3, 3)
    f = x**3 + a*x**2 + b*x + c
    lo = r1 - random.choice([1, 2])
    hi = r2 + random.choice([1, 2])
    pts = [lo, hi] + [r for r in (r1, r2) if lo <= r <= hi]
    vals = [int(f.subs(x, t)) for t in pts]
    mx, mn = max(vals), min(vals)
    kind = random.choice(["max", "min", "diff"])
    ans = mx if kind == "max" else (mn if kind == "min" else mx - mn)
    word = {"max": "최댓값", "min": "최솟값", "diff": "최댓값과 최솟값의 차"}[kind]
    fl = sp.latex(sp.expand(f))
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"닫힌구간 $[{lo},\\,{hi}]$에서 함수 $f(x)={fl}$의 {word}은?")
    e = (f"1단계: $f'(x)=3(x-({r1}))(x-({r2}))$이므로 극점은 $x={r1},\\,{r2}$.\n"
         f"2단계: 양 끝과 극점에서 $f$의 값을 비교 → 최대 ${mx}$, 최소 ${mn}$.\n"
         f"3단계: {word} $={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "닫힌구간에서의 최대·최소", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "closedext", "r1": r1, "r2": r2, "c": c,
                   "lo": lo, "hi": hi, "kind": kind, "ans": ans}
    return rr


def gen_combination_condition():
    """조건부 조합의 경우의 수 (킬러, 상). 조합 합, 정확.

    남학생 a명, 여학생 b명 중 k명을 뽑을 때 여학생이 r명 이상인 경우의 수
    $=\\sum_{j\\ge r}\\binom{b}{j}\\binom{a}{k-j}$.
    """
    from math import comb
    while True:
        A = random.randint(3, 6)
        B = random.randint(3, 6)
        k = random.choice([3, 4])
        r = random.randint(1, min(B, k)-1) if min(B, k) >= 2 else 1
        cnt = sum(comb(B, j)*comb(A, k-j) for j in range(r, min(B, k)+1))
        naive = comb(A+B, k)
        if cnt >= 6 and cnt != naive:
            break
    s = max(3, round(cnt/8))
    dd = [(naive, "조건(여학생 수) 무시 — 전체 경우의 수"),
          cnt-s, cnt+s, cnt-2*s, cnt+2*s]
    choices, lab, traps = make_mc_tagged(sp.Integer(cnt), [d for d in dd])
    q = (f"남학생 ${A}$명과 여학생 ${B}$명 중에서 ${k}$명의 대표를 뽑을 때, "
         f"여학생이 ${r}$명 이상 포함되는 경우의 수는?")
    e = (f"1단계: 여학생 수를 $j\\,(j\\ge{r})$로 나누어 $\\binom{{{B}}}{{j}}\\binom{{{A}}}{{{k}-j}}$ 합산.\n"
         f"2단계: $\\sum_{{j={r}}}^{{{min(B,k)}}}\\binom{{{B}}}{{j}}\\binom{{{A}}}{{{k}-j}}={cnt}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "조건부 조합의 경우의 수", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "combcond", "A": A, "B": B, "k": k, "r": r, "ans": cnt}
    return rr


def gen_log_chain():
    """로그의 연쇄 곱 (밑변환) (킬러, 상). log_a b·log_b c·…=log_a(끝). 정확.

    $\\log_{a_0}a_1\\cdot\\log_{a_1}a_2\\cdots=\\log_{a_0}a_m$. 끝값 $a_m=a_0^K$이면 $K$.
    """
    a0 = random.choice([2, 3])
    K = random.choice([2, 3, 4])
    am = a0**K
    mids = random.sample([5, 6, 7, 10, 12], random.choice([1, 2]))
    chain = [a0] + mids + [am]
    prod = sp.Integer(1)
    for i in range(len(chain)-1):
        prod *= sp.log(chain[i+1])/sp.log(chain[i])
    ans = sp.nsimplify(sp.simplify(prod))
    assert ans == K
    factors = "\\cdot".join(f"\\log_{{{chain[i]}}} {chain[i+1]}" for i in range(len(chain)-1))
    choices, lab = make_mc(sp.Integer(K), [K-1, K+1, K-2 if K > 2 else K+2, K*2, K+3])
    q = (f"${factors}$의 값은?")
    e = (f"1단계: 밑변환 $\\log_a b=\\dfrac{{\\log b}}{{\\log a}}$로 연쇄 곱이 망원으로 약분되어 "
         f"$\\log_{{{a0}}} {am}$.\n"
         f"2단계: ${am}={a0}^{{{K}}}$이므로 $\\log_{{{a0}}} {am}={K}$.\n\n정답: {lab}")
    rr = _rec("대수", "로그의 연쇄와 밑변환", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "logchain", "a0": a0, "K": K, "ans": K}
    return rr


def _trig_sum_contrib(r):
    """sin x = r (0≤x<2π) 해들의 합 (정확, π의 배수)."""
    r = sp.nsimplify(r)
    if sp.im(r) != 0:
        return sp.Integer(0)
    rf = float(r)
    if abs(rf) > 1 + 1e-9:
        return sp.Integer(0)
    if abs(rf - 1) < 1e-9:
        return sp.pi/2                  # x=π/2
    if abs(rf + 1) < 1e-9:
        return 3*sp.pi/2                # x=3π/2
    if abs(rf) < 1e-9:
        return sp.pi                    # x=0, π
    if rf > 0:
        return sp.pi                    # 두 해 합 π
    return 3*sp.pi                      # 두 해 합 3π


def gen_trig_solution_sum():
    """삼각방정식 A sin²x+B sin x+C=0 (0≤x<2π) 의 해의 합 (킬러, 상). 정확.

    sin x=s 치환 → 각 실근의 해들의 합(부호별 π·3π·…)을 합산. π의 배수로 정확.
    """
    s = sp.Symbol("s")
    nice = [sp.Rational(-1, 2), sp.Integer(0), sp.Rational(1, 2),
            sp.Integer(1), sp.Integer(-1), sp.Integer(2)]
    while True:
        s1, s2 = random.sample(nice, 2)
        expr = sp.expand((s - s1)*(s - s2))
        D = sp.ilcm(*[sp.denom(c) for c in expr.as_coefficients_dict().values()])
        P = sp.Poly(sp.expand(expr*D), s)
        A = int(P.coeff_monomial(s**2)); B = int(P.coeff_monomial(s)); C = int(P.coeff_monomial(1))
        ssum = _trig_sum_contrib(s1) + _trig_sum_contrib(s2)
        if A != 0 and ssum != 0:
            break
    ans = sp.nsimplify(ssum)

    def _term(coef, sym):
        if coef == 0:
            return ""
        sign = "+" if coef > 0 else "-"
        mag = abs(coef)
        body = sym if mag == 1 and sym else f"{mag}{sym}"
        return f"{sign}{body}" if sym else f"{sign}{mag}"
    lhs = (_term(A, "\\sin^2 x") + _term(B, "\\sin x") + _term(C, "")).lstrip("+")
    dd = [ans + sp.pi, ans - sp.pi, ans + sp.pi/2, ans - sp.pi/2, 2*sp.pi]
    choices, lab = make_mc(ans, [d for d in dd if d > 0 and d != ans])
    q = (f"$0\\le x<2\\pi$일 때, 방정식 ${lhs}=0$의 모든 해의 합은?")
    e = (f"1단계: $\\sin x=s$로 놓으면 근 $s={sp.latex(s1)},\\,{sp.latex(s2)}$.\n"
         f"2단계: 각 근에서 $\\sin x=s$의 해의 합 — $s={sp.latex(s1)}$: ${sp.latex(_trig_sum_contrib(s1))}$, "
         f"$s={sp.latex(s2)}$: ${sp.latex(_trig_sum_contrib(s2))}$.\n"
         f"3단계: 합 ${sp.latex(ans)}$.\n\n정답: {lab}")
    rr = _rec("대수", "삼각방정식 해의 합", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "trigsum", "A": A, "B": B, "C": C,
                   "ans": str(ans)}
    return rr


def gen_remainder_theorem():
    """나머지정리: f(x)를 (x-p)(x-q)로 나눈 나머지 R(x)=ax+b → R(t) (킬러, 상). 정확.

    $f(p)=u$, $f(q)=v$에서 $ap+b=u$, $aq+b=v$ 연립 → $a,b$. $R(t)=at+b$.
    """
    p, qq = sorted(random.sample([-2, -1, 1, 2, 3], 2))
    a = random.randint(-3, 3)
    b = random.randint(-4, 4)
    u = a*p + b
    v = a*qq + b
    t = random.choice([v_ for v_ in [-3, 0, 4, 5] if v_ not in (p, qq)])
    ans = a*t + b
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"다항식 $f(x)$를 $x-({p})$로 나눈 나머지가 ${u}$, $x-({qq})$로 나눈 나머지가 "
         f"${v}$이다. $f(x)$를 $(x-({p}))(x-({qq}))$로 나눈 나머지를 $R(x)$라 할 때, "
         f"$R({t})$의 값은?")
    e = (f"1단계: 나머지정리에서 $f({p})={u}$, $f({qq})={v}$.\n"
         f"2단계: $R(x)=ax+b$로 놓으면 $a({p})+b={u}$, $a({qq})+b={v}$. "
         f"연립하면 $a={a}$, $b={b}$.\n"
         f"3단계: $R({t})={a}\\cdot{t}+({b})={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "나머지정리와 나눗셈", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "remainder", "p": p, "q": qq, "a": a, "b": b, "t": t, "ans": ans}
    return rr


def gen_bayes_three():
    """3원인 베이즈 정리 조건부확률 (킬러, 상). sp.Rational 정확.

    세 공장 $A,B,C$ 생산비율과 불량률 → 불량품일 때 그것이 $A$ 생산품일 확률.
    """
    while True:
        ps = random.sample([20, 30, 40, 50, 25, 35], 3)
        if sum(ps) == 100:
            break
    rs = [random.choice([1, 2, 3, 4, 5]) for _ in range(3)]
    joints = [sp.Rational(ps[i]*rs[i], 10000) for i in range(3)]
    tot = sum(joints)
    ans = sp.nsimplify(joints[0]/tot)
    naive = sp.Rational(ps[0], 100)             # 함정: 생산비율 그대로
    dd = [(naive, "불량 조건 무시 — 생산비율 그대로 답함"),
          sp.nsimplify(joints[1]/tot), ans*sp.Rational(4, 5), ans*sp.Rational(6, 5)]
    choices, lab, traps = make_mc_tagged(ans, dd)
    q = (f"어떤 제품을 세 공장 $\\mathrm{{A}}$, $\\mathrm{{B}}$, $\\mathrm{{C}}$에서 각각 "
         f"${ps[0]}\\%$, ${ps[1]}\\%$, ${ps[2]}\\%$ 생산하고, 각 공장의 불량률은 "
         f"${rs[0]}\\%$, ${rs[1]}\\%$, ${rs[2]}\\%$이다. 임의로 택한 한 제품이 "
         f"불량품일 때, 이것이 공장 $\\mathrm{{A}}$의 제품일 확률은?")
    e = (f"1단계: $\\mathrm{{P}}(\\mathrm{{A}}\\cap\\text{{불}})={_ltx(joints[0])}$, "
         f"$\\mathrm{{B}}$: ${_ltx(joints[1])}$, $\\mathrm{{C}}$: ${_ltx(joints[2])}$.\n"
         f"2단계: $\\mathrm{{P}}(\\text{{불}})={_ltx(tot)}$.\n"
         f"3단계: $\\mathrm{{P}}(\\mathrm{{A}}\\mid\\text{{불}})="
         f"\\dfrac{{{_ltx(joints[0])}}}{{{_ltx(tot)}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "베이즈 조건부확률(3원인)", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "bayes3", "ps": ps, "rs": rs, "ans": [ans.p, ans.q]}
    return rr


def gen_func_composition_value():
    """일차함수 반복합성 (f∘…∘f)(c) (킬러, 상). 직접 반복으로 정확.

    $f(x)=ax+b$의 $n$번 합성 $f^n(x)=a^n x+b\\dfrac{a^n-1}{a-1}$.
    """
    a = random.choice([2, 3, -1, -2])
    b = random.randint(-4, 4)
    c = random.randint(-3, 3)
    n = random.choice([3, 4, 5])
    val = c
    for _ in range(n):
        val = a*val + b
    ans = val
    choices, lab = make_mc(sp.Integer(ans), [ans-3, ans+3, ans-6, ans+6, -ans])
    fl = f"{a}x{'+' if b>=0 else ''}{b}"
    comp = "\\circ".join(["f"]*n)
    q = (f"함수 $f(x)={fl}$에 대하여 $(\\underbrace{{{comp}}}_{{{n}\\text{{개}}}})({c})$의 값은?")
    e = (f"1단계: $f(x)={fl}$를 ${c}$부터 차례로 ${n}$번 적용한다.\n"
         f"2단계: 반복 계산하면 값은 ${ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "일차함수의 반복합성", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "funccomp", "a": a, "b": b, "c": c, "n": n, "ans": ans}
    return rr


def gen_sn_to_an():
    """수열의 합 Sₙ 으로부터 일반항 aₖ (킬러, 상). aₖ=Sₖ-Sₖ₋₁, a₁=S₁.

    $S_n=an^2+bn+c$ 가 주어질 때 특정 항 $a_k$를 구한다. $c\\ne0$이면 $a_1$이
    일반항 공식에서 벗어나는 함정.
    """
    a = random.choice([1, 2, 3])
    b = random.randint(-4, 4)
    c = random.choice([0, 0, 1, -1, 2, -2])

    def S(nn):
        return a*nn*nn + b*nn + c
    k = random.randint(1, 8)
    ans = S(k) - S(k-1) if k >= 2 else S(1)
    cterm = (f"+{c}" if c > 0 else (f"{c}" if c < 0 else ""))
    bterm = (f"+{b}n" if b > 0 else (f"{b}n" if b < 0 else ""))
    Sexpr = f"{a}n^2{bterm}{cterm}".replace("1n^2", "n^2")
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-2*a, ans+2*a, -ans])
    q = (f"수열 $\\{{a_n\\}}$의 첫째항부터 제$n$항까지의 합 $S_n={Sexpr}$일 때, "
         f"$a_{{{k}}}$의 값은?")
    if k >= 2:
        e = (f"1단계: $a_{{{k}}}=S_{{{k}}}-S_{{{k-1}}}$ ($k\\ge2$).\n"
             f"2단계: $S_{{{k}}}={S(k)}$, $S_{{{k-1}}}={S(k-1)}$이므로 "
             f"$a_{{{k}}}={S(k)}-{S(k-1)}={ans}$.\n\n정답: {lab}")
    else:
        e = (f"1단계: $a_1=S_1={S(1)}={ans}$ ($c\\ne0$이면 일반항 공식과 달라짐에 주의).\n\n"
             f"정답: {lab}")
    rr = _rec("대수", "수열의 합과 일반항", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "sntoan", "a": a, "b": b, "c": c, "k": k, "ans": ans}
    return rr


def gen_antideriv_initial():
    """부정적분 + 초기조건 → 함숫값 (킬러, 상). f(t1)=f(t0)+∫f'. sympy 정확.

    $f'(x)$와 $f(t_0)=v_0$이 주어질 때 $f(t_1)=v_0+\\int_{t_0}^{t_1}f'(x)dx$.
    """
    a = random.choice([3, 6])
    b = random.choice([-4, -2, 2, 4])
    cc = random.randint(-3, 3)
    fp = a*x**2 + b*x + cc
    t0 = random.choice([0, 1, -1])
    t1 = random.choice([2, 3, -2])
    if t1 == t0:
        t1 = t0 + 2
    v0 = random.randint(-5, 5)
    ans = v0 + sp.integrate(fp, (x, t0, t1))
    ans = int(ans)
    fpl = sp.latex(fp)
    choices, lab = make_mc(sp.Integer(ans), [ans-3, ans+3, ans-6, ans+6, -ans])
    q = (f"함수 $f(x)$의 도함수가 $f'(x)={fpl}$이고 $f({t0})={v0}$일 때, "
         f"$f({t1})$의 값은?")
    e = (f"1단계: $f({t1})=f({t0})+\\displaystyle\\int_{{{t0}}}^{{{t1}}}f'(x)\\,dx$.\n"
         f"2단계: $\\int_{{{t0}}}^{{{t1}}}({fpl})\\,dx={sp.integrate(fp,(x,t0,t1))}$.\n"
         f"3단계: $f({t1})={v0}+{sp.integrate(fp,(x,t0,t1))}={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "부정적분과 초기조건", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "antideriv", "a": a, "b": b, "c": cc,
                   "t0": t0, "t1": t1, "v0": v0, "ans": ans}
    return rr


def gen_exp_log_system():
    """지수·로그 연립방정식 → x+y (킬러, 상). 이차방정식으로 정확.

    $\\log_2(xy)=m$, $(\\log_2 x)(\\log_2 y)=p$ → $u=\\log_2 x,v=\\log_2 y$는
    $t^2-mt+p=0$의 두 근. $x+y=2^u+2^v$.
    """
    u, v = sorted(random.sample([1, 2, 3, 4, 5], 2))
    m, p = u+v, u*v
    ans = 2**u + 2**v
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), [
        (sp.Integer(2**(u+v)), "$x+y$가 아니라 $xy=2^{m}$을 구함(합·곱 혼동)"),
        ans-2, ans+2, ans+4, 2**u*2**v + 2])
    q = (f"$1$보다 큰 두 실수 $x$, $y$가 $\\log_2(xy)={m}$, "
         f"$(\\log_2 x)(\\log_2 y)={p}$를 만족시킨다. $x<y$일 때, $x+y$의 값은?")
    e = (f"1단계: $u=\\log_2 x$, $v=\\log_2 y$로 놓으면 $u+v={m}$, $uv={p}$.\n"
         f"2단계: $u$, $v$는 $t^2-{m}t+{p}=0$의 두 근 ${u}$, ${v}$ ($x<y$이므로 $u={u}$, $v={v}$).\n"
         f"3단계: $x=2^{{{u}}}$, $y=2^{{{v}}}$이므로 $x+y={2**u}+{2**v}={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "지수·로그 연립방정식", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "explogsys", "u": u, "v": v, "ans": ans}
    return rr


def gen_normal_interval():
    """정규분포 양쪽 구간확률 P(m-aσ≤X≤m+bσ) (상). 표값 합.

    표준화 후 $P(-a\\le Z\\le b)=P(0\\le Z\\le a)+P(0\\le Z\\le b)$. 주어진 표값 사용.
    """
    TBL = {1: "0.3413", sp.Rational(3, 2): "0.4332", 2: "0.4772"}
    (za, ta), (zb, tb) = random.sample(list(TBL.items()), 2)
    m = random.choice([50, 60, 100])
    s = random.choice([4, 8, 10])
    val = float(ta) + float(tb)
    ans_txt = f"{val:.4f}".rstrip("0").rstrip(".")
    lo = int(m - za*s) if (za*s) == int(za*s) else m - za*s
    hi = int(m + zb*s) if (zb*s) == int(zb*s) else m + zb*s
    # 보기(소수) 직접 조립
    pool = list({ans_txt, ta, tb, f"{abs(float(ta)-float(tb)):.4f}".rstrip("0").rstrip("."),
                 f"{0.5+float(ta)-float(tb):.4f}".rstrip("0").rstrip(".")})
    while len(pool) < 5:
        pool.append(f"{val + 0.05*len(pool):.4f}")
    pool = sorted(set(pool), key=float)[:5]
    if ans_txt not in pool:
        pool[-1] = ans_txt
        pool = sorted(set(pool), key=float)
    idx = pool.index(ans_txt)
    choices = [f"${v}$" for v in pool]
    lab = f"{idx+1}번 (${ans_txt}$)"

    def zt(z):
        return "1" if z == 1 else ("1.5" if z == sp.Rational(3, 2) else "2")
    q = (f"확률변수 $X$가 정규분포 $\\mathrm{{N}}({m},\\,{s}^2)$을 따를 때, "
         f"$\\mathrm{{P}}({lo}\\le X\\le {hi})$의 값은? (단, "
         f"$\\mathrm{{P}}(0\\le Z\\le {zt(za)})={ta}$, $\\mathrm{{P}}(0\\le Z\\le {zt(zb)})={tb}$)")
    e = (f"1단계: $Z=\\dfrac{{X-{m}}}{{{s}}}$로 표준화하면 ${lo}\\le X\\le{hi}$는 "
         f"$-{zt(za)}\\le Z\\le{zt(zb)}$.\n"
         f"2단계: $\\mathrm{{P}}(-{zt(za)}\\le Z\\le{zt(zb)})="
         f"{ta}+{tb}={ans_txt}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "정규분포 구간확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "norminterval", "ta": ta, "tb": tb, "ans": ans_txt}
    return rr


def gen_trig_synthesis_max():
    """삼각함수 합성 a sinx+b cosx+c 의 최대·최소 (킬러, 상). 정확.

    $a\\sin x+b\\cos x=\\sqrt{a^2+b^2}\\sin(x+\\alpha)$. 최댓값 $\\sqrt{a^2+b^2}+c$,
    최솟값 $-\\sqrt{a^2+b^2}+c$. 정수해 위해 $(a,b)$는 피타고라스 쌍.
    """
    a, b = random.choice([(3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5)])
    R = sp.Integer(a*a + b*b)
    R = sp.sqrt(R)                              # 정수 (피타고라스)
    assert R == int(R)
    R = int(R)
    c = random.randint(-5, 5)
    kind = random.choice(["max", "min", "prod"])
    if kind == "max":
        ans = R + c
        query = "최댓값은?"
        tail = f"최댓값 $=\\sqrt{{{a}^2+{b}^2}}+({c})={R}+({c})={ans}$"
    elif kind == "min":
        ans = -R + c
        query = "최솟값은?"
        tail = f"최솟값 $=-\\sqrt{{{a}^2+{b}^2}}+({c})=-{R}+({c})={ans}$"
    else:
        ans = (R + c)*(-R + c)                  # 최대×최소 = c²-(a²+b²)
        query = "최댓값과 최솟값의 곱은?"
        tail = f"(최대)$\\times$(최소)$=({R}+{c})(-{R}+{c})={c}^2-{a*a+b*b}={ans}$"
    cterm = f"+{c}" if c > 0 else (f"{c}" if c < 0 else "")
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans-4, ans+4, -ans])
    q = (f"함수 $f(x)={a}\\sin x+{b}\\cos x{cterm}$의 {query}")
    e = (f"1단계: ${a}\\sin x+{b}\\cos x=\\sqrt{{{a}^2+{b}^2}}\\sin(x+\\alpha)"
         f"={R}\\sin(x+\\alpha)$ (합성), $-{R}\\le{R}\\sin(x+\\alpha)\\le{R}$.\n"
         f"2단계: {tail}.\n\n정답: {lab}")
    rr = _rec("대수", "삼각함수의 합성과 최대·최소", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "trigsyn", "a": a, "b": b, "c": c, "kind": kind, "ans": ans}
    return rr


def gen_ball_draw_prob():
    """주머니 공 조건부확률 (킬러, 상). 조합 브루트포스로 정확.

    흰 공 w, 검은 공 b개에서 k개를 동시에 꺼낼 때, 흰 공이 j₁개 이상이었을 때
    정확히 j₂개일 조건부확률.
    """
    from itertools import combinations
    while True:
        w = random.randint(3, 6)
        b = random.randint(3, 6)
        k = random.choice([3, 4])
        j1 = random.randint(1, min(w, k)-1)
        j2 = random.randint(j1, min(w, k))
        balls = [1]*w + [0]*b
        num = den = 0
        for comb_ in combinations(range(w+b), k):
            nw = sum(balls[i] for i in comb_)
            if nw >= j1:
                den += 1
                if nw == j2:
                    num += 1
        if den and 0 < num < den:
            break
    ans = sp.Rational(num, den)
    dd = [sp.Rational(num+1, den), sp.Rational(num-1, den) if num >= 2 else sp.Rational(num+2, den),
          ans*sp.Rational(1, 2), ans*sp.Rational(3, 2), sp.Rational(num, den+1)]
    choices, lab = make_mc(ans, [d for d in dd if 0 < d < 1 and d != ans])
    q = (f"흰 공 ${w}$개와 검은 공 ${b}$개가 들어 있는 주머니에서 임의로 ${k}$개의 공을 "
         f"동시에 꺼낸다. 꺼낸 흰 공이 ${j1}$개 이상이었을 때, 흰 공이 정확히 ${j2}$개일 "
         f"확률은?")
    e = (f"1단계: (조건) 흰 공이 ${j1}$개 이상인 경우의 수는 ${den}$.\n"
         f"2단계: 그중 흰 공이 정확히 ${j2}$개인 경우의 수는 ${num}$.\n"
         f"3단계: 조건부확률 $=\\dfrac{{{num}}}{{{den}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "주머니 조건부확률", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "balldraw", "w": w, "b": b, "k": k, "j1": j1, "j2": j2,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_distribute_objects():
    """같은 종류의 물건을 여러 사람에게 나눠주기 — 하한·짝수 제약 (상). 브루트포스.

    수능 30번 빈출 '구체적 분배' 상황. 중복조합을 제약과 함께 직접 전수 조사해
    정확한 경우의 수를 센다(공식 추정 아님).
    """
    from itertools import product as _prod
    items = random.choice(["사탕", "초콜릿", "공", "연필", "구슬"])
    while True:
        people = random.choice([3, 4])
        names = ["A", "B", "C", "D"][:people]
        N = random.randint(8, 14)
        lowers = [random.randint(1, 3) for _ in range(people)]
        if sum(lowers) > N:
            continue
        even_idx = random.randrange(people)        # 한 사람은 짝수 개
        sols = [s for s in _prod(range(N + 1), repeat=people) if sum(s) == N]
        cnt = 0
        for s in sols:
            if all(s[i] >= lowers[i] for i in range(people)) and s[even_idx] % 2 == 0:
                cnt += 1
        # 함정: 짝수 조건을 빠뜨린 개수
        naive = sum(1 for s in sols if all(s[i] >= lowers[i] for i in range(people)))
        if 12 <= cnt <= 400 and naive != cnt:
            break
    cond_lines = []
    for i in range(people):
        cond_lines.append(f"{names[i]}는 ${lowers[i]}$개 이상을 받는다.")
    cond_lines.append(f"{names[even_idx]}가 받는 {items}의 개수는 짝수이다.")
    conds = "\n".join(f"({chr(0xAC00+i)}) {c}" if False else f"({'가나다라마'[i]}) {c}"
                      for i, c in enumerate(cond_lines))
    q = (f"같은 종류의 {items} ${N}$개를 {', '.join(names)} "
         f"{people}명에게 남김없이 나누어 줄 때, 다음 조건을 모두 만족시키도록 "
         f"나누어 주는 경우의 수는?\n\n{conds}")
    e = (f"1단계: 각자 받는 개수를 순서쌍으로 보고 합이 ${N}$인 경우를 "
         f"제약(하한·짝수) 아래 전수 조사한다.\n"
         f"2단계: 조건을 모두 만족하는 경우의 수는 ${cnt}$이다.\n\n")
    dd = [(sp.Integer(naive), "짝수 조건을 빠뜨림")]
    dd += [d for d in [cnt-2, cnt+2, cnt-4, cnt+4, cnt+6] if d > 0]
    choices, lab, traps = make_mc_tagged(sp.Integer(cnt), dd)
    e += f"정답: {lab}"
    rr = _rec("확률과 통계", "같은 것을 나누어 주는 경우의 수", "상", 4, "객관식",
              q, choices, lab, e)
    if traps:
        rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "distribute", "N": N, "people": people,
                   "lowers": lowers, "even_idx": even_idx, "ans": cnt}
    return rr


def gen_monotone_function_count():
    """단조 함수의 개수 + 고정점 제약 (킬러, 상). 브루트포스로 정확.

    수능 확통 30번 클래식. 공역으로 가는 (넓은 의미) 증가함수 중 추가 조건을
    만족하는 것을 전수 조사한다. 추상적 'f(f(x))' 류가 아니라 단조성+값 조건.
    """
    from itertools import combinations_with_replacement as _cwr
    while True:
        n = 5
        m = random.choice([5, 6, 7])
        seqs = list(_cwr(range(1, m + 1), n))      # f(1)≤f(2)≤…≤f(5)
        mode = random.choice(["fix_value", "no_fixed", "sum_even"])
        if mode == "fix_value":
            k = random.randint(2, n - 1)
            v = random.randint(2, m - 1)
            good = [s for s in seqs if s[k - 1] == v]
            cond2 = f"$f({k})={v}$"
        elif mode == "no_fixed":
            good = [s for s in seqs if all(s[i] != i + 1 for i in range(n))]
            cond2 = "모든 $x$에 대하여 $f(x)\\ne x$이다."
        else:
            a, b = sorted(random.sample(range(1, n + 1), 2))
            good = [s for s in seqs if (s[a - 1] + s[b - 1]) % 2 == 0]
            cond2 = f"$f({a})+f({b})$는 짝수이다."
        cnt = len(good)
        if 12 <= cnt <= 600:
            break
    Xs = ",".join(str(i) for i in range(1, n + 1))
    Ys = ",".join(str(i) for i in range(1, m + 1))
    q = (f"두 집합 $X=\\{{{Xs}\\}}$, $Y=\\{{{Ys}\\}}$에 대하여 다음 두 조건을 "
         f"모두 만족시키는 함수 $f:X\\to Y$의 개수는?\n\n"
         f"(가) $X$의 임의의 두 원소 $x_1$, $x_2$에 대하여 $x_1<x_2$이면 "
         f"$f(x_1)\\le f(x_2)$이다.\n(나) {cond2}")
    e = (f"1단계: 조건 (가)에서 $f$는 $f(1)\\le f(2)\\le\\cdots\\le f(5)$인 "
         f"(넓은 의미) 증가함수이다.\n"
         f"2단계: 조건 (나)까지 만족하는 경우를 전수 조사하면 ${cnt}$개이다.\n\n")
    dd = [d for d in [cnt-3, cnt+3, cnt-6, cnt+6, cnt+9, cnt-1] if d > 0]
    choices, lab = make_mc(sp.Integer(cnt), dd)
    e += f"정답: {lab}"
    rr = _rec("확률과 통계", "단조함수의 개수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "monotone_func", "n": n, "m": m, "ans": cnt}
    return rr


def gen_card_arrangement():
    """카드(숫자)를 일렬로 나열 — 위치·이웃 제약 (상). 순열 브루트포스로 정확.

    수능 빈출 '구체적 나열' 상황. 6장의 카드를 모두 일렬로 놓는 경우 중 조건을
    만족하는 것을 전수 조사(720가지)한다.
    """
    from itertools import permutations as _perm
    cards = list(range(1, 7))
    while True:
        mode = random.choice(["ends_odd", "evens_apart", "between"])
        cnt = 0
        if mode == "ends_odd":
            for p in _perm(cards):
                if p[0] % 2 == 1 and p[-1] % 2 == 1:
                    cnt += 1
            cond = "양 끝에 놓인 카드에 적힌 수가 모두 홀수이다."
        elif mode == "evens_apart":
            for p in _perm(cards):
                pos = {v: i for i, v in enumerate(p)}
                evens = [pos[v] for v in (2, 4, 6)]
                evens.sort()
                if all(evens[i+1] - evens[i] >= 2 for i in range(2)):
                    cnt += 1
            cond = "짝수가 적힌 카드끼리는 서로 이웃하지 않는다."
        else:
            g = random.choice([1, 2])
            for p in _perm(cards):
                pos = {v: i for i, v in enumerate(p)}
                if abs(pos[1] - pos[2]) == g + 1:
                    cnt += 1
            cond = f"$1$이 적힌 카드와 $2$가 적힌 카드 사이에 다른 카드가 ${g}$장 있다."
        if 30 <= cnt <= 600:
            break
    q = (f"$1$부터 $6$까지의 자연수가 하나씩 적힌 $6$장의 카드를 모두 일렬로 "
         f"나열할 때, 다음 조건을 만족시키도록 나열하는 경우의 수는?\n\n"
         f"({cond})")
    e = (f"1단계: $6$장을 일렬로 놓는 모든 경우($6!=720$가지) 중 조건을 만족하는 "
         f"것을 전수 조사한다.\n2단계: 조건을 만족하는 경우의 수는 ${cnt}$이다.\n\n")
    dd = [d for d in [cnt-12, cnt+12, cnt-24, cnt+24, cnt+36] if d > 0]
    choices, lab = make_mc(sp.Integer(cnt), dd)
    e += f"정답: {lab}"
    rr = _rec("확률과 통계", "조건을 만족시키는 카드 나열", "상", 4, "객관식",
              q, choices, lab, e)
    rr["_meta"] = {"chk": "cardline", "ans": cnt}
    return rr


def gen_two_bag_bayes():
    """두 주머니 + 주사위 선택의 베이즈 조건부확률 (킬러, 상). 조합 브루트포스.

    수능 빈출 '구체적 베이즈' 상황. 주사위 결과로 주머니를 고르고 공을 꺼낸 뒤,
    꺼낸 결과를 보고 어느 주머니였을지의 조건부확률. 분수 정답을 직접 계산.
    """
    from itertools import combinations as _comb
    from fractions import Fraction as _F
    while True:
        wA, bA = random.randint(2, 5), random.randint(2, 5)
        wB, bB = random.randint(2, 5), random.randint(2, 5)
        k = random.choice([2, 3])
        thr = random.choice([2, 3, 4])            # 주사위 thr 이하면 A
        pA, pB = _F(thr, 6), _F(6 - thr, 6)
        # 각 주머니에서 k개 모두 흰색일 확률
        def all_white(w, b):
            tot = len(list(_comb(range(w + b), k)))
            fav = len(list(_comb(range(w), k))) if w >= k else 0
            return _F(fav, tot)
        prA, prB = all_white(wA, bA), all_white(wB, bB)
        denom = pA * prA + pB * prB
        if denom == 0:
            continue
        ans = (pA * prA) / denom                   # P(A | 모두 흰색)
        if 0 < ans < 1 and ans.numerator > 1 and ans.denominator <= 60:
            break
    ansr = sp.Rational(ans.numerator, ans.denominator)
    q = (f"주머니 $A$에는 흰 공 ${wA}$개와 검은 공 ${bA}$개가, 주머니 $B$에는 흰 공 "
         f"${wB}$개와 검은 공 ${bB}$개가 들어 있다. 한 개의 주사위를 던져 나온 눈의 "
         f"수가 ${thr}$ 이하이면 주머니 $A$에서, 그렇지 않으면 주머니 $B$에서 임의로 "
         f"${k}$개의 공을 동시에 꺼낸다. 꺼낸 ${k}$개의 공이 모두 흰 공이었을 때, "
         f"그 공이 주머니 $A$에서 꺼낸 것이었을 확률은?")
    e = (f"1단계: 주머니 $A$를 택할 확률 ${_ltx(sp.Rational(thr,6))}$, 그때 모두 흰 공일 "
         f"확률 ${_ltx(sp.Rational(prA.numerator,prA.denominator))}$.\n"
         f"2단계: 주머니 $B$를 택할 확률 ${_ltx(sp.Rational(6-thr,6))}$, 그때 모두 흰 공일 "
         f"확률 ${_ltx(sp.Rational(prB.numerator,prB.denominator))}$.\n"
         f"3단계: 베이즈 정리로 "
         f"$\\dfrac{{P(A\\cap W)}}{{P(W)}}={_ltx(ansr)}$.\n\n")
    dd = [ansr*sp.Rational(1, 2), 1 - ansr, ansr*sp.Rational(3, 2),
          sp.Rational(ans.numerator + 1, ans.denominator),
          sp.Rational(ans.numerator, ans.denominator + 1)]
    choices, lab = make_mc(ansr, [d for d in dd if 0 < d < 1 and d != ansr])
    e += f"정답: {lab}"
    rr = _rec("확률과 통계", "두 주머니 베이즈 조건부확률", "상", 4, "객관식",
              q, choices, lab, e)
    rr["_meta"] = {"chk": "twobag", "ans": [ansr.p, ansr.q]}
    return rr


def gen_periodic_recurrence():
    """주기성 점화식의 특정 항 (킬러, 상). 직접 반복으로 정확.

    $a_{n+1}=\\dfrac1{1-a_n}$ 은 주기 3. $a_1$로부터 직접 반복해 $a_N$을 구한다.
    """
    a1 = sp.Rational(random.choice([2, 3, -1, -2]), random.choice([1, 2, 3]))
    if a1 == 1:
        a1 = sp.Rational(2)
    N = random.choice([50, 100, 2024, 2025, 30])

    def step(a):
        return sp.Rational(1, 1) / (1 - a)
    a = a1
    seq = [a1]
    for _ in range(2):
        a = step(a)
        seq.append(a)
    # 주기 3 확인
    assert step(seq[2]) == a1, "주기 아님"
    ans = seq[(N-1) % 3]
    dd = [seq[(N) % 3], seq[(N-2) % 3], ans+1, ans-1, -ans]
    choices, lab = make_mc(ans, [d for d in dd if d != ans])
    q = (f"수열 $\\{{a_n\\}}$이 $a_1={_ltx(a1)}$이고, 모든 자연수 $n$에 대하여 "
         f"$a_{{n+1}}=\\dfrac1{{1-a_n}}$을 만족시킬 때, $a_{{{N}}}$의 값은?")
    e = (f"1단계: $a_1={_ltx(seq[0])}$, $a_2={_ltx(seq[1])}$, $a_3={_ltx(seq[2])}$, "
         f"$a_4=a_1$ — 주기가 $3$이다.\n"
         f"2단계: ${N}=3\\times{(N-1)//3}+{((N-1)%3)+1}$이므로 "
         f"$a_{{{N}}}=a_{{{((N-1)%3)+1}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("대수", "주기성 점화식", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "periodrec", "a1": [a1.p, a1.q], "N": N,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_abs_integral_area():
    """정적분 절댓값 넓이 ∫|x(x-c)|dx (킬러, 상). 부호 분할로 정확.

    $f(x)=x(x-c)$ ($c>0$)는 $[0,c]$에서 음, $[c,d]$에서 양. 절댓값 적분을
    근에서 분할해 정확히 계산. sympy 와 대조.
    """
    c = random.choice([2, 3, 4])
    d = c + random.choice([1, 2, 3])
    f = x*(x - c)
    area = sp.integrate(-f, (x, 0, c)) + sp.integrate(f, (x, c, d))
    assert area == sp.integrate(sp.Abs(f), (x, 0, d)), "넓이 불일치"
    ans = area
    fl = sp.latex(sp.expand(f))
    dd = [ans + sp.Rational(c**3, 6), ans - sp.Rational(c**3, 6),
          ans*sp.Rational(1, 2), ans*sp.Rational(3, 2),
          sp.integrate(f, (x, 0, d))]            # 함정: 절댓값 없이 적분
    choices, lab = make_mc(ans, [v for v in dd if v != ans])
    q = (f"곡선 $y={fl}$와 $x$축 및 직선 $x={d}$로 둘러싸인 부분의 넓이는? "
         f"(즉 $\\displaystyle\\int_0^{{{d}}}\\left|{fl}\\right|\\,dx$의 값)")
    e = (f"1단계: $f(x)={fl}=x(x-{c})$는 $0\\le x\\le{c}$에서 $f\\le0$, "
         f"${c}\\le x\\le{d}$에서 $f\\ge0$.\n"
         f"2단계: 넓이 $=\\displaystyle\\int_0^{{{c}}}\\!\\!-f\\,dx+\\int_{{{c}}}^{{{d}}}\\!f\\,dx"
         f"={_ltx(area)}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분과 절댓값 넓이", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "absarea", "c": c, "d": d, "ans": [ans.p, ans.q]}
    return rr


def gen_binomial_coeff():
    """이항정리 일반항: (ax²+b/x)ⁿ 의 특정 항 계수 (킬러, 상). sympy 전개 검증.

    일반항 $T_{r+1}=\\binom{n}{r}(ax^2)^{n-r}(b/x)^r$, $x$의 지수 $2(n-r)-r=2n-3r$.
    목표 지수 $m$이 되는 $r$를 찾아 계수 $\\binom{n}{r}a^{n-r}b^r$ 계산.
    """
    from math import comb
    while True:
        n = random.choice([4, 5, 6])
        a = random.choice([1, 2])
        b = random.choice([1, 2])
        r = random.randint(0, n)
        m = 2*n - 3*r                          # 해당 항의 x 지수
        coef = comb(n, r) * a**(n-r) * b**r
        if coef <= 2000 and abs(m) <= 2*n:
            break
    # sympy 전개로 독립 검증
    exp = sp.expand((a*x**2 + sp.Rational(b, 1)/x)**n)
    assert exp.coeff(x, m) == coef, "이항계수 불일치"
    ans = coef
    aex = "x^2" if a == 1 else f"{a}x^2"
    bex = "\\dfrac1x" if b == 1 else f"\\dfrac{{{b}}}x"
    target = "상수항" if m == 0 else f"$x^{{{m}}}$의 계수"
    qword = "상수항은?" if m == 0 else f"$x^{{{m}}}$의 계수는?"
    dd = [comb(n, r-1)*a**(n-r+1)*b**(r-1) if r >= 1 else coef+comb(n, 1),
          comb(n, r+1)*a**(n-r-1)*b**(r+1) if r+1 <= n else coef*2,
          coef*2, coef//2 if coef % 2 == 0 else coef+3, comb(n, r)]
    choices, lab = make_mc(sp.Integer(ans), [d for d in dd if d > 0 and d != coef])
    q = (f"다항식 $\\left({aex}+{bex}\\right)^{{{n}}}$의 전개식에서 {qword}")
    e = (f"1단계: 일반항 $T_{{r+1}}=\\binom{{{n}}}{{r}}({aex})^{{{n}-r}}"
         f"\\left({bex}\\right)^{{r}}$이고, $x$의 지수는 $2({n}-r)-r={2*n}-3r$이다.\n"
         f"2단계: 지수 $={m}$ 에서 ${2*n}-3r={m}$, 즉 $r={r}$.\n"
         f"3단계: 계수 $=\\binom{{{n}}}{{{r}}}\\cdot{a}^{{{n-r}}}\\cdot{b}^{{{r}}}={ans}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "이항정리의 일반항 계수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "binomcoeff", "n": n, "a": a, "b": b, "m": m, "ans": ans}
    return rr


def gen_telescoping_sum():
    """부분분수(망원급수) 수열의 합 (킬러, 상). sympy 정확 분수.

    등차수열 $a_k=a_1+(k-1)d$에 대하여 $\\sum_{k=1}^{n}\\dfrac1{a_k a_{k+1}}
    =\\dfrac1d\\left(\\dfrac1{a_1}-\\dfrac1{a_{n+1}}\\right)$. 직접합과 대조 검증.
    """
    a1 = random.choice([1, 2, 3])
    d = random.choice([1, 2, 3])
    n = random.choice([4, 5, 6, 7])

    def ak(k):
        return a1 + (k-1)*d
    direct = sum(sp.Rational(1, ak(k)*ak(k+1)) for k in range(1, n+1))
    formula = sp.Rational(1, d)*(sp.Rational(1, a1) - sp.Rational(1, ak(n+1)))
    assert direct == formula, "망원합 불일치"
    ans = direct
    a_n1 = ak(n+1)
    # a_k 의 일반항을 깔끔히: a_k = d·k + (a1-d)
    c0 = a1 - d
    gen_ak = (f"{d}n" if d != 1 else "n") + (f"+{c0}" if c0 > 0 else (f"{c0}" if c0 < 0 else ""))

    def S(nn):  # 상한 nn 까지의 부분합 (망원)
        return sp.Rational(1, d)*(sp.Rational(1, a1) - sp.Rational(1, ak(nn+1)))
    dd = [S(n+1), S(n-1), S(n+2), S(n-2)]
    choices, lab = make_mc(ans, [x for x in dd if 0 < x < 1 and x != ans])
    q = (f"수열 $\\{{a_n\\}}$의 일반항이 $a_n={gen_ak}$일 때, "
         f"$\\displaystyle\\sum_{{k=1}}^{{{n}}}\\dfrac{{1}}{{a_k\\,a_{{k+1}}}}$의 값은?")
    e = (f"1단계: $\\dfrac1{{a_k a_{{k+1}}}}=\\dfrac1{{{d}}}\\left(\\dfrac1{{a_k}}"
         f"-\\dfrac1{{a_{{k+1}}}}\\right)$ (부분분수 분해, 공차 $d={d}$).\n"
         f"2단계: 망원합 $=\\dfrac1{{{d}}}\\left(\\dfrac1{{a_1}}-\\dfrac1{{a_{{{n+1}}}}}\\right)"
         f"=\\dfrac1{{{d}}}\\left(\\dfrac1{{{a1}}}-\\dfrac1{{{a_n1}}}\\right)={_ltx(ans)}$.\n\n"
         f"정답: {lab}")
    rr = _rec("대수", "부분분수 수열의 합(망원급수)", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "telescope", "a1": a1, "d": d, "n": n,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_log_quadratic_ineq():
    """로그 이차부등식의 정수해 개수 (킬러, 상). 치환 + 정수 카운트, 정확.

    $(\\log_p x)^2-(t_1+t_2)\\log_p x+t_1 t_2\\le 0 \\iff \\log_p x\\in[t_1,t_2]
    \\iff x\\in[p^{t_1},p^{t_2}]$. 정수해 개수 $=p^{t_2}-p^{t_1}+1$.
    """
    p = random.choice([2, 3])
    t1 = random.choice([1, 2])
    t2 = t1 + random.choice([2, 3]) if p == 2 else t1 + random.choice([1, 2])
    s = t1 + t2
    pr = t1 * t2
    lo, hi = p**t1, p**t2
    ans = hi - lo + 1
    mid = f"-{s}\\log_{{{p}}} x" if s else ""
    lhs = f"(\\log_{{{p}}} x)^2{mid}+{pr}"
    q = (f"부등식 ${lhs}\\le 0$을 만족시키는 정수 $x$의 개수는?")
    dd = [d for d in [ans-2, ans+2, ans-4, ans+4, hi-lo] if d > 0 and d != ans]
    choices, lab = make_mc(sp.Integer(ans), dd)
    e = (f"1단계: $\\log_{{{p}}} x=t$로 놓으면 $t^2-{s}t+{pr}\\le0$, 즉 $(t-{t1})(t-{t2})\\le0$ "
         f"에서 ${t1}\\le t\\le{t2}$.\n"
         f"2단계: ${t1}\\le\\log_{{{p}}} x\\le{t2}$이므로 ${p}^{{{t1}}}\\le x\\le{p}^{{{t2}}}$, "
         f"즉 ${lo}\\le x\\le{hi}$.\n"
         f"3단계: 이 범위의 정수 $x$의 개수는 ${hi}-{lo}+1={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "로그 이차부등식의 정수해", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "logquadineq", "p": p, "t1": t1, "t2": t2, "ans": ans}
    return rr


def _divisor_count(k):
    return sum(1 for i in range(1, k+1) if k % i == 0)


def gen_log_divisor_count():
    """로그 자연수 조건 + 약수 개수 조건 → n 최솟값 (킬러, 상). 정수론, 정확.

    드릴드1 수1 채굴: (가) log_p n은 자연수 → n=p^k. (나) log_m n이 자연수가 되는
    2 이상 자연수 m의 개수 = (m=p^{k/j}, j|k 이므로) k의 약수의 개수 d(k) = t.
    답 n = p^(최소의 d(k)=t인 k).
    """
    p = random.choice([2, 3])
    t = random.choice([2, 3, 4])
    k = next(kk for kk in range(1, 60) if _divisor_count(kk) == t)
    n = p**k
    divs = [i for i in range(1, k+1) if k % i == 0]
    q = (f"다음 조건을 만족시키는 자연수 $n$의 최솟값을 구하시오.\n\n"
         f"(가) $\\log_{{{p}}} n$은 자연수이다.\n"
         f"(나) $\\log_m n$의 값이 자연수가 되도록 하는 $2$ 이상의 자연수 $m$의 "
         f"값의 개수는 ${t}$이다.")
    e = (f"1단계: (가)에서 $n={p}^k$ ($k$는 자연수).\n"
         f"2단계: $\\log_m n$이 자연수 $\\iff n=m^j \\iff m={p}^{{k/j}}$ (단, $j$는 $k$의 약수). "
         f"따라서 그런 $m$의 개수는 $k$의 약수의 개수 $d(k)$와 같다.\n"
         f"3단계: (나)에서 $d(k)={t}$.\n"
         f"4단계: $d(k)={t}$인 최소의 $k$는 ${k}$ (약수 ${', '.join(map(str, divs))}$).\n"
         f"5단계: $n={p}^{{{k}}}={n}$.\n\n정답: {n}")
    rr = _rec("대수", "로그와 약수의 개수", "상", 4, "주관식", q, None, str(n), e)
    rr["_meta"] = {"chk": "logdivcount", "p": p, "t": t, "ans": n}
    return rr


def _trialseq_pred(spec, n):
    """반복시행 수열 조건 spec → 술어 함수."""
    t = spec["t"]
    if t == "sum_ge":
        return lambda s, v=spec["v"]: sum(s) >= v
    if t == "sum_le":
        return lambda s, v=spec["v"]: sum(s) <= v
    if t == "consec":
        return lambda s, v=spec["v"]: any(s[i] == v and s[i+1] == v
                                          for i in range(len(s)-1))
    if t == "count_eq":
        return lambda s, v=spec["v"], k=spec["k"]: s.count(v) == k
    if t == "count_ge":
        return lambda s, v=spec["v"], k=spec["k"]: s.count(v) >= k
    raise AssertionError(t)


def _trialseq_text(spec, n):
    t = spec["t"]
    if t == "sum_ge":
        return f"확인한 ${n}$개의 수의 합은 ${spec['v']}$ 이상이다."
    if t == "sum_le":
        return f"확인한 ${n}$개의 수의 합은 ${spec['v']}$ 이하이다."
    if t == "consec":
        return f"숫자 ${spec['v']}$가 적힌 카드가 연속하여 나오는 경우가 있다."
    if t == "count_eq":
        return f"숫자 ${spec['v']}$가 적힌 카드가 정확히 ${spec['k']}$번 나온다."
    return f"숫자 ${spec['v']}$가 적힌 카드가 ${spec['k']}$번 이상 나온다."


def gen_seq_trial_prob():
    """반복시행 수열에 대한 두 조건 결합 확률 (킬러, 상). 2ⁿ 브루트포스 정확.

    드릴 확통 채굴: 1,2가 적힌 카드를 꺼내 확인 후 되넣기를 n번 반복(각 1/2),
    (가)(나) 두 조건을 모두 만족시킬 확률. 전체 2ⁿ 수열을 전수 조사해
    favorable/total 을 정확히 계산.
    """
    from itertools import product as _ip
    vals = (1, 2)
    n = random.choice([6, 7])
    seqs = list(_ip(vals, repeat=n))
    total = len(seqs)
    while True:
        types = random.sample(["sum_ge", "sum_le", "consec", "count_ge", "count_eq"], 2)
        specs = []
        for t in types:
            if t in ("sum_ge",):
                specs.append({"t": t, "v": random.randint(n+2, 2*n-1)})
            elif t == "sum_le":
                specs.append({"t": t, "v": random.randint(n+1, 2*n-2)})
            elif t == "consec":
                specs.append({"t": t, "v": random.choice([1, 2])})
            elif t == "count_ge":
                specs.append({"t": t, "v": random.choice([1, 2]), "k": random.randint(2, n-1)})
            else:
                specs.append({"t": t, "v": random.choice([1, 2]), "k": random.randint(2, n-2)})
        preds = [_trialseq_pred(s, n) for s in specs]
        fav = sum(1 for s in seqs if all(p(s) for p in preds))
        naive = sum(1 for s in seqs if preds[0](s))      # (가)만
        if 6 <= fav <= total-6 and naive != fav:
            break
    ans = sp.Rational(fav, total)
    dd = [sp.Rational(f2, total) for f2 in
          [fav-2, fav+2, fav-4, fav+4, naive] if 0 < f2 < total and f2 != fav]
    choices, lab, traps = make_mc_tagged(ans, [
        (sp.Rational(naive, total), "조건 (나)를 빠뜨리고 (가)만으로 계산")] +
        [d for d in dd if d != sp.Rational(naive, total)])
    labels = ["가", "나"]
    cond = "\n".join(f"({labels[i]}) {_trialseq_text(specs[i], n)}" for i in range(2))
    q = (f"주머니에 숫자 $1$, $2$가 하나씩 적힌 카드 $2$장이 들어 있다. 이 주머니에서 "
         f"임의로 한 장의 카드를 꺼내어 카드에 적힌 수를 확인하고 다시 넣는 시행을 "
         f"${n}$번 반복할 때, 다음 조건을 모두 만족시킬 확률은?\n\n{cond}")
    e = (f"1단계: 전체 경우의 수는 $2^{{{n}}}={total}$ (각 시행 독립, 등확률).\n"
         f"2단계: 두 조건을 모두 만족하는 수열을 전수 조사로 세면 ${fav}$개이다.\n"
         f"3단계: 확률 $=\\dfrac{{{fav}}}{{{total}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "반복시행 수열의 조건부 확률", "상", 4, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "seqtrialprob", "n": n, "specs": specs,
                   "fav": fav, "total": total, "ans": [ans.p, ans.q]}
    return rr


def gen_trig_solution_count():
    """삼각방정식 A sin²x+B sin x+C=0 (0≤x<2π) 의 해의 개수 (킬러, 상).

    sin x=s 치환 → 이차식의 실근별 해 개수 합산(정확 대수). 정답은 sympy로 근을
    재계산해 독립 검증. 드릴/수능 단골 유형('해의 개수')으로 엔진에 없던 유형.
    """
    s = sp.Symbol("s")
    nice = [sp.Rational(-1), sp.Rational(-1, 2), sp.Integer(0),
            sp.Rational(1, 2), sp.Integer(1), sp.Integer(2), sp.Integer(-2)]
    while True:
        s1, s2 = random.sample(nice, 2)
        expr = sp.expand((s - s1) * (s - s2))
        d = sp.ilcm(*[sp.denom(c) for c in expr.as_coefficients_dict().values()])
        P = sp.Poly(sp.expand(expr * d), s)
        A = int(P.coeff_monomial(s**2))
        B = int(P.coeff_monomial(s))
        C = int(P.coeff_monomial(1))
        cnt = _trig_count_rule(s1) + _trig_count_rule(s2)
        if A != 0 and 1 <= cnt <= 4:
            break

    def _term(coef, sym):
        if coef == 0:
            return ""
        sign = "+" if coef > 0 else "-"
        mag = abs(coef)
        body = sym if mag == 1 and sym else f"{mag}{sym}"
        return f"{sign}{body}" if sym else f"{sign}{mag}"
    lhs = (_term(A, "\\sin^2 x") + _term(B, "\\sin x") + _term(C, "")).lstrip("+")
    dd = [d2 for d2 in [cnt-1, cnt+1, cnt-2, cnt+2, 6] if 0 < d2 <= 8 and d2 != cnt]
    choices, lab = make_mc(sp.Integer(cnt), dd)
    q = (f"$0 \\le x < 2\\pi$일 때, 방정식 ${lhs}=0$의 서로 다른 실근의 개수는?")
    e = (f"1단계: $\\sin x=s$로 놓으면 ${A}s^2{_term(B,'s')}{_term(C,'')}=0$, "
         f"근은 $s={sp.latex(s1)},\\,{sp.latex(s2)}$.\n"
         f"2단계: $-1\\le\\sin x\\le1$에서 각 근의 해 개수 — "
         f"$s={sp.latex(s1)}$: ${_trig_count_rule(s1)}$개, "
         f"$s={sp.latex(s2)}$: ${_trig_count_rule(s2)}$개.\n"
         f"3단계: 합 ${cnt}$개.\n\n정답: {lab}")
    rr = _rec("대수", "삼각방정식의 해의 개수", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "trigcount", "A": A, "B": B, "C": C, "ans": cnt}
    return rr


def gen_abs_sum_min():
    """f(x)=Σ|x-aᵢ| 의 최솟값 (중상). 중앙값에서 최소, 정확 계산.

    절댓값 합 함수의 최솟값은 aᵢ들의 중앙값에서 달성. 정답=Σ|중앙값-aᵢ|.
    """
    n = random.choice([3, 5])                  # 홀수 → 최소점 유일
    pts = sorted(random.sample(range(-6, 7), n))
    med = pts[n // 2]
    ans = sum(abs(med - a) for a in pts)
    # 함정: 평균 부근/양끝 합 등 흔한 오답
    mean_pt = round(sum(pts) / n)
    trap = sum(abs(mean_pt - a) for a in pts)
    dd = [d for d in [ans+1, ans+2, ans-1, ans-2, trap] if d > 0 and d != ans]
    choices, lab = make_mc(sp.Integer(ans), dd)
    terms = "+".join(f"|x-({a})|" if a < 0 else f"|x-{a}|" for a in pts)
    q = f"함수 $f(x)={terms}$의 최솟값은?"
    e = (f"1단계: 절댓값 합 $\\sum|x-a_i|$는 $a_i$들의 중앙값에서 최소이다 "
         f"(점 {pts}의 중앙값 ${med}$).\n"
         f"2단계: $f({med})=\\sum|{med}-a_i|={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "절댓값 합 함수의 최솟값", "중", 3, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "abssummin", "pts": pts, "ans": ans}
    return rr


def gen_truefalse_cubic():
    """삼차함수 성질에 대한 보기 ㄱㄴㄷ 합답형 (킬러, 상). 각 명제를 sympy로 참거짓 판정.

    실제 수능이 즐겨 쓰는 '<보기>에서 옳은 것만 고른 것은?' 형식. 엔진에 없던 출제
    형식. 각 명제의 참/거짓을 sympy로 엄밀히 계산하고, 정답=참인 보기들의 집합.
    """
    LBL = ["ㄱ", "ㄴ", "ㄷ"]
    while True:
        a = random.randint(-3, 3)
        b = random.randint(-4, 4)
        c = random.randint(-4, 4)
        f = x**3 + a*x**2 + b*x + c
        fp = sp.diff(f, x)
        disc = sp.discriminant(f, x)          # >0 ⇔ 서로 다른 세 실근
        incr = sp.discriminant(fp, x) <= 0    # f'≥0 ∀x ⇔ f' 판별식 ≤0 (3>0)

        def st_value():
            k = random.choice([-2, -1, 1, 2])
            true = random.random() < 0.5
            v = int(f.subs(x, k)) + (0 if true else random.choice([-3, -2, 2, 3]))
            return (f"$f({k})={v}$", v == int(f.subs(x, k)), {"t": "val", "k": k, "v": v})

        def st_deriv():
            k = random.choice([-2, -1, 1, 2])
            true = random.random() < 0.5
            v = int(fp.subs(x, k)) + (0 if true else random.choice([-4, -2, 2, 4]))
            return (f"$f'({k})={v}$", v == int(fp.subs(x, k)), {"t": "der", "k": k, "v": v})

        def st_incr():
            return ("함수 $f(x)$는 실수 전체의 집합에서 증가한다.", bool(incr), {"t": "inc"})

        def st_roots():
            return ("방정식 $f(x)=0$은 서로 다른 세 실근을 갖는다.", bool(disc > 0), {"t": "root"})

        def st_extremum():
            k = random.choice([-1, 0, 1])
            true_at = (fp.subs(x, k) == 0) and (sp.diff(fp, x).subs(x, k) != 0)
            return (f"함수 $f(x)$는 $x={k}$에서 극값을 갖는다.", bool(true_at), {"t": "ext", "k": k})

        pool = [st_value, st_deriv, st_incr, st_roots, st_extremum]
        chosen = random.sample(pool, 3)
        stmts = [fn() for fn in chosen]
        truth = [t for _, t, _ in stmts]
        specs = [sp_ for _, _, sp_ in stmts]
        ntrue = sum(truth)
        if 1 <= ntrue <= 2:                   # 너무 자명(0 또는 3)하지 않게
            break
    true_set = tuple(LBL[i] for i in range(3) if truth[i])
    # 보기(5개): 정답 부분집합 + 다른 비공집합 부분집합 4개
    from itertools import combinations
    allsub = []
    for r in (1, 2, 3):
        for comb in combinations(range(3), r):
            allsub.append(tuple(LBL[i] for i in comb))
    others = [s for s in allsub if s != true_set]
    random.shuffle(others)
    opts = [true_set] + others[:4]
    opts = sorted(set(opts), key=lambda s: (len(s), s))
    while len(opts) < 5:                      # 안전(이론상 항상 7개라 충분)
        for s in allsub:
            if s not in opts:
                opts.append(s); break
        opts = sorted(set(opts), key=lambda s: (len(s), s))
    choices = [", ".join(s) for s in opts]
    idx = opts.index(true_set)
    lab = f"{idx+1}번 ({choices[idx]})"
    boki = "\n".join(f"{LBL[i]}. {stmts[i][0]}" for i in range(3))
    q = (f"삼차함수 $f(x)={sp.latex(f)}$에 대하여 <보기>에서 옳은 것만을 있는 대로 "
         f"고른 것은?\n\n<보기>\n{boki}")
    judge = "\n".join(f"{LBL[i]}: {'참' if truth[i] else '거짓'}" for i in range(3))
    e = (f"1단계: $f'(x)={sp.latex(fp)}$, 판별식 등으로 각 보기를 판정한다.\n"
         f"2단계: 각 보기 참거짓 — {judge.replace(chr(10), ', ')}.\n"
         f"3단계: 옳은 것은 {choices[idx]}.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "삼차함수 보기 합답형", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "tfcubic", "a": a, "b": b, "c": c,
                   "specs": specs, "truth": truth,
                   "true_set": list(true_set), "ans": choices[idx]}
    return rr


def gen_explog_symmetry():
    """지수·로그 y=x 대칭 두 점 사이 거리 (중상). 정확해 + 수치 교차검증.

    P(p, aᵖ)는 y=aˣ 위, 그 y=x 대칭점 Q(aᵖ, p)는 y=log_a x 위. 두 곡선은 서로
    역함수라 y=x 대칭. PQ=|aᵖ-p|√2 (정확해). numverify 로 수치 교차검증.
    """
    from csat_forge import numverify as nv
    a = random.choice([2, 3, 4])
    p = random.choice([2, 3] if a == 2 else [1, 2])
    ap = a**p
    d = abs(ap - p)
    ans = d * sp.sqrt(2)                       # PQ 길이 (정확해)
    # 수치 교차검증: 실제 두 점 좌표로 거리 재계산
    P = (sp.Integer(p), sp.Integer(ap))
    Q = (sp.Integer(ap), sp.Integer(p))
    pq = sp.sqrt((P[0]-Q[0])**2 + (P[1]-Q[1])**2)
    assert nv.verify_value(ans, pq), "수치 교차검증 실패"
    dd = [(d+1)*sp.sqrt(2), (d-1)*sp.sqrt(2) if d > 1 else (d+2)*sp.sqrt(2),
          d*2, sp.sqrt(2)*d/2]
    choices, lab = make_mc(ans, dd)
    q = (f"두 곡선 $y={a}^x$과 $y=\\log_{{{a}}} x$는 직선 $y=x$에 대하여 대칭이다. "
         f"곡선 $y={a}^x$ 위의 점 $\\mathrm{{P}}$의 $x$좌표가 ${p}$이고, 점 $\\mathrm{{P}}$를 "
         f"직선 $y=x$에 대하여 대칭이동한 점 $\\mathrm{{Q}}$가 곡선 $y=\\log_{{{a}}} x$ "
         f"위에 있을 때, 선분 $\\mathrm{{PQ}}$의 길이는?")
    e = (f"1단계: $\\mathrm{{P}}=({p},\\,{a}^{{{p}}})=({p},\\,{ap})$이고, $y=x$ 대칭점 "
         f"$\\mathrm{{Q}}=({ap},\\,{p})$이다.\n"
         f"2단계: $\\mathrm{{PQ}}=\\sqrt{{({p}-{ap})^2+({ap}-{p})^2}}"
         f"=|{ap}-{p}|\\sqrt2={d}\\sqrt2$.\n\n정답: {lab}")
    rr = _rec("대수", "지수·로그함수의 대칭", "중", 3, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "explogsym", "a": a, "p": p, "d": int(d),
                   "ans_sqrt2_coeff": int(d)}
    return rr


def gen_secant_exp_roots():
    """지수곡선과 직선의 교점 — 두 실근의 합 (킬러, 상). 볼록성+수치검증.

    $y=a^x$는 아래로 볼록 → 두 곡선점 $(x_1,a^{x_1})$, $(x_2,a^{x_2})$를 지나는
    직선(할선)은 $y=a^x$와 정확히 그 두 점에서만 만난다. 방정식 $a^x=mx+c$의 두
    실근은 $x_1,x_2$. numverify로 '근이 정확히 2개'를 고정밀 수치 재확인.
    """
    from csat_forge import numverify as nv
    a = random.choice([2, 3])
    pool = [0, 1, 2, 3] if a == 2 else [0, 1, 2, 3]   # 음 아닌 지수 → 정수 좌표
    x1, x2 = sorted(random.sample(pool, 2))
    ax1, ax2 = sp.Integer(a)**x1, sp.Integer(a)**x2
    m = sp.Rational(ax2 - ax1, x2 - x1)
    c = ax1 - m*x1
    # 검증: 직선이 두 곡선점을 정확히 지난다 + 고정밀 수치 확인
    assert m*x1 + c == ax1 and m*x2 + c == ax2
    assert nv.close(sp.Integer(a)**x1 - (m*x1 + c), 0)
    ans = x1 + x2
    line = sp.latex(sp.nsimplify(m*x + c))
    dd = [(sp.Integer(x1*x2), "두 근의 곱과 혼동"),
          (sp.Integer(int(ax1 + ax2)), "근이 아니라 함숫값($a^{x}$)의 합을 구함"),
          ans-2, ans+2, ans-1]
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), [d for d in dd])
    q = (f"곡선 $y={a}^x$과 직선 $y={line}$이 만나는 두 점의 $x$좌표를 각각 "
         f"$\\alpha$, $\\beta$라 할 때, $\\alpha+\\beta$의 값은?")
    e = (f"1단계: $y={a}^x$은 아래로 볼록하므로 직선과 많아야 두 점에서 만난다.\n"
         f"2단계: 직선이 곡선 위의 두 점 $({x1},\\,{int(ax1)})$, $({x2},\\,{int(ax2)})$를 "
         f"지나므로 교점의 $x$좌표는 ${x1}$, ${x2}$.\n"
         f"3단계: $\\alpha+\\beta={x1}+{x2}={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "지수곡선과 직선의 교점", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "secantexp", "a": a, "x1": x1, "x2": x2, "ans": ans}
    return rr


def gen_exp_power_intsols():
    """지수=거듭제곱 방정식의 정수해 (킬러, 상). 브루트포스, 정확.

    $a^x=c\\,x^k$ (대수적으로 못 푸는 초월방정식)의 정수인 실근을 모두 찾아 합한다.
    """
    while True:
        a = random.choice([2, 3, 4])
        c = random.choice([1, 2, 3, 4])
        k = random.choice([1, 2, 3, 4])
        sols = [xx for xx in range(1, 20) if a**xx == c*xx**k]
        if len(sols) == 2:
            break
    ans = sum(sols)
    rhs = ("" if c == 1 else str(c)) + ("x" if k == 1 else f"x^{k}")
    dd = [(sp.Integer(sols[0]*sols[1]), "정수해의 합이 아니라 곱을 구함"),
          ans-1, ans+1, ans-2, ans+2]
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), [d for d in dd])
    q = (f"방정식 ${a}^x={rhs}$을 만족시키는 정수 $x$의 값을 모두 더한 값은?")
    e = (f"1단계: 정수 $x$를 차례로 대입하면 $x={sols[0]}$, $x={sols[1]}$에서 "
         f"${a}^x={rhs}$이 성립한다.\n"
         f"2단계: ${sols[0]}+{sols[1]}={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "지수=거듭제곱 방정식의 정수해", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "exppow", "a": a, "c": c, "k": k, "ans": ans}
    return rr


def gen_exp_inequality_intcount():
    """지수부등식의 정수해 개수 (킬러, 상). 브루트포스, 정확.

    $2^x\\le mx+c$ 를 만족시키는 정수 $x$의 개수(유계 구간)를 전수로 센다.
    """
    from fractions import Fraction as Fr
    while True:
        m = random.choice([2, 3, 4])
        c = random.randint(3, 10)
        sols = [xx for xx in range(-20, 30) if Fr(2)**xx <= m*xx + c]
        if 3 <= len(sols) <= 9:
            break
    ans = len(sols)
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, ans+3])
    q = (f"부등식 $2^x\\le {m}x+{c}$를 만족시키는 정수 $x$의 개수는?")
    e = (f"1단계: $2^x$은 급격히 증가하므로 해는 유계 구간의 정수들이다.\n"
         f"2단계: 정수를 대입해 세면 $x={sols[0]}$부터 $x={sols[-1]}$까지 ${ans}$개.\n\n정답: {lab}")
    rr = _rec("대수", "지수부등식의 정수해", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "expineq", "m": m, "c": c, "ans": ans}
    return rr


def gen_trig_addition():
    """삼각함수의 덧셈정리 (킬러, 상). 피타고라스 수, 정확.

    예각 $A$, $B$의 삼각비(피타고라스 수)에서 $\\sin(A\\pm B)$ 또는 $\\cos(A\\pm B)$.
    """
    triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
    (pa, qa, ha) = random.choice(triples)
    (pb, qb, hb) = random.choice(triples)
    sA, cA = sp.Rational(pa, ha), sp.Rational(qa, ha)
    sB, cB = sp.Rational(pb, hb), sp.Rational(qb, hb)
    sign = random.choice([1, -1])
    so = "+" if sign == 1 else "-"
    kind = random.choice(["sin", "cos"])
    if kind == "sin":
        ans = sA*cB + sign*cA*sB
        tgt = f"\\sin(A{so}B)"
        trap = (cA*cB + sign*sA*sB, "사인 덧셈정리에 코사인 공식을 잘못 적용")
    else:
        ans = cA*cB - sign*sA*sB
        tgt = f"\\cos(A{so}B)"
        trap = (sA*cB - sign*cA*sB, "코사인 덧셈정리에 사인 공식을 잘못 적용")
    ans = sp.nsimplify(ans)
    dd = [(sp.nsimplify(trap[0]), trap[1]),
          ans+sp.Rational(1, ha*hb), ans-sp.Rational(1, ha*hb), -ans, ans*sp.Rational(1, 2)]
    choices, lab, traps = make_mc_tagged(ans, [d for d in dd])
    q = (f"두 예각 $A$, $B$에 대하여 $\\sin A=\\dfrac{{{pa}}}{{{ha}}}$, "
         f"$\\cos B=\\dfrac{{{qb}}}{{{hb}}}$일 때, ${tgt}$의 값은?")
    e = (f"1단계: $A$, $B$가 예각이므로 $\\cos A=\\dfrac{{{qa}}}{{{ha}}}$, "
         f"$\\sin B=\\dfrac{{{pb}}}{{{hb}}}$.\n"
         f"2단계: 덧셈정리로 ${tgt}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("대수", "삼각함수의 덧셈정리", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "trigadd", "tri_a": [pa, qa, ha], "tri_b": [pb, qb, hb],
                   "sign": sign, "kind": kind, "ans": [ans.p, ans.q]}
    return rr


def gen_exp_monotone_root():
    """지수방정식 a^x+bx=k 의 실근 (킬러, 상). 강증가→유일근, 수치검증.

    $h(x)=a^x+bx$는 $a>1,b>0$에서 강증가 → 실근 유일. $k=h(x_0)$로 구성하면 근 $x_0$.
    """
    from csat_forge import numverify as nv
    a = random.choice([2, 3])
    b = random.choice([1, 2, 3])
    x0 = random.choice([-1, 0, 1, 2, 3])
    k = sp.Integer(a)**x0 + b*x0
    assert nv.close(sp.Integer(a)**x0 + b*x0 - k, 0)
    ans = x0
    bterm = f"+{b}x" if b > 0 else f"{b}x"
    extra = sp.Integer(int(k)) if (int(k) == k and int(k) not in
                                   (ans-1, ans+1, ans-2, ans+2)) else ans+3
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, extra])
    q = (f"방정식 ${a}^x{bterm}={_ltx(k)}$의 실근은?")
    e = (f"1단계: $h(x)={a}^x{bterm}$은 ${a}^x$도 $x$도 증가하므로 강증가함수 → "
         f"$h(x)={_ltx(k)}$의 실근은 단 하나.\n"
         f"2단계: $x={x0}$을 대입하면 ${a}^{{{x0}}}+{b}\\cdot({x0})={_ltx(k)}$ 성립. "
         f"따라서 실근은 ${x0}$.\n\n정답: {lab}")
    rr = _rec("대수", "지수방정식의 실근(단조성)", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "expmono", "a": a, "b": b, "x0": x0, "ans": ans}
    return rr


def gen_log_monotone_root():
    """로그방정식 log_a x+bx=k 의 실근 (킬러, 상). 강증가→유일근, 수치검증.

    $h(x)=\\log_a x+bx$는 $x>0$에서 강증가 → 유일근. $x_0=a^j$로 구성하면 근 $a^j$.
    """
    from csat_forge import numverify as nv
    a = random.choice([2, 3])
    b = random.choice([1, 2])
    j = random.choice([1, 2, 3] if a == 2 else [1, 2])
    x0 = a**j
    k = sp.Integer(j) + b*x0
    assert nv.close(sp.log(x0, a) + b*x0 - k, 0)
    ans = x0
    bterm = f"+{b}x" if b > 1 else "+x"
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-a, ans+a, j])
    q = (f"방정식 $\\log_{{{a}}} x{bterm}={k}$의 실근은?")
    e = (f"1단계: $h(x)=\\log_{{{a}}} x{bterm}$은 $x>0$에서 강증가 → 실근은 유일.\n"
         f"2단계: $x={x0}$을 대입하면 $\\log_{{{a}}} {x0}+{b}\\cdot{x0}={j}+{b*x0}={k}$ 성립. "
         f"실근은 ${x0}$.\n\n정답: {lab}")
    rr = _rec("대수", "로그방정식의 실근(단조성)", "상", 4, "객관식", q, choices, lab, e)
    rr["_meta"] = {"chk": "logmono", "a": a, "b": b, "j": j, "ans": ans}
    return rr


def gen_secant_log_roots():
    """로그곡선과 직선의 교점 — 두 실근의 합 (킬러, 상). 오목성+수치검증.

    $y=\\log_a x$는 위로 볼록(오목) → 두 곡선점 $(a^{j_1},j_1)$, $(a^{j_2},j_2)$를
    지나는 직선은 곡선과 정확히 그 두 점에서만 만난다. 교점의 $x$좌표 합 $=a^{j_1}+a^{j_2}$.
    """
    from csat_forge import numverify as nv
    a = random.choice([2, 3])
    pool = [0, 1, 2, 3] if a == 2 else [0, 1, 2]
    j1, j2 = sorted(random.sample(pool, 2))
    x1, x2 = a**j1, a**j2                     # 곡선 위 점의 x좌표(=a^j)
    m = sp.Rational(j2 - j1, x2 - x1)
    c = sp.Rational(j1) - m*x1
    assert m*x1 + c == j1 and m*x2 + c == j2
    assert nv.close(sp.log(x1, a) - (m*x1 + c), 0)
    ans = x1 + x2
    line = sp.latex(sp.nsimplify(m*x + c))
    dd = [(sp.Integer(x1*x2), "두 근의 곱과 혼동"),
          (sp.Integer(j1 + j2), "$x$좌표가 아니라 $y$좌표(로그값)의 합을 구함"),
          ans-2, ans+2, ans-1]
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), [d for d in dd])
    q = (f"곡선 $y=\\log_{{{a}}} x$과 직선 $y={line}$이 만나는 두 점의 $x$좌표를 각각 "
         f"$\\alpha$, $\\beta$라 할 때, $\\alpha+\\beta$의 값은?")
    e = (f"1단계: $y=\\log_{{{a}}} x$은 위로 볼록하므로 직선과 많아야 두 점에서 만난다.\n"
         f"2단계: 직선이 곡선 위의 두 점 $({x1},\\,{j1})$, $({x2},\\,{j2})$를 지나므로 "
         f"교점의 $x$좌표는 ${x1}$, ${x2}$.\n"
         f"3단계: $\\alpha+\\beta={x1}+{x2}={ans}$.\n\n정답: {lab}")
    rr = _rec("대수", "로그곡선과 직선의 교점", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "secantlog", "a": a, "j1": j1, "j2": j2, "ans": ans}
    return rr


def gen_even_odd_integral():
    """우함수·기함수 대칭성을 이용한 정적분 ∫_{-a}^a f dx (중상). 드릴 수2 채굴.

    f(x)=c3x³+c2x²+c1x+c0 에서 홀수차항은 [-a,a]에서 0, 짝수차항만 2∫_0^a.
    정답은 sympy 정적분으로 보장(c2는 3의 배수로 잡아 정수해).
    """
    a = random.choice([1, 2, 3])
    c3 = random.choice([-3, -2, -1, 1, 2, 3])
    c2 = random.choice([-6, -3, 3, 6])         # 3의 배수 → 정수해
    c1 = random.choice([-5, -3, -1, 2, 4])
    c0 = random.choice([-2, -1, 1, 2, 4])
    f = c3*x**3 + c2*x**2 + c1*x + c0
    ans = int(sp.integrate(f, (x, -a, a)))     # sympy 가 정답 보장
    half = ans // 2 if ans % 2 == 0 else None  # 함정: ×2 누락
    full_anti = int(sp.integrate(f, (x, 0, a)))  # 함정: 0~a 만 계산
    dd = [(sp.Integer(full_anti), "구간을 $0$~$a$ 로만 잡아 적분 (대칭성 미사용)")]
    if half is not None and half != full_anti:
        dd.append((sp.Integer(half), "짝수항 $2\\int_0^a$ 의 계수 $2$ 누락"))
    dd += [ans+2*a, ans-2*a, ans+4]
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), dd)
    fl = sp.latex(f)
    q = f"함수 $f(x)={fl}$에 대하여 $\\displaystyle\\int_{{-{a}}}^{{{a}}} f(x)\\,dx$의 값은?"
    e = (f"1단계: 기함수(홀수차) 항 ${sp.latex(c3*x**3)}$, ${sp.latex(c1*x)}$는 "
         f"$\\int_{{-{a}}}^{{{a}}}=0$이다.\n"
         f"2단계: 우함수(짝수차) 항 ${sp.latex(c2*x**2)}$, ${c0}$만 남아 "
         f"$\\int_{{-{a}}}^{{{a}}}=2\\int_0^{{{a}}}({sp.latex(c2*x**2+c0)})\\,dx$이다.\n"
         f"3단계: 계산하면 ${ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분의 대칭성(우함수·기함수)", "중", 3, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "evenodd", "a": a, "c3": c3, "c2": c2, "c1": c1,
                   "c0": c0, "ans": ans}
    return rr


def gen_func_codomain_range():
    """공역이 다른 f:X→Y 치역 크기 + 함숫값 조건 개수 (킬러, 상). 브루트포스.

    드릴 확통 채굴 유형: X={1..n}, Y={1..m}(m≠n 허용)에서 (가) |치역|=r
    (나) f(p)=v 를 모두 만족하는 함수의 개수. itertools.product 로 전수.
    """
    from itertools import product as _ip
    while True:
        n = random.choice([4, 5])
        m = random.choice([4, 5, 6])
        r = random.choice([2, 3])
        if r > min(n, m):
            continue
        p = random.randint(1, n)
        v = random.randint(1, m)
        cnt = sum(1 for f in _ip(range(1, m + 1), repeat=n)
                  if len(set(f)) == r and f[p-1] == v)
        # 함정: (나) 누락(치역크기 조건만)
        naive = sum(1 for f in _ip(range(1, m + 1), repeat=n) if len(set(f)) == r)
        if 8 <= cnt <= 4000 and naive != cnt:
            break
    s = max(4, round(cnt/10))
    dd = [(naive, "조건 (나) $f(p)=v$ 를 빠뜨리고 치역 크기만 셈")]
    dd += [d for d in [cnt-s, cnt+s, cnt-2*s, cnt+2*s] if d > 0]
    choices, lab, traps = make_mc_tagged(sp.Integer(cnt), dd)
    q = (f"두 집합 $X=\\{{1,2,\\cdots,{n}\\}}$, $Y=\\{{1,2,\\cdots,{m}\\}}$에 대하여 "
         f"다음 조건을 모두 만족시키는 함수 $f:X\\to Y$의 개수는?\n\n"
         f"(가) $f$의 치역의 원소의 개수는 ${r}$이다.\n(나) $f({p})={v}$이다.")
    e = (f"1단계: (가) 치역으로 쓸 $Y$의 원소 ${r}$개를 고르고, (나) $f({p})={v}$는 "
         f"그 치역에 ${v}$가 반드시 포함되어야 함을 뜻한다.\n"
         f"2단계: 두 조건을 모두 만족하는 함수를 전수 조사(포함배제)로 세면 ${cnt}$개이다.\n"
         f"(참고: (나)를 빼면 ${naive}$개.)\n\n정답: {lab}")
    rr = _rec("확률과 통계", "공역이 다른 함수의 개수", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "funcxyrange", "n": n, "m": m, "r": r, "p": p, "v": v,
                   "ans": cnt}
    return rr


def gen_graph_inverse_compose():
    """그래프·역함수·합성 세 집합 A,B,C 조건 → a+b+c (킬러, 상).

    드릴5 수2 채굴 유형. f(x)=x³+a, g(x)=bx+q.
      A={(x,f(x))}=y=f(x) 그래프, B={(f(x),x)}=역(직선 y=x 대칭),
      C={(x,g(y))|(x,y)∈A}=y=g(f(x)) 그래프.
    A∩B={(p,c)} (p는 |p|≥2 → 유일) 와 (x0,y0)∈C 로 a,b,c 결정. 정답은 정수.
    """
    p = random.choice([2, 3, -2, -3])          # |p|≥2 ⇒ A∩B 유일(전수검증됨)
    a = p - p**3                               # f(p)=p ⇒ a=p-p³, 그리고 c=p
    c = p
    b = random.choice([-3, -2, 2, 3])
    q = random.choice([1, 3, 5, 7])
    x0 = random.choice([-2, -1, 1, 2])
    fx0 = x0**3 + a
    y0 = b*fx0 + q                             # (x0,y0)∈C 조건값
    ans = a + b + c
    choices, lab = make_mc(sp.Integer(ans), [ans-1, ans+1, ans-2, ans+2, -ans])
    qtxt = (f"두 함수 $f(x)=x^3+a$, $g(x)=bx+{q}$에 대하여 "
            f"세 집합 $A$, $B$, $C$가\n"
            f"$A=\\{{(x,\\,f(x))\\mid x\\text{{는 실수}}\\}}$,\n"
            f"$B=\\{{(f(x),\\,x)\\mid x\\text{{는 실수}}\\}}$,\n"
            f"$C=\\{{(x,\\,g(y))\\mid x,\\,y\\text{{는 실수이고 }}(x,y)\\in A\\}}$\n"
            f"이다. $A\\cap B=\\{{({p},\\,c)\\}}$이고 $({x0},\\,{y0})\\in C$일 때, "
            f"상수 $a$, $b$, $c$에 대하여 $a+b+c$의 값은?")
    e = (f"1단계: $B$는 $A$를 직선 $y=x$에 대칭시킨 집합이므로 $A\\cap B$는 직선 "
         f"$y=x$ 위에 있다. $({p},c)\\in A$에서 $c=f({p})={p}^3+a$, "
         f"$({p},c)\\in B$에서 $f(c)={p}$. 두 식에서 $c={p}$, 따라서 "
         f"$a={p}-{p}^3={a}$.\n"
         f"2단계: $C$는 $y=g(f(x))$의 그래프이다. $({x0},{y0})\\in C$이므로 "
         f"${y0}=g(f({x0}))=b\\cdot f({x0})+{q}$. $f({x0})={x0}^3+({a})={fx0}$이므로 "
         f"${y0}={fx0}\\,b+{q}$에서 $b={b}$.\n"
         f"3단계: $a+b+c=({a})+({b})+({c})={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "그래프·역함수·합성 조건", "상", 4, "객관식",
              qtxt, choices, lab, e)
    rr["_meta"] = {"chk": "graphinvcomp", "p": p, "a": a, "b": b, "q": q,
                   "x0": x0, "ans": ans}
    return rr


# ======================================================================
#  확률과 통계 신규 킬러 (21/28번급 조건결합형)
# ======================================================================
def gen_indep_events_hidden():
    """위장된 독립 조건 + 곱/여곱 → 이차방정식으로 P(A) 결정 (킬러, 상).

    (가) P(B|A)=P(B|A^c)  ← 'A와 B가 독립'의 위장 표현 (핵심 통찰)
    (나) P(A∩B), P(A^c∩B^c) 값 제시
    → x+y = 1 + P(A∩B) - P(A^c∩B^c), xy = P(A∩B) 인 이차방정식의 작은 근.
    """
    pool = [sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(3, 10),
            sp.Rational(2, 5), sp.Rational(1, 2), sp.Rational(3, 5),
            sp.Rational(7, 10), sp.Rational(3, 4)]
    x_, y_ = sorted(random.sample(pool, 2))          # P(A)=x_ < P(B)=y_
    v1 = x_ * y_                                     # P(A∩B) (독립)
    v2 = (1 - x_) * (1 - y_)                         # P(A^c∩B^c)
    ans = x_
    # 유일성 자가검증: t² - st + v1 = 0 의 두 근이 (x_, y_) 와 일치
    tt = sp.symbols("tt")
    roots = sorted(sp.solve(tt**2 - (1 + v1 - v2)*tt + v1, tt))
    assert roots == sorted([x_, y_]), f"근 불일치 {roots}"
    choices, lab, traps = make_mc_tagged(ans, [
        (y_, "P(A)<P(B) 조건 무시 — 이차방정식의 큰 근을 택함"),
        (v1, "P(A∩B)를 P(A)로 착각"),
        ans*sp.Rational(4, 5), ans*sp.Rational(6, 5)])
    q = (f"두 사건 $A$, $B$가 다음 조건을 만족시킨다.\n\n"
         f"(가) $\\mathrm{{P}}(B\\mid A)=\\mathrm{{P}}(B\\mid A^c)$\n"
         f"(나) $\\mathrm{{P}}(A\\cap B)={_ltx(v1)}$, "
         f"$\\mathrm{{P}}(A^c\\cap B^c)={_ltx(v2)}$\n\n"
         f"$\\mathrm{{P}}(A)$의 값은? (단, $\\mathrm{{P}}(A)<\\mathrm{{P}}(B)$)")
    e = (f"1단계: (가)에서 $\\dfrac{{\\mathrm{{P}}(A\\cap B)}}{{\\mathrm{{P}}(A)}}"
         f"=\\dfrac{{\\mathrm{{P}}(A^c\\cap B)}}{{\\mathrm{{P}}(A^c)}}$를 정리하면 "
         f"$\\mathrm{{P}}(A\\cap B)=\\mathrm{{P}}(A)\\mathrm{{P}}(B)$ — 두 사건은 독립.\n"
         f"2단계: $x=\\mathrm{{P}}(A)$, $y=\\mathrm{{P}}(B)$로 놓으면 $xy={_ltx(v1)}$, "
         f"$(1-x)(1-y)={_ltx(v2)}$. 전개하면 $x+y=1+{_ltx(v1)}-{_ltx(v2)}={_ltx(x_+y_)}$.\n"
         f"3단계: $t^2-{_ltx(x_+y_)}\\,t+{_ltx(v1)}=0$의 두 근은 "
         f"${_ltx(x_)}$, ${_ltx(y_)}$. $\\mathrm{{P}}(A)<\\mathrm{{P}}(B)$이므로 "
         f"$\\mathrm{{P}}(A)={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "사건의 독립과 조건부확률", "상", 4, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "indephidden", "v1": [v1.p, v1.q], "v2": [v2.p, v2.q],
                   "ans": [ans.p, ans.q]}
    return rr


def gen_dist_table_unknowns():
    """이산확률분포표 미정계수 (가)(나) 연립 → V(2X+1) (킬러, 상).

    분포표의 두 확률 a, b 가 미지수. (가) 합=1 (나) E(X) 조건을 연립해
    a, b 를 sp.solve 로 결정한 뒤 V(2X+1)=4V(X) 를 계산.
    """
    while True:
        xs = sorted(random.sample([0, 1, 2, 3, 4, 5, 6], 4))
        D = random.choice([8, 10])
        # 확률 분자 4개 (각 ≥1, 합 = D)
        cuts = sorted(random.sample(range(1, D), 3))
        ps = [cuts[0], cuts[1]-cuts[0], cuts[2]-cuts[1], D-cuts[2]]
        if min(ps) >= 1:
            break
    probs = [sp.Rational(p, D) for p in ps]
    E = sum(x_*p_ for x_, p_ in zip(xs, probs))
    V = sum(x_**2*p_ for x_, p_ in zip(xs, probs)) - E**2
    ans = 4 * V                                     # V(2X+1)
    # 유일성 자가검증: a,b 를 미지수로 두고 (합=1, 기댓값) 연립 → 구성값과 일치
    aa, bb = sp.symbols("aa bb")
    unk = [probs[1], probs[3]]                      # 분포표에서 a, b 자리
    sol = sp.solve([aa + bb - (unk[0] + unk[1]),
                    xs[1]*aa + xs[3]*bb - (xs[1]*unk[0] + xs[3]*unk[1])],
                   [aa, bb], dict=True)
    assert len(sol) == 1 and sol[0][aa] == unk[0] and sol[0][bb] == unk[1]
    choices, lab, traps = make_mc_tagged(ans, [
        (2*V, "$V(2X+1)=2V(X)$로 잘못 계산 — 분산은 계수의 제곱배"),
        (4*V + 1, "상수항 $+1$을 분산에 더함 — 상수는 분산에 영향 없음"),
        V, 4*V + 2])
    rows_x = " & ".join(str(x_) for x_ in xs)
    rows_p = " & ".join([_ltx(probs[0]), "a", _ltx(probs[2]), "b"])
    q = (f"확률변수 $X$의 확률분포를 표로 나타내면 다음과 같다.\n\n"
         f"$\\begin{{array}}{{c|cccc|c}} X & {rows_x} & \\text{{합계}} \\\\\\hline "
         f"\\mathrm{{P}}(X=x) & {rows_p} & 1 \\end{{array}}$\n\n"
         f"$\\mathrm{{E}}(X)={_ltx(E)}$일 때, $\\mathrm{{V}}(2X+1)$의 값은? "
         f"(단, $a$, $b$는 상수)")
    e = (f"1단계: 확률의 합이 $1$이므로 $a+b={_ltx(unk[0]+unk[1])}$.\n"
         f"2단계: $\\mathrm{{E}}(X)={_ltx(E)}$ 조건과 연립하면 "
         f"$a={_ltx(unk[0])}$, $b={_ltx(unk[1])}$.\n"
         f"3단계: $\\mathrm{{E}}(X^2)={_ltx(V + E**2)}$이므로 "
         f"$\\mathrm{{V}}(X)={_ltx(V)}$, "
         f"$\\mathrm{{V}}(2X+1)=4\\mathrm{{V}}(X)={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "이산확률분포의 미정계수", "상", 4, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "disttable", "xs": xs, "ps": ps, "D": D,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_perm_constraints():
    """제약조건 결합 일렬 배열 (가)(나) (킬러, 상). 브루트포스."""
    from itertools import permutations
    n = random.choice([6, 7])
    variant = random.choice(["adj", "ends", "nonadj2"])
    # 사람 1=A, 2=B, 3=C, 4=D
    if variant == "adj":
        cond_txt = "(나) $C$와 $D$는 서로 이웃한다."
        def cond(p):
            return abs(p.index(3) - p.index(4)) == 1
    elif variant == "ends":
        cond_txt = "(나) $C$는 양 끝에 서지 않는다."
        def cond(p):
            return p.index(3) not in (0, n - 1)
    else:
        cond_txt = "(나) $C$와 $D$도 서로 이웃하지 않는다."
        def cond(p):
            return abs(p.index(3) - p.index(4)) != 1
    cnt = naive = 0
    for p in permutations(range(1, n + 1)):
        ga = abs(p.index(1) - p.index(2)) != 1      # (가) A,B 비인접
        if ga:
            naive += 1                               # (나) 누락 오답
            if cond(p):
                cnt += 1
    dd = [cnt - 24, cnt + 24, cnt - 48, cnt + 48]
    if naive != cnt:
        dd = [(naive, "조건 (나)를 빠뜨리고 (가)만 적용")] + dd
    choices, lab, traps = make_mc_tagged(sp.Integer(cnt), dd)
    q = (f"$A$, $B$, $C$, $D$를 포함한 ${n}$명이 다음 조건을 만족시키도록 "
         f"일렬로 설 때, 그 경우의 수는?\n\n"
         f"(가) $A$와 $B$는 서로 이웃하지 않는다.\n{cond_txt}")
    e = (f"1단계: 전체 ${n}!$가지 중 (가)를 만족하는 배열을 센다 "
         f"(전체 − $A,B$ 인접 $=({n}-1)!\\times2$ 제외).\n"
         f"2단계: 그중 (나) 조건을 추가로 만족하는 경우를 케이스로 나눠 센다.\n"
         f"3단계: 두 조건을 동시에 만족하는 경우의 수는 ${cnt}$이다.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "제약조건 순열", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "permcons", "n": n, "variant": variant, "ans": cnt}
    return rr


def gen_func_composite_count():
    """합성함수 함숫값 조건 + 일대일대응 아님 함수 개수 (킬러, 상). 브루트포스.

    X={1..5}에서 f:X→X 중 (가) f(f(p))=a (나) f가 일대일대응이 아닌 것의 개수.
    `gen_involution_count` 스타일로 itertools.product 로 전체 함수를 열거해 카운트.
    함정: (나)를 빠뜨리고 (가)만 적용한 개수(naive)를 오답으로 태깅.
    """
    from itertools import product as _ip
    p = random.randint(1, 5)
    a = random.randint(1, 5)
    cnt = naive = 0
    for f in _ip(range(1, 6), repeat=5):
        if f[f[p-1]-1] == a:                          # (가) f(f(p))=a
            naive += 1
            if len(set(f)) < 5:                       # (나) 일대일대응(=순열) 아님
                cnt += 1
    s = max(8, round(cnt / 12))                       # 보기 간격(개수에 비례)
    dd = [cnt - s, cnt + s, cnt - 2*s, cnt + 2*s]
    if naive != cnt:
        dd = [(naive, "일대일대응이 아니다(나) 조건 누락 — (가)만 적용해 순열까지 포함")] + dd
    choices, lab, traps = make_mc_tagged(sp.Integer(cnt), dd)
    q = (f"집합 $X=\\{{1,2,3,4,5\\}}$에 대하여 다음 조건을 만족시키는 함수 "
         f"$f:X\\to X$의 개수는?\n\n"
         f"(가) $f(f({p}))={a}$\n(나) $f$는 일대일대응이 아니다.")
    e = (f"1단계: (나) '일대일대응이 아니다'는 치역의 원소가 $5$개 미만, 즉 "
         f"$f$가 순열이 아님을 뜻한다.\n"
         f"2단계: 전체 $5^5$개의 함수 중 (가) $f(f({p}))={a}$를 만족하는 함수를 모두 "
         f"열거하면 ${naive}$개이다.\n"
         f"3단계: 그중 일대일대응(순열)인 경우를 제외하면 "
         f"${naive}-{naive-cnt}={cnt}$개이다.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "합성함수 조건의 함수의 개수", "상", 4, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "compcount", "p": p, "a": a, "ans": cnt}
    return rr


def gen_binom_conditional():
    """반복시행(이항분포) + 조건부확률 P(X=k | X≥m) (킬러, 상). 정확 분수.

    독립시행을 n번 했을 때 성공 횟수 X~B(n,p). X≥m 이라는 조건 아래 X=k 일
    확률을 sp.Rational 로 정확히 계산. 함정: 조건을 무시한 P(X=k).
    """
    from math import comb
    n = random.choice([4, 5, 6])
    _pn, _pd = random.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4)])
    p = sp.Rational(_pn, _pd)
    m = random.randint(1, n - 1)                      # 조건 X≥m
    k = random.randint(m, n)                          # 구하는 X=k (k≥m)

    def pmf(j):
        return sp.Rational(comb(n, j)) * p**j * (1 - p)**(n - j)
    denom = sum(pmf(j) for j in range(m, n + 1))      # P(X≥m)
    ans = sp.nsimplify(pmf(k) / denom)
    naive = pmf(k)                                    # 함정: 조건 무시
    above1 = sp.nsimplify((ans + 1) / 2)              # ans<·<1 보장
    above2 = sp.nsimplify(ans + (1 - ans) * sp.Rational(1, 3))
    choices, lab, traps = make_mc_tagged(ans, [
        (naive, "조건부확률에서 조건 $(X\\ge m)$ 무시 — $\\mathrm{P}(X=k)$만 계산"),
        ans * sp.Rational(4, 5), above1, above2])
    coin = "한 개의 동전" if p == sp.Rational(1, 2) else "한 개의 주사위"
    setup = (f"{coin}을 ${n}$번 던질 때 앞면이 나오는 횟수를 $X$"
             if p == sp.Rational(1, 2)
             else f"매회 성공할 확률이 ${_ltx(p)}$인 독립시행을 ${n}$번 반복할 때 "
                  f"성공하는 횟수를 $X$")
    q = (f"{setup}라 하자. $X\\ge{m}$일 때, $X={k}$일 확률은?")
    e = (f"1단계: $X$는 이항분포 $\\mathrm{{B}}\\!\\left({n},\\,{_ltx(p)}\\right)$를 따른다. "
         f"$\\mathrm{{P}}(X=j)=\\binom{{{n}}}{{j}}({_ltx(p)})^j({_ltx(1-p)})^{{{n}-j}}$.\n"
         f"2단계: 조건 $\\mathrm{{P}}(X\\ge{m})=\\sum_{{j={m}}}^{{{n}}}\\mathrm{{P}}(X=j)"
         f"={_ltx(denom)}$, 분자 $\\mathrm{{P}}(X={k})={_ltx(naive)}$.\n"
         f"3단계: $\\mathrm{{P}}(X={k}\\mid X\\ge{m})"
         f"=\\dfrac{{{_ltx(naive)}}}{{{_ltx(denom)}}}={_ltx(ans)}$.\n\n정답: {lab}")
    rr = _rec("확률과 통계", "이항분포와 조건부확률", "상", 4, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "binomcond", "n": n, "p": [p.p, p.q], "m": m, "k": k,
                   "ans": [ans.p, ans.q]}
    return rr


def gen_multiset_perm_constraint():
    """같은 것이 있는 순열 + 비인접·양끝 제약 (킬러, 상). 브루트포스.

    중복 숫자가 있는 카드 묶음을 일렬 배열. (가) 특정 숫자 d 카드끼리 비인접
    (나) 양 끝 카드의 수의 홀짝 조건. set(permutations)로 서로 다른 배열만 카운트.
    함정: (나)를 빠뜨리고 (가)만 적용한 개수(naive).
    """
    from itertools import permutations
    decks = [(1, 1, 2, 2, 3), (1, 1, 2, 3, 4), (1, 2, 2, 3, 3),
             (1, 1, 2, 2, 3, 3), (1, 1, 2, 2, 3, 4), (1, 1, 2, 3, 4, 5)]
    while True:
        deck = random.choice(decks)
        d = random.choice(sorted({v for v in deck if deck.count(v) >= 2}))
        parity = random.choice(["odd", "even"])
        perms = set(permutations(deck))

        def _noadj(perm):
            pos = sorted(i for i, v in enumerate(perm) if v == d)
            return all(b - a > 1 for a, b in zip(pos, pos[1:]))

        def _ends(perm):
            e0, e1 = perm[0] % 2, perm[-1] % 2
            return (e0 == 1 and e1 == 1) if parity == "odd" else (e0 == 0 and e1 == 0)
        naive = sum(1 for perm in perms if _noadj(perm))
        cnt = sum(1 for perm in perms if _noadj(perm) and _ends(perm))
        if cnt >= 5 and naive > cnt:                  # 풀만한 난이도 + 함정 보장
            break
    par_txt = "홀수" if parity == "odd" else "짝수"
    n = len(deck)
    cards = ", ".join(f"${v}$" for v in deck)
    s = max(2, round(cnt / 6))
    choices, lab, traps = make_mc_tagged(sp.Integer(cnt), [
        (naive, "조건 (나)(양 끝 홀짝)를 빠뜨리고 (가)만 적용"),
        cnt - s, cnt + s, cnt - 2*s, cnt + 2*s])
    q = (f"숫자 {cards}이 하나씩 적힌 ${n}$장의 카드를 모두 일렬로 나열할 때, "
         f"다음 조건을 만족시키도록 나열하는 경우의 수는?\n\n"
         f"(가) ${d}$이 적힌 카드끼리는 서로 이웃하지 않는다.\n"
         f"(나) 양 끝에는 모두 {par_txt}가 적힌 카드가 온다.")
    e = (f"1단계: 같은 숫자가 있으므로 서로 다른 나열은 모두 ${len(perms)}$가지이다.\n"
         f"2단계: 그중 (가) ${d}$끼리 이웃하지 않는 나열은 ${naive}$가지이다.\n"
         f"3단계: 그중 (나) 양 끝이 모두 {par_txt}인 경우만 세면 ${cnt}$가지이다.\n\n"
         f"정답: {lab}")
    rr = _rec("확률과 통계", "같은 것이 있는 순열의 제약", "상", 4, "객관식",
              q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["_meta"] = {"chk": "msperm", "deck": list(deck), "d": d,
                   "parity": parity, "ans": cnt}
    return rr


# ======================================================================
#  그림·그래프 제시형 (figure spec 포함 — 렌더링은 src/figure_render.py)
# ======================================================================
def gen_fig_parabola_area():
    """[그림] 포물선과 x축 사이 넓이 (중상). 그림은 spec 으로 첨부."""
    kf, a = random.choice([(3, 2), (2, 3), (3, 4), (4, 3), (1, 6), (6, 2)])
    area = sp.Rational(kf * a**3, 6)
    ans = int(area)
    choices, lab = make_mc(sp.Integer(ans), [ans-2, ans+2, ans*2, max(1, ans//2), ans+4])
    fl = sp.latex(sp.expand(kf*x*(a - x)))
    q = (f"그림과 같이 곡선 $y={fl}$와 $x$축으로 둘러싸인 부분의 넓이는?")
    e = (f"1단계: 곡선은 $x=0$, $x={a}$에서 $x$축과 만난다.\n"
         f"2단계: 넓이 $=\\displaystyle\\int_0^{{{a}}}({fl})\\,dx"
         f"={kf}\\cdot\\dfrac{{{a}^3}}{{6}}={ans}$.\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "정적분·넓이(그림)", "중", 3, "객관식", q, choices, lab, e)
    rr["completion"]["figure"] = {"kind": "parabola_region", "k": kf, "a": a}
    rr["completion"]["figure_desc"] = (
        f"위로 볼록한 포물선 y={kf}x({a}-x)가 원점과 ({a},0)에서 x축과 만나고, "
        f"곡선과 x축 사이 영역에 빗금이 그어져 있다.")
    rr["_meta"] = {"chk": "figarea", "k": kf, "a": a, "ans": ans}
    return rr


def gen_fig_velocity_graph():
    """[그림] 속도 그래프(꺾은선) → 움직인 거리 (상). 그림 없이는 풀 수 없는 유형."""
    p = random.choice([1, 2])
    qq = p + random.choice([1, 2])
    r = qq + random.choice([1, 2])
    s = r + random.choice([1, 2])
    h = random.choice([2, 4])
    g = random.choice([2, 4])
    dist1 = h * (r + qq - p) // 2          # 사다리꼴 (0,0)-(p,h)-(q,h)-(r,0)
    dist2 = g * (s - r) // 2               # 삼각형 (r,0)-(s,-g) 의 |v| 넓이
    ans = dist1 + dist2
    disp = dist1 - dist2                   # 함정: 변위
    choices, lab, traps = make_mc_tagged(sp.Integer(ans), [
        (sp.Integer(disp), "거리·변위 혼동 — v<0 구간의 넓이를 빼서 변위를 구함"),
        ans-2, ans+2, ans-4, ans+4])
    q = (f"수직선 위를 움직이는 점 P의 시각 $t\\,(0\\le t\\le {s})$에서의 속도 $v(t)$의 "
         f"그래프가 그림과 같다. $t=0$에서 $t={s}$까지 점 P가 움직인 거리는?")
    e = (f"1단계: 움직인 거리는 $\\int_0^{{{s}}}|v(t)|\\,dt$, 즉 그래프와 $t$축 사이 넓이의 합.\n"
         f"2단계: i) $0\\le t\\le{r}$: 사다리꼴 넓이 $\\dfrac{{({r}+{qq-p})\\cdot{h}}}2={dist1}$, "
         f"ii) ${r}\\le t\\le{s}$: 삼각형 넓이 $\\dfrac{{{s-r}\\cdot{g}}}2={dist2}$.\n"
         f"3단계: 거리 $={dist1}+{dist2}={ans}$. (변위 ${disp}$와 혼동 주의)\n\n정답: {lab}")
    rr = _rec("미적분Ⅰ", "속도와 거리(그래프)", "상", 4, "객관식", q, choices, lab, e)
    rr["completion"]["trap_map"] = traps
    rr["completion"]["figure"] = {
        "kind": "polyline", "points": [[0, 0], [p, h], [qq, h], [r, 0], [s, -g]],
        "xlabel": "t", "ylabel": "v(t)"}
    rr["completion"]["figure_desc"] = (
        f"v(t) 그래프: (0,0)→({p},{h})→({qq},{h})→({r},0)을 잇는 꺾은선 후 "
        f"({s},{-g})까지 직선으로 내려간다.")
    rr["_meta"] = {"chk": "figvel", "p": p, "q": qq, "r": r, "s": s,
                   "h": h, "g": g, "ans": ans}
    return rr


# ======================================================================
#  레지스트리 & 검증
# ======================================================================
GENERATORS = {
    "log_equation": gen_log_equation, "geom_seq": gen_geom_seq, "sigma": gen_sigma,
    "trig_equation": gen_trig_equation, "tangent_cubic": gen_tangent_cubic,
    "extremum_diff": gen_extremum_diff, "area_between": gen_area_between,
    "motion": gen_motion, "cubic_roots_count": gen_cubic_roots_count,
    "complement_dice": gen_complement_dice, "conditional_prob": gen_conditional_prob,
    "binomial_param": gen_binomial_param, "normal_tail": gen_normal_tail,
    "limit_poly": gen_limit_poly, "arith_seq": gen_arith_seq,
    "def_integral_func": gen_def_integral_func, "log_inequality": gen_log_inequality,
    "combination_sum3": gen_combination_sum3, "dice_event_prob": gen_dice_event_prob,
    "abs_cubic_nondiff": gen_abs_cubic_nondiff, "tangent_count_cubic": gen_tangent_count_cubic,
    "binomial_prob": gen_binomial_prob,
    "root_count_cubic": gen_root_count_cubic, "quartic_symmetric": gen_quartic_symmetric,
    "seq_recurrence_sum": gen_seq_recurrence_sum,
    "funceq_integral": gen_funceq_integral, "func_count_odd": gen_func_count_odd,
    "cond_prob_dice": gen_cond_prob_dice, "seq_count_target": gen_seq_count_target,
    "intdef_extremum": gen_intdef_extremum, "comb_bounded": gen_comb_bounded,
    "log_floor_sum": gen_log_floor_sum,
    # 하 난이도 2점 (2028 예시문항 1~3번 체제)
    "exp_calc": gen_exp_calc, "diff_at_point": gen_diff_at_point,
    "arith_term": gen_arith_term, "log_calc": gen_log_calc,
    "limit_simple": gen_limit_simple,
    # 구체적 상황 확통 킬러(수능 30번 스타일 — 분배·나열·베이즈·단조함수)
    "distribute_objects": gen_distribute_objects,
    "monotone_function_count": gen_monotone_function_count,
    "card_arrangement": gen_card_arrangement,
    "two_bag_bayes": gen_two_bag_bayes,
    # (가)(나) 조건 결합형 킬러
    "cond_cubic_integral": gen_cond_cubic_integral,
    "piecewise_diff": gen_piecewise_diff,
    "involution_count": gen_involution_count,
    "graph_inverse_compose": gen_graph_inverse_compose,
    "func_codomain_range": gen_func_codomain_range,
    "even_odd_integral": gen_even_odd_integral,
    "secant_exp_roots": gen_secant_exp_roots,
    "secant_log_roots": gen_secant_log_roots,
    "exp_power_intsols": gen_exp_power_intsols,
    "exp_inequality_intcount": gen_exp_inequality_intcount,
    "trig_addition": gen_trig_addition,
    "exp_monotone_root": gen_exp_monotone_root,
    "log_monotone_root": gen_log_monotone_root,
    "truefalse_cubic": gen_truefalse_cubic,
    "explog_symmetry": gen_explog_symmetry,
    "trig_solution_count": gen_trig_solution_count,
    "abs_sum_min": gen_abs_sum_min,
    "seq_trial_prob": gen_seq_trial_prob,
    "log_divisor_count": gen_log_divisor_count,
    "log_quadratic_ineq": gen_log_quadratic_ineq,
    "telescoping_sum": gen_telescoping_sum,
    "binomial_coeff": gen_binomial_coeff,
    "sn_to_an": gen_sn_to_an, "antideriv_initial": gen_antideriv_initial,
    "exp_log_system": gen_exp_log_system, "normal_interval": gen_normal_interval,
    "trig_solution_sum": gen_trig_solution_sum, "remainder_theorem": gen_remainder_theorem,
    "bayes_three": gen_bayes_three, "func_composition_value": gen_func_composition_value,
    "trig_identity_sincos": gen_trig_identity_sincos,
    "closed_interval_extreme": gen_closed_interval_extreme,
    "combination_condition": gen_combination_condition, "log_chain": gen_log_chain,
    "factorial_equation": gen_factorial_equation, "expected_value": gen_expected_value,
    "at_least_one_prob": gen_at_least_one_prob, "log_equation_domain": gen_log_equation_domain,
    "tangent_yintercept": gen_tangent_yintercept,
    "increasing_cubic_param": gen_increasing_cubic_param,
    "three_dice_max": gen_three_dice_max, "group_partition": gen_group_partition,
    "lattice_path": gen_lattice_path, "circular_arrangement": gen_circular_arrangement,
    "recurrence_telescope": gen_recurrence_telescope,
    "inverse_function_value": gen_inverse_function_value,
    "units_digit": gen_units_digit, "combination_symmetry": gen_combination_symmetry,
    "definite_integral_param": gen_definite_integral_param,
    "without_replacement_prob": gen_without_replacement_prob,
    "factor_theorem": gen_factor_theorem, "geometric_three": gen_geometric_three,
    "quadratic_root_relation": gen_quadratic_root_relation,
    "abs_value_equation": gen_abs_value_equation,
    "quadratic_inequality_intcount": gen_quadratic_inequality_intcount,
    "synthetic_division": gen_synthetic_division,
    "variance_direct": gen_variance_direct, "binomial_normal_approx": gen_binomial_normal_approx,
    "combination_repetition": gen_combination_repetition,
    "arith_three_condition": gen_arith_three_condition,
    "confidence_interval": gen_confidence_interval, "sample_mean_prob": gen_sample_mean_prob,
    "derivative_limit": gen_derivative_limit, "extreme_value_sum": gen_extreme_value_sum,
    "normal_unknown_mean": gen_normal_unknown_mean,
    "multiset_permutation": gen_multiset_permutation,
    "arithmetic_sum_max_n": gen_arithmetic_sum_max_n,
    "parabola_line_area": gen_parabola_line_area,
    "trig_synthesis_max": gen_trig_synthesis_max,
    "ball_draw_prob": gen_ball_draw_prob,
    "periodic_recurrence": gen_periodic_recurrence,
    "abs_integral_area": gen_abs_integral_area,
    # 확률과 통계 신규 킬러 (조건결합형)
    "indep_events_hidden": gen_indep_events_hidden,
    "dist_table_unknowns": gen_dist_table_unknowns,
    "perm_constraints": gen_perm_constraints,
    "func_composite_count": gen_func_composite_count,
    "binom_conditional": gen_binom_conditional,
    "multiset_perm_constraint": gen_multiset_perm_constraint,
    # 그림·그래프 제시형 (LLM 학습 제외, 문제은행·서빙용)
    "fig_parabola_area": gen_fig_parabola_area,
    "fig_velocity_graph": gen_fig_velocity_graph,
}

# 킬러 토픽 집합 (난이도/평가용)
KILLER_TOPICS = {
    "실근의 개수 함수 g(t)", "대칭 사차함수", "수열의 귀납적 정의",
    "절댓값함수의 미분가능성", "곡선 밖의 점에서의 접선",
    "정적분 함수방정식", "조건부 함수의 개수", "조건부확률",
    "정적분으로 정의된 함수의 극값", "중복조합의 경우의 수", "로그와 가우스 기호",
    "조건을 만족시키는 삼차함수", "구간별 함수의 미분가능성", "대합 함수의 개수",
    "사건의 독립과 조건부확률", "이산확률분포의 미정계수", "제약조건 순열",
    "합성함수 조건의 함수의 개수", "이항분포와 조건부확률", "같은 것이 있는 순열의 제약",
    "그래프·역함수·합성 조건", "공역이 다른 함수의 개수", "삼차함수 보기 합답형",
    "삼각방정식의 해의 개수", "반복시행 수열의 조건부 확률", "로그와 약수의 개수",
    "로그 이차부등식의 정수해", "부분분수 수열의 합(망원급수)",
    "이항정리의 일반항 계수", "삼각함수의 합성과 최대·최소", "주머니 조건부확률",
    "주기성 점화식", "정적분과 절댓값 넓이", "삼각방정식 해의 합",
    "나머지정리와 나눗셈", "베이즈 조건부확률(3원인)", "일차함수의 반복합성",
    "삼각함수의 항등식(sin+cos)", "닫힌구간에서의 최대·최소",
    "조건부 조합의 경우의 수", "로그의 연쇄와 밑변환",
    "순열·조합 방정식", "확률변수의 기댓값", "여사건의 확률(적어도 하나)",
    "로그방정식과 진수조건", "접선의 y절편", "증가함수가 되는 조건",
    "주사위 최댓값의 확률", "조 나누기(분할)",
    "격자 최단경로의 경우의 수", "원순열과 이웃 조건", "계차수열(망원) 점화식",
    "역함수의 함숫값", "정규분포의 평균 결정", "같은 것이 있는 순열",
    "등차수열 부분합의 최대", "포물선과 직선 사이의 넓이",
    "모평균의 신뢰구간", "표본평균의 확률", "미분계수의 극한 정의",
    "극댓값과 극솟값의 합", "확률변수의 분산", "이항분포의 정규근사",
    "중복조합", "등차수열을 이루는 세 수", "이차방정식 근과 계수의 관계",
    "절댓값 방정식", "이차부등식의 정수해", "다항식의 나눗셈(조립제법)",
    "지수곡선과 직선의 교점", "로그곡선과 직선의 교점",
    "정적분과 미정 상한", "비복원 추출의 조건부확률", "인수정리와 나누어떨어짐",
    "등비수열을 이루는 세 수",
    # 구체적 상황 확통 킬러(수능 30번 스타일)
    "같은 것을 나누어 주는 경우의 수", "단조함수의 개수",
    "조건을 만족시키는 카드 나열", "두 주머니 베이즈 조건부확률",
}


def verify(rec) -> bool:
    """구조 검증.
    객관식: 정답 라벨 'N번 ($v$)' 이 보기 안에 정확히 위치.
    주관식: 답이 0~999 정수(수능 단답형 규격)이고 해설에 등장."""
    c = rec["completion"]
    ans = c["answer"]
    if not c.get("choices"):                       # 주관식(단답형)
        if rec.get("spec", {}).get("problem_type") != "주관식":
            return False
        if not _re.fullmatch(r"\d{1,3}", str(ans)):
            return False
        return str(ans) in (c.get("explanation") or "")
    if "번 (" not in ans:
        return False
    # 'N번 ($v$)' 에서 N과 값 추출
    nlab = int(ans.split("번")[0])
    if not (1 <= nlab <= len(c["choices"])):
        return False
    val = ans.split("(", 1)[1].rsplit(")", 1)[0]
    if c["choices"][nlab-1] != val:
        return False
    return True


_MC_ANS_RE = _re.compile(r"^(\d+)번 \(\$(-?\d+)\$\)$")


def make_subjective(rec):
    """객관식 → 주관식(단답형) 변환. 수능 단답형 규격(답 0~999 정수)일 때만.
    변환 불가(분수·음수·π 등)면 None."""
    m = _MC_ANS_RE.match(rec["completion"].get("answer", ""))
    if not m:
        return None
    val = int(m.group(2))
    if not (0 <= val <= 999):
        return None
    rec = copy.deepcopy(rec)
    c = rec["completion"]
    c["choices"] = None
    c["answer"] = str(val)
    c["explanation"] = _re.sub(r"정답: \d+번 \(\$-?\d+\$\)\s*$",
                               f"정답: {val}", c["explanation"])
    rec["spec"]["problem_type"] = "주관식"
    return rec


def generate(name=None, seed=None, subjective=False):
    if seed is not None:
        random.seed(seed)
    if name is None:
        name = random.choice(list(GENERATORS))
    rec = GENERATORS[name]()
    rec["generator"] = name        # 진단·재출제(diagnosis)용 출처
    if subjective:
        sub = make_subjective(rec)
        if sub is not None:
            sub["generator"] = name
            return sub
    return rec


def generate_batch(per_topic=1, seed=0):
    random.seed(seed)
    out = []
    for name in GENERATORS:
        for _ in range(per_topic):
            out.append(GENERATORS[name]())
    return out
