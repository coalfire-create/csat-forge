"""고정밀 수치 검증 레이어.

검증 엔진은 정답을 sympy 정확해(유리수·라디칼)로 보장하는 게 1차 원칙이지만,
초월함수(지수·로그·삼각)가 섞인 문제는 닫힌형 해가 없을 수 있다. 이때 정답을
'고정밀 수치(기본 40자리)'로 독립 재계산해 대조하고, 두 값이 허용오차 안에서
일치할 때만 채택한다. (모호/불일치 → reject)

원칙: 추정·근사로 '정답을 정하지' 않는다. 정답은 구성(secret-first) 또는 정확해로
정하고, 이 모듈은 그 값이 조건과 실제로 일치하는지 '검산'만 한다.
"""
from __future__ import annotations
import sympy as sp

DIGITS = 40
TOL = sp.Float(10) ** (-25)


def close(a, b, digits: int = DIGITS, tol=TOL) -> bool:
    """|a-b| < tol 를 고정밀로 판정."""
    return abs(sp.N(sp.sympify(a) - sp.sympify(b), digits)) < tol


def verify_value(actual, expected, digits: int = DIGITS, tol=TOL) -> bool:
    """actual(정확해/구성값) 과 expected(독립 재계산) 가 고정밀로 일치하는지."""
    return close(actual, expected, digits, tol)


def solve_real(expr, var, guess, digits: int = DIGITS):
    """expr(var)=0 의 실근을 guess 근방에서 고정밀 수치해로. 실패 시 None."""
    try:
        root = sp.nsolve(expr, var, guess, prec=digits)
        if abs(sp.N(expr.subs(var, root), digits)) < TOL:
            return root
    except Exception:
        return None
    return None


def is_integer(value, digits: int = DIGITS, tol=TOL) -> bool:
    """수치값이 정수에 충분히 가까운지(정답을 정수 라벨로 낼 수 있는지)."""
    v = sp.N(value, digits)
    return abs(v - sp.nsimplify(round(float(v)))) < tol
