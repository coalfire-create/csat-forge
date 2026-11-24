"""오답 진단 → 맞춤 재출제.

차별화 핵심: 엔진의 오답 보기는 '어떤 실수에서 나오는 값'인지 설계 시점에
태깅돼 있다(trap_map). 학생이 고른 보기 번호만으로
  1) 왜 틀렸는지(실수 유형) 진단하고
  2) 같은 실수를 찌르는 변형 문제를 즉시 재출제한다.
문제은행형(주제만 앎)·LLM형(오답에 의도 없음) 서비스가 못 하는 기능.

사용:
    from csat_forge.diagnosis import diagnose, followup
    d = diagnose(rec, chosen=2)      # {"correct": False, "mistake": "...", "advice": "..."}
    nxt = followup(rec)              # 같은 유형(같은 생성기, 다른 파라미터) 변형
"""
from __future__ import annotations
from typing import Optional

# 실수 유형 → 교정 조언 (라벨 부분 일치로 매칭)
ADVICE = {
    "거리·변위 혼동": "움직인 거리는 ∫|v|dt — 속도가 음수인 구간의 넓이도 '더해야' 합니다. 변위(∫v dt)와 구분하세요.",
    "조건부확률 대상": "P(A|B)에서 무엇이 조건(B)이고 무엇을 구하는지(A)부터 표시하고 시작하세요.",
    "가중 무시": "전체에서 뽑을 때는 각 집단의 비율(가중치)을 곱해야 합니다. 표를 그려 분모를 먼저 구하세요.",
    "여사건": "여사건으로 구한 확률은 마지막에 반드시 1에서 빼야 합니다.",
    "진수조건": "로그가 정의되려면 진수>0. 부등식을 풀기 전에 정의역부터 적어두는 습관을 들이세요.",
    "표값": "표준정규분포표는 P(0≤Z≤z)를 줍니다. 꼬리 확률은 0.5에서 빼야 합니다.",
    "P(X≥)와 P(X≤) 혼동": "부등호 방향을 그림으로 그려 어느 쪽 꼬리인지 확인하세요.",
}


def _correct_index(completion: dict) -> Optional[int]:
    ans = str(completion.get("answer", ""))
    if "번" in ans:
        try:
            return int(ans.split("번")[0])
        except ValueError:
            return None
    return None


def _advice_for(label: str) -> Optional[str]:
    for key, adv in ADVICE.items():
        if key in label:
            return adv
    return None


def diagnose(rec: dict, chosen: int) -> dict:
    """rec: 엔진 레코드(또는 completion 포함 dict), chosen: 학생이 고른 보기 번호(1~5).

    반환: {correct, mistake(실수 라벨 또는 None), advice, correct_answer}
    """
    c = rec.get("completion", rec)
    ci = _correct_index(c)
    if ci is not None and chosen == ci:
        return {"correct": True, "mistake": None, "advice": None,
                "correct_answer": c.get("answer")}
    label = (c.get("trap_map") or {}).get(str(chosen))
    return {
        "correct": False,
        "mistake": label,                       # None 이면 '설계된 함정 외 오답'
        "advice": _advice_for(label) if label else None,
        "correct_answer": c.get("answer"),
    }


def followup(rec: dict):
    """같은 유형(같은 생성기, 새 파라미터) 변형 재출제. 생성기 미상이면 None."""
    from csat_forge import problem_engine as pe
    name = rec.get("generator")
    if not name or name not in pe.GENERATORS:
        return None
    orig_q = rec.get("completion", rec).get("question")
    for _ in range(6):                          # 같은 문제 재출제 방지
        nxt = pe.generate(name)
        if nxt["completion"]["question"] != orig_q:
            return nxt
    return None
