"""평문(라벨) 출력 포맷 — JSON 직접 출력의 대안.

작은 모델(3B/7B)은 LaTeX 가 섞인 엄격한 JSON 출력에 약하다
(01c 실험: 7B JSON 직접 출력 실패율 ~50%, 평문 라벨 방식 ~0%).
그래서 모델은 사람이 읽는 라벨 형식으로 출력하고, JSON 은 코드가 조립한다.

형식:
    문제: <문제 내용>
    보기1: <첫 번째 보기>     ← 객관식만, 보기1~보기5
    ...
    정답: <정답>
    해설: <단계별 해설, 여러 줄 가능>

직렬화(to_plain)는 학습 데이터 생성에, 파싱(parse_plain)은 추론 결과
해석에 쓰여 학습-추론 포맷이 항상 일치한다.
"""
from __future__ import annotations
import re
from typing import Optional

PLAIN_SYSTEM_PROMPT = """당신은 전문 수능 수학 문제 출제자입니다.
주어진 조건에 맞는 고품질 문제를 생성합니다.

규칙:
1. 문제는 명확하고 모호하지 않아야 합니다
2. 정답은 반드시 정확해야 합니다
3. 해설은 1단계/2단계/3단계로 나눠 상세하게 작성합니다
4. 수식은 LaTeX($...$)로 표기합니다
5. 객관식은 보기 5개를 오름차순으로 만듭니다

출력은 JSON 이 아니라, 반드시 아래 라벨 형식으로만 응답합니다:

문제: <문제 내용>
보기1: <첫 번째 보기>
보기2: <두 번째 보기>
보기3: <세 번째 보기>
보기4: <네 번째 보기>
보기5: <다섯 번째 보기>
정답: <정답>
해설: <단계별 해설>

주관식이면 보기 줄을 생략합니다. 라벨 외의 다른 말은 쓰지 않습니다."""

# 해설 앞에 오는 라벨들 (해설은 마지막 필드로, 끝까지 전부 해설로 취급)
_HEAD_LABEL_RE = re.compile(r"^(문제|보기[1-9]|정답|힌트)\s*[:：]\s*", re.M)
_EXPL_LABEL_RE = re.compile(r"^해설\s*[:：]\s*", re.M)


def to_plain(completion: dict) -> str:
    """completion dict → 평문 라벨 텍스트 (학습 타겟용)."""
    lines = [f"문제: {completion.get('question', '')}"]
    for i, c in enumerate(completion.get("choices") or [], 1):
        lines.append(f"보기{i}: {c}")
    lines.append(f"정답: {completion.get('answer', '')}")
    if completion.get("hint"):
        lines.append(f"힌트: {completion['hint']}")
    lines.append(f"해설: {completion.get('explanation', '')}")
    return "\n".join(lines)


def parse_plain(text: str) -> Optional[dict]:
    """평문 라벨 텍스트 → completion dict. 실패 시 None.

    해설은 마지막 필드로 보고 '해설:' 이후 전체를 해설로 취급한다
    (해설 내부의 '정답: ...' 마무리 줄이 필드로 오인되지 않도록).
    """
    if not text or "문제" not in text:
        return None
    m_expl = _EXPL_LABEL_RE.search(text)
    head = text[:m_expl.start()] if m_expl else text
    explanation = text[m_expl.end():].strip() if m_expl else ""

    matches = list(_HEAD_LABEL_RE.finditer(head))
    if not matches:
        return None
    fields: dict[str, str] = {}
    choices: dict[int, str] = {}
    for m, nxt in zip(matches, list(matches[1:]) + [None]):
        label = m.group(1)
        end = nxt.start() if nxt else len(head)
        val = head[m.end():end].strip()
        if label.startswith("보기"):
            choices.setdefault(int(label[2:]), val)
        else:
            fields.setdefault(label, val)  # 같은 라벨 중복 시 첫 값
    if not fields.get("문제") or not fields.get("정답"):
        return None
    ch = [choices[k] for k in sorted(choices)] if choices else None
    return {
        "question": fields["문제"],
        "choices": ch,
        "answer": fields["정답"],
        "explanation": explanation,
        "hint": fields.get("힌트"),
        "tags": [],
        "trap_element": None,
    }
