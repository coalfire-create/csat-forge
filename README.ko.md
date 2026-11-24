# csat-forge

**정답을 "맞히는" 게 아니라 "계산하는" 수능 수학 문제 생성기.**
SymPy로 정답을 구성·계산하고, 오답 보기마다 '어떤 실수에서 나온 값인지' 태그를 달고, 단계별 풀이까지 자동으로 만들어요. (가)(나)(다) 조건 합성형 킬러 문제도 무한 생성돼요.

**[▶ 라이브 데모 (브라우저에서 바로 실행)](https://coalfire-create.github.io/csat-forge/)**

```bash
pip install git+https://github.com/coalfire-create/csat-forge
csat-forge killer --domain cubic --seed 11
```

## 왜 만들었나
LLM이 낸 문제는 그럴듯한데 틀린 경우가 많아요. csat-forge는 순서를 뒤집어요. **정답(비밀 함수)을 먼저 정하고**, 그에 대해 참인 조건을 만든 다음 지문을 렌더링해요. 그래서 정답이 구성상 맞아요.

- 생성기 121종 (2028 수능 범위: 대수 + 미적분Ⅰ + 확률과 통계)
- 킬러 엔진: 조건 블록 조합 + 개념 융합·연쇄 → 사실상 무한한 신유형
- 오답 의도 태깅 + **오답 진단**: 학생이 고른 보기 → 실수 유형·조언·변형 재출제
- 단답형(0~999) 변환, SVG 그림 생성기

## 사용법
```bash
csat-forge list
csat-forge gen --topic conditional_prob --seed 0
csat-forge gen -n 20 --subjective --json > set.jsonl
csat-forge killer -n 5
csat-forge diagnose --topic conditional_prob --seed 0 --chosen 4
```

## 정확도에 대해 솔직하게
- 생성기당 60문제, 총 7,260문제 실행에서 `verify()` 전부 통과, 생성 오류 0건.
- `verify()`는 **구조 검증**이에요(정답이 보기에 있는지, 단답형 규격 등). 수학적 정확성은 정답 우선 구성 + 정확해/고정밀 교차검증 + 렌더된 지문을 다시 풀어보는 회귀 테스트로 확보해요. 형식적 증명은 아니에요.
- 틀린 정답이나 모호한 지문을 발견하면 주제와 seed를 적어서 이슈로 남겨 주세요. 가장 큰 도움이 돼요.

기출 문제 원문은 포함하지 않아요. MIT 라이선스.
