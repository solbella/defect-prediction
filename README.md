# 공정·설비 데이터로 불량/이상 조기 판별

반도체 공정(SECOM) 센서 데이터를 중심으로, **표 형태의 공정·설비 데이터에서 불량/고장/이상을 미리 걸러내는 방법**을 하루하루 쌓아가며 익히는 자율 학습 프로젝트입니다. 외부(Notion) 강의 노트를 따라가며, 매일 배운 개념을 실제 데이터에 코드로 적용하고 실행 결과를 노트북에 그대로 남겨두는 방식으로 진행했습니다.

## 무엇을 다루는 프로젝트인가

- **주 데이터셋**: SECOM 반도체 공정 센서 기록 — 공정에서 측정한 센서값으로, 검사 전에 불량 가능성이 높은 제품을 미리 골라내는 문제
- **접근 방식**: 정답이 있는 지도학습(로지스틱 회귀·의사결정나무)부터, 정답 없이 이상만 잡아내는 비지도학습(IsolationForest·LocalOutlierFactor), 시간 흐름을 보는 관리도(SPC)까지 순서대로 확장
- **부가 실습("미션")**: 같은 순서를 온실 작물, 배터리 셀, 냉동 물류창고 등 다른 산업 데이터에도 그대로 적용해보며 일반화 연습
- **개인 프로젝트**: 설비 고장 예측(`day06/project_v1`)을 처음부터 끝까지 혼자 기획·실행한 결과물

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `data/` | 실습에 쓰는 데이터셋 (아래 표 참고) |
| `day01/` | pandas 기초 — 표 다루기, 관리도(SPC) 기초, 센서 진단 |
| `day02/` | 데이터 정제 — 결측/상수/중복 센서 제거, 상관관계로 피처 추리기 (`secom_clean.csv` 생성) |
| `day03/` | 첫 분류 모델 — 학습/시험 분리, 베이스라인, 혼동행렬·정밀도·재현율. `mission03_greenhouse/`: 같은 순서를 온실 데이터로 |
| `day04/` | 클래스 불균형 대응(`class_weight`), 하이퍼파라미터 자동 탐색, 교차검증. `mission04_battery/`: 배터리 셀 검사 데이터로 |
| `day05/` | 정답 없이 찾기 — IsolationForest·LocalOutlierFactor 이상탐지, 시계열/관리도로 흐름 보기. `mission05_coldchain/`: 냉동 물류창고 데이터로. `my_project/`: 신용카드 데이터로 자유 실습 |
| `day06/` | `project_v1/`: 설비 고장 예측을 처음부터 끝까지 혼자 진행한 개인 포트폴리오 프로젝트 |
| `CLAUDE.md` | 이 저장소에서 AI 코딩 도구(Claude Code)가 지켜야 할 규칙 — 데이터 흐름, 노트북 작성 규칙 정리 |

각 `labNN_*`/`missionNN_*`/`project_v1` 폴더 안의 `results/`에는 그 실습에서 실제로 생성한 그래프·표·리포트가 들어 있습니다.

## 데이터셋

| 파일 | 출처/주제 | 크기 | 결과(정답) 열 |
|---|---|---|---|
| `data/04_secom.csv` | SECOM 반도체 공정 센서 (원본) | 1567행 × 592열 | `result` (양품/불량, 불량 6.64%) |
| `data/day01_bottling.csv` | 음료 병입 공정 (day01 실습용) | - | - |
| `data/day03_greenhouse.csv` | 스마트팜 온실 출하 기록 | 2000행 × 13열 | `result` (상품/등외, 등외 6.95%) |
| `data/day04_battery.csv` | 배터리 셀 최종검사 기록 | 2847행 × 14열 | `result` (합격/불합격, 불합격 4.67%) |
| `data/day05_coldchain.csv` | 냉동 물류창고 온습도·설비 로그 | 3412행 × 12열 | `disposal` (정상/폐기, 폐기 3.52%) |
| `data/16_machine-failure.csv` | 설비(기계) 운전 조건과 고장 기록 | 10000행 × 7열 | `Machine failure` (정상/고장, 고장 3.39%) |
| `data/default_of_credit_card_clients.csv` | 대만 신용카드 고객 채무불이행 (UCI) | 30000행 × 25열 (`header=1`로 읽기) | `default payment next month` (0/1, 불이행 22.12%) |

`04_secom.csv`를 제외한 나머지는 각 day의 미션/자유 실습에서 쓰인 보조 데이터셋입니다.

## 실습 목차

| 실습 | 배운 것 |
|---|---|
| day01 lab01~05 | pandas 기초, 공정 카드 정리, 관리도(UCL/LCL), 센서 진단 |
| day02 lab06 | 결측/상수/중복 센서 제거 → 상관관계 기준 상위 피처 추리기 (`secom_clean.csv` 생성) |
| day03 lab07~09 | 학습·시험 분리, 베이스라인 모델, 혼동행렬·정밀도·재현율·문턱 조정 |
| day04 lab10~12 | `class_weight="balanced"` 불균형 대응, `GridSearchCV` 하이퍼파라미터 탐색, `cross_validate` 교차검증 |
| day05 lab13~15 | IsolationForest·LocalOutlierFactor 이상탐지, 두 방법 겹침 비교, 이동평균·관리도로 시간 흐름 보기 |
| day06 project_v1 | 설비 고장 예측 — 문제 정의부터 베이스라인·개선 모델·결과 요약까지 개인 진행 |

## 환경

- Python 3.14.7 / pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter
- 별도 빌드·테스트 명령 없음 — 노트북 셀을 위에서 아래로 순서대로 실행
- Windows에서 한글이 포함된 그래프를 그릴 때는 `plt.rcParams["font.family"] = "Malgun Gothic"` 설정 필요

자세한 규칙은 [`CLAUDE.md`](./CLAUDE.md)에 정리되어 있습니다.
