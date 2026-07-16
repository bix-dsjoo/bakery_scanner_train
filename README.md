# Bakery Scanner Train

고정 카메라로 촬영한 트레이 이미지에서 여러 빵의 위치와 종류를 식별하고, 신규 빵 클래스를 추가했을 때 기존 성능이 얼마나 유지되는지 검증하는 프로젝트입니다.

현재 저장소에는 데이터셋 무결성 검사, COCO 검증, 학습 경로 안전장치와 scene 단위 split을 제공하는 Python 기반이 구현되어 있습니다. 모델 학습·추론 코드는 아직 구현되지 않았습니다.

## 목표

입력은 여러 종류의 빵이 놓인 트레이 RGB 이미지 한 장입니다. 시스템은 검출된 빵마다 다음 값을 반환합니다.

- `bbox`: `[x, y, width, height]`
- `class`: 빵 클래스
- `score`: detector와 classifier의 신뢰도를 결합한 값

Base 15개 클래스로 초기 모델을 학습한 뒤 Incremental 5개 클래스를 추가해 총 20개 클래스로 확장합니다. Incremental 학습에서도 Base 데이터를 재사용할 수 있습니다.

## 모델 구조

```text
트레이 이미지
    -> class-agnostic bread detector
    -> 빵 bbox와 objectness
    -> 검출 영역 일괄 crop
    -> bread classifier
    -> bbox, class, score
```

Detector는 빵 종류를 구분하지 않고 모든 빵을 하나의 `bread` 클래스로 찾습니다. Classifier는 detector가 찾은 crop을 Base 단계에서 15개, Incremental 단계에서 20개 클래스로 분류합니다.

Incremental 단계의 기본 전략은 detector를 고정하고 classifier만 갱신하는 것입니다. Detector 재학습은 신규 빵에 대한 class-agnostic Recall이 부족할 때 비교 실험으로 수행합니다.

## 데이터셋

| 구분 | 클래스 | 단일 객체 학습 이미지 | 장면 이미지 |
|---|---:|---:|---:|
| Base | 15 | 1,260장 | scene train 9장, test 9장 |
| Incremental | 5 | 35장 | test 12장 |
| 전체 | 20 | 1,295장 | 30장 |

클래스 정의와 모델 출력 순서는 [`datasets/class_registry.json`](datasets/class_registry.json)에서 관리합니다. COCO `category_id`와 모델의 `model_index`는 같은 값이 아니므로 반드시 레지스트리 매핑을 사용해야 합니다.

### 데이터 경로와 용도

| 경로 | 실제 용도 | 학습 사용 |
|---|---|---|
| `datasets/base/bread_*` | Base classifier 학습 | 허용 |
| `datasets/base/val` | 실제 장면형 학습 데이터 | 허용 |
| `datasets/base/test` | Base 평가 | 금지 |
| `datasets/incremental/bread_*` | Incremental classifier 학습 | 허용 |
| `datasets/incremental/test` | Incremental 평가 | 금지 |

`datasets/base/val`은 폴더명을 변경하지 않지만 프로젝트에서는 `scene_train`으로 취급합니다.

### Test 격리 원칙

`datasets/base/test`와 `datasets/incremental/test`는 최종 평가 전용입니다. 다음 작업에 사용할 수 없습니다.

- 모델 학습과 미세조정
- early stopping
- 하이퍼파라미터, threshold 및 augmentation 선택
- checkpoint 또는 모델 선택

Validation은 학습 가능한 데이터 내부에서 별도로 구성합니다. 같은 scene ID의 `e/m/h` 이미지는 하나의 그룹으로 묶어 같은 split에 배치합니다.

## Detector 학습 데이터

Class-agnostic detector는 다음 데이터를 함께 사용합니다.

1. `datasets/base/val`의 실제 장면 bbox를 모두 `bread` 클래스로 변환한 데이터
2. 단일 객체를 트레이 배경에 배치한 합성 장면과 자동 생성 bbox
3. 고정 카메라 환경에서 추가 촬영하고 COCO bbox로 라벨링한 실제 트레이 장면

합성 장면은 객체 수, 위치, 크기, 회전, 밝기, 색온도, 겹침과 난이도를 변화시킵니다. 생성 seed, 원본 객체, 배경, 변환과 bbox를 manifest에 기록해 재현 가능하게 만듭니다.

기존 원본 데이터는 수정하지 않습니다. 합성 데이터는 `datasets/derived/synthetic/`, 추가 촬영 데이터는 `datasets/collected/scene_train/`에서 별도로 관리합니다.

## 학습 절차

### Base 단계

1. 실제 장면, 합성 장면, 추가 촬영 장면으로 class-agnostic detector를 학습합니다.
2. Base 단일 객체와 장면 bbox crop으로 15-class classifier를 학습합니다.
3. Train-side validation으로 설정과 checkpoint를 결정합니다.
4. 설정을 고정한 뒤 Base test를 평가합니다.

### Incremental 단계

1. 기본 실험에서는 Base detector를 고정합니다.
2. Base와 Incremental classifier 데이터를 함께 사용합니다.
3. 클래스별 데이터 수 차이를 보정합니다.
4. 20-class classifier를 학습합니다.
5. 설정을 고정한 뒤 Base test와 Incremental test를 평가합니다.

비교할 classifier 전략은 전체 재학습, head-only 재학습, backbone 일부 미세조정, cosine/prototype classifier와 지식 증류입니다.

## 평가

성능 목표값은 첫 실험 전에 임의로 정하지 않습니다. 재현 가능한 기준선을 먼저 만들고 이후 성공 기준을 결정합니다.

- Detector: class-agnostic AP50, Recall@IoU 0.5, 미검출률, 난이도별 Recall
- Classifier: 정답 bbox crop 기준 Top-1, Macro F1, 클래스별 Precision/Recall
- End-to-end: mAP50, mAP50:95, 클래스별 수량 정확도
- Incremental: Base 성능 변화와 신규 5개 클래스 성능

정답 bbox crop과 detector 예측 bbox crop을 모두 평가해 classifier 오류와 detector 오류를 분리합니다.

## 실행 정책

- 학습: GPU 사용 허용
- 추론 성능 측정: GPU를 비활성화하고 CPU만 사용
- CPU 기준 환경: Intel Core Ultra 9 285K, 24 cores / 24 logical processors
- 학습 GPU: NVIDIA GeForce RTX 5080 16 GB
- Python: 3.11.9

CPU benchmark는 warm-up 이후 반복 실행하며 평균, P50과 P95를 기록합니다. Detector, crop·전처리, classifier batch, 후처리 및 전체 시간을 각각 측정합니다. 이 결과는 현재 PC에서 알고리즘을 상대 비교하기 위한 것이며 특정 POS 성능을 의미하지 않습니다.

## 프로젝트 구조

```text
datasets/                         원본 및 프로젝트 데이터
  base/                           Base 단일 객체, scene train, test
  incremental/                    Incremental 단일 객체와 test
  class_registry.json             클래스 ID와 모델 인덱스 매핑
  derived/synthetic/              재생성 가능한 합성 장면
  collected/scene_train/          추가 촬영한 실제 학습 장면
src/bakery_scanner/               데이터 검증, 안전장치, split과 audit CLI
tests/                            pytest 자동 테스트
docs/superpowers/specs/           승인된 설계 문서
docs/superpowers/plans/           구현 계획
pyproject.toml                    Python 패키지와 실행 명령 정의
```

`derived/`와 `collected/` 경로는 해당 데이터가 생성되거나 수집될 때 추가합니다.

## Python 환경 준비

Python 3.11 이상에서 프로젝트 루트 기준으로 editable 설치합니다.

```powershell
python -m pip install -e ".[test]"
```

Pillow는 COCO에 기록된 이미지 크기와 실제 디코딩 크기가 같은지 확인하는 데 사용합니다. 모델 학습 라이브러리는 아직 포함하지 않습니다.

## 데이터 audit

전체 레지스트리, 클래스 폴더, 단일 객체 이미지 수와 세 COCO split을 읽기 전용으로 검사합니다.

```powershell
bakery-audit --dataset-root datasets
```

설치된 console script 대신 Python module entry point를 사용할 수도 있습니다.

```powershell
python -m bakery_scanner --dataset-root datasets
```

자동 처리용 JSON 출력과 제안 train-side split의 seed 및 validation 비율을 지정할 수 있습니다.

```powershell
bakery-audit --dataset-root datasets --json --seed 42 --validation-fraction 0.2
```

Audit는 다음 조건을 하나라도 발견하면 exit code `1`로 실패합니다.

- 중복 ID, 0부터 19까지 연속적이지 않은 `model_index`, Base 15개/Incremental 5개 구성 불일치
- 레지스트리 phase와 `bread_*` 클래스 폴더 불일치
- COCO image/category 참조 오류, 디렉터리의 누락·미등록 이미지, 실제 이미지 크기 불일치
- 0 이하 면적, 유한하지 않은 값 또는 이미지 경계를 벗어난 bbox
- 잘못되었거나 불완전한 `scene_e_*`, `scene_m_*`, `scene_h_*` 그룹

Audit는 무결성 확인을 위해 평가 전용 split도 읽지만, 이 동작은 학습 사용 권한을 부여하지 않습니다. 향후 학습 entry point는 `bakery_scanner.safety.assert_training_paths_safe`를 호출해 `datasets/base/test`와 `datasets/incremental/test` 및 그 하위 경로를 즉시 거부해야 합니다.

기본 설정(`seed=42`, `validation_fraction=0.2`)으로 현재 `datasets/base/val`에 제안되는 scene split은 다음과 같습니다. 이 명령은 split 파일을 쓰거나 원본 데이터를 변경하지 않습니다.

- train scene ID: `0503`, `0510`
- validation scene ID: `0509`

## 자동 테스트

전체 테스트를 실행합니다.

```powershell
python -m pytest -q
```

테스트는 임시 데이터만 생성하며 기존 `datasets` 원본을 변경하지 않습니다.

## 설계 문서

상세한 데이터 경계, 오류 처리와 검증 요구사항은 [`docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`](docs/superpowers/specs/2026-07-16-bakery-scanner-design.md)를 따릅니다.
