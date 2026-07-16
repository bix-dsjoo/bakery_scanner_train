# Project Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `README.md` for project onboarding and an `AGENTS.md` that preserves the approved data, training, evaluation, and CPU benchmark rules during future implementation.

**Architecture:** `README.md` explains the project to humans from first principles. `AGENTS.md` converts the same design into mandatory repository-working rules for coding agents. Both documents use `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md` as the source of truth and must agree on every data split and runtime constraint.

**Tech Stack:** Markdown, PowerShell verification commands, existing COCO JSON dataset metadata

## Global Constraints

- `datasets/base/test` and `datasets/incremental/test` are evaluation-only and must never influence training, tuning, thresholds, early stopping, augmentation, checkpoint selection, or model selection.
- `datasets/base/val` is treated semantically as scene training data even though its physical folder name remains unchanged.
- Base data may be reused when training the 20-class Incremental classifier.
- The default Incremental experiment freezes the class-agnostic detector and updates the classifier.
- Training may use the NVIDIA GPU; inference benchmarks must use CPU only.
- Current CPU benchmarks are relative comparisons on Intel Core Ultra 9 285K and are not POS deployment claims.
- Existing source images, annotations, and `datasets/class_registry.json` remain immutable unless the user explicitly requests a dataset correction.
- Do not initialize Git automatically. The current directory is not a Git repository.

---

### Task 1: Create the project README

**Files:**
- Create: `README.md`
- Reference: `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`

**Interfaces:**
- Consumes: approved project design and current dataset inventory
- Produces: the human-facing project contract used for onboarding, experiment interpretation, and future command documentation

- [ ] **Step 1: Verify that no README exists and that referenced dataset paths are present**

Run:

```powershell
Test-Path README.md
Test-Path datasets\class_registry.json
Test-Path datasets\base\val\instances_val.json
Test-Path datasets\base\test\instances_test.json
Test-Path datasets\incremental\test\instances_test.json
```

Expected:

```text
False
True
True
True
True
```

- [ ] **Step 2: Create `README.md` with the complete approved project description**

Write exactly this content:

````markdown
# Bakery Scanner Train

고정 카메라로 촬영한 트레이 이미지에서 여러 빵의 위치와 종류를 식별하고, 신규 빵 클래스를 추가했을 때 기존 성능이 얼마나 유지되는지 검증하는 프로젝트입니다.

현재 저장소에는 데이터셋과 설계 문서가 있으며 학습·추론 코드는 아직 구현되지 않았습니다.

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
docs/superpowers/specs/           승인된 설계 문서
docs/superpowers/plans/           구현 계획
```

`derived/`와 `collected/` 경로는 해당 데이터가 생성되거나 수집될 때 추가합니다.

## 설계 문서

상세한 데이터 경계, 오류 처리와 검증 요구사항은 [`docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`](docs/superpowers/specs/2026-07-16-bakery-scanner-design.md)를 따릅니다.
````

- [ ] **Step 3: Verify README structure and critical policy text**

Run:

```powershell
rg -n '^# Bakery Scanner Train$|^## 목표$|^## 모델 구조$|^## 데이터셋$|^## Detector 학습 데이터$|^## 학습 절차$|^## 평가$|^## 실행 정책$' README.md
rg -n 'datasets/base/test.*금지|datasets/incremental/test.*금지|CPU만 사용|detector를 고정|class_registry.json' README.md
```

Expected: every heading is printed once and every critical policy pattern has at least one match.

### Task 2: Create repository instructions for coding agents

**Files:**
- Create: `AGENTS.md`
- Reference: `README.md`
- Reference: `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`

**Interfaces:**
- Consumes: project contract from Task 1 and approved design
- Produces: mandatory instructions governing all future code, data generation, training, evaluation, and documentation changes in this repository

- [ ] **Step 1: Create `AGENTS.md` with complete repository rules**

Write exactly this content:

````markdown
# AGENTS.md

## 적용 범위

이 파일의 규칙은 저장소 전체에 적용한다. 더 하위 경로에 별도의 `AGENTS.md`가 생기면 해당 파일은 이 규칙을 유지하면서 그 경로에 필요한 세부 규칙만 추가한다.

## 프로젝트 목적

고정 카메라 트레이 이미지에서 모든 빵의 bbox와 종류를 반환하는 모델을 개발한다. 기본 구조는 class-agnostic bread detector와 별도 bread classifier이다. Base 15개 클래스 학습 후 Incremental 5개 클래스를 추가하며 기존 및 신규 클래스 성능을 함께 평가한다.

## 변경 불가 원칙

1. `datasets/base/test`와 `datasets/incremental/test`는 평가 전용이다.
2. Test 데이터와 결과를 학습, 미세조정, early stopping, threshold, augmentation, checkpoint, 하이퍼파라미터 또는 모델 선택에 사용하지 않는다.
3. `datasets/base/val`은 물리적 이름과 무관하게 장면형 학습 데이터로 취급한다.
4. 기존 이미지, COCO JSON과 `datasets/class_registry.json`은 사용자가 데이터 수정을 명시적으로 요청하지 않는 한 변경하지 않는다.
5. COCO `category_id`를 모델 출력 인덱스로 사용하지 않는다. 항상 레지스트리의 `model_index`를 사용한다.
6. Incremental 단계에서는 Base 데이터를 재사용할 수 있다.
7. 기본 Incremental 실험에서는 detector를 고정하고 classifier를 갱신한다.
8. 학습에는 GPU를 사용할 수 있지만 추론 benchmark는 CPU만 사용한다.
9. 현재 CPU benchmark 결과를 특정 POS 장치의 성능으로 표현하지 않는다.

## 데이터 및 split 규칙

- Train-side validation은 학습 가능한 데이터에서 만든다.
- 같은 scene ID의 `scene_e_*`, `scene_m_*`, `scene_h_*`는 하나의 그룹으로 분할한다.
- 학습 설정이 `datasets/base/test` 또는 `datasets/incremental/test`를 참조하면 즉시 실패시킨다.
- 합성 데이터는 `datasets/derived/synthetic/`에 저장하고 생성 seed, 원본, 배경, 변환, bbox와 생성기 버전을 manifest에 기록한다.
- 추가 촬영한 실제 장면은 `datasets/collected/scene_train/`에 저장하고 COCO bbox를 제공한다.
- 파생 데이터 생성 과정에서 원본 데이터 파일을 덮어쓰지 않는다.

## 모델 경계

### Detector

- 모든 빵 category를 단일 `bread` label로 통합한다.
- bbox와 objectness를 출력한다.
- Base 및 Incremental class group별 class-agnostic Recall을 측정한다.
- Incremental 단계에서의 detector 재학습은 별도 ablation으로 취급한다.

### Classifier

- Detector bbox crop을 입력받는다.
- Base 단계는 15개, Incremental 단계는 20개 출력을 사용한다.
- 출력 순서는 `class_registry.json`의 `model_index`와 정확히 일치해야 한다.
- Base 84장/클래스와 Incremental 7장/클래스의 불균형을 보정한다.
- 장면의 crop은 가능한 경우 하나의 batch로 추론한다.

## 구현 규칙

- 데이터셋 원본, 파생 데이터, 모델 코드, 설정, 결과물을 명확히 분리한다.
- 재현에 영향을 주는 seed, dependency, hardware, 입력 크기, batch와 thread 수를 기록한다.
- 경로나 클래스 매핑을 코드에 중복 하드코딩하지 않는다.
- 학습 및 평가 entry point는 사용하는 split을 시작 시점에 출력하고 검증해야 한다.
- 잘못된 COCO 참조, 범위를 벗어난 bbox, 중복 클래스 ID, 출력 차원 불일치와 누락 manifest는 경고가 아니라 오류로 처리한다.
- 검출 결과가 없는 정상 이미지는 빈 결과 목록으로 처리한다.
- 아직 존재하지 않는 학습·추론 명령을 문서에 만들어 내지 않는다. 구현된 명령만 `README.md`에 추가한다.

## 평가 규칙

- Detector: AP50, Recall@IoU 0.5, 미검출률과 난이도별 Recall을 기록한다.
- Classifier: 정답 bbox crop 기준 Top-1, Macro F1과 클래스별 Precision/Recall을 기록한다.
- End-to-end: mAP50, mAP50:95와 클래스별 수량 정확도를 기록한다.
- Base 모델과 Incremental 모델의 Base test 차이를 기록한다.
- Incremental test의 신규 5개 클래스 성능을 별도로 기록한다.
- 모든 비교 실험은 동일한 split을 사용한다.
- Test 결과를 확인한 뒤 설정을 다시 선택하지 않는다.

## CPU benchmark 규칙

- 모델과 tensor를 명시적으로 CPU에 배치한다.
- ONNX Runtime은 `CPUExecutionProvider`만 허용한다.
- GPU provider가 활성화되면 benchmark를 실패시킨다.
- warm-up은 통계에서 제외한다.
- 평균, P50, P95와 반복 횟수를 기록한다.
- detector, crop·전처리, classifier batch, 후처리와 end-to-end 시간을 분리해 기록한다.
- CPU 모델, thread 수, 입력 크기와 batch 크기를 결과에 포함한다.

## 필수 검증

변경 범위에 맞춰 다음을 검증한다.

- 클래스 레지스트리 유일성과 연속적인 `model_index`
- COCO 이미지 참조, category 참조와 bbox 경계
- train/validation/test 경로 누수
- scene ID 그룹 split
- 합성 데이터 재현성과 bbox 일치
- 15-class에서 20-class 확장
- detector 고정 실험에서 detector 가중치 불변
- CPU-only inference provider
- 평가 결과의 환경 및 metric metadata

## 문서 일관성

프로젝트 목적이나 데이터 정책을 바꾸는 작업은 다음 파일을 함께 검토한다.

- `README.md`
- `AGENTS.md`
- `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`

문서와 코드가 다르면 구현을 사실로 간주하지 말고, 승인된 설계와 사용자 의도를 확인해 둘을 함께 수정한다.
````

- [ ] **Step 2: Verify AGENTS scope and immutable constraints**

Run:

```powershell
rg -n '^# AGENTS.md$|^## 적용 범위$|^## 변경 불가 원칙$|^## 데이터 및 split 규칙$|^## 모델 경계$|^## 구현 규칙$|^## 평가 규칙$|^## CPU benchmark 규칙$|^## 필수 검증$|^## 문서 일관성$' AGENTS.md
rg -n 'datasets/base/test|datasets/incremental/test|class_registry.json|model_index|CPUExecutionProvider|detector를 고정' AGENTS.md
```

Expected: every required heading and every protected constraint is present.

### Task 3: Cross-document consistency verification

**Files:**
- Verify: `README.md`
- Verify: `AGENTS.md`
- Verify: `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`

**Interfaces:**
- Consumes: completed human and agent documentation
- Produces: evidence that paths, split roles, model boundary, and compute policy are consistent across all three documents

- [ ] **Step 1: Verify that all local Markdown links resolve**

Run:

```powershell
$required = @(
  'datasets\class_registry.json',
  'docs\superpowers\specs\2026-07-16-bakery-scanner-design.md'
)
$required | ForEach-Object {
  if (-not (Test-Path $_)) { throw "Missing linked path: $_" }
}
Write-Output 'All required local links resolve.'
```

Expected:

```text
All required local links resolve.
```

- [ ] **Step 2: Verify split roles across every policy document**

Run:

```powershell
$files = @(
  'README.md',
  'AGENTS.md',
  'docs\superpowers\specs\2026-07-16-bakery-scanner-design.md'
)
foreach ($file in $files) {
  $text = Get-Content -Raw -Encoding UTF8 $file
  foreach ($required in @('datasets/base/val', 'datasets/base/test', 'datasets/incremental/test')) {
    if (-not $text.Contains($required)) { throw "$file missing $required" }
  }
}
Write-Output 'Split roles are documented in all policy files.'
```

Expected:

```text
Split roles are documented in all policy files.
```

- [ ] **Step 3: Scan for incomplete markers and contradictory compute claims**

Run:

```powershell
$docs = @(
  'README.md',
  'AGENTS.md',
  'docs\superpowers\specs\2026-07-16-bakery-scanner-design.md'
)
$incompletePattern = @(('T' + 'BD'), ('TO' + 'DO'), ('FIX' + 'ME'), ('fill' + ' in')) -join '|'
$incompleteHits = Select-String -Path $docs -Pattern $incompletePattern -Encoding UTF8
if ($incompleteHits) { $incompleteHits; throw 'Documentation contains incomplete markers.' }

$readme = Get-Content -Raw -Encoding UTF8 README.md
$agents = Get-Content -Raw -Encoding UTF8 AGENTS.md
if (-not ($readme.Contains('학습: GPU 사용 허용') -and $readme.Contains('CPU만 사용'))) {
  throw 'README compute policy is incomplete.'
}
if (-not ($agents.Contains('학습에는 GPU') -and $agents.Contains('추론 benchmark는 CPU'))) {
  throw 'AGENTS compute policy is incomplete.'
}
Write-Output 'No incomplete markers or compute-policy contradictions found.'
```

Expected:

```text
No incomplete markers or compute-policy contradictions found.
```

- [ ] **Step 4: Report completion without committing**

Run:

```powershell
if (Test-Path .git) {
  throw 'Repository state changed unexpectedly; review Git status before committing.'
}
Get-Item README.md,AGENTS.md,docs\superpowers\specs\2026-07-16-bakery-scanner-design.md |
  Select-Object FullName,Length
```

Expected: all three files exist and have non-zero length. Do not initialize Git or create a commit without explicit user authorization.
