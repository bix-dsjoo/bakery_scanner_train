# Bakery Scanner Train

고정 카메라로 촬영한 트레이 이미지에서 여러 빵의 위치와 종류를 식별하고, 신규 빵 클래스를 추가했을 때 기존 성능이 얼마나 유지되는지 검증하는 프로젝트입니다.

현재 저장소에는 데이터셋 무결성 검사, COCO 검증, 학습 경로 안전장치, scene 단위 split, detector용 합성 장면 생성·replay 검증, YOLO 데이터 변환과 YOLO11n detector 학습·train-side 평가, Base 15-class와 Incremental 20-class classifier 학습 데이터 조립·검증, Base ResNet18 classifier 학습·train-side 평가와 Base end-to-end 추론·평가가 구현되어 있습니다.

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
| `datasets/collected/backgrounds` | 합성 장면용 빈 트레이 배경 | 허용 |

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

기존 원본 데이터는 수정하지 않습니다. 합성 데이터는 `datasets/derived/synthetic/`, 빈 트레이 배경은 `datasets/collected/backgrounds/`, 추가 촬영 데이터는 `datasets/collected/scene_train/`에서 별도로 관리합니다.

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
  derived/detector/               class-agnostic detector train/validation COCO
  derived/yolo/                   검증된 detector COCO의 YOLO 파생 데이터
  derived/classifier/             검증된 15/20-class classifier 파생 데이터
  collected/backgrounds/          합성 장면용 실제 빈 트레이 배경
  collected/scene_train/          추가 촬영한 실제 학습 장면
src/bakery_scanner/               데이터 검증, split, 합성·detector·classifier 데이터와 CLI
configs/detector/                 재현 가능한 detector 학습 설정
models/pretrained/                다운로드한 사전학습 모델 가중치
runs/detector/                    로컬 checkpoint, 예측, metric과 환경 metadata
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

Pillow는 이미지 검증과 합성 장면 생성에 사용합니다. PyYAML은 학습 설정을 읽고, Ultralytics 8.x와 PyTorch는 YOLO11n detector 학습·예측에 사용합니다.

## 합성 장면 생성과 검증

`bakery-synthetic`는 등록된 단일 객체 이미지를 명시적으로 지정한 빈 트레이 배경에 배치합니다. `datasets/collected/backgrounds/`에는 빵이 없는 실제 트레이 배경 3장이 포함되어 있습니다. `datasets/base/test`와 `datasets/incremental/test` 또는 그 하위 경로를 배경으로 지정하면 즉시 거부됩니다.

다음 명령은 Base 단일 객체로 장면 100장, 장면당 객체 5개를 생성합니다. 출력은 `datasets/derived/synthetic/base_seed42/`에만 기록됩니다.

```powershell
bakery-synthetic generate `
  --dataset-root datasets `
  --background-dir datasets/collected/backgrounds `
  --run-name base_seed42 `
  --phase base `
  --seed 42 `
  --scene-count 100 `
  --objects-per-scene 5
```

같은 CLI를 Python module로 실행할 수도 있습니다.

```powershell
python -m bakery_scanner.synthetic_cli generate --dataset-root datasets --background-dir datasets/collected/backgrounds --run-name base_seed42 --seed 42 --scene-count 100 --objects-per-scene 5
```

기존 run 디렉터리는 기본적으로 덮어쓰지 않습니다. 동일 입력과 seed로 명시적으로 재생성할 때만 `generate`에 `--overwrite`를 추가합니다. `--size-fraction-min/max`, `--rotation-min/max`, `--brightness-min/max`, `--contrast-min/max`와 `--foreground-threshold`로 기본 변환 범위를 조절할 수 있습니다.

생성된 run을 사용하기 전에 manifest, 원본 경로, 이미지, 해시와 bbox를 replay 검증합니다.

```powershell
bakery-synthetic validate --dataset-root datasets --run-name base_seed42
```

자동 처리에는 두 subcommand 모두 `--json`을 사용할 수 있습니다. 검증은 manifest에 기록된 변환을 다시 적용해 PNG bytes와 bbox를 재계산하므로 이미지나 annotation이 바뀌면 exit code `1`로 실패합니다.

각 run의 `manifest.json`에는 manifest/generator/Pillow version, master seed, 전체 생성 설정과 장면 목록이 기록됩니다. 장면에는 파생 seed, 출력 파일·크기·SHA-256, 배경 경로가, 객체에는 원본 경로, COCO `category_id`, 위치·크기·회전·밝기·대비 변환과 `[x, y, width, height]` bbox가 포함됩니다. 모델 출력 순서인 `model_index`는 합성 annotation으로 사용하지 않습니다.

현재 합성기 구현은 흰색 또는 near-white 배경의 단일 객체 이미지를 전제로 alpha mask를 계산하고, 완전히 보이는 bbox를 위해 객체 간 겹침을 허용하지 않습니다. 실제 tray 배경 분리, 가림·밀집도 난이도와 색온도 변환은 후속 범위입니다. 합성 run 자체는 COCO를 쓰지 않으며, 다음 detector 데이터 조립 단계가 실제 장면과 함께 단일 `bread` COCO로 변환합니다.

## Detector train/validation COCO 조립과 검증

`bakery-detector-data`는 실제 train-side 장면인 `datasets/base/val/instances_val.json`과 이미 생성·검증된 합성 run 하나를 조립합니다. detector 학습이나 추론은 실행하지 않습니다. 먼저 위의 `bakery-synthetic validate`가 통과하는 합성 run이 있어야 합니다.

다음 예시는 `base_seed42` 합성 run을 사용해 `datasets/derived/detector/base_seed42_detector_origin_aware/`를 생성합니다.

```powershell
bakery-detector-data generate `
  --dataset-root datasets `
  --synthetic-run base_seed42 `
  --run-name base_seed42_detector_origin_aware `
  --seed 42 `
  --validation-fraction 0.2
```

설치된 console script 대신 module로도 같은 작업을 실행할 수 있습니다.

```powershell
python -m bakery_scanner.detector_cli generate --dataset-root datasets --synthetic-run base_seed42 --run-name base_seed42_detector_origin_aware --seed 42 --validation-fraction 0.2
```

완성된 run의 manifest, 입력 provenance, 입력·출력 SHA-256, 이미지 inventory, 두 COCO, bbox와 split 누수를 독립적으로 다시 검사합니다.

```powershell
bakery-detector-data validate `
  --dataset-root datasets `
  --run-name base_seed42_detector_origin_aware
```

출력 구조는 다음과 같습니다.

```text
datasets/derived/detector/base_seed42_detector_origin_aware/
  manifest.json
  train/
    images/
    instances.json
  validation/
    images/
    instances.json
```

두 COCO의 category는 `{"id": 1, "name": "bread"}` 하나뿐입니다. 원본 COCO `category_id`는 sample별 `original_annotations`와 합성 객체 provenance에만 보존되며 `model_index`로 변환하거나 혼용하지 않습니다. annotation이 없는 정상 이미지도 빈 annotation 목록으로 유지됩니다.

분할은 이미지별 독립 난수가 아니라 누수 자원의 전역 연결 component 단위로 수행합니다. 실제 `scene_e/m/h`의 동일 scene ID, 합성 원본 객체의 정규화 경로와 SHA-256, 합성 배경의 정규화 경로와 SHA-256, 동일 이미지 SHA-256을 공유하는 장면은 origin과 무관하게 반드시 같은 split으로 이동합니다. 이 component 중 실제 장면을 포함한 그룹과 합성 전용 그룹을 seed 기반으로 각각 최적화합니다. 따라서 독립적인 실제 scene group이 둘 이상이면 실제 장면이 train과 validation에 모두 포함되며, 하나뿐인 합성 component는 train에 유지됩니다. 어느 origin group도 자체 validation을 만들 수 없을 때만 전역 component 선택으로 안전하게 fallback합니다. 모든 입력이 하나의 component로 연결되어 train/validation을 모두 만들 수 없으면 오류로 종료합니다.

생성은 `datasets/derived/detector/` 아래 임시 staging run에서 전체 검증을 통과한 뒤 원자적으로 publish합니다. 기존 run은 `--overwrite` 없이는 거부하며, 교체 실패 시 기존 run을 복구합니다. test 경로, 잘못된 bbox, 누락·추가·변조 파일, source hash 변경과 split 누수는 모두 exit code `1` 오류입니다. 두 subcommand 모두 자동 처리용 `--json`을 지원합니다.

## YOLO11n detector 기준선 학습과 평가

`bakery-detector train`은 설정에 지정된 detector COCO run을 먼저 독립 검증합니다. YOLO 파생 run이 없으면 `datasets/derived/yolo/`에 이미지·label·`data.yaml`·provenance manifest를 만들고, 이미 있으면 source hash, 전체 inventory, 이미지, label과 bbox를 다시 검증합니다. 원본 COCO와 기존 detector run은 수정하지 않습니다.

기본 기준선은 `configs/detector/yolo11n_base.yaml`에 고정되어 있습니다. `models/pretrained/yolo11n.pt`, 입력 크기 640, epoch 50, batch 16, seed 42, CUDA device 0과 train-side early stopping을 사용합니다. 다운로드한 사전학습 가중치는 `models/pretrained/`에 두며 Git에 포함하지 않습니다.

```powershell
bakery-detector train --config configs/detector/yolo11n_base.yaml
```

명령은 backend 작업 전에 dataset root, source detector run, train split, validation split, model과 출력 경로를 출력합니다. `datasets/base/test` 또는 `datasets/incremental/test`를 가리키는 dataset, pretrained model, checkpoint나 출력 설정은 즉시 거부합니다. Checkpoint와 설정 선택에는 detector run의 train-side validation만 사용합니다.

완성된 로컬 run은 다음 구조를 가집니다. `runs/detector/`는 Git에 포함되지 않습니다.

```text
runs/detector/yolo11n_base_seed42/
  config.yaml
  metadata.json
  predictions.json
  metrics.json
  checkpoints/
    best.pt
    last.pt
```

`metadata.json`은 입력 manifest와 checkpoint SHA-256, seed, dependency, Python, OS, CPU, GPU, CUDA, 입력 크기, batch, worker 수와 backend 인자를 기록합니다. `metrics.json`은 validation AP50, Recall@IoU 0.5, 미검출률, easy/medium/hard Recall과 Base/Incremental phase Recall을 기록합니다. 정답 객체가 없는 group은 0점으로 만들지 않고 객체 수 0과 `null` metric으로 저장합니다.

저장된 checkpoint를 train-side validation에서 독립적으로 다시 평가할 수 있습니다.

```powershell
bakery-detector evaluate `
  --config configs/detector/yolo11n_base.yaml `
  --checkpoint runs/detector/yolo11n_base_seed42/checkpoints/best.pt
```

두 subcommand 모두 `--json`을 지원합니다. 이 결과는 3장의 실제 validation 장면에 대한 최초 재현 가능한 train-side 기준선이며 test 성능이나 특정 POS 장치의 성능을 의미하지 않습니다. Test 평가는 기준 설정을 고정한 뒤 별도 단계에서만 수행합니다.

## Classifier 학습 데이터 조립과 검증

`bakery-classifier-data`는 레지스트리에 등록된 단일 객체 이미지와
`datasets/base/val`의 정답 bbox crop을 모델 출력 인덱스별 파생 데이터로
조립합니다. 원본 이미지, COCO JSON과 클래스 레지스트리는 수정하지 않으며
출력은 `datasets/derived/classifier/<run-name>/`에만 저장합니다.

Base run은 15개 출력 클래스를 사용하고 단일 객체 이미지는 train에만
배치합니다. 실제 장면 crop은 동일 scene ID의 `e/m/h` 이미지가 나뉘지
않도록 train과 validation으로 분할합니다.

```powershell
bakery-classifier-data generate `
  --dataset-root datasets `
  --run-name base_seed42 `
  --phase base `
  --seed 42 `
  --validation-fraction 0.2

bakery-classifier-data validate `
  --dataset-root datasets `
  --run-name base_seed42
```

Incremental run은 Base 데이터를 재사용하면서 20개 출력 클래스를 구성합니다.
Incremental 클래스의 7장 중 seed로 결정된 1장은 임시 train-side validation,
나머지 6장은 train에 배치합니다. 실제 Incremental 장면 validation이 수집되기
전까지 이 결과는 단일 객체 도메인 검증으로 명시적으로 구분됩니다.
기본 데이터 계약은 Base 클래스당 84장과 Incremental 클래스당 7장을
요구하며, 개수가 달라지면 암묵적으로 비율을 바꾸지 않고 즉시 실패합니다.

```powershell
bakery-classifier-data generate `
  --dataset-root datasets `
  --run-name incremental_seed42 `
  --phase incremental `
  --seed 42 `
  --validation-fraction 0.2

bakery-classifier-data validate `
  --dataset-root datasets `
  --run-name incremental_seed42
```

두 subcommand 모두 `--json`을 지원합니다. Manifest는 source/output hash,
COCO bbox, `category_id`와 `model_index`, split, validation 도메인과 클래스별
표본 수, Python·platform·Pillow 버전을 기록합니다. Test 경로, 변조된
원본·crop, 누락·추가 파일, Pillow replay 버전과 레지스트리 매핑 불일치는
경고가 아니라 오류입니다.

## Base 15-class classifier 학습과 검증

`bakery-classifier train`은 검증된 `base_seed42` classifier 데이터만 입력으로
사용해 ImageNet 사전학습 ResNet18 전체를 15개 출력으로 미세조정합니다. 기본
설정은 입력 224, epoch 30, batch 64, seed 42, CUDA device 0이며 AdamW와
클래스 빈도 보정 loss를 사용합니다. Best checkpoint와 early stopping은
train-side validation loss로만 결정합니다.

공식 torchvision 가중치 `resnet18-f37072fd.pth`를
`models/pretrained/`에 준비한 뒤 실행합니다. 가중치와 `runs/classifier/`
산출물은 Git에 포함되지 않습니다.

```powershell
bakery-classifier train --config configs/classifier/resnet18_base.yaml
```

완성된 run에는 설정, 환경·하드웨어·의존성, 입력 manifest와 가중치/checkpoint
SHA-256, epoch history, validation 예측과 Top-1·Macro F1·클래스별
Precision/Recall/F1이 기록됩니다. 저장된 checkpoint를 동일한 train-side
validation에서 독립적으로 다시 평가할 수 있습니다.

```powershell
bakery-classifier evaluate `
  --config configs/classifier/resnet18_base.yaml `
  --checkpoint runs/classifier/resnet18_base_seed42/checkpoints/best.pt
```

두 subcommand 모두 `--json`을 지원합니다. 이 평가는 test 성능이 아니며 Base
또는 Incremental test를 checkpoint 선택, threshold, hyperparameter나 설정
선택에 사용하지 않습니다.

## Base end-to-end 추론과 평가

`bakery-e2e evaluate`는 고정된 YOLO11n bread detector와 Base ResNet18
classifier를 결합합니다. Detector가 찾은 한 장면의 모든 bbox crop은 가능한
경우 하나의 classifier batch로 처리하며, 결과는 bbox, `model_index`, detector
신뢰도, classifier 신뢰도와 두 신뢰도의 곱을 포함합니다.

```powershell
bakery-e2e evaluate --config configs/e2e/base_resnet18.yaml
```

명령은 detector 학습 때 seed 42로 고정된 train-side validation scene group을
detector manifest에서 가져옵니다. 모델 결과나 test split으로 평가 장면을
선택하지 않습니다. 입력 manifest와 checkpoint SHA-256, classifier registry
mapping·config를 검증하고 detector checkpoint가 추론 전후 바뀌지 않았는지
확인한 뒤 `runs/e2e/<run-name>/`에 설정, metadata, 예측과 지표를 원자적으로
저장합니다.

지표는 class-aware 101-point mAP50, mAP50:95와 클래스별 이미지 단위 exact
count accuracy입니다. AP는 detector confidence floor `0.001` 이상의 전체
예측으로 계산하고, exact-count에는 운영 confidence `0.25`를 적용합니다.
기본 실제 run은 `scene_e/h/m_0509` 3장, 정답 15개를 사용한 train-side
기준선이며 mAP50 `0.534653`, mAP50:95 `0.476370`, 지원
클래스 Macro exact-count accuracy `0.466667`입니다. 이 값은 test 또는 특정
POS 장치 성능이 아닙니다. `--json`도 지원합니다.

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

Audit는 무결성 확인을 위해 평가 전용 split도 읽지만, 이 동작은 학습 사용 권한을 부여하지 않습니다. 학습·train-side 평가 entry point는 `bakery_scanner.safety.assert_training_paths_safe`를 호출해 `datasets/base/test`와 `datasets/incremental/test` 및 그 하위 경로를 즉시 거부합니다.

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

- [에이전트 전용 Git 작업 흐름](docs/agent-git-workflow.md): 브랜치, 커밋, PR, 독립 리뷰와 자율 병합 규칙
