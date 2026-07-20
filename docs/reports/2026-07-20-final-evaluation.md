# 2026-07-20 동결 최종 평가 보고서

## 결론

동결된 YOLO11n class-agnostic detector, Base 15-class ResNet18 classifier와
Incremental 20-class ResNet18 classifier의 one-shot 최종 평가를 완료했다.
평가 파이프라인과 재현성 경계는 정상 동작했지만, Incremental 모델의 신규 5개
클래스 GT-crop Top-1은 `0.1000`, E2E mAP50은 `0.1061`이므로 현재 모델은 배포
준비 상태가 아니다.

Base test에서는 Incremental 모델의 classifier Top-1이 Base 모델보다 `0.0222`
낮아졌지만 Macro F1은 `0.0170`, E2E mAP50은 `0.0603`, E2E mAP50:95는
`0.0212` 높았다. Incremental test에서 detector Recall은 `1.0000`이지만 AP50은
`0.4989`이고 운영 confidence에서 30개 정답에 36개 오탐이 있었다. 신규 성능의
핵심 병목은 classifier이며 detector 정밀도도 개선 대상이다.

이 결과를 확인한 뒤 모델, checkpoint, threshold, 코드 또는 frozen config를
변경하지 않았다. 이후 실험 선택에도 이 test 결과를 사용하지 않는다.

## 실행 경계와 계보

- evaluator가 포함된 `main`: `2a453fb1d8927aceb68891282d3175b8e84a9c3b`
- 평가 ID: `bakery_scanner_frozen_v1`
- 실행 상태: one-shot lock `completed`; 재실행하지 않음
- test 접근 시작: `2026-07-20T11:53:16.518173+00:00`
- 평가 완료: `2026-07-20T11:54:05.740568+00:00`
- frozen config SHA-256: `ae3c72c448dc081e80e5ab1ef649b040eb82ac896275c2e655ba0db5f634b236`
- registry SHA-256: `9bf47dbfcb29d45401878544cd919418cd172782dafe93699cf0a5dab352841d`
- detector checkpoint SHA-256: `ca109b8a3cebb92c31a11d0b82dd532e9943e59a0e009095bfaada106c0e151b`
- Base classifier checkpoint SHA-256: `934b7fb31aebb70099ec149fd6e6d7e1c5a762e48e96e3c225bf718fc7f55763`
- Incremental classifier checkpoint SHA-256: `b9384bbf6fd3d2725d2c8534e751e235d6a9fcd716fad8057f0a8521d29e7d8b`

평가 전후와 현재 checkpoint hash가 모두 일치했고, 게시된 frozen config 복사본도
원본과 일치했다. Base test는 9장·45개 객체, Incremental test는 12장·30개
객체다. 설정과 checkpoint 선택은 test 접근 전에 train-side validation만으로
완료했다.

## 실행 환경과 동결 설정

- OS/Python: Windows 10 build 26200 / Python 3.11.9
- CPU/GPU: Intel64 Family 6 Model 198 / NVIDIA GeForce RTX 5080
- PyTorch/CUDA/Ultralytics: 2.13.0+cu130 / 13.0 / 8.4.91
- detector 입력·device: 640 / CUDA device 0
- classifier 입력·batch: 224 / scene당 하나의 동적 batch
- detector confidence floor·운영 confidence: 0.001 / 0.25
- NMS IoU·matching IoU: 0.7 / 0.5
- E2E score: detector confidence × classifier confidence

## Detector 결과

| Split | AP50 | Recall@0.5 | 미검출률 | 정답 | TP | FP |
|---|---:|---:|---:|---:|---:|---:|
| Base test | 1.0000 | 1.0000 | 0.0000 | 45 | 45 | 2 |
| Incremental test | 0.4989 | 1.0000 | 0.0000 | 30 | 30 | 36 |

두 split 모두 easy, medium, hard Recall이 각각 `1.0000`이다. Base phase Recall은
Base test에서 `1.0000`, Incremental phase Recall은 Incremental test에서
`1.0000`이다. 각 test에는 반대 phase 정답이 없어 해당 group metric은 `null`이다.

Recall이 완전하더라도 Incremental AP50이 낮다는 점이 중요하다. 운영 confidence
`0.25`에서 모든 빵을 찾았지만 정답보다 많은 오탐을 함께 반환했다.

## GT bbox crop classifier 결과

| 조합 | 표본 | Top-1 | Macro F1 |
|---|---:|---:|---:|
| Base model · Base test | 45 | 0.7556 | 0.7219 |
| Incremental model · Base test | 45 | 0.7333 | 0.7389 |
| Incremental model · Incremental test | 30 | 0.1000 | 0.1333 |

### Base test 클래스별 Precision / Recall / F1

| idx | 클래스 | Base model P/R/F1 | Incremental model P/R/F1 |
|---:|---|---:|---:|
| 0 | Croffle | — / 0.0000 / 0.0000 | — / 0.0000 / 0.0000 |
| 1 | Scone | 0.2000 / 0.3333 / 0.2500 | 1.0000 / 1.0000 / 1.0000 |
| 2 | Half-moon Croissant | 1.0000 / 0.3333 / 0.5000 | 0.7500 / 1.0000 / 0.8571 |
| 3 | Croissant | 0.7500 / 1.0000 / 0.8571 | 0.5000 / 0.1667 / 0.2500 |
| 4 | Flower Bread | 0.4286 / 1.0000 / 0.6000 | 1.0000 / 1.0000 / 1.0000 |
| 5 | Almond Scone | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.6667 / 0.8000 |
| 6 | Dinner Roll | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.6667 / 0.8000 |
| 7 | Sugar Donut | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 8 | Bagel | — / — / — | — / — / — |
| 9 | Egg Tart | 1.0000 / 1.0000 / 1.0000 | 0.5000 / 0.6667 / 0.5714 |
| 10 | Muffin | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 11 | Burger | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| 12 | Sandwich | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.6667 / 0.8000 |
| 13 | Mini Bread | 1.0000 / 0.3333 / 0.5000 | 0.4286 / 1.0000 / 0.6000 |
| 14 | Pastry Bread | 0.5000 / 0.3333 / 0.4000 | 0.5000 / 1.0000 / 0.6667 |

Bagel은 Base test 정답 support가 없어 Precision, Recall과 F1을 계산하지 않았다.

### Incremental test 신규 클래스별 결과

| idx | 클래스 | Precision | Recall | F1 | Support |
|---:|---|---:|---:|---:|---:|
| 15 | Walnut Donut | — | 0.0000 | 0.0000 | 6 |
| 16 | Waffle | — | 0.0000 | 0.0000 | 6 |
| 17 | Grain Campagne | 1.0000 | 0.5000 | 0.6667 | 6 |
| 18 | Almond Campagne | — | 0.0000 | 0.0000 | 6 |
| 19 | Plain Bread | 0.0000 | 0.0000 | 0.0000 | 6 |

신규 5개 중 Grain Campagne만 정답 6개 가운데 3개를 맞혔다. 나머지 네 클래스는
true positive가 0이므로, 7장/클래스의 적은 Incremental 학습 데이터에 대한
일반화가 현재 학습 방식으로 확보되지 않았다.

## End-to-end 결과

| 조합 | mAP50 | mAP50:95 | 전체 Macro 수량 정확도 | 지원 클래스 Macro 수량 정확도 |
|---|---:|---:|---:|---:|
| Base model · Base test | 0.7254 | 0.6670 | 0.8370 | 0.8254 |
| Incremental model · Base test | 0.7858 | 0.6881 | 0.8778 | 0.8254 |
| Incremental model · Incremental test | 0.1061 | 0.0929 | 0.6958 | 0.5333 |

### Base retention delta (Incremental - Base)

| 지표 | 변화 |
|---|---:|
| GT-crop Top-1 | -0.0222 |
| GT-crop Macro F1 | +0.0170 |
| E2E mAP50 | +0.0603 |
| E2E mAP50:95 | +0.0212 |
| 지원 클래스 Macro 수량 정확도 | 0.0000 |

### Base test 클래스별 E2E AP와 수량 정확도

| idx | 클래스 | Base AP50/AP50:95/Count | Incremental AP50/AP50:95/Count |
|---:|---|---:|---:|
| 0 | Croffle | 0.0000 / 0.0000 / 0.6667 | 0.0000 / 0.0000 / 0.6667 |
| 1 | Scone | 0.3777 / 0.3483 / 0.6667 | 1.0000 / 0.9168 / 1.0000 |
| 2 | Half-moon Croissant | 0.3366 / 0.3366 / 0.7778 | 0.8342 / 0.8342 / 0.8889 |
| 3 | Croissant | 0.7356 / 0.5929 / 0.5556 | 0.3333 / 0.1004 / 0.5556 |
| 4 | Flower Bread | 1.0000 / 0.9337 / 0.6667 | 1.0000 / 0.9112 / 1.0000 |
| 5 | Almond Scone | 1.0000 / 1.0000 / 1.0000 | 0.6634 / 0.6634 / 0.8889 |
| 6 | Dinner Roll | 1.0000 / 0.8053 / 1.0000 | 0.9158 / 0.4168 / 0.7778 |
| 7 | Sugar Donut | 1.0000 / 0.9663 / 1.0000 | 1.0000 / 0.9554 / 1.0000 |
| 8 | Bagel | — / — / 1.0000 | — / — / 1.0000 |
| 9 | Egg Tart | 1.0000 / 0.8611 / 1.0000 | 0.8317 / 0.7099 / 0.7778 |
| 10 | Muffin | 1.0000 / 1.0000 / 0.8889 | 1.0000 / 1.0000 / 1.0000 |
| 11 | Burger | 1.0000 / 0.9168 / 1.0000 | 1.0000 / 0.9112 / 1.0000 |
| 12 | Sandwich | 1.0000 / 0.9000 / 1.0000 | 0.8076 / 0.6557 / 0.8889 |
| 13 | Mini Bread | 0.3693 / 0.3399 / 0.6667 | 0.6987 / 0.6843 / 0.5556 |
| 14 | Pastry Bread | 0.3366 / 0.3366 / 0.6667 | 0.9158 / 0.8743 / 0.5556 |

### 신규 5개 클래스별 E2E 결과

| idx | 클래스 | AP50 | AP50:95 | Exact count accuracy | 정답 수 |
|---:|---|---:|---:|---:|---:|
| 15 | Walnut Donut | 0.0000 | 0.0000 | 0.5000 | 6 |
| 16 | Waffle | 0.0000 | 0.0000 | 0.5000 | 6 |
| 17 | Grain Campagne | 0.5303 | 0.4646 | 0.6667 | 6 |
| 18 | Almond Campagne | 0.0000 | 0.0000 | 0.5000 | 6 |
| 19 | Plain Bread | 0.0000 | 0.0000 | 0.5000 | 6 |

## CPU-only 기준선

최종 정확도 평가는 GPU에서 실행했으며 latency benchmark와 목적이 다르다. 별도로
동결 Incremental 모델을 Intel Core Ultra 9 285K, PyTorch CPU, intra/inter-op
thread 4/1, detector 640, classifier 224, scene batch 5에서 측정한 train-side
기준선은 다음과 같다. 특정 POS 장치 성능으로 해석하지 않는다.

| 구간 | Mean (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|
| Detector | 97.700 | 97.762 | 101.671 |
| Crop·전처리 | 110.866 | 110.611 | 113.145 |
| Classifier batch | 31.964 | 33.488 | 37.109 |
| 후처리 | 0.021 | 0.020 | 0.023 |
| End-to-end | 240.567 | 241.043 | 248.645 |

## 독립 검산

`predictions.json`과 두 test COCO를 다시 읽어 detector, classifier와 E2E metric을
각각 재계산했다. 재계산한 세 metric JSON과 summary는 게시 결과와 정확히
일치했다. 또한 다음 항목을 확인했다.

- Base classifier 표본 45개와 Incremental classifier 표본 30개의 ID·정답 결합
- frozen config 원본·게시 복사본 SHA-256 일치
- detector와 두 classifier checkpoint의 평가 전·후·현재 SHA-256 일치
- one-shot lock 상태 `completed`
- `configuration_changed_after_test: false`

원시 산출물은 로컬 `runs/final_evaluation/frozen_v1/`의 `predictions.json`,
`detector_metrics.json`, `classifier_metrics.json`, `e2e_metrics.json`,
`summary.json`, `metadata.json`, `frozen_config.yaml`과 `report.md`에 보존된다.
`runs/`는 모델·데이터 산출물과 마찬가지로 Git에 포함하지 않으며, 이 문서는 해당
산출물에서 독립 검산한 영구 요약이다.

## 해석과 다음 목표

이 평가는 현재 동결 모델의 최초 test 기준선이며 표본 수가 작다. 특히 Base test는
클래스별 support가 불균일하고 Bagel 정답이 없으며, Incremental test는 신규
클래스당 6개 객체뿐이다. 따라서 작은 변화나 클래스별 수치를 일반적인 운영
성능으로 확대 해석할 수 없다.

다음 개발 사이클은 현재 test 결과로 설정을 고르는 방식이 아니라 아래 순서로
진행해야 한다.

1. 신규 5개 클래스의 다양한 실제 장면과 단일 객체 train-side 데이터를 추가한다.
2. scene ID로 그룹화한 train-side validation을 확대하고 클래스별 support를
   확보한다.
3. 동일 train-side split에서 ResNet18과 YOLO classification 등 classifier 후보,
   불균형 보정과 증류 전략을 ablation한다.
4. Incremental detector 오탐 원인을 train-side 장면에서 분석하되 기존 test로
   threshold를 조정하지 않는다.
5. 다음 모델 선택이 끝나면 새 평가 프로토콜과 가능하면 별도 holdout을 먼저
   동결한다. 현재 test는 이미 관측됐으므로 재사용 점수는 독립적인 모델 선택
   증거로 취급하지 않는다.

ResNet18은 YOLO classification보다 test 성능이 우수해서 선택한 것이 아니다.
15→20 출력 확장, `model_index` 순서, Base head 보존, detector 고정과 checkpoint
계보를 명시적으로 검증하기 쉬운 독립 classifier 기준선으로 먼저 채택했다.
YOLO classification은 위 3번의 train-side ablation 후보로 남아 있다.
