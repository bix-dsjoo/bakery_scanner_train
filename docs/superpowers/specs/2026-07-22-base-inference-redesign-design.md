# Base 정확도·FP 개선 추론 아키텍처 재설계

**Date:** 2026-07-22
**Status:** 사용자 승인 설계

## 1. 목적

Base 15-class 단계의 추론 구조를 재설계해 제한된 train-side 평가에서 다음 운영
목표를 동시에 만족하는 모델을 만든다.

- 최종 bbox `Recall@IoU 0.5 >= 0.95`
- 최종 bbox false positive 수 `0`
- 현재 ResNet18 기준선보다 GT-crop과 detector-crop 분류 성능 개선
- 현재 CPU 기준선 대비 기본 추론 경로의 평균과 P95 지연을 1.5배 이내로 제한

이 설계에서 FP는 detector가 만든 후보 가운데 IoU 0.5에서 어떤 정답과도 매칭되지
않고 최종 결과에 남은 bbox를 뜻한다. 목표는 후보 detector가 아니라 breadness
verifier와 중복 제거를 통과한 최종 bbox 출력에 적용한다.

추가 이미지 촬영은 하지 않는다. 기존 학습 허용 이미지, 파생 합성 데이터와
train-side hard negative만 사용한다. `datasets/base/test`와
`datasets/incremental/test` 및 이미 관측된 test 결과는 학습, threshold,
augmentation, checkpoint 또는 모델 선택에 사용하지 않는다.

## 2. 현재 기준선과 해석 경계

현재 동결 기준선은 YOLO11n class-agnostic detector와 Base 15-class ResNet18
classifier다.

- train-side validation: 3장, 15개 정답, Recall `1.0000`, FP `0`
- 동결 Base test: 9장, 45개 정답, Recall `1.0000`, FP `2`
- Base GT-crop classifier Top-1 `0.7556`, Macro F1 `0.7219`
- 게시된 Incremental CPU end-to-end 참고값 평균 `240.567ms`, P95 `248.645ms`

게시된 CPU 수치도 Base 모델 합격선이 아니라 초기 예산을 잡기 위한 참고값이다.
E0에서 새 split의 Base 기준선을 CPU로 다시 측정하고, 실제 지연 gate는 그 E0
평균과 P95의 각각 1.5배로 동결한다.

Test 수치는 이미 동결된 모델의 진단 기록일 뿐 이 설계의 선택 근거가 아니다.
Train-side validation도 FP가 0이고 표본이 작으므로, 기존 운영 threshold에서
verifier의 효과를 직접 확인할 수 없다. 새 사이클은 낮은 후보 threshold의
out-of-fold proposal과 negative-only 장면을 포함해 FP 억제 능력을 평가한다.

학습 가능한 실제 장면은 9장, scene ID 그룹은 3개뿐이며 모두 과거 학습 또는
validation에 사용됐다. 따라서 새 cycle holdout도 완전히 미관측인 실제 데이터는
아니다. 이 설계에서 얻은 FP 0은 제한된 cycle holdout의 관측 결과이며 실제 운영
환경의 절대적인 FP 확률 0을 보증하지 않는다.

## 3. 설계 결정

기본 추론 경로는 단일 detector, 단일 binary verifier와 단일 fine-grained
classifier로 제한한다.

```text
입력 이미지 검증
  -> 고정 트레이 ROI와 입력 정규화
  -> 단일 class-agnostic YOLO
  -> bbox 경계 보정과 중복 제거
  -> bread/background verifier 일괄 추론
  -> 통과 bbox 일괄 crop과 전처리
  -> 단일 Base 15-class classifier 일괄 추론
  -> bbox, model_index, 단계별 score 반환
```

DINO, SAM 2, detector ensemble과 classifier ensemble은 상시 실행하지 않는다.
기본 cascade가 정확도 목표를 달성하지 못하고 해당 구성의 개선 효과가 동일
train-side split에서 확인될 때만 별도 고비용 ablation으로 승격한다.

이 결정은 역할을 다음처럼 분리한다.

- detector: 낮은 confidence에서 빵 후보를 놓치지 않는다.
- verifier: 후보 bbox를 `bread` 또는 `background`로 판정해 FP를 억제한다.
- classifier: verifier를 통과한 빵을 15개 Base 종류 중 하나로 분류한다.
- orchestrator: ROI, 중복 제거, threshold, mapping과 결과 계보를 통제한다.

Classifier confidence 거부는 detector FP 해결책으로 간주하지 않는다. 향후 선택적
거부를 추가하면 coverage와 최종 Recall 감소를 별도 지표로 기록해야 한다.

## 4. 구성요소 경계

### 4.1 Tray normalizer

Tray normalizer는 이미지 형식과 크기를 검증하고 고정 카메라의 트레이 ROI를
적용한다. ROI 밖 픽셀은 detector 입력에서 마스킹하고 ROI 밖에 중심이 있는 후보는
제거한다.

카메라 위치 검사는 기존 빈 트레이 기준 영상과 재현 가능한 정합 방법으로 허용
범위를 검증할 수 있을 때만 활성화한다. 정합 근거가 없는 휴리스틱 이동 감지는
기본 경로에 넣지 않는다. 유효한 이미지에서 검출이 없는 것은 정상적인 빈 결과
목록이고, 손상되거나 읽을 수 없는 이미지는 오류다.

### 4.2 Class-agnostic detector

모든 detector는 단일 클래스 `{0: "bread"}`만 출력한다. 같은 split과 seed에서
다음의 제한된 비교군을 사용한다.

1. YOLO11n: 현재 기준선
2. YOLO26s: 중간 용량 후보
3. YOLO26m: 상위 용량 후보

Detector 운영 confidence는 out-of-fold 후보 Recall을 우선해 선택한다. 개발 실제
장면의 정답이 30개라면 후보 단계는 30/30 검출을 요구한다. Recall이 같은 후보
가운데 prediction 수, FP 수, 최악 fold Recall과 CPU 지연 순으로 선택한다. 현재
Recall이 이미 1.0이므로 단순한 Recall 증가를 승격 조건으로 사용하지 않는다.

### 4.3 Breadness verifier

Verifier는 detector 후보의 bbox crop과 확장 context crop을 입력받아
`bread/background` 확률을 출력한다. 같은 장면의 후보는 하나의 batch로 처리한다.
Verifier는 15-class classifier와 별도 checkpoint, config와 metadata를 가진다.

비교군은 다음 두 개로 제한한다.

1. YOLO26n-cls binary classifier
2. ConvNeXt-Tiny binary classifier

Verifier threshold는 pooled out-of-fold 예측에서 최종 Recall 0.95 이상을 만족하는
값 중 FP가 가장 적은 값으로 선택한다. FP가 같으면 최악 fold Recall, seed 안정성,
CPU P95 순으로 선택한다.

### 4.4 Fine-grained classifier

Base classifier는 정확히 15개 출력을 사용하고 출력 순서는
`datasets/class_registry.json`의 Base `model_index` 0부터 14까지와 일치해야 한다.
다음 세 후보를 동일 조건에서 비교한다.

1. ResNet18: 현재 기준선
2. YOLO26m-cls
3. ConvNeXt-Tiny

ViT-B/16은 위 세 후보가 분류 성능 목표를 개선하지 못했을 때만 추가 ablation으로
허용한다. YOLO classification의 기본 random/center crop은 비정방형 빵 crop의
일부를 자를 수 있으므로 모델별 기본 transform을 그대로 비교하지 않는다. 모든
후보는 동일한 aspect ratio 보존 resize, padding, 정규화와 crop jitter 정책을
사용한다.

기본 배포 후보는 최고 단일 classifier다. 상위 두 classifier 앙상블은 단일 최고
모델보다 detector-crop Macro F1과 클래스별 최악 Recall을 개선하고 CPU 지연 gate도
통과할 때만 허용한다.

### 4.5 Decision orchestrator

Orchestrator는 단계별 checkpoint와 config hash, ROI 버전, detector score,
verifier score, classifier probability, bbox 변환과 거부 사유를 보존한다. 최종
결과는 `bbox`, `model_index`, 결합 score와 단계별 score를 포함한다.

정상적인 빈 장면은 빈 목록을 반환한다. Detector와 verifier가 모두 통과한 crop은
장면당 하나의 classifier batch로 처리한다. 결합 score 공식은 train-side
calibration에서 동결하며 모델별 score를 근거 없이 평균하지 않는다.

## 5. 파생 데이터 설계

원본 이미지, 기존 COCO JSON과 `datasets/class_registry.json`은 수정하지 않는다.
새 crop, hard negative와 합성 장면은 `datasets/derived/` 아래에 원자적으로
게시하고 generator version, seed, 원본 경로/hash, 변환, bbox와 split을 manifest에
기록한다.

### 5.1 Positive

- 실제 장면의 GT bbox crop
- 합성 장면의 GT bbox crop
- Base 단일 객체 이미지
- GT bbox를 1.0, 1.2, 1.4배 확장한 context crop
- detector 위치 오차를 모사한 범위 제한 crop jitter

### 5.2 Negative

- 기존 빈 트레이 배경 crop
- 실제 장면에서 GT 밖으로 검증된 배경 crop
- 합성 장면의 비객체 영역
- out-of-fold detector가 반환했지만 GT와 매칭되지 않은 proposal
- 빵과 교차하지 않지만 트레이 무늬나 반사광에 반응한 고신뢰 proposal

IoU 하나만으로 negative를 판정하지 않는다. Candidate crop이 background가 되려면
모든 GT에 대해 다음 검사를 통과해야 한다.

- crop-GT IoU 제한
- crop 면적 중 GT 교차 비율 제한
- crop 중심이 GT 내부에 있지 않음
- GT 경계와의 최소 거리

구현 계획은 이 네 제한값을 config에 명시하고 단위 테스트로 고정해야 한다. Mining
중 조건을 위반한 후보는 negative 대상에서 제외한다. 게시 대상이나 manifest가 이
조건을 위반하면 파생 데이터 생성을 실패시켜 잘못된 background label이 조용히
게시되지 않게 한다. 하나의 GT를 중복 검출한 후보는 verifier negative가 아니라
중복 제거 평가 대상으로 취급한다.

### 5.3 Scene과 background 격리

동일한 `scene_e_*`, `scene_m_*`, `scene_h_*` ID는 항상 같은 split에 둔다. 새
사이클 시작 전에 한 scene ID 그룹과 한 원본 빈 배경을 cycle holdout으로 잠그고,
그 원본이나 변형을 학습, 합성, mining 또는 calibration에 사용하지 않는다.

현재 파생 합성 run은 세 빈 배경을 모두 사용하므로 holdout 배경을 정한 뒤 새
합성 run을 만들어야 한다. Holdout 선택은 성능을 보기 전에 config와 hash로
동결한다. 나머지 두 실제 scene 그룹은 서로를 validation으로 사용하는 2-fold
out-of-fold 개발에 사용한다.

기존 project detector/classifier checkpoint는 이미 세 실제 scene 그룹 또는 세
배경의 영향을 받았으므로 새 사이클 모델의 초기값으로 사용하지 않는다. E0과 모든
후보는 일반 공개 pretrained weight에서 시작하고 해당 weight의 이름, 출처와
SHA-256을 기록한다. 기존 project checkpoint는 과거 기준선 기록으로만 보존한다.

단일 객체 이미지를 classifier train/validation으로 나눌 때도 같은 원본에서 나온
crop이나 변형이 양쪽에 걸치지 않게 source path를 그룹 키로 사용한다. Scene crop은
항상 상위 scene ID split을 따른다.

## 6. 학습과 선택 흐름

1. Cycle holdout scene/background, 개발 fold와 seed를 동결한다.
2. 새 split으로 YOLO11n과 ResNet18 기준선을 다시 학습해 비교 기준을 만든다.
3. YOLO11n, YOLO26s와 YOLO26m의 out-of-fold 후보 예측을 생성한다.
4. 후보 Recall 30/30을 유지하는 detector와 operating confidence를 선택한다.
5. 선택 detector의 out-of-fold unmatched proposal로 verifier 데이터를 만든다.
6. YOLO26n-cls와 ConvNeXt-Tiny verifier를 같은 데이터와 seed로 비교한다.
7. 최종 Recall 0.95 이상에서 FP가 가장 적은 verifier와 threshold를 선택한다.
8. ResNet18, YOLO26m-cls와 ConvNeXt-Tiny 15-class classifier를 비교한다.
9. 선택된 단일 detector, verifier와 classifier를 개발 데이터 전체로 재학습한다.
10. ROI, 전처리, checkpoint, threshold, mapping과 artifact hash를 동결한다.
11. Cycle holdout을 한 번 평가한다.
12. 합격 구성이 생기면 별도 `frozen_v2` test protocol을 구현·리뷰·동결한 뒤
    test를 한 번 실행한다.

각 모델은 최소 세 seed로 실행한다. 모델 선택은 평균만 보지 않고 최악 seed와
최악 fold를 함께 기록한다. 같은 holdout을 확인한 뒤 설정을 바꾸거나 다시 선택하지
않는다.

## 7. 실험 승격 순서

### E0: 새 split 기준선과 병목 재현

- YOLO11n + ResNet18 재학습
- train-side 정확도와 CPU benchmark 재현
- detector, crop/전처리, classifier와 후처리 시간 분리

### E1: crop·전처리 최적화

- 장면 단위 crop batch
- 중복 이미지 decode와 색상 변환 제거
- verifier와 classifier가 재사용할 수 있는 crop inventory 생성
- 정확도와 crop 좌표가 byte-level 또는 허용 오차 내에서 동일함을 검증

### E2: 단일 detector 비교

- YOLO11n, YOLO26s, YOLO26m
- 후보 Recall 30/30을 만족하는 가장 단순하고 안정적인 모델 선택

### E3: Breadness verifier

- YOLO26n-cls와 ConvNeXt-Tiny binary 비교
- out-of-fold hard negative 사용
- 최종 Recall 0.95 제약 아래 FP 최소화

### E4: Base classifier 비교

- ResNet18, YOLO26m-cls, ConvNeXt-Tiny
- GT-crop과 detector-crop Top-1, Macro F1과 클래스별 Recall 기록
- 최고 단일 모델을 기본 선택

### E5: 기본 cascade 통합

- 단일 detector + 단일 verifier + 단일 classifier
- 정확도와 CPU 지연 gate를 동시에 검증

### E6: 선택적 고비용 ablation

기본 cascade가 목표를 충족하지 못할 때만 다음을 독립적으로 비교한다.

- 후보 Recall 부족: DINO-R50 또는 detector ensemble
- classifier 성능 부족: 상위 두 classifier 앙상블
- bbox 배경 혼입이 확인됨: SAM 2 mask crop

SAM 2는 detector FP 제거기로 사용하지 않는다. Mask 품질 검사 실패는 bbox crop
fallback으로 처리하고 mask 경로가 detector-crop 분류 성능을 개선하지 않으면
최종 구조에서 제거한다.

### E7: Cycle holdout과 `frozen_v2`

Holdout 합격 구성만 새 final-evaluation protocol로 전달한다. 기존 `frozen_v1`
lock과 config는 재사용하거나 덮어쓰지 않는다. 이미 관측된 test는 새 모델 선택의
독립 근거가 아니며 `frozen_v2` 결과는 회귀 보고로만 해석한다.

## 8. 평가와 합격 조건

### 8.1 단계별 지표

- detector 후보: AP50, Recall@0.5, prediction 수, FP, 난이도별 Recall
- verifier 이후: Recall@0.5, FP, Precision, 이미지당 FP
- classifier: GT-crop/detector-crop Top-1, Macro F1, 클래스별 Precision/Recall
- end-to-end: mAP50, mAP50:95, 클래스별 수량 정확도
- 안정성: seed/fold별 값, 평균과 최악 값
- latency: detector, crop/전처리, verifier batch, classifier batch, 후처리,
  end-to-end 평균/P50/P95

### 8.2 Cycle holdout 합격

실제 holdout scene 그룹은 3장, 15개 객체이므로 한 개를 놓치면 Recall은
`14/15 = 0.9333`이다. 합격하려면 다음을 모두 만족해야 한다.

- 실제 장면 GT 15/15 검출
- 실제 장면 최종 FP 0
- 격리한 원본 negative-only 배경 최종 FP 0
- easy/medium/hard Recall 각각 1.0
- checkpoint/config/hash가 holdout 접근 전후 동일
- CPU 평균과 P95가 각각 동결된 E0 Base 기준선의 `1.5배 이하`

게시된 Incremental 참고값을 단순 적용하면 평균 약 `361ms`, P95 약 `373ms`지만,
이는 계획용 수치이고 최종 합격선은 E0 Base 재측정값에서 계산한다. 지연 기준은
특정 POS 장치의 성능 요구가 아니다.
최종 CPU benchmark는 모델과 tensor를 명시적으로 CPU에 배치한다. ONNX Runtime을
사용하면 `CPUExecutionProvider`만 허용하고 다른 provider가 활성화되면 실패한다.

Holdout 실패 후 동일 holdout을 보고 재튜닝하지 않는다. 추가 촬영이 없는 조건에서
새 독립 holdout을 만들 수 없으므로 실패는 목표 미달로 기록한다.

## 9. Fail-closed 조건

다음 조건은 경고가 아니라 오류다.

- 학습, mining, calibration 또는 checkpoint 선택 설정이 test 경로를 참조함
- 동일 scene ID나 holdout 배경의 원본/변형이 개발 split에 포함됨
- GT와 교차하는 crop이 background negative로 게시됨
- detector 출력 mapping이 `{0: "bread"}`와 다름
- Base classifier 출력 차원 또는 순서가 registry의 0~14와 다름
- 파생 crop/합성 데이터의 manifest, seed, 원본 hash 또는 변환 기록 누락
- checkpoint, pretrained weight, config, registry 또는 ROI hash 불일치
- holdout 접근 전에 freeze manifest나 one-shot lock이 없음
- holdout 접근 전후 모델, threshold, config 또는 코드 hash가 달라짐
- CPU benchmark에서 모델/tensor가 CPU가 아니거나 GPU provider가 활성화됨

SAM 2가 선택적 ablation에 포함되면 mask 생성 모델/version/checkpoint hash와 bbox
prompt를 manifest에 기록한다. Mask 품질 실패는 bbox fallback으로 처리하지만
누락된 mask provenance는 오류다.

## 10. 검증 범위

- ROI 적용과 bbox 좌표 복원
- valid empty scene과 invalid image 구분
- class-agnostic detector mapping
- IoU matching, 중복 제거, FP와 Recall 계산
- GT 교차 negative 생성 거부
- scene/background split 격리와 test 경로 누수 차단
- YOLO-CLS 포함 모든 classifier의 동일 전처리
- Base 15-output mapping과 향후 20-output 확장 경계
- verifier/classifier 장면 단위 batch
- mask fallback과 선택적 경로 비활성화
- checkpoint/config/manifest hash와 atomic publication
- holdout one-shot lock과 동결 전후 불변성
- CPU-only provider와 단계별 지연 metadata

## 11. Incremental 단계 경계

이 사이클의 모델 선택과 합격 조건은 Base 15-class에 한정한다. Incremental 기본
실험에서는 선택된 detector와 breadness verifier를 detector 경로의 일부로 보고
우선 동결하며 classifier만 20개 출력으로 확장한다.

동결 verifier가 신규 5개 빵을 background로 거부하는지는 Incremental
train-side validation에서 별도로 측정한다. 부족할 경우 verifier 갱신은 detector
재학습과 마찬가지로 별도 ablation으로 취급하고 Base/Incremental Recall과 FP를
함께 평가한다.

## 12. 문서와 구현 영향

이 문서는 승인된 후속 설계이며 현재 구현이나 README 명령을 변경하지 않는다.
구현 계획은 새 entry point와 config가 실제로 만들어지는 작업에서만 README를
갱신한다. 구현이 프로젝트의 기본 아키텍처나 데이터 정책을 변경하면 다음 문서를
같은 변경에서 함께 검토한다.

- `README.md`
- `AGENTS.md`
- `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`

## 13. 참고 자료

- Ultralytics classification: <https://docs.ultralytics.com/tasks/classify/>
- Ultralytics YOLO26: <https://docs.ultralytics.com/models/yolo26/>
- DINO: <https://arxiv.org/abs/2203.03605>
- SAM 2: <https://arxiv.org/abs/2408.00714>
