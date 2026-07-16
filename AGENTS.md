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
