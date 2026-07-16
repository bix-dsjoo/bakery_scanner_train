# Bakery Scanner 설계

## 1. 목적

이 프로젝트는 고정 카메라로 촬영한 트레이 이미지에서 여러 빵의 위치와 종류를 식별하는 학습·추론 방식을 검증한다.

시스템은 검출된 빵마다 다음 값을 출력한다.

- 위치: `bbox = [x, y, width, height]`
- 종류: 20개 빵 클래스 중 하나
- 신뢰도: detector objectness와 classifier confidence

초기 단계에서는 Base 15개 클래스로 모델을 만든다. 이후 Incremental 5개 클래스를 추가하고, 신규 클래스 학습 성능과 기존 클래스 유지 성능을 함께 측정한다. Incremental 학습에서도 Base 데이터를 재사용할 수 있다.

초기 연구 단계에서는 고정된 합격 성능을 두지 않는다. 재현 가능한 기준선을 먼저 측정한 뒤 후속 실험에서 성공 기준을 정한다.

## 2. 데이터셋

### 2.1 클래스 구성

- Base: 15개 클래스, 단일 객체 이미지 1,260장
- Incremental: 5개 클래스, 단일 객체 이미지 35장
- 전체: 20개 클래스

클래스의 영속 식별자는 `datasets/class_registry.json`에서 관리한다. `category_id`는 COCO 주석 식별자이고 `model_index`는 모델 출력 인덱스이므로 서로 대신 사용할 수 없다.

### 2.2 경로별 용도

| 경로 | 용도 | 학습 사용 |
|---|---|---|
| `datasets/base/bread_*` | Base classifier 학습 | 허용 |
| `datasets/base/val` | 실제 장면형 학습 데이터 | 허용 |
| `datasets/base/test` | Base 평가 | 금지 |
| `datasets/incremental/bread_*` | Incremental classifier 학습 | 허용 |
| `datasets/incremental/test` | Incremental 평가 | 금지 |

`datasets/base/val`은 기존 폴더명을 유지하지만 프로젝트에서는 `scene_train` 역할로 해석한다.

### 2.3 Test 격리 규칙

`datasets/base/test`와 `datasets/incremental/test`는 최종 평가에만 사용한다. 다음 작업에서 test 이미지, 주석 또는 test 결과를 사용하지 않는다.

- 모델 학습과 미세조정
- early stopping
- 하이퍼파라미터 선택
- confidence 및 NMS threshold 선택
- augmentation 정책 선택
- 학습 모델 또는 checkpoint 선택

학습용 설정이 test 경로를 참조하면 실행을 중단해야 한다.

### 2.4 Train-side validation

모델 설정은 학습 가능한 데이터에서 만든 validation으로 결정한다.

- 장면 데이터는 파일 단위가 아니라 scene ID 단위로 분할한다.
- 동일한 ID를 공유하는 `scene_e_*`, `scene_m_*`, `scene_h_*`는 같은 split에 배치한다.
- 합성 장면은 원본 객체와 배경이 train/validation 사이에 과도하게 공유되지 않도록 manifest를 기준으로 그룹 분할한다.
- 추가 촬영 데이터도 같은 물리적 트레이 배치에서 파생된 이미지를 하나의 그룹으로 취급한다.

## 3. 시스템 아키텍처

```text
트레이 이미지
    -> class-agnostic bread detector
    -> 빵 bbox와 objectness
    -> bbox 영역 일괄 crop 및 전처리
    -> bread classifier
    -> 빵 클래스와 confidence
    -> bbox, class, combined score
```

### 3.1 Class-agnostic detector

Detector는 제품 종류를 구분하지 않고 모든 빵을 단일 `bread` 클래스로 탐지한다.

- 입력: 트레이 RGB 이미지 1장
- 출력: 0개 이상의 bbox와 objectness score
- 주요 목표: Base 및 아직 분류기에 추가되지 않은 빵에 대한 높은 Recall
- Incremental 단계의 기본 정책: detector 가중치 고정

Detector가 특정 Base 클래스의 형태에만 의존하지 않도록 학습 주석의 모든 빵 category를 하나의 `bread` label로 통합한다.

### 3.2 Bread classifier

Classifier는 detector가 찾은 bbox crop을 입력받아 빵 종류를 예측한다.

- Base 출력: 15개 클래스
- Incremental 출력: 20개 클래스
- 동일 장면의 모든 crop은 가능한 경우 하나의 batch로 처리한다.
- 클래스 매핑은 `class_registry.json`의 `model_index`를 따른다.

최종 신뢰도 결합 방식은 train-side validation에서 결정한다. 기본 후보는 `objectness * classifier_confidence`이다.

### 3.3 Incremental 업데이트 경계

기본 Incremental 실험은 detector를 고정하고 classifier만 갱신한다. Base 데이터와 Incremental 데이터를 함께 사용할 수 있다.

다음 classifier 전략을 동일한 split과 평가 조건에서 비교한다.

1. 전체 classifier 재학습
2. classifier head만 재학습
3. backbone 마지막 블록과 head 미세조정
4. cosine 또는 prototype classifier
5. 기존 classifier를 teacher로 사용하는 지식 증류

Base는 클래스당 84장이고 Incremental은 클래스당 7장이므로 class-balanced sampling 또는 동등한 불균형 보정이 필요하다.

Detector 재학습은 기본 방식이 아니라 ablation으로 분리한다. Train-side validation에서 신규 빵의 class-agnostic Recall이 부족할 때만 detector 미세조정을 비교한다.

## 4. Detector 학습 데이터

### 4.1 기존 실제 장면

`datasets/base/val/instances_val.json`의 모든 bbox category를 단일 `bread` 클래스로 변환해 사용한다.

### 4.2 합성 장면

Base 및 Incremental 단일 객체 이미지를 트레이 배경에 배치해 detector 학습용 장면과 bbox를 생성한다.

합성기는 다음 요소를 제어할 수 있어야 한다.

- 장면당 객체 수
- 위치, 크기, 회전
- 밝기, 대비, 색온도
- 부분 겹침과 가림
- easy, medium, hard 수준의 밀집도
- 배경 이미지

배치 좌표로 bbox를 자동 생성하고 각 장면에 대해 다음 정보를 manifest에 기록한다.

- 생성 seed
- 원본 객체 경로와 category
- 배경 경로
- 객체별 변환 파라미터
- 생성 bbox
- 생성기 버전

같은 입력과 seed로 같은 결과를 재생성할 수 있어야 한다. 생성물은 원본 `datasets/base`와 `datasets/incremental`을 수정하지 않는 별도 파생 데이터 경로에 저장한다.

### 4.3 추가 촬영한 실제 장면

고정 카메라와 실제 트레이 환경에서 장면을 촬영하고 COCO bbox로 라벨링한다.

- Base와 Incremental 빵을 모두 포함할 수 있다.
- 다양한 객체 수, 회전, 위치, 밀집도와 부분 가림을 포함한다.
- 촬영 그룹 단위로 train/validation을 나눈다.
- test 장면을 복제하거나 재라벨링해 학습 데이터로 사용하지 않는다.

## 5. 학습 흐름

### 5.1 Base 단계

1. 실제 장면, 합성 장면, 추가 촬영 장면을 통합한다.
2. 모든 bbox를 `bread`로 변환해 detector를 학습한다.
3. Base 단일 객체, 실제 bbox crop, 합성 bbox crop으로 15-class classifier를 학습한다.
4. Train-side validation으로 모델, checkpoint와 threshold를 결정한다.
5. 설정을 고정한 뒤 Base test를 한 번의 평가 단계로 실행한다.

### 5.2 Incremental 단계

1. 기본 실험에서는 Base detector를 고정한다.
2. Base 및 Incremental classifier 데이터를 함께 구성한다.
3. 데이터 불균형을 보정해 20-class classifier를 학습한다.
4. 모든 설정을 train-side validation에서 결정한다.
5. 설정을 고정한 뒤 Base test와 Incremental test를 평가한다.
6. Base 성능 변화와 신규 클래스 성능을 함께 기록한다.

## 6. 추론 흐름

1. 입력 이미지 형식과 크기를 검증한다.
2. Detector로 모든 빵 bbox를 예측한다.
3. 유효한 bbox만 이미지 경계 안으로 정규화하고 crop한다.
4. 모든 crop을 batch로 classifier에 입력한다.
5. detector와 classifier score를 결합한다.
6. 각 객체의 bbox, class, score를 반환한다.

검출 결과가 없으면 빈 목록을 정상 결과로 반환한다. 잘못된 이미지, 로드 실패, 모델·레지스트리 클래스 수 불일치와 CPU 추론 설정 위반은 명시적인 오류로 처리한다.

## 7. 평가

### 7.1 Detector

- class-agnostic AP50
- Recall@IoU 0.5
- 미검출률
- easy, medium, hard별 Recall
- Base와 Incremental class group별 Recall

### 7.2 Classifier

- 정답 bbox crop 기준 Top-1 Accuracy
- Macro F1
- 클래스별 Precision과 Recall

정답 crop 평가는 detector 오류와 분리된 순수 classifier 성능을 나타낸다.

### 7.3 End-to-end

- 예측 bbox 기준 mAP50
- mAP50:95
- 클래스별 Precision과 Recall
- 클래스별 수량 정확도
- Base 학습 직후 대비 Incremental 학습 후 Base 성능 변화
- Incremental 5개 클래스 성능

### 7.4 비교 원칙

- 모든 방법은 같은 데이터 split을 사용한다.
- seed와 실행 설정을 기록한다.
- test 결과를 보고 하이퍼파라미터를 다시 선택하지 않는다.
- 고정된 합격선을 두지 않고 최초 재현 가능한 실험을 기준선으로 사용한다.

## 8. 실행 및 성능 측정 정책

### 8.1 학습

- GPU 사용을 허용한다.
- 실행 환경, GPU, dependency 버전, seed와 설정 파일을 기록한다.

### 8.2 추론

- 성능 측정에서는 현재 PC의 GPU를 사용하지 않는다.
- 모델과 입력 tensor를 명시적으로 CPU에 배치한다.
- ONNX Runtime을 사용하면 `CPUExecutionProvider`만 허용한다.
- CPU 모델, thread 수, batch 크기와 입력 해상도를 결과에 기록한다.

현재 기준 환경은 다음과 같다.

- CPU: Intel Core Ultra 9 285K, 24 cores / 24 logical processors
- RAM: 약 64 GB
- 학습 GPU: NVIDIA GeForce RTX 5080, 16 GB
- Python: 3.11.9

### 8.3 지연시간 측정

- warm-up 실행은 통계에서 제외한다.
- 동일 입력 조건으로 반복 측정한다.
- 평균, P50과 P95를 기록한다.
- 다음 구간을 개별 및 전체로 측정한다.
  - detector
  - crop 및 전처리
  - classifier batch
  - 후처리
  - end-to-end

## 9. 오류 및 무결성 처리

다음 조건은 경고로 넘기지 않고 실행 실패로 처리한다.

- 학습 설정에서 test 경로 참조
- 이미지 파일과 COCO `images` 항목 불일치
- 존재하지 않는 `image_id` 또는 `category_id`
- 이미지 경계를 벗어나거나 면적이 0 이하인 bbox
- `class_registry.json`과 classifier 출력 차원 불일치
- 중복된 `category_id` 또는 `model_index`
- 합성 데이터 manifest 누락
- CPU benchmark에서 GPU provider 활성화

## 10. 검증 범위

프로젝트 구현은 최소한 다음 자동 검증을 제공해야 한다.

- 클래스 레지스트리 유일성과 연속적인 `model_index` 검사
- COCO 파일 참조와 bbox 경계 검사
- train/validation/test 경로 누수 검사
- scene ID 그룹 분할 검사
- 동일 seed 합성 결과 재현 검사
- 합성 bbox와 객체 배치의 일치 검사
- 15-class에서 20-class classifier 확장 검사
- detector 고정 실험에서 detector 가중치 불변 검사
- CPU-only 추론 provider 검사
- 평가 결과에 필수 환경 및 metric 필드가 포함되는지 검사

## 11. 범위 제외

현재 단계에서는 특정 POS 하드웨어에 대한 성능 목표와 배포 최적화를 다루지 않는다. 현재 PC의 CPU-only 측정은 알고리즘과 구현 간 상대 비교를 위한 기준선으로만 사용한다.
