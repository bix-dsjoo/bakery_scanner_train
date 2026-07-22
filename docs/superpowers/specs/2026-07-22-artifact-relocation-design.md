# Artifact Relocation Design

## 목적

학습·평가 artifact를 생성한 Git worktree에서 저장소 루트로 복사한 뒤에도 기존
manifest, metadata, checkpoint와 one-shot 평가 결과의 bytes와 SHA-256을 바꾸지
않고 provenance 검증을 수행할 수 있게 한다. 검증이 끝나면 더 이상 필요하지 않은
`classifier-foundation` worktree를 안전하게 제거한다.

이 변경은 모델, 데이터 내용, metric, frozen config, threshold 또는 checkpoint를
변경하지 않는다. Test 결과를 학습이나 설정 선택에 사용하지 않는다.

## 문제와 root cause

현재 provenance 검증은 기록된 절대경로를 현재 파일의 절대경로와 직접 비교한다.
예를 들어 YOLO manifest는 생성 당시 detector manifest를 다음과 같이 기록한다.

```text
C:\workspace\bakery_scanner_train\.worktrees\classifier-foundation\
datasets\derived\detector\base_seed42_detector_origin_aware\manifest.json
```

같은 파일을 저장소 루트의 `datasets/derived/...`로 복사해 SHA-256이 동일해도
`validate_yolo_dataset()`은 `YOLO source manifest path changed`로 실패한다.
Classifier와 detector 파생 데이터 자체는 루트에서 검증되므로, 실패 원인은 파일
내용이나 매핑이 아니라 checkout 위치를 provenance identity로 사용한 데 있다.

## 선택한 접근법

Artifact identity를 다음 두 조건의 결합으로 검증한다.

1. 현재 저장소 안에서의 허용된 상대경로가 동일하다.
2. 기존 검증 지점에서 요구하는 SHA-256이 동일하다.

기존 절대경로가 현재 경로와 정확히 일치하면 기존과 동일하게 통과한다. 경로가
다를 때만 제한된 worktree relocation 문법을 적용한다. Artifact 파일과 기록된
JSON/YAML은 다시 쓰지 않는다.

Manifest와 metadata를 루트 경로로 직접 다시 쓰는 방법은 hash 계보를 바꾸므로
사용하지 않는다. 이전 경로에 junction을 유지하는 방법도 숨은 경로 의존성을
남기므로 사용하지 않는다.

## 경로 일치 계약

새 모듈 `bakery_scanner.artifact_paths`는 다음 인터페이스를 제공한다.

```python
def recorded_artifact_path_matches(
    recorded_path: str | Path,
    actual_path: Path,
    *,
    project_root: Path,
) -> bool:
    ...
```

함수는 filesystem을 수정하지 않으며 다음 순서로 판정한다.

1. 기록 경로가 문자열 또는 `Path`가 아니면 거부한다.
2. `actual_path`와 `project_root`를 `resolve(strict=False)`로 정규화한다.
3. `actual_path`가 `project_root` 아래에 없으면 거부한다.
4. 현재 상대경로의 첫 segment가 `datasets`, `runs`, `configs`, `models` 중 하나가
   아니면 거부한다.
5. 기록 경로와 현재 project root를 `/`와 `\` 모두 인식하는 segment 목록으로
   정규화하고, `.` 또는 `..`가 있으면 거부한다.
6. 기록 경로의 전체 prefix가 현재 project root의 drive·parent·저장소 이름을
   포함한 모든 segment와 일치하지 않으면 거부한다.
7. 기록 경로가 현재 절대경로와 같으면 통과한다.
8. project-root prefix 뒤의 tail이 현재 상대경로와 같으면 통과한다.
9. tail이 `.worktrees/<worktree-name>/<현재 상대경로>`와 정확히 같으면
   relocation으로 통과한다.
10. 빈 worktree 이름, 추가 중간 segment, 다른 drive·parent·저장소 또는 다른
    상대경로는 거부한다.

Windows checkout의 drive와 대소문자 차이는 정규화하되 path segment의 수와 순서는
완전히 같아야 한다. 함수는 basename이나 suffix 일부만으로 일치시키지 않는다.

이 함수는 path identity만 판정한다. 호출자는 기존 SHA-256 검사를 반드시 함께
유지한다. 단, post-merge 루트 검증에서 발견된 YOLO `data.yaml.path`는 provenance가
아니라 Ultralytics operational run root이며 자체 hash 필드가 없다. 이 필드는 source
manifest hash와 전체 image/label hash·inventory를 먼저 검증하고 `train`, `val`,
`names`가 exact일 때만 동일한 제한 경로 문법을 적용한다. 그 밖의 hash 없는 임의
경로 검증은 relocation-aware로 완화하지 않는다.

## 적용 지점

다음과 같이 저장된 provenance 경로를 현재 artifact와 비교하는 지점에만 공통
함수를 적용한다.

- `yolo_dataset.py`
  - YOLO manifest의 source detector manifest
- `e2e_inference.py`
  - detector metadata의 YOLO manifest
  - YOLO manifest의 source detector manifest
- `cpu_benchmark.py`
  - Incremental classifier metadata의 classifier manifest
  - Incremental classifier metadata의 frozen detector checkpoint
- `final_evaluation.py`
  - Base/Incremental classifier metadata의 classifier manifest
  - Incremental classifier metadata의 frozen detector checkpoint
- `yolo_dataset.py`의 `data.yaml.path`
  - source manifest, 전체 sample hash와 inventory 검증 뒤의 operational run root

현재 실행 config에서 계산하는 checkpoint, output과 dataset root 경로는 relocation
대상이 아니다. 이 경로들은 기록된 과거 provenance가 아니라 현재 실행 선택이므로
기존의 strict absolute-path 검사를 유지한다.

## 데이터 및 provenance 불변성

다음 파일은 수정하지 않는다.

- `datasets/class_registry.json`
- 원본 Base/Incremental 이미지와 COCO JSON
- `datasets/derived/**/manifest.json`
- `runs/**/metadata.json`, metric과 prediction artifact
- detector와 classifier checkpoint
- `configs/final_evaluation/frozen_v1.yaml`
- one-shot start lock과 최종 평가 결과

루트에 복사된 파일은 worktree 원본과 SHA-256이 같아야 한다. Relocation 기능은
hash 불일치, 누락 파일, 잘못된 model mapping이나 checkpoint context를 허용하지
않는다.

## 오류 처리

기존 `DataValidationError` 계약과 메시지 범주를 유지한다. 다음 조건은 계속
실패한다.

- 현재 artifact가 프로젝트 밖에 있음
- 허용하지 않은 top-level anchor
- 다른 저장소 이름 또는 다른 상대경로
- `.worktrees` 뒤 worktree 이름 누락
- `.` 또는 `..` segment
- 기록된 manifest/checkpoint SHA-256 불일치
- checkpoint, registry mapping, output dimension 또는 frozen detector context 불일치

## 테스트 전략

### 공통 경로 helper

- 동일 절대경로 통과
- root와 `.worktrees/<name>` 사이의 동일 상대경로 통과
- Windows separator와 drive 대소문자 정규화
- 다른 repository 이름, 다른 anchor, 다른 filename과 추가 segment 거부
- project root 밖 actual path와 traversal segment 거부

### 통합 회귀 테스트

- 복사된 YOLO manifest가 루트 detector manifest와 같은 hash일 때 검증 통과
- detector metadata의 기록 경로가 worktree이고 현재 YOLO manifest가 루트일 때
  provenance 통과
- classifier metadata의 manifest와 frozen detector 경로 relocation 통과
- YOLO `data.yaml.path` relocation 통과 및 다른 repository root 거부
- 위 각 경로에서 hash가 다르면 기존처럼 실패
- exact-path 기반 기존 테스트가 모두 계속 통과

테스트는 임시 디렉터리와 synthetic artifact만 사용한다. 실제 Base/Incremental
test는 읽거나 변경하지 않는다.

## 실제 migration 검증

PR이 독립 리뷰와 squash merge를 마친 뒤 루트 `main`에서 다음을 확인한다.

1. classifier Base/Incremental dataset validation
2. detector dataset validation
3. YOLO dataset validation
4. detector/classifier metadata provenance validation
5. 복사된 `datasets/derived/classifier`, `datasets/derived/yolo`, ResNet18 pretrained
   weight와 `runs` artifact의 source/destination SHA-256 일치
6. 전체 pytest와 compileall
7. frozen config, checkpoint와 one-shot 결과 hash 불변

검증이 모두 통과한 경우에만
`C:\workspace\bakery_scanner_train\.worktrees\classifier-foundation`을 정확한 절대
경로로 확인하고 `git worktree remove --force`로 제거한다. 제거 전 worktree에만
남은 의미 있는 파일이 없음을 다시 검사한다. 제거 후 로컬 작업 브랜치를 삭제하고
`git worktree list`와 루트 validation을 다시 확인한다.

## Git 및 리뷰 절차

- 최신 `main`에서 만든 `codex/fix-artifact-relocation` 브랜치만 수정한다.
- 설계, 구현 계획과 TDD 구현을 커밋한다.
- Ready PR에서 구현자와 다른 새 컨텍스트가 독립 리뷰한다.
- diff가 변경되면 영향 검증과 독립 리뷰를 반복한다.
- 별도 병합 담당자가 최신 main, 필수 검증, 해결되지 않은 대화와 리뷰 증거를
  확인한 뒤 한국어 제목으로 squash merge한다.
- 원격 작업 브랜치를 삭제한 뒤 실제 migration과 이전 worktree 제거를 수행한다.
