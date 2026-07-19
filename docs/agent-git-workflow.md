# 에이전트 전용 Git 작업 흐름

## 적용 범위와 우선순위

이 문서는 이 저장소에서 코드, 설정, 문서, 파생 데이터 생성기와 CI를 변경하는 모든 에이전트 작업에 적용한다. `AGENTS.md`의 데이터·평가 정책이 이 문서보다 우선한다.

## 역할 분리

구현 에이전트는 작업 브랜치를 만들고 변경·검증·PR 작성을 수행한다. 독립 리뷰 에이전트는 새 컨텍스트에서 PR 설명, 전체 diff, 검증 증거와 정책 준수를 검토하며 PR 브랜치를 수정하거나 병합하지 않는다. 병합 에이전트는 최신 상태와 병합 조건을 확인하고 squash merge와 브랜치 삭제를 수행한다. 한 에이전트 인스턴스는 같은 PR에서 두 역할을 겸할 수 없다.

GitHub에서 승인 규칙을 사용할 때는 구현 에이전트와 다른 승인 가능한 봇 또는 서비스 계정이 독립 리뷰를 승인해야 한다.

## 브랜치와 커밋

작업 브랜치는 `codex/<type>-<short-description>` 형식을 사용한다. `<type>`에는 `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `experiment`, `perf`, `build`, `ci`, `chore`만 사용한다. 이는 Git 식별자이며 한국어 문서·커밋 규칙의 예외다.

`main`에는 직접 commit, direct push, force push를 하지 않는다. 작업 브랜치의 중간 commit은 허용하지만 `main`에는 squash merge만 사용한다. squash commit은 `<유형>[선택 범위]: <명령형 한국어 요약>` 형식을 사용하고, PR 제목·본문도 한국어로 작성한다.

허용 유형은 `기능`, `수정`, `개선`, `시험`, `문서`, `데이터`, `실험`, `성능`, `빌드`, `자동화`, `정리`, `되돌림`이다. 예를 들어 `수정(데이터): 평가 전용 split 참조를 거부한다`와 같이 작성한다.

## PR 생성과 검증 증거

PR은 목적, 변경 범위, 검증 명령과 결과, 실행하지 못한 검증과 이유, 데이터·평가 정책 확인, 영향·롤백, 관련 작업을 포함한다. 로직 변경은 관련 test를 포함한다. 데이터 또는 모델 관련 PR은 seed, dependency, hardware, 입력 크기, batch와 thread 수처럼 재현에 필요한 항목을 기록한다.

PR은 작업 중에는 Draft로 둘 수 있다. Ready for review로 전환하기 전에는 PR 설명이 실제 diff와 일치하는지 확인하고, 필수 검증 증거를 모두 기록한다.

## 독립 리뷰와 병합

Ready for review 상태의 PR은 구현 에이전트와 다른 에이전트 인스턴스가 검토한다. 리뷰 에이전트는 정책 위반, test split 누수, 원본 데이터·레지스트리 변경, 출력 차원·`model_index` 불일치, 누락된 manifest, 검증 증거 부족, 되돌릴 수 없는 변경을 확인한다. 구현 에이전트가 diff를 바꾸면 영향을 받는 검증과 독립 리뷰를 다시 실행한다.

병합 에이전트는 최신 `main`을 반영하고 필수 CI가 통과했으며, 독립 리뷰가 완료되고 모든 대화가 해결됐고 PR 템플릿의 필수 항목이 채워졌는지 확인한다. 확인 뒤 squash merge하고 원격 작업 브랜치를 삭제한다.

## 고위험 변경

원본 `datasets/`, COCO JSON, `datasets/class_registry.json`, 평가 전용 split 참조, train/validation split, threshold·checkpoint 선택, 모델 출력 차원·`model_index`, detector 고정 원칙, CI, `AGENTS.md`, PR 템플릿은 고위험 변경이다. PR에 고위험 변경을 표시하고 관련 정책·검증 증거를 독립 리뷰에서 명시적으로 확인한다.

원본 데이터와 레지스트리는 사용자가 명시적으로 요청한 경우에만 변경할 수 있다. `datasets/base/test`와 `datasets/incremental/test` 및 그 결과는 학습, 미세 조정, 설정 선택, 모델 선택에 사용하지 않는다.

## 실패와 롤백

병합 전 조건을 만족하지 못하면 병합을 보류한다. 병합 실패나 병합 뒤 문제는 새 `codex/fix-...` 브랜치에서 수정하거나 GitHub revert PR로 되돌린다. `main`의 force push 또는 이력 재작성은 금지한다.

## GitHub 설정 체크리스트

- `main`에 PR 필수
- direct push와 force push 차단
- 필수 CI와 상태 검사
- 구현 에이전트와 다른 승인 가능한 계정의 독립 리뷰
- 새 push 시 승인 무효화 또는 최신 push 승인 요구
- 해결되지 않은 대화 차단
- linear history와 squash merge만 허용
- 병합 후 작업 브랜치 자동 삭제

GitHub 플랜 또는 권한으로 강제할 수 없는 항목은 PR 템플릿과 병합 에이전트가 동일하게 확인한다. 이 문서는 GitHub 설정을 자동으로 변경하지 않는다.
