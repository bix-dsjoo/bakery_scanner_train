# 에이전트 전용 Git 작업 흐름 구현 계획

> **에이전트 작업자용:** 필수 하위 스킬: 이 계획을 작업별로 구현할 때 `superpowers:subagent-driven-development`(권장) 또는 `superpowers:executing-plans`를 사용한다. 진행 상태는 체크박스로 기록한다.

**목표:** 모든 개발·리뷰·병합을 에이전트가 수행하는 저장소에 한국어 Git 규칙, 상세 절차, PR 증거 양식을 제공한다.

**구조:** `AGENTS.md`에는 에이전트가 매 작업에서 즉시 적용할 짧은 강제 규칙을 추가한다. `docs/agent-git-workflow.md`는 브랜치 소유권, 독립 리뷰, 병합·롤백과 GitHub 설정을 상세히 정의한다. PR 템플릿은 에이전트가 검증 증거와 데이터 정책 확인을 빠뜨리지 않게 한다.

**기술 구성:** Git, GitHub Pull Request, GitHub ruleset 또는 branch protection, Markdown

## 전역 제약

- 모든 새 문서 문장과 PR 제목·본문 예시는 한국어로 작성한다. 커밋 메시지는 영어 유형과 한국어 범위·요약을 사용한다.
- 브랜치 식별자는 Codex 기본 접두사인 `codex/`와 ASCII 유형을 사용한다.
- `main` 직접 push, force push, 직접 커밋을 금지하고 모든 변경은 PR을 통해 squash merge한다.
- 구현, 독립 리뷰, 병합은 서로 다른 에이전트 인스턴스가 담당한다. 리뷰 승인이 필요한 GitHub 설정에서는 구현 에이전트와 다른 승인 가능한 봇 또는 서비스 계정을 사용한다.
- `datasets/base/test`와 `datasets/incremental/test`는 평가 전용이며, 원본 데이터·COCO JSON·`datasets/class_registry.json`은 명시적 사용자 요청 없이는 변경하지 않는다.
- 기존 데이터·평가·모델·CPU benchmark 규칙은 새 Git 작업 규칙보다 우선한다.

---

### 작업 1: 설계 문서에 한국어 메시지 규칙 반영

**파일:**
- 수정: `docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md`

**연결:**
- 입력: 승인된 에이전트 전용 Git 작업 흐름 설계
- 출력: 한국어 커밋·PR 표기 규칙을 명시한 설계 문서

- [ ] **1단계: 기존 영어 유형과 예시를 확인한다**

실행:

```powershell
rg -n "feat|fix|refactor|test|docs|data|experiment|perf|build|ci|chore|Conventional" docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md
```

예상 결과: 영어 브랜치 유형과 영어 squash 커밋 예시가 표시된다.

- [ ] **2단계: 영어 커밋 유형과 한국어 제목 형식으로 교체한다**

설계 문서의 커밋 형식을 다음으로 바꾼다.

```text
<type>[선택 범위]: <명령형 한국어 요약>
```

허용 커밋 유형은 다음으로 명시한다.

```text
feat, fix, refactor, test, docs, data, experiment, perf, build, ci, chore, revert
```

영어 예시 `fix(dataset): reject evaluation split references`는 다음으로 교체한다.

```text
fix(데이터): 평가 전용 split 참조를 거부한다
```

PR 제목과 본문도 한국어로 작성한다는 문장을 PR 계약에 추가한다. 브랜치 이름의 ASCII 유형은 기술 식별자임을 명시해 커밋 메시지 규칙과 구분한다.

- [ ] **3단계: 설계 문서 검증을 실행한다**

실행:

```powershell
git diff --check -- docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md
rg -n "fix\(dataset\)|Conventional Commits" docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md
```

예상 결과: `git diff --check`는 출력 없이 성공하고, 두 번째 명령은 이전 영어 커밋 예시를 찾지 못해 종료 코드 1을 반환한다.

- [ ] **4단계: 설계 정렬 변경을 커밋한다**

```powershell
git add -- docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md
git commit -m "docs: 영어 Git 유형 규칙을 설계에 반영한다"
```

### 작업 2: AGENTS.md에 실행 규칙 추가

**파일:**
- 수정: `AGENTS.md`

**연결:**
- 입력: `docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md`
- 출력: 에이전트가 모든 작업에서 읽는 최소 Git 작업 규칙

- [ ] **1단계: AGENTS.md의 기존 데이터·문서 규칙 위치를 확인한다**

실행:

```powershell
Get-Content -Raw -Encoding utf8 AGENTS.md
```

예상 결과: 데이터 원본·test split·레지스트리 보호와 문서 일관성 규칙이 표시된다.

- [ ] **2단계: 문서 끝에 `에이전트 Git 작업 규칙` 절을 추가한다**

추가할 절은 다음 요구를 정확히 포함한다.

```markdown
## 에이전트 Git 작업 규칙

- 모든 에이전트 변경은 최신 `main`에서 만든 `codex/<type>-<short-description>` 작업 브랜치와 PR을 통해서만 `main`에 병합한다.
- `main`에는 직접 commit, direct push, force push를 하지 않는다. 병합은 squash merge만 사용한다.
- 구현 에이전트는 자신이 만든 작업 브랜치의 유일한 작성자다. 같은 브랜치를 여러 구현 에이전트가 병렬로 수정하지 않는다.
- PR이 Ready for review가 되면 구현 에이전트와 다른 에이전트 인스턴스가 새 컨텍스트에서 독립 리뷰를 수행한다. diff가 바뀌면 리뷰와 영향을 받는 검증을 다시 수행한다.
- 병합 에이전트는 최신 `main` 반영, 필수 CI, 해결되지 않은 대화 없음, 독립 리뷰, PR 검증 증거를 확인한 뒤에만 병합한다.
- 작업 브랜치의 중간 commit은 허용하지만 `main`의 squash commit과 PR 제목·본문은 한국어로 작성한다. commit 형식은 `<type>[선택 범위]: <명령형 한국어 요약>`이며, 허용 유형은 `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `experiment`, `perf`, `build`, `ci`, `chore`, `revert`다.
- 병합 뒤 원격 작업 브랜치를 삭제한다. 문제를 되돌릴 때는 새 `codex/fix-...` 작업 브랜치 또는 GitHub revert PR을 사용하며 `main` 이력을 재작성하지 않는다.
- 상세 절차와 GitHub 설정 요구는 `docs/agent-git-workflow.md`를 따른다.
```

기존 프로젝트 목적, 데이터 정책, 실행 명령은 바꾸지 않는다.

- [ ] **3단계: 문서 형식과 규칙 충돌을 확인한다**

실행:

```powershell
git diff --check -- AGENTS.md
rg -n "에이전트 Git 작업 규칙|datasets/base/test|datasets/incremental/test|docs/agent-git-workflow.md" AGENTS.md
```

예상 결과: 공백 오류 없이 성공하고, 새 절·두 평가 split·상세 문서 경로가 모두 표시된다.

- [ ] **4단계: AGENTS 규칙을 커밋한다**

```powershell
git add -- AGENTS.md
git commit -m "docs: 에이전트 Git 작업 규칙을 추가한다"
```

### 작업 3: 상세 에이전트 Git 작업 흐름 문서 작성

**파일:**
- 생성: `docs/agent-git-workflow.md`

**연결:**
- 입력: `AGENTS.md`의 강제 규칙과 승인된 설계 문서
- 출력: 에이전트 역할과 PR 상태 전이를 설명하는 기준 문서

- [ ] **1단계: 상세 문서의 필수 절을 작성한다**

다음 제목과 내용을 포함한 한국어 Markdown 문서를 만든다.

```markdown
# 에이전트 전용 Git 작업 흐름

## 적용 범위와 우선순위

이 문서는 이 저장소에서 코드, 설정, 문서, 파생 데이터 생성기와 CI를 변경하는 모든 에이전트 작업에 적용한다. `AGENTS.md`의 데이터·평가 정책이 이 문서보다 우선한다.

## 역할 분리

구현 에이전트는 작업 브랜치를 만들고 변경·검증·PR 작성을 수행한다. 독립 리뷰 에이전트는 새 컨텍스트에서 PR 설명, 전체 diff, 검증 증거와 정책 준수를 검토하며 PR 브랜치를 수정하거나 병합하지 않는다. 병합 에이전트는 최신 상태와 병합 조건을 확인하고 squash merge와 브랜치 삭제를 수행한다. 한 에이전트 인스턴스는 같은 PR에서 두 역할을 겸할 수 없다.

## 브랜치와 커밋

작업 브랜치는 `codex/<type>-<short-description>` 형식을 사용한다. `<type>`에는 `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `experiment`, `perf`, `build`, `ci`, `chore`만 사용한다. 이는 Git 식별자이며 한국어 문서·커밋 규칙의 예외다.

`main`에는 직접 commit, direct push, force push를 하지 않는다. 작업 브랜치의 중간 commit은 허용하지만 `main`에는 squash merge만 사용한다. squash commit은 `<type>[선택 범위]: <명령형 한국어 요약>` 형식을 사용하고, PR 제목·본문도 한국어로 작성한다.

## PR 생성과 검증 증거

PR은 목적, 변경 범위, 검증 명령과 결과, 실행하지 못한 검증과 이유, 데이터·평가 정책 확인, 영향·롤백, 관련 작업을 포함한다. 로직 변경은 관련 test를 포함한다. 데이터 또는 모델 관련 PR은 seed, dependency, hardware, 입력 크기, batch와 thread 수처럼 재현에 필요한 항목을 기록한다.

## 독립 리뷰와 병합

Ready for review 상태의 PR은 구현 에이전트와 다른 에이전트 인스턴스가 검토한다. 리뷰 에이전트는 정책 위반, test split 누수, 원본 데이터·레지스트리 변경, 출력 차원·`model_index` 불일치, 누락된 manifest, 검증 증거 부족, 되돌릴 수 없는 변경을 확인한다. 구현 에이전트가 diff를 바꾸면 영향을 받는 검증과 독립 리뷰를 다시 실행한다.

병합 에이전트는 최신 `main`을 반영하고 필수 CI가 통과했으며, 독립 리뷰가 완료되고 모든 대화가 해결됐고 PR 템플릿의 필수 항목이 채워졌는지 확인한다. 확인 뒤 squash merge하고 원격 작업 브랜치를 삭제한다.

## 고위험 변경

원본 `datasets/`, COCO JSON, `datasets/class_registry.json`, 평가 전용 split 참조, train/validation split, threshold·checkpoint 선택, 모델 출력 차원·`model_index`, detector 고정 원칙, CI, `AGENTS.md`, PR 템플릿은 고위험 변경이다. PR에 고위험 변경을 표시하고 관련 정책·검증 증거를 독립 리뷰에서 명시적으로 확인한다.

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

GitHub 플랜 또는 권한으로 강제할 수 없는 항목은 PR 템플릿과 병합 에이전트가 동일하게 확인한다.
```

- [ ] **2단계: 상세 문서의 문장·경로·형식을 검증한다**

실행:

```powershell
git diff --check -- docs/agent-git-workflow.md
rg -n "구현 에이전트|독립 리뷰 에이전트|병합 에이전트|squash merge|datasets/class_registry.json|GitHub 설정 체크리스트" docs/agent-git-workflow.md
```

예상 결과: 공백 오류 없이 성공하고, 역할 세 가지·병합 방식·고위험 경로·설정 체크리스트가 표시된다.

- [ ] **3단계: 상세 작업 흐름 문서를 커밋한다**

```powershell
git add -- docs/agent-git-workflow.md
git commit -m "docs: 에이전트 전용 Git 작업 흐름을 추가한다"
```

### 작업 4: 한국어 PR 템플릿과 README 안내 추가

**파일:**
- 생성: `.github/pull_request_template.md`
- 수정: `README.md`

**연결:**
- 입력: `docs/agent-git-workflow.md`
- 출력: PR 작성 시 자동으로 주입되는 증거 양식과 기준 문서 링크

- [ ] **1단계: 한국어 PR 템플릿을 만든다**

다음 내용을 `.github/pull_request_template.md`에 작성한다.

```markdown
## 목적

<!-- 해결하려는 문제와 변경 이유를 한국어로 작성한다. -->

## 변경 범위

- 변경한 항목:
- 의도적으로 제외한 항목:

## 검증

- 실행 명령:
- 결과:
- 실행하지 못한 검증과 이유:

## 데이터·평가 정책 확인

- [ ] `datasets/base/test`와 `datasets/incremental/test`를 학습, 설정 선택 또는 모델 선택에 사용하지 않았다.
- [ ] 원본 데이터, COCO JSON, `datasets/class_registry.json`을 변경하지 않았거나 사용자 요청과 검증 근거를 기록했다.
- [ ] 파생 데이터를 변경한 경우 seed, 원본, 변환, bbox, 생성기 버전 manifest를 기록했다.
- [ ] 모델 출력을 변경한 경우 `model_index` 순서와 출력 차원을 검증했다.
- [ ] CPU benchmark를 변경한 경우 CPU-only provider와 환경 메타데이터를 확인했다.

## 영향과 롤백

- 영향:
- 롤백 방법:

## 리뷰·병합 확인

- [ ] 구현 에이전트와 다른 에이전트 인스턴스가 독립 리뷰를 수행했다.
- [ ] 새 diff가 생긴 뒤 필요한 검증과 독립 리뷰를 다시 수행했다.
- [ ] 병합 전 최신 `main`, 필수 CI, 해결되지 않은 대화 없음을 확인했다.

## 관련 작업

<!-- 관련 이슈, 선행·후속 PR을 적고 없으면 `없음`으로 작성한다. -->
```

- [ ] **2단계: README에 기준 문서 링크를 추가한다**

README의 프로젝트 문서 안내 절에 다음 링크를 추가한다.

```markdown
- [에이전트 전용 Git 작업 흐름](docs/agent-git-workflow.md): 브랜치, 커밋, PR, 독립 리뷰와 자율 병합 규칙
```

기존 구현된 명령 목록은 변경하지 않는다.

- [ ] **3단계: 템플릿과 README를 검증한다**

실행:

```powershell
git diff --check -- .github/pull_request_template.md README.md
rg -n "데이터·평가 정책 확인|독립 리뷰|최신 `main`" .github/pull_request_template.md
rg -n "에이전트 전용 Git 작업 흐름" README.md
```

예상 결과: 공백 오류 없이 성공하고, PR 템플릿의 정책·리뷰·병합 확인과 README 링크가 표시된다.

- [ ] **4단계: PR 템플릿과 README 안내를 커밋한다**

```powershell
git add -- .github/pull_request_template.md README.md
git commit -m "docs: 에이전트 PR 양식과 안내를 추가한다"
```

### 작업 5: 전체 문서 일관성 검토와 최종 검증

**파일:**
- 검토: `AGENTS.md`
- 검토: `README.md`
- 검토: `docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md`
- 검토: `docs/agent-git-workflow.md`
- 검토: `.github/pull_request_template.md`

**연결:**
- 입력: 작업 1~4의 문서 변경
- 출력: 한국어 표현, 경로, 데이터 정책, 역할 분리가 일치하는 문서 세트

- [ ] **1단계: 필수 문구와 금지된 영어 커밋 예시를 검사한다**

실행:

```powershell
rg -n "에이전트 Git 작업 규칙|에이전트 전용 Git 작업 흐름|독립 리뷰|squash merge|datasets/base/test|datasets/incremental/test" AGENTS.md README.md docs/agent-git-workflow.md .github/pull_request_template.md
rg -n "fix\(dataset\): reject evaluation split references|docs: define agent-only Git workflow" AGENTS.md README.md docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md docs/agent-git-workflow.md .github/pull_request_template.md
```

예상 결과: 첫 번째 명령은 모든 필수 문구를 표시한다. 두 번째 명령은 이전 영어 메시지를 찾지 못해 종료 코드 1을 반환한다.

- [ ] **2단계: 전체 diff와 공백 오류를 확인한다**

실행:

```powershell
git diff --check main...HEAD
git status --short
git log --format=%s main..HEAD
```

예상 결과: 공백 오류가 없고, 변경 파일만 표시되며, 계획 이후 추가한 커밋 제목은 모두 한국어다.

- [ ] **3단계: 최종 검증 커밋을 만든다**

검토에서 수정이 필요했다면 해당 파일을 stage한 뒤 다음 메시지로 커밋한다.

```powershell
git add -- AGENTS.md README.md docs/superpowers/specs/2026-07-20-agent-git-workflow-design.md docs/agent-git-workflow.md .github/pull_request_template.md
git commit -m "docs: 에이전트 Git 규칙의 일관성을 검토한다"
```

수정이 없으면 이 단계에서는 새 커밋을 만들지 않는다.
