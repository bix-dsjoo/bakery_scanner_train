# Artifact Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every existing artifact byte and SHA-256 while allowing provenance validation after the artifact tree moves from `.worktrees/<name>/` to the repository root.

**Architecture:** Add one pure path-identity helper that accepts only an exact path or the same repository-relative artifact path with one `.worktrees/<name>` segment removed. Integrate it only where stored provenance paths are already paired with SHA-256 validation; keep current runtime config/output/checkpoint selection strict.

**Tech Stack:** Python 3.11, `pathlib`, pytest 9, existing `DataValidationError` contracts, Git worktrees.

## Global Constraints

- Do not modify original images, COCO JSON, `datasets/class_registry.json`, any derived manifest, run metadata, checkpoint, frozen config, one-shot lock, metric or prediction artifact.
- Do not read or use Base/Incremental test data for implementation or model selection.
- Relocation is valid only for the exact repository-relative path under `datasets`, `runs`, `configs` or `models` and only when the caller's existing SHA-256 check also passes.
- Reject project-root escape, traversal segments, another repository name, missing worktree name, extra path segments and a different artifact filename.
- Keep runtime-selected config, dataset root, output and checkpoint paths strict; only stored historical provenance paths become relocation-aware.
- Work only on `codex/fix-artifact-relocation`, obtain independent Ready-PR review, squash merge with a Korean title and remove the remote branch.
- Remove the legacy `classifier-foundation` worktree only after the merged `main` passes root artifact validation and source/destination hash checks.

---

### Task 1: Pure artifact-path identity helper

**Files:**

- Create: `src/bakery_scanner/artifact_paths.py`
- Create: `tests/test_artifact_paths.py`

**Interfaces:**

- Consumes: recorded path string or `Path`, current artifact `Path`, explicit repository root `Path`.
- Produces: `recorded_artifact_path_matches(recorded_path, actual_path, *, project_root: Path) -> bool`.

- [x] **Step 1: Write failing tests for exact and relocated identity**

```python
from pathlib import Path

import pytest

from bakery_scanner.artifact_paths import recorded_artifact_path_matches


def test_exact_artifact_path_matches(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"
    actual = root / "datasets" / "derived" / "manifest.json"

    assert recorded_artifact_path_matches(actual, actual, project_root=root)


def test_linked_worktree_artifact_path_matches_root_copy(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"
    actual = root / "datasets" / "derived" / "manifest.json"
    recorded = (
        root
        / ".worktrees"
        / "classifier-foundation"
        / "datasets"
        / "derived"
        / "manifest.json"
    )

    assert recorded_artifact_path_matches(recorded, actual, project_root=root)
```

- [x] **Step 2: Write parameterized rejection tests**

```python
@pytest.mark.parametrize(
    "recorded,actual",
    [
        ("other_repo/.worktrees/w/datasets/a.json", "datasets/a.json"),
        ("bakery_scanner_train/.worktrees/datasets/a.json", "datasets/a.json"),
        ("bakery_scanner_train/.worktrees/w/extra/datasets/a.json", "datasets/a.json"),
        ("bakery_scanner_train/.worktrees/w/datasets/b.json", "datasets/a.json"),
        ("bakery_scanner_train/.worktrees/w/other/a.json", "other/a.json"),
        ("bakery_scanner_train/.worktrees/w/datasets/../a.json", "datasets/a.json"),
    ],
)
def test_artifact_path_rejects_non_identity(
    tmp_path: Path, recorded: str, actual: str
) -> None:
    root = tmp_path / "bakery_scanner_train"

    assert not recorded_artifact_path_matches(
        recorded, root / actual, project_root=root
    )


def test_artifact_path_rejects_actual_outside_project(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"

    assert not recorded_artifact_path_matches(
        root / "datasets" / "a.json",
        tmp_path / "outside" / "datasets" / "a.json",
        project_root=root,
    )
```

- [x] **Step 3: Run the new tests and verify RED**

Run:

```powershell
pytest -q tests/test_artifact_paths.py
```

Expected: collection fails with `ModuleNotFoundError: bakery_scanner.artifact_paths`.

- [x] **Step 4: Implement the minimal pure helper**

```python
from __future__ import annotations

from pathlib import Path


_ARTIFACT_ANCHORS = frozenset({"datasets", "runs", "configs", "models"})


def _portable_parts(value: str | Path) -> tuple[str, ...] | None:
    parts = tuple(part for part in str(value).replace("\\", "/").split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        return None
    return parts


def _same_parts(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    return tuple(item.casefold() for item in first) == tuple(
        item.casefold() for item in second
    )


def recorded_artifact_path_matches(
    recorded_path: str | Path,
    actual_path: Path,
    *,
    project_root: Path,
) -> bool:
    root = Path(project_root).resolve(strict=False)
    actual = Path(actual_path).resolve(strict=False)
    try:
        relative = actual.relative_to(root)
    except ValueError:
        return False
    if not relative.parts or relative.parts[0].casefold() not in _ARTIFACT_ANCHORS:
        return False
    if Path(recorded_path).resolve(strict=False) == actual:
        return True
    recorded_parts = _portable_parts(recorded_path)
    if recorded_parts is None:
        return False
    root_positions = [
        index
        for index, part in enumerate(recorded_parts)
        if part.casefold() == root.name.casefold()
    ]
    if not root_positions:
        return False
    tail = recorded_parts[root_positions[-1] + 1 :]
    expected = tuple(relative.parts)
    if _same_parts(tail, expected):
        return True
    return (
        len(tail) >= 3
        and tail[0].casefold() == ".worktrees"
        and tail[1] not in {"", ".", ".."}
        and _same_parts(tail[2:], expected)
    )
```

- [x] **Step 5: Run helper tests and full tests**

Run:

```powershell
pytest -q tests/test_artifact_paths.py
pytest -q
```

Expected: all helper tests pass and the existing 246-test baseline remains green.

- [x] **Step 6: Commit the helper**

```powershell
git add src/bakery_scanner/artifact_paths.py tests/test_artifact_paths.py
git commit -m "feat(artifact): worktree relocation 경로를 판정한다"
```

---

### Task 2: YOLO and detector provenance relocation

**Files:**

- Modify: `src/bakery_scanner/yolo_dataset.py:193-197`
- Modify: `src/bakery_scanner/e2e_inference.py:194-274,405-413,430-439`
- Modify: `tests/test_yolo_dataset.py`
- Modify: `tests/test_e2e_inference.py`

**Interfaces:**

- Consumes: Task 1 `recorded_artifact_path_matches()`.
- Produces: relocated YOLO source and detector metadata paths pass only with unchanged manifest hash.

- [x] **Step 1: Add a failing relocated YOLO-manifest test**

```python
def test_validate_yolo_dataset_accepts_same_hashed_relocated_source(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "relocated")
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    source_manifest = dataset_root / "derived" / "detector" / source_run / "manifest.json"
    manifest["source"]["manifest_path"] = str(
        dataset_root.parent
        / ".worktrees"
        / "old"
        / source_manifest.relative_to(dataset_root.parent)
    )
    report.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validated = validate_yolo_dataset(dataset_root, "relocated")

    assert validated.output_dir == report.output_dir
```

- [x] **Step 2: Add failing e2e provenance relocation tests**

Extend `_validate_detector_checkpoint_provenance()` and
`_validate_yolo_source_binding()` tests so their recorded manifest path is
`project_root/.worktrees/old/<actual relative path>`, while the actual file and
hash remain at the project root. Pass `project_root=dataset_root.parent` to both
functions and assert no exception. Add a paired test that writes different
manifest bytes and still expects `DataValidationError` containing `SHA-256` or
`YOLO source`.

- [x] **Step 3: Run targeted tests and verify RED**

Run:

```powershell
pytest -q tests/test_yolo_dataset.py::test_validate_yolo_dataset_accepts_same_hashed_relocated_source
pytest -q tests/test_e2e_inference.py -k relocation
```

Expected: YOLO test fails with `YOLO source manifest path changed`; e2e tests
fail because the validation functions do not accept `project_root` or reject
the relocated paths.

- [x] **Step 4: Integrate the helper in `yolo_dataset.py`**

Import `recorded_artifact_path_matches` and replace the direct path equality:

```python
if not recorded_artifact_path_matches(
    source["manifest_path"],
    source_report.manifest_path,
    project_root=dataset_root.parent,
):
    raise DataValidationError("YOLO source manifest path changed")
```

Keep the following SHA-256 comparison unchanged.

- [x] **Step 5: Integrate the helper in `e2e_inference.py`**

Add explicit keyword-only `project_root: Path` parameters to
`_validate_detector_checkpoint_provenance()` and `_validate_yolo_source_binding()`.
Use the helper for the stored YOLO-manifest and source-detector-manifest paths,
then update all production and test call sites to pass `dataset_root.parent`.
Do not change expected runtime checkpoint equality at lines 243-251.

- [x] **Step 6: Run targeted and full tests**

Run:

```powershell
pytest -q tests/test_yolo_dataset.py tests/test_e2e_inference.py
pytest -q
```

Expected: relocated and negative hash tests pass; the full suite is green.

- [x] **Step 7: Commit YOLO/detector integration**

```powershell
git add src/bakery_scanner/yolo_dataset.py src/bakery_scanner/e2e_inference.py tests/test_yolo_dataset.py tests/test_e2e_inference.py
git commit -m "fix(artifact): detector provenance relocation을 허용한다"
```

---

### Task 3: Classifier provenance relocation in benchmark and final evaluation

**Files:**

- Modify: `src/bakery_scanner/cpu_benchmark.py:299-344`
- Modify: `src/bakery_scanner/final_evaluation.py:431-482`
- Modify: `tests/test_cpu_benchmark.py`
- Modify: `tests/test_final_evaluation.py`

**Interfaces:**

- Consumes: Task 1 helper and current classifier manifest/checkpoint hashes.
- Produces: copied classifier metadata remains valid at root without weakening context, output dimension or detector-freeze checks.

- [x] **Step 1: Add failing CPU benchmark provenance tests**

Create a project root fixture with the actual classifier manifest under
`datasets/derived/classifier/incremental_seed42/manifest.json` and detector
checkpoint under `runs/detector/.../best.pt`. Put the corresponding recorded
paths under `project_root/.worktrees/old/`, preserve the real SHA-256 values,
and call `_validate_classifier_checkpoint_provenance(...,
project_root=project_root)`. Assert success. In a separate test change the
manifest bytes and assert `DataValidationError`.

- [x] **Step 2: Add failing final-evaluation classifier metadata tests**

Extend `_validate_classifier_metadata()` coverage with relocated
`dataset.manifest_path` and relocated Incremental `frozen_detector.checkpoint`.
Pass `project_root=tmp_path / "bakery_scanner_train"` and assert success only
when the recorded manifest hash and detector before/after hashes are exact.

- [x] **Step 3: Run targeted tests and verify RED**

Run:

```powershell
pytest -q tests/test_cpu_benchmark.py -k relocation
pytest -q tests/test_final_evaluation.py -k relocation
```

Expected: tests fail because the validation functions do not accept
`project_root` or still compare stored paths literally.

- [x] **Step 4: Integrate relocation in `cpu_benchmark.py`**

Add `project_root: Path` to `_validate_classifier_checkpoint_provenance()`.
Replace `dataset.get("manifest_path") != str(classifier_manifest_path)` and the
frozen-detector direct `Path.resolve()` comparison with
`recorded_artifact_path_matches()`. Keep manifest SHA-256, registry mapping,
output dimension and detector hash checks unchanged. Pass
`project_root=dataset_root.parent` from the benchmark preflight call site.

- [x] **Step 5: Integrate relocation in `final_evaluation.py`**

Add `project_root: Path` to `_validate_classifier_metadata()`. Use the helper
for `dataset.manifest_path` and Incremental `frozen_detector.checkpoint`, retain
all current hash/context checks, and pass `project_root=dataset_root.parent`
for Base and Incremental calls in `_prepare_non_test_inputs()`.

- [x] **Step 6: Run targeted and full tests**

Run:

```powershell
pytest -q tests/test_cpu_benchmark.py tests/test_final_evaluation.py tests/test_final_evaluation_cli.py
pytest -q
python -m compileall -q src tests
```

Expected: all tests and compileall pass without accessing either real test split.

- [x] **Step 7: Commit classifier provenance integration**

```powershell
git add src/bakery_scanner/cpu_benchmark.py src/bakery_scanner/final_evaluation.py tests/test_cpu_benchmark.py tests/test_final_evaluation.py
git commit -m "fix(artifact): classifier provenance relocation을 허용한다"
```

---

### Task 4: Documentation, exact-head verification and PR

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-22-artifact-relocation.md`

**Interfaces:**

- Consumes: all relocation behavior and test evidence from Tasks 1-3.
- Produces: user-facing relocation contract and a Ready PR eligible for independent review.

- [x] **Step 1: Document the relocation rule**

Add an `Artifact 경로 이동` paragraph near the dataset/run provenance sections:

```markdown
동일 저장소의 `.worktrees/<name>/`에서 루트로 artifact를 이동한 경우 검증기는
`datasets`, `runs`, `configs`, `models` 이하의 동일 상대경로와 기존 SHA-256이
모두 일치할 때만 같은 provenance로 인정합니다. Manifest, metadata와 checkpoint는
경로 이동을 위해 다시 쓰지 않습니다.
```

- [x] **Step 2: Run final verification**

Run:

```powershell
pytest -q
python -m compileall -q src tests
git diff --check origin/main...HEAD
git status --short
```

Expected: full suite passes, compileall and diff-check are clean, and only the
approved source/tests/docs are changed.

- [x] **Step 3: Confirm protected artifacts are untouched**

Run:

```powershell
git diff --name-only origin/main...HEAD -- datasets runs models configs/final_evaluation/frozen_v1.yaml
```

Expected: no output.

- [x] **Step 4: Commit documentation and completed plan checkboxes**

```powershell
git add README.md docs/superpowers/plans/2026-07-22-artifact-relocation.md
git commit -m "docs(artifact): relocation 검증 규칙을 문서화한다"
```

- [ ] **Step 5: Push and open a Korean Ready PR**

Push `codex/fix-artifact-relocation`, open a Ready PR titled
`fix(artifact): worktree artifact relocation을 지원한다`, and include root
cause, exact allowed path grammar, unchanged SHA enforcement, protected-file
diff evidence, full test count and unexecuted real migration.

- [ ] **Step 6: Obtain exact-head independent review and separate merge**

The reviewer must inspect the helper for path-broadening bugs and confirm every
relocated path remains hash-bound. If the diff changes, rerun affected tests and
request a fresh exact-head review. A separate merge agent verifies current
`main`, unresolved conversations, CI/review evidence, then squash-merges and
deletes the remote branch.

---

### Task 5: Post-merge root validation and legacy worktree removal

**Files:**

- Read only before cleanup: root `datasets/`, `models/`, `runs/`
- Remove after all gates pass: `C:\workspace\bakery_scanner_train\.worktrees\classifier-foundation`

**Interfaces:**

- Consumes: merged relocation code and the already copied root artifacts.
- Produces: current root is canonical; no legacy classifier-foundation worktree or local branch remains.

- [ ] **Step 1: Fast-forward root main and verify a clean tracked tree**

```powershell
git fetch origin --prune
git merge --ff-only origin/main
git status --short --branch
```

Expected: root `main` equals `origin/main` and has no tracked changes.

- [ ] **Step 2: Run root data validations without final-evaluator rerun**

```powershell
$env:PYTHONPATH='src'
python -m bakery_scanner.classifier_data_cli validate --dataset-root datasets --run-name base_seed42
python -m bakery_scanner.classifier_data_cli validate --dataset-root datasets --run-name incremental_seed42
python -m bakery_scanner.detector_cli validate --dataset-root datasets --run-name base_seed42_detector_origin_aware
python -c "from bakery_scanner.yolo_dataset import validate_yolo_dataset; validate_yolo_dataset('datasets','base_seed42_detector_origin_aware_yolo'); print('YOLO validation: ok')"
```

Expected: all four validations pass. Do not execute `bakery-final-eval run`.

- [ ] **Step 3: Verify copied artifact completeness and hashes**

Compare every file under legacy worktree `runs`,
`datasets/derived/classifier`, `datasets/derived/yolo` and
`models/pretrained/resnet18-f37072fd.pth` with its root-relative destination.
Require identical file counts, relative paths and SHA-256. Confirm that any
worktree-only remainder outside these trees is limited to cache/build files.

- [ ] **Step 4: Recheck frozen artifacts**

Require:

- one-shot lock status `completed`;
- frozen config SHA-256
  `ae3c72c448dc081e80e5ab1ef649b040eb82ac896275c2e655ba0db5f634b236`;
- detector checkpoint SHA-256
  `ca109b8a3cebb92c31a11d0b82dd532e9943e59a0e009095bfaada106c0e151b`;
- Base classifier checkpoint SHA-256
  `934b7fb31aebb70099ec149fd6e6d7e1c5a762e48e96e3c225bf718fc7f55763`;
- Incremental classifier checkpoint SHA-256
  `b9384bbf6fd3d2725d2c8534e751e235d6a9fcd716fad8057f0a8521d29e7d8b`.

- [ ] **Step 5: Remove the exact legacy worktree**

Resolve and print the target first. Require it to equal
`C:\workspace\bakery_scanner_train\.worktrees\classifier-foundation` and to
appear in `git worktree list --porcelain`. Then run:

```powershell
git worktree remove --force C:\workspace\bakery_scanner_train\.worktrees\classifier-foundation
git branch -D codex/docs-final-results
git worktree prune
```

This deletion is not recoverable from the removed worktree itself; recovery is
through the hash-verified root copies and merged Git history.

- [ ] **Step 6: Verify post-removal state**

Run root YOLO/classifier/detector validations again, confirm key files and
hashes, ensure `git worktree list` no longer contains `classifier-foundation`,
and verify root `main` remains clean and equal to `origin/main`.

## Plan self-review

- Spec coverage: helper grammar, all five provenance integration points,
  negative hash behavior, protected artifact immutability, independent review
  and post-merge cleanup are covered.
- Placeholder scan: no deferred implementation or unspecified error handling.
- Type consistency: every integration consumes
  `recorded_artifact_path_matches(recorded_path, actual_path,
  project_root=project_root) -> bool` and passes an explicit `Path` root.
