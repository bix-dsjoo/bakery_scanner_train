# Frozen Final Test Evaluation Implementation Plan

> Implement and review without reading either real test split. Use only
> synthetic pytest fixtures until the evaluator PR is merged.

**Goal:** Ship a one-shot final evaluator for the already frozen detector,
Base classifier, and Incremental classifier, then run it exactly once from the
merged `main` and publish a documentation-only result report.

**Design:** `docs/superpowers/specs/2026-07-20-final-evaluation-design.md`

## Constraints

- `configs/final_evaluation/frozen_v1.yaml` is immutable after commit
  `ec7a0c2`; do not change its models, hashes, thresholds, paths, or inference
  choices.
- Do not read/list/stat/decode either real test path before the evaluator PR is
  independently reviewed and merged.
- Preflight may inspect only non-test configs, metadata, checkpoints, registry,
  CUDA, output, and lock paths.
- The real run creates a persistent start lock before test access and cannot be
  repeated.
- After the real run, change documentation only.

## Task 1: Strict frozen config and non-test preflight

**Files:**

- Create: `src/bakery_scanner/final_evaluation.py`
- Create: `tests/test_final_evaluation.py`

- [ ] Write failing tests for exact nested schema, freeze declarations, valid
  phases/output dimensions, numeric thresholds, SHA-256 values, and run names.
- [ ] Implement immutable config dataclasses and strict YAML loading.
- [ ] Write a sentinel test whose test paths raise on any filesystem access;
  prove `preflight_final_evaluation()` never touches them.
- [ ] Implement preflight hash/provenance/context/CUDA/output validation using
  only non-test paths.
- [ ] Test config/checkpoint/metadata/context drift and existing lock/output
  failures.
- [ ] Commit: `feat(eval): 최종 평가 사전 검증을 추가한다`.

## Task 2: Test COCO loading and one-shot lock

**Files:**

- Modify: `src/bakery_scanner/final_evaluation.py`
- Modify: `tests/test_final_evaluation.py`

- [ ] Build synthetic Base/Incremental COCO fixtures outside real datasets.
- [ ] Write failing tests for COCO validation, registry category-to-model-index
  conversion, difficulty parsing, detector/end-to-end truth, and GT crop order.
- [ ] Implement post-lock test loading through `validate_coco()` plus strict
  payload conversion.
- [ ] Write failing tests proving the lock is created before the first test
  read, persists on success/failure, refuses a second run, and records config
  hash/status/timestamps.
- [ ] Implement atomic lock create/update and staging cleanup.
- [ ] Commit: `feat(eval): one-shot test 입력 경계를 구현한다`.

## Task 3: Frozen GPU inference backend

**Files:**

- Modify: `src/bakery_scanner/final_evaluation.py`
- Modify: `tests/test_final_evaluation.py`

- [ ] Write failing adapter tests using fake detector/classifiers for detector
  floor/NMS/device arguments, detector reuse, GT crops, one scene batch, empty
  detections, and Base classifier exclusion from Incremental test.
- [ ] Implement strict model loading on CUDA device 0 and per-split detector
  inference reuse.
- [ ] Implement GT-crop classifier predictions and end-to-end classifications
  for the three frozen combinations.
- [ ] Verify all emitted model indices, confidences, image/sample IDs, bbox
  bounds, and batch sizes.
- [ ] Commit: `feat(eval): 동결 모델의 최종 추론을 구현한다`.

## Task 4: Metrics, deltas, raw artifacts, and CLI

**Files:**

- Modify: `src/bakery_scanner/final_evaluation.py`
- Create: `src/bakery_scanner/final_evaluation_cli.py`
- Modify: `pyproject.toml`
- Create: `tests/test_final_evaluation_cli.py`
- Modify: `tests/test_final_evaluation.py`
- Modify: `README.md`

- [ ] Write failing orchestration tests for both detector reports, all three
  classifier reports, all three end-to-end reports, Base-retention deltas, and
  new-five separation.
- [ ] Implement required metrics by reusing validated detector, classifier, and
  end-to-end evaluation functions.
- [ ] Publish frozen config, metadata, raw predictions, metric files, summary,
  and Korean report atomically; verify exact completed file set.
- [ ] Add `bakery-final-eval preflight/run --config ...` CLI tests and entry
  point. Print an explicit irreversible one-shot warning for `run`.
- [ ] Document only the implemented preflight/run commands and the two-PR
  freeze workflow; do not include test results yet.
- [ ] Run full tests, compileall, diff-check, and verify no process accessed the
  real test paths.
- [ ] Commit, push, open a Korean Ready evaluator PR, obtain independent review,
  repeat review after diff changes, and use a separate merge agent.

## Task 5: One-shot final run from merged main

**Prerequisite:** Task 4 PR merged; new `codex/docs-final-results` branch created
from that exact latest `origin/main`; no code/config diff.

- [ ] Run `bakery-final-eval preflight --config configs/final_evaluation/frozen_v1.yaml`.
- [ ] Record clean worktree, exact main commit, absent output, and absent lock.
- [ ] Run exactly once:

  ```powershell
  bakery-final-eval run --config configs/final_evaluation/frozen_v1.yaml
  ```

- [ ] Verify lock status completed, every expected artifact exists, all raw
  prediction IDs/counts match COCO, and checkpoint before/after/current hashes
  are identical.
- [ ] Independently recompute all detector/classifier/end-to-end metrics and
  deltas from raw predictions without changing any setting.
- [ ] Record actual results and limitations in README and a dated final report.
- [ ] Confirm the result branch changes documentation only.
- [ ] Run documentation and artifact consistency checks, push, open a Korean
  Ready results PR, obtain independent review, and use a separate merge agent.

## Task 6: Final audit

- [ ] Confirm latest `origin/main` contains both evaluator and result squash
  commits and no remote work branch remains.
- [ ] Confirm all seven top-level project stages are complete.
- [ ] Confirm no test-derived adjustment commit exists after the start-lock
  timestamp.
- [ ] Report final checkpoint hashes, core metrics, CPU benchmark, PR links,
  merge SHAs, and remaining scientific limitations to the user.
