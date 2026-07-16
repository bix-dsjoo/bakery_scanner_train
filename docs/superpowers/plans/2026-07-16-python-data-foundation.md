# Python Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkboxes for tracking.

**Goal:** Build a tested Python package and audit CLI that validates the current bakery datasets, blocks test data from training, and creates deterministic scene-group train/validation splits.

**Architecture:** A `src/bakery_scanner` package separates registry, COCO, path-safety, split, audit, and CLI responsibilities. Public validation functions raise `DataValidationError`; the audit composes them into JSON-serializable statistics, and the CLI converts failures to exit code 1.

**Tech Stack:** Python 3.11, standard library, Pillow 12+, pytest 9+, `argparse`, `pathlib`, `dataclasses`

## Global Constraints

- Never modify existing files under `datasets/`.
- `datasets/base/test` and `datasets/incremental/test` are evaluation-only and forbidden for training, tuning, early stopping, thresholds, augmentation, checkpoints, or model selection.
- COCO `category_id` is never used as a model output index; only registry `model_index` defines output order.
- `datasets/base/val` is train-side scene data despite its physical name.
- Do not implement detector or classifier training or inference.
- Do not commit or push.

---

### Task 1: Project metadata and registry validation

**Files:**
- Create: `pyproject.toml`
- Create: `src/bakery_scanner/__init__.py`
- Create: `src/bakery_scanner/errors.py`
- Create: `src/bakery_scanner/registry.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Produces: `DataValidationError`, `ClassRecord`, `ClassRegistry`, `load_class_registry(path)` and `validate_class_directories(dataset_root, registry)`.

- [x] Write tests for a valid 15/5 registry, category-to-model mapping, duplicate IDs, non-continuous indices, invalid phase counts, missing class directories, and phase/folder mismatches.
- [x] Run `python -m pytest tests/test_registry.py -q` and confirm failure because the package API does not exist.
- [x] Add minimal package metadata and registry implementation.
- [x] Re-run the focused tests and confirm all pass.

### Task 2: COCO validation

**Files:**
- Create: `src/bakery_scanner/coco.py`
- Create: `tests/test_coco.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `ClassRegistry`, `DataValidationError`.
- Produces: `CocoStats` and `validate_coco(annotation_path, registry, expected_phase)`.

- [x] Write tests using real temporary JPEGs for valid COCO, missing/extra files, duplicate IDs, invalid image/category references, registry mismatch, decoded size mismatch, zero-area and out-of-bounds bboxes.
- [x] Run `python -m pytest tests/test_coco.py -q` and confirm failure because `coco` is absent.
- [x] Implement strict structural, reference, image, category, and bbox validation.
- [x] Re-run focused and registry tests and confirm all pass.

### Task 3: Training-path safety

**Files:**
- Create: `src/bakery_scanner/safety.py`
- Create: `tests/test_safety.py`

**Interfaces:**
- Produces: `assert_training_paths_safe(paths, dataset_root)`.

- [x] Write tests allowing Base/Incremental class folders and `base/val`, while rejecting both test roots, descendants, normalized paths, and case-variant Windows paths.
- [x] Run `python -m pytest tests/test_safety.py -q` and confirm failure because `safety` is absent.
- [x] Implement canonical path containment checks without requiring paths to exist.
- [x] Re-run focused tests and confirm all pass.

### Task 4: Scene-group split

**Files:**
- Create: `src/bakery_scanner/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Produces: `SceneSplit`, `scene_id_from_path(path)`, and `split_scene_paths(paths, validation_fraction, seed)`.

- [x] Write tests for ID extraction, e/m/h group integrity, deterministic seed behavior, complete input preservation, invalid names/fractions, duplicate paths, and too few groups.
- [x] Run `python -m pytest tests/test_splits.py -q` and confirm failure because `splits` is absent.
- [x] Implement grouped deterministic splitting with post-condition checks.
- [x] Re-run focused tests and confirm all pass.

### Task 5: Audit composition and CLI

**Files:**
- Create: `src/bakery_scanner/audit.py`
- Create: `src/bakery_scanner/cli.py`
- Create: `src/bakery_scanner/__main__.py`
- Create: `tests/test_audit.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: registry, COCO, and split APIs.
- Produces: `AuditReport`, `audit_dataset(dataset_root, validation_fraction, seed)`, `main(argv=None)` and console command `bakery-audit`.

- [x] Write integration tests for report statistics, proposed scene split, JSON output, human output, and non-zero failure behavior.
- [x] Run the focused tests and confirm failure because audit/CLI APIs are absent.
- [x] Implement dataset discovery, report serialization, CLI formatting, and error-to-exit-code handling.
- [x] Re-run focused tests and then the full suite.

### Task 6: README and real-data verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents only commands proven to exist: editable installation, full tests, text audit, JSON audit, and module fallback.

- [x] Add setup, audit, test, output interpretation, and explicit read-only/test-isolation notes.
- [x] Run `python -m pip install -e .`, `python -m pytest -q`, `bakery-audit --dataset-root datasets`, and `python -m bakery_scanner --dataset-root datasets --json`.
- [x] Confirm all commands exit 0, record exact test totals and real audit statistics, inspect the worktree, and verify the audit leaves dataset files unchanged.

## Verification Evidence

- `python -m pip check`: no broken requirements.
- `python -m pytest -q`: 51 tests passed.
- Real audit: 20 registry classes (15 Base, 5 Incremental); 1,260 Base and 35 Incremental single-object images.
- COCO audit: Base scene-train 9 images/45 annotations/15 categories; Base test 9/45/15; Incremental test 12/30/5.
- Proposed scene split at seed 42 and fraction 0.2: train IDs `0503`, `0510`; validation ID `0509`.
- Invalid dataset CLI invocation: exit code 1 with an explicit audit failure.
- Read-only check: all 1,329 files under `datasets/` retained the same length and modification time through the final verification.
- Git: no commit or push was performed.
