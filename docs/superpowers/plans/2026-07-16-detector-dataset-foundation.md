# Detector Dataset Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble and independently validate leakage-safe class-agnostic detector COCO datasets from real Base scenes and one validated synthetic run.

**Architecture:** A focused `detector_dataset` module loads and validates both sources, converts samples into immutable provenance-rich records, groups them through a leakage-resource graph, and atomically materializes split COCO datasets plus a manifest. A separate CLI module exposes generation and validation without adding any detector model code.

**Tech Stack:** Python 3.11, standard library (`argparse`, `hashlib`, `json`, `pathlib`, `shutil`), Pillow, pytest.

## Global Constraints

- Never use `datasets/base/test` or `datasets/incremental/test` for generation or split selection.
- Treat `datasets/base/val` as train-side scene data and keep each `scene_e/m/h` ID group together.
- Preserve original `category_id` only as provenance; detector COCO uses category 1 named `bread` and never uses `model_index`.
- Do not modify original data, source COCO, `class_registry.json`, or the selected synthetic run.
- Write completed outputs only below `datasets/derived/detector/`, atomically.
- Permit valid images with zero annotations.
- Do not implement detector training or inference.
- Do not commit, push, or open a PR.

---

### Task 1: Test fixtures and core conversion contract

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_detector_dataset.py`
- Create: `src/bakery_scanner/detector_dataset.py`

**Interfaces:**
- Produces: `DetectorDatasetConfig(seed: int, validation_fraction: float)`, `build_detector_dataset(dataset_root, synthetic_run, run_name, config, overwrite=False) -> DetectorBuildReport`.
- Consumes: existing `validate_coco`, `validate_synthetic_dataset`, `load_class_registry`, `assert_training_paths_safe`, and `scene_id_from_path`.

- [ ] Add a fixture that generates a valid synthetic run alongside the existing two-group real COCO fixture, including an optional empty real scene.
- [ ] Write a failing test that calls `build_detector_dataset`, expects both split directories, and asserts every output annotation uses `category_id == 1` with exactly `[{"id": 1, "name": "bread"}]`.
- [ ] Run `python -m pytest tests/test_detector_dataset.py -q` and confirm failure is caused by the missing module/API.
- [ ] Implement only enough input loading, conversion, split materialization, and report fields to pass the conversion test.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Leakage graph and deterministic safe partition

**Files:**
- Modify: `tests/test_detector_dataset.py`
- Modify: `src/bakery_scanner/detector_dataset.py`

**Interfaces:**
- Produces: deterministic component assignment internal to `build_detector_dataset`; manifest sample fields `split`, `origin`, `scene_id`, `background`, `objects`, and SHA-256 values.

- [ ] Write failing tests showing same-seed order independence, real `e/m/h` grouping, shared synthetic source grouping, shared background grouping, and image-hash grouping.
- [ ] Write a failing test where every sample belongs to one leakage component and assert `DataValidationError` reports an impossible safe split with no final run.
- [ ] Run the focused tests and confirm each fails for missing grouping behavior.
- [ ] Implement union-find resource grouping and seed-controlled subset-sum selection of a non-empty validation component subset nearest the target image count.
- [ ] Re-run focused tests, then refactor resource-key creation while keeping them green.

### Task 3: Manifest and independent validation

**Files:**
- Modify: `tests/test_detector_dataset.py`
- Modify: `src/bakery_scanner/detector_dataset.py`

**Interfaces:**
- Produces: `validate_detector_dataset(dataset_root, run_name) -> DetectorValidationReport` and manifest version 1.

- [ ] Write failing validation tests for a valid run, original `category_id` provenance without `model_index`, empty scenes, missing/extra images, changed image or COCO bytes, changed source asset or manifest, invalid bbox, and cross-split provenance leakage.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement strict manifest parsing, source revalidation, inventory/hash/dimension/COCO checks, manifest-to-COCO comparison, and split leakage checks.
- [ ] Re-run the focused tests and refactor shared bbox/hash/path helpers only after green.

### Task 4: Atomic replacement and safety failures

**Files:**
- Modify: `tests/test_detector_dataset.py`
- Modify: `src/bakery_scanner/detector_dataset.py`

**Interfaces:**
- Extends: `build_detector_dataset(..., overwrite: bool = False)`.

- [ ] Write failing tests for forbidden test-path configuration, unsafe run names, existing output without overwrite, and a forced failed overwrite preserving the previous run byte-for-byte.
- [ ] Run focused tests and confirm failure reasons.
- [ ] Add direct-child path enforcement, staging generation, complete pre-publish validation, backup/rename rollback, and staging cleanup.
- [ ] Re-run focused tests and confirm no partial or backup directories remain.

### Task 5: CLI integration

**Files:**
- Create: `tests/test_detector_cli.py`
- Create: `src/bakery_scanner/detector_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `bakery-detector-data generate --dataset-root ... --synthetic-run ... --run-name ... --seed ... --validation-fraction ... [--overwrite] [--json]` and `bakery-detector-data validate --dataset-root ... --run-name ... [--json]`.

- [ ] Write failing CLI tests for generate JSON, validate JSON, and exit code 1 with stderr on invalid input.
- [ ] Run `python -m pytest tests/test_detector_cli.py -q` and confirm the module/entry point is missing.
- [ ] Implement argparse routing and structured report output; add the console script to `pyproject.toml`.
- [ ] Re-run CLI and detector-focused tests.

### Task 6: Documentation and actual-data verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents only commands that exist after Task 5.

- [ ] Add exact PowerShell generation and validation commands, output layout, split leakage guarantees, provenance semantics, empty-scene behavior, and the failure policy.
- [ ] Build an isolated temporary dataset using the actual `datasets/base/val`, actual registry/source images/background, and a small generated synthetic run; run generation followed by a separate validation command.
- [ ] Record command output and verify all created persistent repository data, if any, is below `datasets/derived/detector/`.
- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Review `git diff`, source/input hashes, and the requirements checklist before reporting results.
