# Classifier Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently validate reproducible Base 15-class and Incremental 20-class train-side classifier datasets.

**Architecture:** A focused classifier dataset module converts registered single-object files and validated Base scene annotations into an atomic derived run with a strict provenance manifest. A separate CLI exposes generation and validation; model training consumes only validated manifests in later units.

**Tech Stack:** Python 3.11, Pillow 12, dataclasses, hashlib, JSON, argparse, pytest 9.

## Global Constraints

- Never use `datasets/base/test` or `datasets/incremental/test` for generation, validation selection, or configuration.
- Preserve all original images, COCO JSON, and `datasets/class_registry.json` byte-for-byte.
- Use registry `model_index`, never COCO `category_id`, as the output directory and model label.
- Keep all `scene_e/m/h` variants sharing one scene ID in the same split.
- Treat missing manifests, invalid bbox, mapping mismatch, and path traversal as errors.
- Require the configured default counts of 84 Base and 7 Incremental images per class.
- Write derived data only below `datasets/derived/classifier/` and publish atomically.

---

### Task 1: Define classifier dataset contract and Base build

**Files:**
- Create: `src/bakery_scanner/classifier_dataset.py`
- Create: `tests/test_classifier_dataset.py`

**Interfaces:**
- Consumes: `load_class_registry`, `validate_class_directories`, `validate_coco`, `split_scene_paths`, `assert_training_paths_safe`.
- Produces: `ClassifierDatasetConfig`, `ClassifierDatasetReport`, `ClassifierValidationReport`, `build_classifier_dataset`, `validate_classifier_dataset`.

- [ ] **Step 1: Write failing tests for the Base manifest and exact crops**

  Create a fixture dataset with 15 registered Base classes, two complete scene
  groups, valid COCO annotations, and tiny RGB images. Assert that a Base run
  contains model-index directories `0..14`, byte-identical single-object files,
  exact PNG scene crops, a validation scene group isolated from training, and a
  manifest whose category IDs map to the expected model indices.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `python -m pytest tests/test_classifier_dataset.py -q`

  Expected: collection fails because `bakery_scanner.classifier_dataset` does
  not exist.

- [ ] **Step 3: Implement the minimal Base builder**

  Implement strict dataclasses, safe run-name validation, dataset-root-relative
  paths, SHA-256 helpers, Base single-object discovery, Base COCO validation,
  scene-group splitting, exact RGB crops, manifest serialization, staging-tree
  validation, and atomic publication.

- [ ] **Step 4: Run focused and full tests**

  Run: `python -m pytest tests/test_classifier_dataset.py -q`

  Expected: PASS.

  Run: `python -m pytest -q`

  Expected: all prior and new tests PASS.

### Task 2: Add Incremental split and strict replay validation

**Files:**
- Modify: `src/bakery_scanner/classifier_dataset.py`
- Modify: `tests/test_classifier_dataset.py`

**Interfaces:**
- Consumes: Task 1 manifest schema and path/hash helpers.
- Produces: deterministic 20-class Incremental runs and independent replay validation.

- [ ] **Step 1: Write failing Incremental and tamper tests**

  Assert six of seven Incremental images per class are in training, one is in
  validation, model indices are `0..19`, identical seeds reproduce identical
  manifests, and validators reject output mutation, source mutation, registry
  mutation, missing/extra files, invalid manifest fields, test paths before
  decode, and mutation of annotation-free scene sources.
  Add one extra class image and assert count drift fails before publication.

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `python -m pytest tests/test_classifier_dataset.py -q`

  Expected: failures show missing Incremental partitioning and replay checks.

- [ ] **Step 3: Implement Incremental partitioning and replay validation**

  Derive a stable per-class seed with SHA-256, select validation files after a
  deterministic shuffle, record validation domain, replay every scene crop,
  compare every source/output hash, record all scene hashes including empty
  normal scenes, enforce exact manifest keys and inventory, and verify
  split/count summaries from records. Treat staging rename as the commit point
  and make post-commit backup cleanup best-effort.

- [ ] **Step 4: Run focused and full tests**

  Run: `python -m pytest tests/test_classifier_dataset.py -q`

  Expected: PASS.

  Run: `python -m pytest -q`

  Expected: all tests PASS.

### Task 3: Add CLI and repository documentation

**Files:**
- Create: `src/bakery_scanner/classifier_data_cli.py`
- Create: `tests/test_classifier_data_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 and Task 2 public builder/validator functions.
- Produces: `bakery-classifier-data generate` and `bakery-classifier-data validate`.

- [ ] **Step 1: Write failing CLI tests**

  Assert human and JSON success output, concrete split/path reporting, exit code
  1 for unsafe or invalid input, and forwarding of phase, seed, validation
  fraction, and overwrite settings.

- [ ] **Step 2: Run CLI tests and verify RED**

  Run: `python -m pytest tests/test_classifier_data_cli.py -q`

  Expected: collection fails because `classifier_data_cli` does not exist.

- [ ] **Step 3: Implement CLI and register the console script**

  Add strict `generate` and `validate` parsers, concise error handling for
  `DataValidationError`, stable JSON payloads, and the
  `bakery-classifier-data` project script. Document only these implemented
  commands and distinguish scene validation from Incremental single-object
  validation.

- [ ] **Step 4: Run CLI and full tests**

  Run: `python -m pytest tests/test_classifier_data_cli.py -q`

  Expected: PASS.

  Run: `python -m pytest -q`

  Expected: all tests PASS.

### Task 4: Validate the repository dataset and review the diff

**Files:**
- No production file changes beyond Tasks 1-3.

**Interfaces:**
- Consumes: checked-in dataset and the new CLI.
- Produces: validated local Base and Incremental classifier dataset manifests.

- [ ] **Step 1: Generate and validate local runs**

  Run:

  ```powershell
  $env:PYTHONPATH='src'
  python -m bakery_scanner.classifier_data_cli generate --dataset-root datasets --run-name base_seed42 --phase base --seed 42 --validation-fraction 0.2 --json
  python -m bakery_scanner.classifier_data_cli validate --dataset-root datasets --run-name base_seed42 --json
  python -m bakery_scanner.classifier_data_cli generate --dataset-root datasets --run-name incremental_seed42 --phase incremental --seed 42 --validation-fraction 0.2 --json
  python -m bakery_scanner.classifier_data_cli validate --dataset-root datasets --run-name incremental_seed42 --json
  ```

  Expected: both validations report `status: ok`, Base output dimension 15,
  Incremental output dimension 20, and no test paths.

- [ ] **Step 2: Run final verification**

  Run: `python -m pytest -q`

  Expected: all tests PASS.

  Run: `git diff --check`

  Expected: no output and exit code 0.

- [ ] **Step 3: Commit the independently testable unit**

  ```powershell
  git add src/bakery_scanner/classifier_dataset.py src/bakery_scanner/classifier_data_cli.py tests/test_classifier_dataset.py tests/test_classifier_data_cli.py pyproject.toml README.md docs/superpowers/specs/2026-07-20-classifier-data-foundation-design.md docs/superpowers/plans/2026-07-20-classifier-data-foundation.md
  git commit -m "feat(classifier): 학습 데이터 기반을 추가한다"
  ```
