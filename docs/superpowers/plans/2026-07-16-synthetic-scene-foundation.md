# Synthetic Scene Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic detector-scene generation and replay validation under `datasets/derived/synthetic/` without implementing model code.

**Architecture:** A focused `synthetic.py` module extracts white-background object masks, applies recorded transforms, writes stable PNG scenes and a versioned manifest, and validates runs by exact replay. A separate CLI module exposes generation and validation while reusing the package's registry, safety, and error APIs.

**Tech Stack:** Python 3.11, Pillow 12, standard-library `argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`, `random`, and pytest 9

## Global Constraints

- Never read `datasets/base/test` or `datasets/incremental/test` for generation.
- Never modify existing dataset images, COCO JSON, or `datasets/class_registry.json`.
- Write generated artifacts only below `datasets/derived/synthetic/`.
- Store COCO `category_id` in annotations; never substitute registry `model_index`.
- Fail on missing manifests, invalid references, invalid bboxes, or unsafe paths.
- Do not implement detector/classifier training or inference.
- Do not commit or push.

---

### Task 1: Public generation contract and manifest

**Files:**
- Create: `tests/test_synthetic.py`
- Create: `src/bakery_scanner/synthetic.py`

**Interfaces:**
- Produces: `GENERATOR_VERSION`, `SyntheticConfig`, `SyntheticGenerationReport`, and `generate_synthetic_dataset(dataset_root, background_dir, run_name, config, overwrite=False)`.

- [x] Write temporary-dataset tests asserting that generation writes PNGs plus `manifest.json` only under `derived/synthetic/<run-name>`, records seed/source/background/category/transform/bbox/version fields, and uses registry `category_id` values.
- [x] Run `python -m pytest tests/test_synthetic.py -q` and confirm import or missing-API failure.
- [x] Implement validated config, source/background discovery, white-background mask extraction, seeded selection, resize/rotation/brightness/contrast, non-overlapping placement, bbox calculation, PNG hashing, and manifest output.
- [x] Re-run the focused tests and confirm they pass.

### Task 2: Deterministic replay and validation

**Files:**
- Modify: `tests/test_synthetic.py`
- Modify: `src/bakery_scanner/synthetic.py`

**Interfaces:**
- Produces: `SyntheticValidationReport` and `validate_synthetic_dataset(dataset_root, run_name)`.

- [x] Add tests for identical output from repeated seed/config inputs, correct bbox bounds around transformed foreground, and failures after image, bbox, source-path, or output-file tampering.
- [x] Run the new focused tests and confirm expected missing-validation failures.
- [x] Implement strict manifest parsing, safe relative path resolution, registry category checks, exact scene replay, bbox comparison, SHA-256 comparison, and missing/extra file detection.
- [x] Re-run all synthetic tests and existing safety/registry tests.

### Task 3: CLI

**Files:**
- Create: `tests/test_synthetic_cli.py`
- Create: `src/bakery_scanner/synthetic_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `bakery-synthetic generate ...`, `bakery-synthetic validate ...`, and `python -m bakery_scanner.synthetic_cli ...`.

- [x] Add CLI tests for successful JSON generation/validation and exit code `1` with stderr on invalid input.
- [x] Run `python -m pytest tests/test_synthetic_cli.py -q` and confirm the CLI is absent.
- [x] Implement subcommand parsing, config construction, JSON/text reports, and console-script registration.
- [x] Re-run focused CLI and synthetic tests.

### Task 4: README and verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents the exact background requirements, output layout, generation command, validation command, manifest semantics, and current limitations.

- [x] Add only commands now implemented by `bakery-synthetic`.
- [x] Run a real generation against registered non-test objects with a temporary plain background, then run validation on that output.
- [x] Run `python -m pytest -q` and `python -m pip check` fresh.
- [x] Inspect `git diff`, dataset changes, and generated locations; report exact evidence and limitations without committing or pushing.

## Verification Evidence

- TDD RED checks observed missing generation and CLI modules, then missing validation behavior, before implementation.
- Focused synthetic generation/validation and CLI suite: 21 tests passed.
- Full suite: 72 tests passed.
- Real registered Base-source smoke: 2 scenes and 6 objects generated and replay-validated with generator `1.0.0`.
- Same-seed overwrite regeneration retained both PNG SHA-256 values exactly.
- `python -m pip check`: no broken requirements.
- Independent code review found no remaining Critical or Important issues after regression fixes.
- Verification-only artifacts were removed; existing source images, COCO files, test splits, and `class_registry.json` were not modified.
- The initial implementation was later published in draft PR #2; subsequent background assets are added on the same feature branch.
