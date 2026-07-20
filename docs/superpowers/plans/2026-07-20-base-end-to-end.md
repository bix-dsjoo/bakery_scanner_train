# Base End-to-End Implementation Plan

**Goal:** Evaluate the frozen Base detector and classifier together on the fixed train-side scene validation split.

**Architecture:** Pure class-aware scene metrics are independent from inference. A strict orchestrator validates manifests, registry mapping, checkpoint hashes and paths, then delegates detector-plus-batched-crop inference to a backend and atomically publishes replayable artifacts.

**Tech Stack:** Python 3.11, PyTorch 2.13, torchvision 0.28, Ultralytics 8.4, Pillow 12, PyYAML 6, pytest 9.

### Task 1: Pure end-to-end metrics

**Files:**
- Create: `src/bakery_scanner/e2e_evaluation.py`
- Create: `tests/test_e2e_evaluation.py`

- [x] Write failing tests for class-aware AP50/AP50:95, wrong-class boxes, ranking, exact-count accuracy, empty predictions, absent classes and invalid records.
- [x] Implement immutable truth/prediction records, strict validation, 101-point AP and count metrics.
- [x] Run focused and full tests.

### Task 2: Strict orchestration and inference backend

**Files:**
- Create: `src/bakery_scanner/e2e_inference.py`
- Create: `tests/test_e2e_inference.py`
- Create: `configs/e2e/base_resnet18.yaml`
- Modify: `src/bakery_scanner/classifier_training.py`
- Modify: `.gitignore`

- [x] Write failing tests for exact config, pre-read test safety, manifest/checkpoint hashes, Base 15-class mapping, frozen detector hash, per-scene classifier batching, empty detections and failed-run cleanup.
- [x] Implement strict config, validation-scene loader, reusable strict classifier model loading, Ultralytics/torch backend, metadata, artifact validation and atomic publish.
- [x] Run focused and full tests.

### Task 3: CLI, docs and real Base run

**Files:**
- Create: `src/bakery_scanner/e2e_cli.py`
- Create: `tests/test_e2e_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [x] Write failing human/JSON CLI and invalid-config tests.
- [x] Implement `bakery-e2e evaluate`, concrete preflight output and README documentation.
- [x] Rebuild/replay-validate the detector dataset split and copy the ignored detector run into this worktree.
- [x] Execute the three-scene CUDA run and independently replay metrics and hashes.
- [ ] Run full tests, compileall and diff-check; obtain fresh independent review, fix findings, push a Korean PR and squash merge.
