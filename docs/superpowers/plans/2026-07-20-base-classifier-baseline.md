# Base Classifier Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and independently evaluate a reproducible Base 15-class ResNet18 classifier using only validated train-side crop data.

**Architecture:** Pure metric code is isolated from torch training. The training orchestrator strictly validates YAML, source manifests and paths, delegates tensor work through a backend protocol, validates a staged run, and atomically publishes artifacts. The CLI only exposes implemented train and validation-evaluate commands.

**Tech Stack:** Python 3.11, PyTorch 2.13, torchvision 0.28, Pillow 12, PyYAML 6, pytest 9.

## Global Constraints

- Never read or resolve a Base/Incremental test path before safety validation.
- Consume only classifier runs that pass independent manifest replay validation.
- Base output dimension is exactly 15 and targets use registry `model_index`.
- Select checkpoints and early stopping from train-side validation loss only.
- Record seeds, dependency versions, hardware, input size, batch, workers and hashes.
- Training may use CUDA; later inference benchmark remains CPU-only.
- Publish complete runs atomically under `runs/classifier/`; never commit weights.

---

### Task 1: Pure classifier evaluation metrics

**Files:**
- Create: `src/bakery_scanner/classifier_evaluation.py`
- Create: `tests/test_classifier_evaluation.py`

**Interfaces:**
- Produces: `ClassifierPrediction`, `evaluate_classifier_predictions(predictions, output_dimension)`.

- [x] Write failing tests for Top-1, supported-class Macro F1, per-class precision/recall/F1, absent classes, empty input and invalid indices.
- [x] Run `python -m pytest tests/test_classifier_evaluation.py -q` and confirm RED because the module is missing.
- [x] Implement immutable prediction records, strict validation, confusion counts and JSON-safe `null` metrics.
- [x] Run focused tests and the full suite; expect PASS.

### Task 2: Strict configuration and orchestration

**Files:**
- Create: `src/bakery_scanner/classifier_training.py`
- Create: `tests/test_classifier_training.py`
- Create: `configs/classifier/resnet18_base.yaml`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `ClassifierTrainingConfig`, `ClassifierBackend`, `BackendTrainingResult`, `train_classifier`, `evaluate_classifier_checkpoint`, `TorchvisionClassifierBackend`.
- Consumes: `validate_classifier_dataset`, `load_class_registry`, `assert_training_paths_safe`, Task 1 metrics.

- [x] Write failing tests for exact YAML fields/ranges, path safety before reads, Base run/output dimension, CUDA availability, backend arguments, staged run contents, hashes and failed-run nonpublication.
- [x] Run `python -m pytest tests/test_classifier_training.py -q` and confirm RED.
- [x] Implement strict config parsing, environment/hash capture, validated manifest dataset records, backend protocol, staging orchestration, artifact validation and atomic publication.
- [x] Add direct torch/torchvision dependency ranges, Base YAML, and ignored `runs/classifier/` plus pretrained weights.
- [x] Run focused tests and full suite; expect PASS.

### Task 3: Torchvision ResNet18 backend

**Files:**
- Modify: `src/bakery_scanner/classifier_training.py`
- Modify: `tests/test_classifier_training.py`

**Interfaces:**
- Consumes: Task 2 backend contract.
- Produces: CUDA training, deterministic transforms, checkpoints, history and validation predictions.

- [x] Write failing tests for strict ImageNet state loading, 15-output head, seed/device/optimizer arguments, checkpoint schema and batch prediction normalization using tiny generated fixtures.
- [x] Run focused tests and confirm RED for missing backend behavior.
- [x] Implement manifest dataset, transforms, DataLoaders, AdamW loop, validation-loss early stopping, checkpoint payloads and prediction batching.
- [x] Run focused and full tests; expect PASS without requiring a real GPU job in unit tests.

### Task 4: CLI and documentation

**Files:**
- Create: `src/bakery_scanner/classifier_train_cli.py`
- Create: `tests/test_classifier_train_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: `bakery-classifier train` and `bakery-classifier evaluate`.

- [x] Write failing human/JSON CLI tests, concrete selection output tests and invalid-config exit tests.
- [x] Run `python -m pytest tests/test_classifier_train_cli.py -q` and confirm RED.
- [x] Implement CLI, register console script and document only implemented commands and train-side metric meaning.
- [x] Run focused and full tests; expect PASS.

### Task 5: Real Base baseline and final verification

**Files:**
- Local ignored input: `models/pretrained/resnet18-f37072fd.pth`
- Local ignored output: `runs/classifier/resnet18_base_seed42/`

**Interfaces:**
- Produces: the first reproducible Base classifier run and independent validation metrics.

- [x] Copy the official cached torchvision weight to the configured ignored path and record its SHA-256.
- [x] Run `bakery-classifier train --config configs/classifier/resnet18_base.yaml`; expect a complete CUDA run.
- [x] Run `bakery-classifier evaluate --config configs/classifier/resnet18_base.yaml --checkpoint runs/classifier/resnet18_base_seed42/checkpoints/best.pt`; expect validation-only metrics.
- [x] Compare stored and reevaluated checkpoint hashes and metrics, run `python -m pytest -q`, `python -m compileall -q src tests`, and `git diff --check`.
- [ ] Request independent review, fix all Critical/Important issues, commit with Korean imperative messages, push and create a Korean PR through the required workflow.
