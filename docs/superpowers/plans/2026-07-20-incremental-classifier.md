# Incremental 20-Class Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the approved Base ResNet18 classifier to a reproducible 20-class Incremental model while reusing Base data and proving the detector checkpoint remains frozen.

**Architecture:** Preserve the existing Base config and checkpoint context unchanged. Add an explicit Incremental config that strict-loads the Base checkpoint, copies the backbone and first 15 head rows into a 20-output model, trains with recorded inverse-frequency class weights, publishes Base/new validation groups, and hashes an out-of-band detector checkpoint before and after the run.

**Tech Stack:** Python 3.11, PyTorch 2.13, torchvision 0.28, Pillow 12, PyYAML 6, pytest 9.

## Global Constraints

- Never read `datasets/base/test` or `datasets/incremental/test` for training, validation, selection, thresholds, augmentation, checkpoint choice, or model choice.
- Preserve the existing Base YAML schema and checkpoint context so the approved Base checkpoint remains loadable.
- Use registry `model_index` 0 through 19; never use COCO `category_id` as a model output index.
- Reuse Base train-side data from the validated Incremental classifier manifest.
- The default Incremental experiment updates only the classifier and proves the detector checkpoint bytes are unchanged.
- Use CUDA for training; do not claim these results are CPU or POS benchmark results.
- Publish only atomic, complete runs and never overwrite source data or an existing run.

---

### Task 1: Close classifier dataset pre-read safety gap

**Files:**
- Modify: `src/bakery_scanner/classifier_dataset.py`
- Modify: `tests/test_classifier_dataset.py`

**Interfaces:**
- Consumes: `assert_training_paths_safe(paths, dataset_root)`.
- Produces: `build_classifier_dataset(config)` that rejects a redirected scene COCO path before `_scene_payload()` reads it.

- [x] **Step 1: Write the failing pre-read safety test**

Add a test that monkeypatches `_scene_payload` to fail if called, redirects the configured scene COCO candidate under `datasets/base/test`, and asserts `DataValidationError` contains `evaluation-only`.

```python
def test_build_classifier_dataset_checks_scene_coco_before_read(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "datasets"
    forbidden = dataset_root / "base" / "test" / "instances_test.json"
    monkeypatch.setattr(
        "bakery_scanner.classifier_dataset._scene_coco_path",
        lambda root: forbidden,
    )
    monkeypatch.setattr(
        "bakery_scanner.classifier_dataset._scene_payload",
        lambda path: pytest.fail("scene COCO must not be read"),
    )
    with pytest.raises(DataValidationError, match="evaluation-only"):
        build_classifier_dataset(_config(dataset_root, "unsafe", "base"))
```

- [x] **Step 2: Run the focused test and confirm RED**

Run: `python -m pytest tests/test_classifier_dataset.py::test_build_classifier_dataset_checks_scene_coco_before_read -q`

Expected: failure because `_scene_payload()` runs before the safety guard.

- [x] **Step 3: Move safety validation before the first COCO read**

Resolve the registry, Base class directories, Incremental class directories when applicable, scene COCO, source images, and output root first. Call `assert_training_paths_safe` before `_scene_payload(scene_coco)`. Do not weaken the later complete-path validation.

- [x] **Step 4: Run dataset tests**

Run: `python -m pytest tests/test_classifier_dataset.py tests/test_classifier_data_cli.py -q`

Expected: all tests pass, including deterministic Base and Incremental replay validation.

- [x] **Step 5: Confirm the safety fix already exists on latest main**

```powershell
git show c0dbe327 -- src/bakery_scanner/classifier_dataset.py tests/test_classifier_dataset.py
```

---

### Task 2: Add Incremental config and phase metrics without changing Base context

**Files:**
- Modify: `src/bakery_scanner/classifier_training.py`
- Modify: `src/bakery_scanner/classifier_evaluation.py`
- Modify: `src/bakery_scanner/classifier_train_cli.py`
- Modify: `tests/test_classifier_training.py`
- Modify: `tests/test_classifier_evaluation.py`
- Modify: `tests/test_classifier_train_cli.py`
- Create: `configs/classifier/resnet18_incremental.yaml`

**Interfaces:**
- Produces: `IncrementalClassifierTrainingConfig` with `phase`, `base_checkpoint`, and `frozen_detector_checkpoint`.
- Produces: `load_classifier_experiment_config(path) -> ClassifierTrainingConfig | IncrementalClassifierTrainingConfig`.
- Produces: `evaluate_classifier_predictions(...)["phase"]` for 20-output runs.
- Preserves: `_config_payload(ClassifierTrainingConfig)` exactly, so the approved Base checkpoint context remains unchanged.

- [x] **Step 1: Write failing strict-config compatibility tests**

Test that the existing Base YAML still loads as `ClassifierTrainingConfig` with the exact legacy payload. Test an Incremental YAML with these exact fields:

```yaml
phase: incremental
dataset_root: datasets
source_classifier_run: incremental_seed42
output_root: runs/classifier
run_name: resnet18_incremental_seed42
architecture: resnet18
base_checkpoint: runs/classifier/resnet18_base_seed42/checkpoints/best.pt
frozen_detector_checkpoint: runs/detector/yolo11n_base_seed42/checkpoints/best.pt
image_size: 224
epochs: 30
batch_size: 64
seed: 42
device: "0"
patience: 5
workers: 8
learning_rate: 0.001
weight_decay: 0.0001
```

Reject missing, extra, empty, non-Incremental phase, CPU device, and test-path artifact values.

- [x] **Step 2: Write failing phase metric tests**

Use 20 outputs with correct and incorrect predictions in both ranges. Require:

```python
assert metrics["phase"]["base"] == {
    "sample_count": 2,
    "top1": 0.5,
    "macro_f1": expected_base_macro,
}
assert metrics["phase"]["incremental"]["sample_count"] == 2
```

For 15 outputs, require that `phase` contains only `base`. Other dimensions
retain the generic metric payload without phase groups for unit-level reuse.

- [x] **Step 3: Run the config and metric tests and confirm RED**

Run: `python -m pytest tests/test_classifier_training.py tests/test_classifier_evaluation.py tests/test_classifier_train_cli.py -q`

Expected: failures for missing Incremental config types/loader and phase metrics.

- [x] **Step 4: Implement the separate schema and grouped metrics**

Keep `ClassifierTrainingConfig` and its field order unchanged. Add a separate field order and serializer for `IncrementalClassifierTrainingConfig`. Dispatch only when the YAML contains `phase: incremental`; otherwise call the legacy loader.

Compute phase metrics from validated predictions and existing per-class F1 values:

```python
groups = {"base": range(15)}
if output_dimension == 20:
    groups["incremental"] = range(15, 20)
```

Top-1 uses only samples whose target is in the group. Macro F1 averages every registry class in the group, including a zero F1 class.

- [x] **Step 5: Route CLI train/evaluate through the experiment loader**

The CLI must print `Classifier phase: base` or `Classifier phase: incremental` before backend work. Preserve the existing command names and JSON/human output contracts.

- [x] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_classifier_evaluation.py tests/test_classifier_training.py tests/test_classifier_train_cli.py -q`

Expected: all focused tests pass and Base config serialization assertions remain unchanged.

- [x] **Step 7: Commit config and metric support**

```powershell
git add src/bakery_scanner/classifier_training.py src/bakery_scanner/classifier_evaluation.py src/bakery_scanner/classifier_train_cli.py tests/test_classifier_training.py tests/test_classifier_evaluation.py tests/test_classifier_train_cli.py configs/classifier/resnet18_incremental.yaml
git commit -m "feat(classifier): Incremental 설정과 단계별 지표를 추가한다"
```

---

### Task 3: Expand the Base checkpoint and record class balancing

**Files:**
- Modify: `src/bakery_scanner/classifier_training.py`
- Modify: `tests/test_classifier_training.py`

**Interfaces:**
- Produces: `_build_incremental_resnet18(base_checkpoint, output_dimension, checkpoint_context, image_size)`.
- Produces: `_balanced_class_statistics(train_samples, output_dimension) -> tuple[counts, weights]`.
- Changes backend input name from `pretrained_model` to `initial_model` while keeping Base behavior identical.

- [x] **Step 1: Write failing exact 15-to-20 expansion tests**

Create a valid 15-output checkpoint payload with distinctive backbone tensors and head rows. Build the 20-output model and assert:

```python
for key, tensor in base_model.state_dict().items():
    if not key.startswith("fc."):
        assert torch.equal(expanded.state_dict()[key], tensor)
assert torch.equal(expanded.fc.weight[:15], base_model.fc.weight)
assert torch.equal(expanded.fc.bias[:15], base_model.fc.bias)
assert expanded.fc.weight.shape[0] == 20
```

Add rejection tests for output dimension other than 20, malformed schema, Base output other than 15, registry SHA mismatch, first-15 mapping mismatch, and image-size mismatch.

- [x] **Step 2: Write failing balance-statistics tests**

For counts `[4, 2, 1]`, require weights `[7/(3*4), 7/(3*2), 7/(3*1)]`. Reject an output index outside range and any class with zero train support.

- [x] **Step 3: Run the new tests and confirm RED**

Run: `python -m pytest tests/test_classifier_training.py -q`

Expected: failures for missing expansion and balance functions.

- [x] **Step 4: Implement strict checkpoint expansion**

Reuse the existing checkpoint schema validator. Load a 15-output ResNet18 strictly, create the seeded 20-output model, copy non-head state and rows 0 through 14 under `torch.no_grad()`, then return the model plus initialization evidence:

```python
{
    "source_output_dimension": 15,
    "target_output_dimension": 20,
    "copied_base_rows": 15,
    "new_rows": 5,
}
```

- [x] **Step 5: Use one balancing implementation for metadata and loss**

Compute counts and weights before backend invocation, pass the immutable weights in backend arguments, and construct `CrossEntropyLoss` from those exact values. Record counts and weights in metadata.

- [x] **Step 6: Run classifier tests**

Run: `python -m pytest tests/test_classifier_training.py tests/test_classifier_evaluation.py -q`

Expected: all tests pass, including legacy `_build_resnet18` and Base training tests.

- [x] **Step 7: Commit expansion and weighting**

```powershell
git add src/bakery_scanner/classifier_training.py tests/test_classifier_training.py
git commit -m "feat(classifier): Base checkpoint를 20개 출력으로 확장한다"
```

---

### Task 4: Enforce the frozen detector and publish Incremental runs

**Files:**
- Modify: `src/bakery_scanner/classifier_training.py`
- Modify: `tests/test_classifier_training.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: validated detector `best.pt` and adjacent `metadata.json`.
- Produces: Incremental training metadata with detector before/after hashes and `detector_unchanged: true`.
- Produces: atomic 20-output run artifacts using the existing layout.

- [x] **Step 1: Write failing frozen-detector orchestration tests**

Require path safety before dataset validation, adjacent detector metadata hash/class-name verification, hash capture before backend work, hash capture after best-checkpoint evaluation, cleanup on mutation, and metadata fields:

```python
assert metadata["frozen_detector"] == {
    "checkpoint": str(detector_checkpoint.resolve()),
    "sha256_before": expected_hash,
    "sha256_after": expected_hash,
    "detector_unchanged": True,
}
```

Assert the backend call has no detector object, detector path, or detector tensor argument.

- [x] **Step 2: Write failing Incremental dataset and checkpoint tests**

Require phase `incremental`, output dimension 20, train support for all 20
classes, validation support for all five new classes, Base checkpoint SHA and
mapping validation, and complete run metadata. Reject a Base source run for
Incremental config and an Incremental source run for Base config.

- [x] **Step 3: Run orchestration tests and confirm RED**

Run: `python -m pytest tests/test_classifier_training.py tests/test_classifier_train_cli.py -q`

Expected: failures until Incremental dispatch and detector hashing exist.

- [x] **Step 4: Implement Incremental train/evaluate dispatch**

Share atomic publication and `_write_evaluation`, but select phase-specific initialization and provenance. Hash the frozen detector before dataset reads and after evaluation. On any mismatch, raise `DataValidationError` and remove staging output.

- [x] **Step 5: Update README with only implemented commands**

Document:

```powershell
bakery-classifier train --config configs/classifier/resnet18_incremental.yaml
bakery-classifier evaluate `
  --config configs/classifier/resnet18_incremental.yaml `
  --checkpoint runs/classifier/resnet18_incremental_seed42/checkpoints/best.pt
```

State that Base data is replayed, the detector is frozen, all metrics are train-side, and no test or POS result is reported.

- [x] **Step 6: Run focused and full verification**

Run:

```powershell
python -m pytest tests/test_classifier_training.py tests/test_classifier_evaluation.py tests/test_classifier_train_cli.py -q
python -m pytest -q
python -m compileall -q src tests
git diff --check
```

Expected: all commands succeed.

- [x] **Step 7: Commit orchestration and docs**

```powershell
git add src/bakery_scanner/classifier_training.py tests/test_classifier_training.py README.md docs/superpowers/plans/2026-07-20-incremental-classifier.md
git commit -m "feat(classifier): Incremental 학습과 detector 고정을 검증한다"
```

---

### Task 5: Execute, independently review, and merge the real Incremental baseline

**Files:**
- Runtime only: `datasets/derived/classifier/incremental_seed42/`
- Runtime only: `runs/classifier/resnet18_incremental_seed42/`
- Modify after measured results: `README.md`

**Interfaces:**
- Produces: approved real 20-output checkpoint and train-side metrics for later Incremental E2E and CPU benchmark stages.

- [x] **Step 1: Replay-validate the actual Incremental classifier dataset**

Run:

```powershell
bakery-classifier-data validate --dataset-root datasets --run-name incremental_seed42 --json
```

Require output dimension 20, all model indices 0 through 19, no test path, and deterministic replay equality.

- [x] **Step 2: Verify source artifacts before training**

Check the Base classifier checkpoint hash against its metadata and the detector checkpoint hash against its metadata. Record both hashes in the work log.

- [x] **Step 3: Execute the CUDA Incremental run once**

Run:

```powershell
bakery-classifier train --config configs/classifier/resnet18_incremental.yaml --json
```

Do not inspect any test result. If the command fails for an implementation defect, fix under TDD and restart to a new run name; do not overwrite a completed run.

- [x] **Step 4: Independently replay train-side metrics and provenance**

Recompute predictions/metrics from `best.pt`, compare the JSON payload exactly, verify first-15 mapping, 20-output shape, Base/new group metrics, class counts/weights, and unchanged detector hash.

- [x] **Step 5: Record measured train-side results**

Update README with the actual overall Top-1/Macro F1 and Base/new group values. Label them train-side and do not claim CPU/POS performance.

- [x] **Step 6: Final local verification**

Run full tests, compileall, diff-check, config safety checks, detector hash checks, and tracked-worktree status. Preserve ignored datasets and runs.

- [ ] **Step 7: Publish and merge through the repository workflow**

Commit any measured documentation update, push `codex/feat-incremental-classifier`, open a Korean Ready PR with full evidence, obtain a fresh independent review after the final diff, fix and re-review any finding, then have a separate merge agent squash merge and delete the remote branch.
