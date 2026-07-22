# Detector Candidate Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate and freeze a two-member YOLO26s candidate ensemble that reaches Base development OOF Recall at least 95% with zero false positives while recording its CPU cost.

**Architecture:** A strict ensemble config references two existing selected detector configs and immutable checkpoint/config SHA-256 values. Each member produces class-agnostic candidates at the frozen confidence floor; a pure merger validates image IDs and class 0, concatenates candidates in configured member order, then applies the frozen aspect filter and cross-member NMS before the existing detector evaluator computes metrics. A detector-only CPU benchmark loads the same frozen members on CPU and measures the complete sequential ensemble invocation including merge/postprocess.

**Tech Stack:** Python 3.11, PyYAML, Ultralytics YOLO, PyTorch CPU/CUDA, pytest.

## Global Constraints

- Never read `datasets/base/test`, `datasets/incremental/test`, or Base cycle holdout scene `0510` during configuration, evaluation, tuning, or benchmarking.
- Every detector member must emit only class index `0` (`bread`).
- Both members must use identical dataset root, validation scene, image size, device, confidence floor, operating confidence, NMS IoU, matching IoU, and maximum symmetric aspect ratio.
- Member order is configuration order and is recorded; candidate sorting and NMS remain deterministic.
- Checkpoint and selected-config SHA-256 values are verified before and after inference.
- Empty detections are valid and produce an empty result list.
- CPU benchmarking explicitly uses `device="cpu"`, eight threads, one warm-up per image, three measured repetitions per image, and records mean/P50/P95; warm-up is excluded.
- The benchmark is a current development-PC measurement, not a specific POS-device claim.

---

### Task 1: Strict ensemble config and deterministic candidate merger

**Files:**
- Create: `src/bakery_scanner/detector_ensemble.py`
- Create: `tests/test_detector_ensemble.py`

**Interfaces:**
- Produces: `DetectorEnsembleMember(config_path: str, config_sha256: str, checkpoint_path: str, checkpoint_sha256: str)`.
- Produces: `DetectorEnsembleConfig(dataset_root: str, output_root: str, run_name: str, members: tuple[DetectorEnsembleMember, ...], cpu_threads: int, cpu_warmups: int, cpu_repetitions: int)`.
- Produces: `load_detector_ensemble_config(path: str | Path) -> DetectorEnsembleConfig`.
- Produces: `merge_member_predictions(member_predictions: Sequence[Mapping[str, Sequence[Detection]]], image_ids: Sequence[str], postprocess: DetectorPostprocessConfig) -> dict[str, tuple[Detection, ...]]`.

- [ ] **Step 1: Write failing strict-loader and merger tests**

```python
def test_load_ensemble_config_rejects_one_member_and_invalid_sha(tmp_path: Path) -> None:
    # Write exact-schema YAML fixtures; require exactly two unique members and 64 lowercase hex hashes.

def test_merge_member_predictions_is_ordered_deterministic_and_handles_empty() -> None:
    # Supply overlapping class-0 detections in two mappings plus an empty image.
    # Assert highest-confidence cross-member box survives NMS and empty image stays empty.

def test_merge_member_predictions_rejects_image_or_class_mismatch() -> None:
    # Missing/extra image IDs and class_index != 0 must raise DataValidationError.
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_detector_ensemble.py -q`

Expected: collection failure because `bakery_scanner.detector_ensemble` does not exist.

- [ ] **Step 3: Implement the strict schema and merger**

```python
@dataclass(frozen=True, slots=True)
class DetectorEnsembleMember:
    config_path: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str

def merge_member_predictions(member_predictions, image_ids, postprocess):
    expected = tuple(image_ids)
    merged = {}
    for image_id in expected:
        candidates = [item for member in member_predictions for item in member[image_id]]
        merged[image_id] = filter_detections(candidates, postprocess)
    return merged
```

The loader accepts exactly these top-level fields: `dataset_root`, `output_root`, `run_name`, `members`, `cpu_threads`, `cpu_warmups`, `cpu_repetitions`. It rejects test path components using `assert_training_paths_safe`, duplicate member config/checkpoint paths, non-positive thread/repetition values, negative warm-ups, and any member count other than two.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_detector_ensemble.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/bakery_scanner/detector_ensemble.py tests/test_detector_ensemble.py
git commit -m "feat(detector): 후보 앙상블 설정과 병합기를 추가한다"
```

### Task 2: Provenance-bound ensemble evaluation

**Files:**
- Modify: `src/bakery_scanner/detector_ensemble.py`
- Modify: `tests/test_detector_ensemble.py`

**Interfaces:**
- Consumes: Task 1 config and merger.
- Produces: `DetectorEnsembleReport(output_dir, metadata_path, predictions_path, metrics_path, benchmark_path | None)` with `to_dict()`.
- Produces: `evaluate_detector_ensemble(config: DetectorEnsembleConfig, backend: DetectorBackend | None = None) -> DetectorEnsembleReport`.

- [ ] **Step 1: Write failing integration tests with a recording backend**

```python
def test_evaluate_ensemble_merges_members_writes_metrics_and_preserves_hashes(...):
    # Use two fixture selected configs/checkpoints and identical validation samples.
    # Return complementary detections from two backend calls.
    # Assert TP recovery, FP=0, member order/hashes in metadata, and unchanged files.

def test_evaluate_ensemble_rejects_threshold_validation_or_hash_drift(...):
    # Parameterize mismatched thresholds, validation image/label hashes, config SHA, and checkpoint SHA.

def test_evaluate_ensemble_accepts_normal_empty_predictions(...):
    # Both members return empty tuples; report prediction_count is zero.
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_detector_ensemble.py -q`

Expected: failures because `evaluate_detector_ensemble` is missing.

- [ ] **Step 3: Implement preparation, evaluation, and atomic artifacts**

The evaluator must:

1. Resolve every path and reject unsafe/test paths before reading member files.
2. Verify configured SHA-256 for each selected config and checkpoint.
3. Load both selected configs through `load_detector_training_config`.
4. Validate both YOLO datasets and require identical validation `image_path`, `image_sha256`, `label_path`, `label_sha256`, dimensions, and annotation counts.
5. Require the frozen inference arguments to match exactly across members.
6. Call the existing `DetectorBackend.predict` once per member with confidence floor `0.001`, NMS IoU `0.15`, image size `640`, and CUDA device `0`.
7. Merge with maximum symmetric aspect ratio `2.0`, evaluate at operating confidence `0.04796672239899635` and matching IoU `0.5`, and write deterministic `predictions.json`, `metrics.json`, and `metadata.json` in a staging directory before atomic publish.
8. Re-hash every config and checkpoint after inference and fail closed if any changed.

- [ ] **Step 4: Run focused detector tests**

Run: `python -m pytest tests/test_detector_ensemble.py tests/test_detector_training.py tests/test_detector_postprocess.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/bakery_scanner/detector_ensemble.py tests/test_detector_ensemble.py
git commit -m "feat(detector): 후보 앙상블 평가를 provenance에 결합한다"
```

### Task 3: CPU-only ensemble benchmark and CLI

**Files:**
- Modify: `src/bakery_scanner/detector_ensemble.py`
- Create: `src/bakery_scanner/detector_ensemble_cli.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_detector_ensemble.py`
- Create: `tests/test_detector_ensemble_cli.py`

**Interfaces:**
- Produces: `benchmark_detector_ensemble_cpu(config: DetectorEnsembleConfig, backend: DetectorEnsembleCpuBackend | None = None) -> Path`.
- Produces CLI commands `bakery-detector-ensemble evaluate --config <yaml> [--json]` and `bakery-detector-ensemble benchmark --config <yaml> [--json]`.

- [ ] **Step 1: Write failing benchmark and CLI tests**

```python
def test_cpu_benchmark_records_complete_ensemble_samples_and_excludes_warmup(...):
    # Fake backend asserts device='cpu', returns one timing per full two-member invocation.
    # Assert six images * three repetitions = 18 measured samples and six warm-ups.

def test_cpu_benchmark_rejects_non_cpu_backend_or_checkpoint_mutation(...):
    # GPU provider/device declaration and post-run SHA drift must fail.

def test_ensemble_cli_prints_selected_members_and_json_report(...):
    # Patch evaluator/benchmark and assert both paths, validation split, and metrics are printed.
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_detector_ensemble.py tests/test_detector_ensemble_cli.py -q`

Expected: failures because benchmark and CLI interfaces are missing.

- [ ] **Step 3: Implement CPU benchmark and CLI**

The real CPU backend loads both YOLO checkpoints, calls each sequentially with `device="cpu"`, `imgsz=640`, `conf=0.001`, and `iou=0.15`, then includes cross-member merge/postprocess in one scene timing. Call `torch.set_num_threads(8)`. Exclude one warm-up per image and record 18 measured samples with mean/P50/P95, dependency versions, CPU/platform, member/config/checkpoint hashes, input size, thread count, image count, and the non-POS disclaimer.

- [ ] **Step 4: Run focused tests and CLI help**

Run: `python -m pytest tests/test_detector_ensemble.py tests/test_detector_ensemble_cli.py -q`

Run: `python -m bakery_scanner.detector_ensemble_cli --help`

Expected: tests pass and help lists `evaluate` and `benchmark`.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/bakery_scanner/detector_ensemble.py src/bakery_scanner/detector_ensemble_cli.py pyproject.toml tests/test_detector_ensemble.py tests/test_detector_ensemble_cli.py
git commit -m "perf(detector): 후보 앙상블 CPU benchmark를 추가한다"
```

### Task 4: Freeze seed 42+44 fold configs and verify actual results

**Files:**
- Create: `configs/detector_ensemble/yolo26s_s42_s44_val0503.yaml`
- Create: `configs/detector_ensemble/yolo26s_s42_s44_val0509.yaml`
- Modify: `tests/test_detector_ensemble.py`

**Interfaces:**
- Consumes: Tasks 1-3 CLI and evaluator.
- Produces: two immutable fold configs and ignored run artifacts under `runs/detector_ensemble/base_v2/`.

- [ ] **Step 1: Add a failing repository-config loading test**

```python
def test_repository_ensemble_configs_are_frozen_and_loadable() -> None:
    # Load both configs and assert ordered seeds 42,44, exact hashes, threads=8, warmups=1, repetitions=3.
```

- [ ] **Step 2: Write exact fold configs**

Use the following immutable member hashes:

- val0503 seed42 checkpoint `370986e06f6bd9b60bc389937e6f74b986d54f1ef2235c8728817a466dcead92`, config `f176a807c2e10332af41eb9aeab44229a94d66b31a5374165230ec3b29b395d9`.
- val0503 seed44 checkpoint `3c611b1124e7de9e7bd5ac7141a3e155f4ae45e1e3d1c57ee426ff5dcef9f7f6`, config `62a0a86548e483f92e197d92d52a1bbe395f847ace1c1d4025b704277c03ef2b`.
- val0509 seed42 checkpoint `b11b56600f6920e9015c31d0fe48af91f7773369cecb3d8c15d3338409b880f6`, config `9960b18cc607c6dc55dfc0b4cf6722571bbe0b28395fab972ff98d5c85255ccc`.
- val0509 seed44 checkpoint `b97feb4fa0a5df6798ccff90e39f5e79a9478b263b7a2bfcdd9fb533db4acb85`, config `9d584b7a0f8011deb130240a0b2787c79e153fe8b0c6a0ad2147efaf7d5d1f75`.

- [ ] **Step 3: Run the repository-config test**

Run: `python -m pytest tests/test_detector_ensemble.py -q`

Expected: all tests pass.

- [ ] **Step 4: Run actual CUDA evaluation without test or holdout access**

```powershell
python -m bakery_scanner.detector_ensemble_cli evaluate --config configs/detector_ensemble/yolo26s_s42_s44_val0503.yaml --json
python -m bakery_scanner.detector_ensemble_cli evaluate --config configs/detector_ensemble/yolo26s_s42_s44_val0509.yaml --json
```

Expected per fold: GT 15, TP 15, FP 0, Recall 1.0. Combined OOF: GT 30, TP 30, FP 0.

- [ ] **Step 5: Run the actual CPU benchmark**

```powershell
python -m bakery_scanner.detector_ensemble_cli benchmark --config configs/detector_ensemble/yolo26s_s42_s44_val0503.yaml --json
python -m bakery_scanner.detector_ensemble_cli benchmark --config configs/detector_ensemble/yolo26s_s42_s44_val0509.yaml --json
```

Expected: CPU-only device/provider evidence, nine measured samples per fold, and combined 18-sample summary computable from recorded raw timings.

- [ ] **Step 6: Run final verification**

Run: `python -m pytest -q`

Run: `git diff --check`

Expected: full suite passes and no whitespace errors.

- [ ] **Step 7: Commit Task 4**

```powershell
git add configs/detector_ensemble/yolo26s_s42_s44_val0503.yaml configs/detector_ensemble/yolo26s_s42_s44_val0509.yaml tests/test_detector_ensemble.py
git commit -m "experiment(detector): Base 후보 앙상블 구성을 동결한다"
```
