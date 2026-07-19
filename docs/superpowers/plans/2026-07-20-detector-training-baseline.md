# Detector Training Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the first reproducible single-class YOLO11n bread-detector baseline using only train-authorized data, producing validated checkpoints, predictions, train-side metrics, and environment metadata.

**Architecture:** Convert the independently validated detector COCO run into a separately validated YOLO run, then pass it through a small training-backend boundary. Keep metric calculation backend-independent so Ultralytics predictions and test fixtures share one contract. Publish converted datasets and completed model runs atomically, and fail closed on evaluation-only paths before backend work begins.

**Tech Stack:** Python 3.11, Pillow 12.x, PyYAML 6.x, PyTorch 2.13, Ultralytics 8.4, pytest 9, CUDA 13, NVIDIA GeForce RTX 5080.

## Global Constraints

- `datasets/base/test` and `datasets/incremental/test` are evaluation-only and must never affect training, early stopping, thresholds, augmentation, checkpoints, hyperparameters, or model selection.
- `datasets/base/val` is train-side scene data despite its physical name.
- Detector labels contain exactly one class, `bread`, with YOLO class index `0`.
- Original COCO `category_id` is provenance only; phase lookup uses `datasets/class_registry.json` and never treats `category_id` as `model_index`.
- Existing images, COCO JSON, synthetic runs, detector runs, and `class_registry.json` are immutable inputs.
- GPU is allowed for training; this plan does not implement or claim a CPU benchmark.
- Entry points print and validate concrete train and validation paths before work begins.
- Missing/extra files, malformed boxes, manifest drift, output-dimension mismatch, and missing metadata are errors.
- Empty normal images and empty prediction lists are valid.
- Fixed baseline: `yolo11n.pt`, image size 640, 50 epochs, batch 16, seed 42, CUDA device 0, patience 10, AP floor 0.001, operating confidence 0.25, NMS IoU 0.7, matching IoU 0.5.

## File Map

- Create `src/bakery_scanner/yolo_dataset.py`: COCO-to-YOLO conversion, validation, hashing, atomic publication.
- Create `src/bakery_scanner/detector_evaluation.py`: contracts, IoU matching, AP50, Recall, miss rate, group aggregation.
- Create `src/bakery_scanner/detector_training.py`: strict config, backend protocol, Ultralytics adapter, metadata, orchestration.
- Create `src/bakery_scanner/detector_train_cli.py`: `train` and `evaluate` commands.
- Create `configs/detector/yolo11n_base.yaml`.
- Create `tests/test_yolo_dataset.py`, `tests/test_detector_evaluation.py`, `tests/test_detector_training.py`, `tests/test_detector_train_cli.py`.
- Modify `pyproject.toml`, `.gitignore`, and `README.md`.

---

### Task 1: Validated YOLO Dataset Conversion

**Files:**
- Create: `src/bakery_scanner/yolo_dataset.py`
- Create: `tests/test_yolo_dataset.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `validate_detector_dataset(dataset_root, source_run)` and the source detector run.
- Produces: `YoloDatasetReport`, `build_yolo_dataset(dataset_root, source_run, run_name, overwrite=False)`, `validate_yolo_dataset(dataset_root, run_name)`.

- [ ] **Step 1: Write the failing public-contract test**

Create a tiny source run using existing synthetic and detector builders, then add this assertion:

```python
def test_build_yolo_dataset_converts_boxes_and_records_provenance(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "yolo-fixture")
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))

    assert report.image_count == report.train_image_count + report.validation_image_count
    assert manifest["manifest_version"] == 1
    assert manifest["source"]["run_name"] == source_run
    assert manifest["source"]["manifest_sha256"]
    sample = manifest["samples"][0]
    fields = (report.output_dir / sample["label_path"]).read_text().splitlines()[0].split()
    assert fields[0] == "0"
    assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_yolo_dataset.py::test_build_yolo_dataset_converts_boxes_and_records_provenance -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'bakery_scanner.yolo_dataset'`.

- [ ] **Step 3: Implement the public types and minimal conversion**

```python
CONVERTER_VERSION = "1.0.0"
MANIFEST_VERSION = 1

@dataclass(frozen=True, slots=True)
class YoloDatasetReport:
    output_dir: Path
    manifest_path: Path
    image_count: int
    annotation_count: int
    train_image_count: int
    validation_image_count: int
    converter_version: str = CONVERTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "converter_version": self.converter_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "train_image_count": self.train_image_count,
            "validation_image_count": self.validation_image_count,
        }

```

Implement public functions `build_yolo_dataset(dataset_root: str | Path, source_run: str, run_name: str, overwrite: bool = False) -> YoloDatasetReport` and `validate_yolo_dataset(dataset_root: str | Path, run_name: str) -> YoloDatasetReport`. The builder calls `validate_detector_dataset` first, converts `[x, y, width, height]` to normalized `0 x_center y_center width height`, copies images, creates empty labels for empty scenes, and records source/output hashes and original annotations.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_yolo_dataset.py::test_build_yolo_dataset_converts_boxes_and_records_provenance -v`

Expected: PASS.

- [ ] **Step 5: Add failing validation and atomicity tests**

Cover deterministic output, empty labels, test-path provenance, invalid labels, tampering, missing/extra files, existing-run refusal, and rollback. The tampering case is exact:

```python
def test_validate_yolo_dataset_rejects_tampered_label(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "tampered")
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    label_path = report.output_dir / manifest["samples"][0]["label_path"]
    label_path.write_text("0 0.5 0.5 2.0 0.5\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="label|hash|bounds"):
        validate_yolo_dataset(dataset_root, "tampered")
```

- [ ] **Step 6: Verify RED, implement strict validation, then verify GREEN**

Run before implementation: `python -m pytest tests/test_yolo_dataset.py -q`

Expected: new validation tests FAIL.

Implement strict manifest keys, source revalidation/hash comparison, exact inventory, decoded dimensions, five finite label fields, class zero, positive normalized boxes within image bounds, `data.yaml` agreement, and staging/backup/restore publication matching `detector_dataset.py`.

Run after implementation: `python -m pytest tests/test_yolo_dataset.py tests/test_detector_dataset.py tests/test_safety.py -q`

Expected: all selected tests PASS.

- [ ] **Step 7: Ignore generated artifacts and commit**

Add:

```gitignore
datasets/derived/yolo/
runs/detector/
```

Run:

```powershell
git add .gitignore src/bakery_scanner/yolo_dataset.py tests/test_yolo_dataset.py
git commit -m "feat: add validated YOLO dataset conversion"
```

---

### Task 2: Backend-Independent Detector Metrics

**Files:**
- Create: `src/bakery_scanner/detector_evaluation.py`
- Create: `tests/test_detector_evaluation.py`

**Interfaces:**
- Produces: `EvaluationObject`, `EvaluationImage`, `Detection`, `EvaluationThresholds`, `evaluate_detector_predictions(images, predictions, thresholds)`.

- [ ] **Step 1: Write the failing global-metric test**

```python
def test_evaluate_detector_predictions_computes_global_metrics() -> None:
    images = (
        EvaluationImage("scene_e_1.jpg", "easy", (EvaluationObject((0, 0, 10, 10), 2, "base"),)),
        EvaluationImage("scene_h_1.jpg", "hard", (EvaluationObject((20, 20, 30, 30), 4, "base"),)),
    )
    predictions = {
        "scene_e_1.jpg": (Detection((0, 0, 10, 10), 0.9), Detection((0, 0, 10, 10), 0.8)),
        "scene_h_1.jpg": (),
    }
    metrics = evaluate_detector_predictions(images, predictions, EvaluationThresholds())

    assert metrics["global"]["ground_truth_count"] == 2
    assert metrics["global"]["true_positive_count"] == 1
    assert metrics["global"]["recall"] == pytest.approx(0.5)
    assert metrics["global"]["miss_rate"] == pytest.approx(0.5)
    assert 0.0 < metrics["global"]["ap50"] <= 1.0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_detector_evaluation.py::test_evaluate_detector_predictions_computes_global_metrics -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement exact contracts and global metrics**

```python
@dataclass(frozen=True, slots=True)
class EvaluationObject:
    bbox_xyxy: tuple[float, float, float, float]
    category_id: int
    phase: str

@dataclass(frozen=True, slots=True)
class EvaluationImage:
    image_id: str
    difficulty: str | None
    objects: tuple[EvaluationObject, ...]

@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_index: int = 0

@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    confidence_floor: float = 0.001
    operating_confidence: float = 0.25
    nms_iou: float = 0.7
    matching_iou: float = 0.5

```

Implement `evaluate_detector_predictions(images: Sequence[EvaluationImage], predictions: Mapping[str, Sequence[Detection]], thresholds: EvaluationThresholds) -> dict[str, Any]`. Validate finite values, unique image IDs, exact prediction-key coverage, class zero, and xyxy geometry. Sort detections by confidence with deterministic tie-breakers, greedily match to one unmatched object per image, and calculate AP50 from 101 interpolated recall points.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_detector_evaluation.py::test_evaluate_detector_predictions_computes_global_metrics -v`

Expected: PASS.

- [ ] **Step 5: Add failing group/empty tests and implement aggregation**

```python
def test_phase_without_objects_uses_null_metrics() -> None:
    images = (EvaluationImage("scene_m_1.jpg", "medium", (EvaluationObject((1, 1, 5, 5), 2, "base"),)),)
    metrics = evaluate_detector_predictions(images, {"scene_m_1.jpg": ()}, EvaluationThresholds())
    assert metrics["phase"]["incremental"] == {
        "sample_count": 0,
        "ground_truth_count": 0,
        "true_positive_count": 0,
        "recall": None,
        "miss_rate": None,
    }
```

Run before implementation: `python -m pytest tests/test_detector_evaluation.py -q`

Expected: group tests FAIL.

Implement easy/medium/hard grouping, Base/Incremental grouping, normal empty images, and `null` metrics for groups with no objects.

Run after implementation: `python -m pytest tests/test_detector_evaluation.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/bakery_scanner/detector_evaluation.py tests/test_detector_evaluation.py
git commit -m "feat: add detector validation metrics"
```

---

### Task 3: Strict Configuration and Training Orchestration

**Files:**
- Create: `src/bakery_scanner/detector_training.py`
- Create: `tests/test_detector_training.py`

**Interfaces:**
- Produces: `DetectorTrainingConfig`, `BackendTrainingResult`, `DetectorBackend`, `DetectorTrainingReport`, `DetectorEvaluationReport`, `load_detector_training_config`, `train_detector`, `evaluate_detector_checkpoint`, `UltralyticsBackend`.

- [ ] **Step 1: Write failing strict-config tests**

```python
def test_load_detector_training_config_accepts_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.yaml"
    path.write_text(BASELINE_YAML, encoding="utf-8")
    config = load_detector_training_config(path)
    assert config.model == "yolo11n.pt"
    assert config.image_size == 640
    assert config.device == "0"
    assert config.thresholds == EvaluationThresholds()
```

Also parameterize unknown keys, booleans used as integers, unsafe run names, invalid thresholds, and non-CUDA device values.

- [ ] **Step 2: Verify RED, implement strict config, verify GREEN**

Run before: `python -m pytest tests/test_detector_training.py -k config -q`

Expected: FAIL because the module does not exist.

Implement:

```python
@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    dataset_root: str
    source_detector_run: str
    yolo_run_name: str
    output_root: str
    run_name: str
    model: str
    image_size: int
    epochs: int
    batch_size: int
    seed: int
    device: str
    patience: int
    workers: int
    thresholds: EvaluationThresholds
```

Implement `load_detector_training_config(path: str | Path) -> DetectorTrainingConfig`. Reject every unknown/missing key, require positive integer sizes, nonnegative patience/workers, device exactly `"0"`, safe run names, and valid threshold ordering.

Run after: `python -m pytest tests/test_detector_training.py -k config -q`

Expected: config tests PASS.

- [ ] **Step 3: Write failing orchestration tests with a recording backend**

```python
class RecordingBackend:
    def cuda_available(self, device: str) -> bool:
        return True

    def train(self, *, model: str, data_yaml: Path, output_dir: Path,
              arguments: Mapping[str, object]) -> BackendTrainingResult:
        best = output_dir / "best.pt"
        last = output_dir / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        return BackendTrainingResult(best, last, ("bread",), Path(model))

    def predict(self, *, checkpoint: Path, image_paths: Sequence[Path],
                confidence_floor: float, nms_iou: float, image_size: int,
                device: str) -> Mapping[str, Sequence[Detection]]:
        return {path.name: () for path in image_paths}
```

Assert exact backend arguments, required metadata, checkpoint hashes, CUDA gating, existing-run refusal, atomic publication, and failure cleanup.

- [ ] **Step 4: Verify RED and implement orchestration contracts**

Run before: `python -m pytest tests/test_detector_training.py -k "train or evaluate" -q`

Expected: orchestration tests FAIL.

Implement:

```python
@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    best_checkpoint: Path
    last_checkpoint: Path
    class_names: tuple[str, ...]
    pretrained_checkpoint: Path

class DetectorBackend(Protocol):
    def cuda_available(self, device: str) -> bool:
        raise NotImplementedError

    def train(self, *, model: str, data_yaml: Path, output_dir: Path,
              arguments: Mapping[str, object]) -> BackendTrainingResult:
        raise NotImplementedError

    def predict(self, *, checkpoint: Path, image_paths: Sequence[Path],
                confidence_floor: float, nms_iou: float, image_size: int,
                device: str) -> Mapping[str, Sequence[Detection]]:
        raise NotImplementedError
```

Implement `train_detector(config: DetectorTrainingConfig, backend: DetectorBackend | None = None) -> DetectorTrainingReport` and `evaluate_detector_checkpoint(config: DetectorTrainingConfig, checkpoint: str | Path, backend: DetectorBackend | None = None, output_dir: str | Path | None = None) -> DetectorEvaluationReport`. Training validates safe paths and the YOLO run, stages under `runs/detector/.<run>.tmp-<uuid>`, verifies `("bread",)`, normalizes predictions, writes config/metadata/predictions/metrics with hashes, and publishes only after complete validation.

- [ ] **Step 5: Implement the Ultralytics adapter**

Lazy-import `torch` and `ultralytics`. Call `YOLO(model).train` with `data=str(data_yaml)`, `imgsz=arguments["image_size"]`, `epochs=arguments["epochs"]`, `batch=arguments["batch_size"]`, `seed=arguments["seed"]`, `device=arguments["device"]`, `patience=arguments["patience"]`, `workers=arguments["workers"]`, `deterministic=True`, `project=str(output_dir)`, `name="backend"`, and `exist_ok=False`. Normalize `boxes.xyxy`, `boxes.conf`, and `boxes.cls` into `Detection`; preserve empty tuples.

- [ ] **Step 6: Verify GREEN and commit**

Run: `python -m pytest tests/test_detector_training.py tests/test_detector_evaluation.py tests/test_yolo_dataset.py -q`

Expected: all selected tests PASS without a GPU training job.

```powershell
git add src/bakery_scanner/detector_training.py tests/test_detector_training.py
git commit -m "feat: orchestrate detector training and evaluation"
```

---

### Task 4: CLI, Baseline Configuration, and Dependencies

**Files:**
- Create: `src/bakery_scanner/detector_train_cli.py`
- Create: `tests/test_detector_train_cli.py`
- Create: `configs/detector/yolo11n_base.yaml`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `bakery-detector train --config <path>` and `bakery-detector evaluate --config <path> --checkpoint <path>`, both with `--json`.

- [ ] **Step 1: Write failing CLI tests**

Patch only `train_detector`/`evaluate_detector_checkpoint`. Construct report dataclasses directly and assert human split/path output, JSON-only output, and exit code 1 for `DataValidationError`.

```python
def test_train_cli_returns_failure_for_invalid_config(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("dataset_root: datasets\n", encoding="utf-8")
    exit_code = detector_train_cli.main(["train", "--config", str(path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed" in captured.err.lower()
```

- [ ] **Step 2: Verify RED and implement CLI**

Run before: `python -m pytest tests/test_detector_train_cli.py -q`

Expected: FAIL because the CLI module does not exist.

Use `argparse`, catch `DataValidationError`, serialize `report.to_dict()` as sorted UTF-8 JSON, and ensure JSON mode emits JSON only.

- [ ] **Step 3: Add the exact baseline YAML**

```yaml
dataset_root: datasets
source_detector_run: base_seed42_detector_origin_aware
yolo_run_name: base_seed42_detector_origin_aware_yolo
output_root: runs/detector
run_name: yolo11n_base_seed42
model: yolo11n.pt
image_size: 640
epochs: 50
batch_size: 16
seed: 42
device: "0"
patience: 10
workers: 8
thresholds:
  confidence_floor: 0.001
  operating_confidence: 0.25
  nms_iou: 0.7
  matching_iou: 0.5
```

- [ ] **Step 4: Declare runtime dependencies and console script**

```toml
dependencies = [
  "Pillow>=12,<13",
  "PyYAML>=6,<7",
  "ultralytics>=8.4,<9",
]

[project.scripts]
bakery-audit = "bakery_scanner.cli:main"
bakery-synthetic = "bakery_scanner.synthetic_cli:main"
bakery-detector-data = "bakery_scanner.detector_cli:main"
bakery-detector = "bakery_scanner.detector_train_cli:main"
```

- [ ] **Step 5: Verify GREEN, packaging, and commit**

Run:

```powershell
python -m pytest tests/test_detector_train_cli.py -q
python -m pip install -e ".[test]"
bakery-detector --help
```

Expected: tests PASS, install succeeds, help lists `train` and `evaluate`.

```powershell
git add pyproject.toml configs/detector/yolo11n_base.yaml src/bakery_scanner/detector_train_cli.py tests/test_detector_train_cli.py
git commit -m "feat: expose detector baseline commands"
```

---

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Verify: `AGENTS.md`
- Verify: `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`
- Verify: `docs/superpowers/specs/2026-07-20-detector-training-baseline-design.md`

- [ ] **Step 1: Document only implemented commands**

```powershell
bakery-detector train --config configs/detector/yolo11n_base.yaml
bakery-detector evaluate `
  --config configs/detector/yolo11n_base.yaml `
  --checkpoint runs/detector/yolo11n_base_seed42/checkpoints/best.pt
```

Explain YOLO derivative creation/validation, train-side-only checkpoint selection, completed run files, and that results are neither test scores nor POS benchmarks.

- [ ] **Step 2: Verify policy and documentation consistency**

Run: `rg -n "bakery-detector|base/test|incremental/test|CPU|POS" README.md AGENTS.md docs/superpowers/specs`

Expected: commands exist and every test path remains evaluation-only.

- [ ] **Step 3: Run the full suite and whitespace check**

Run:

```powershell
python -m pytest -q
git diff --check
```

Expected: the prior 103 tests plus all new tests PASS; no whitespace errors.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md
git commit -m "docs: describe detector baseline workflow"
```

---

### Task 6: Produce and Validate the Actual GPU Baseline

**Files:**
- Generate, ignored: `datasets/derived/yolo/base_seed42_detector_origin_aware_yolo/`
- Generate, ignored: `runs/detector/yolo11n_base_seed42/`
- Do not modify tracked source in this task.

- [ ] **Step 1: Verify runtime before training**

Run:

```powershell
git status --short --branch
python -m pytest -q
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: clean tracked worktree, tests PASS, CUDA is `True`, GPU is RTX 5080.

- [ ] **Step 2: Train the baseline**

Run: `bakery-detector train --config configs/detector/yolo11n_base.yaml`

Expected: it prints detector source, YOLO train/validation paths, model, GPU, and output path; completes with both checkpoints and validation metrics; never references a test split.

- [ ] **Step 3: Independently re-evaluate best.pt**

Run:

```powershell
bakery-detector evaluate `
  --config configs/detector/yolo11n_base.yaml `
  --checkpoint runs/detector/yolo11n_base_seed42/checkpoints/best.pt `
  --json
```

Expected: JSON has `status: ok`, `split: validation`, AP50, Recall, miss rate, difficulty records, Base metrics, and Incremental object count 0 with `null` metrics.

- [ ] **Step 4: Validate generated data and source immutability**

Run:

```powershell
python -c "from bakery_scanner.yolo_dataset import validate_yolo_dataset; print(validate_yolo_dataset('datasets', 'base_seed42_detector_origin_aware_yolo').to_dict())"
python -m bakery_scanner.detector_cli validate --dataset-root datasets --run-name base_seed42_detector_origin_aware --json
git status --short --branch
```

Expected: both validations PASS and no tracked input/source file changed.

- [ ] **Step 5: Final verification and handoff**

Run:

```powershell
python -m pytest -q
git diff --check
git status --short --branch
```

Expected: full suite PASS, no whitespace errors, tracked worktree clean. Report actual train-side metrics, checkpoint/run paths and hashes, package versions, duration, and the three-image validation limitation. Do not call it a test score or POS benchmark.

---

## Plan Self-Review Checklist

- Spec coverage: Tasks 1-6 cover conversion, validation, metrics, training, evaluation, CLI, metadata, documentation, and actual GPU execution.
- Test isolation: data entry points validate safety before backend/output work; executable commands never name evaluation-only data.
- Type consistency: `EvaluationThresholds`, `Detection`, `DetectorTrainingConfig`, backend methods, and report names match across tasks.
- Artifact separation: YOLO derivatives stay under `datasets/derived/yolo`; checkpoints/results stay under ignored `runs/detector`.
- Placeholder scan: no deferred behavior or unnamed error handling remains.
