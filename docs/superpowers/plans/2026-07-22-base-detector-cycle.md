# Base Detector Cycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the Base-v2 development-only detector cycle that retrains the YOLO11n baseline and compares YOLO26s/YOLO26m over two real-scene folds and seeds 42/43/44 without using the cycle holdout or either evaluation-only test tree.

**Architecture:** The frozen `base_v2` manifest remains the only assignment authority. A cycle-aware synthetic entry point uses only the two development backgrounds, and a backwards-compatible detector dataset mode assigns one development scene group to train and the other to validation while keeping every synthetic scene train-only. An experiment runner reuses the existing YOLO training backend, aggregates the two fold predictions into six-image/30-object out-of-fold metrics, selects the highest confidence threshold that preserves 30/30 Recall, and records detector-only CPU mean/P50/P95 before ranking candidates.

**Tech Stack:** Python 3.11, PyYAML 6.x, Pillow 12.x, PyTorch 2.13, torchvision 0.28, Ultralytics 8.4.x, CUDA GPU training, CPU-only detector benchmark, pytest 9.x.

## Global Constraints

- Work from the checked-in `configs/base_cycle/base_v2.yaml` and the immutable `datasets/derived/base_cycle/base_v2/` artifact.
- Never read, list, stat, resolve, decode, train on, mine from, calibrate on, or select with `datasets/base/test` or `datasets/incremental/test`.
- Never pass cycle-holdout scene `0510`, its three images, or `tray_wood_white_surface.png` to synthetic generation, detector dataset generation, training, prediction, threshold selection, or CPU benchmark.
- Automated Base-cycle integrity validation may verify hashes/dimensions as already approved, but this phase emits no holdout pixels, predictions, scores, or metrics.
- Use only development scene IDs `0503` and `0509`; fold `val0503` trains on `0509`, and fold `val0509` trains on `0503`.
- Use exactly seeds `42`, `43`, and `44` for the completed comparison.
- Use only class mapping `{0: "bread"}` in every detector/YOLO dataset, checkpoint, prediction, and metric artifact.
- Generate a new synthetic run per seed from the two development backgrounds. Do not reuse `base_seed42`, because it was generated from a pool containing the holdout background.
- Synthetic scenes are train-only. Validation contains exactly one complete real e/m/h scene group and 15 GT objects per fold.
- Aggregate both folds before selecting a threshold. Each model/seed must cover six unique validation images and exactly 30 GT objects.
- A detector model/seed is Recall-eligible only if some threshold at or above confidence floor `0.001` yields 30/30 Recall at IoU `0.5`.
- Select the highest eligible threshold, which removes the most lower-confidence proposals while preserving 30/30 Recall. Record FP, prediction count, AP50, miss rate, and easy/medium/hard Recall at that threshold.
- A model is eligible for the default cascade only when all three seeds meet 30/30 Recall. Rank eligible models by total OOF FP, development-background FP, worst-seed AP50, detector CPU P95, then model order `yolo11n`, `yolo26s`, `yolo26m`.
- GPU is allowed only for training/prediction generation. CPU benchmark explicitly uses `device="cpu"`; warm-up samples are excluded.
- Record Python, OS/architecture, CPU, GPU, CUDA, PyTorch, torchvision, Ultralytics, Pillow, PyYAML, input size, batch size, worker/thread count, warm-up count, repetitions, and every model/config/checkpoint/source SHA-256.
- Never overwrite a completed synthetic, detector dataset, YOLO dataset, training, prediction, benchmark, or summary run. A changed configuration uses a new run name.
- Keep original images, COCO JSON, `datasets/class_registry.json`, Base-cycle artifacts, and existing project checkpoints unchanged.
- README gains commands only after the corresponding entry point exists and CLI tests pass.
- This plan is implemented from updated `main` on `codex/feat-base-detector-cycle`; implementation is not committed to this documentation branch or directly to `main`.

---

## File Map

- Modify `src/bakery_scanner/safety.py`: make the shared training-path guard lexical-first and link-safe without resolving evaluation roots.
- Modify `tests/test_safety.py`: prove a successful safe call never touches evaluation subtrees and dot-segment/link aliases fail before target access.
- Modify `src/bakery_scanner/synthetic.py`: add explicit-background generation while preserving the existing directory API and manifest format.
- Modify `tests/test_synthetic.py`: validate exact background allowlists and replay compatibility.
- Modify `src/bakery_scanner/detector_dataset.py`: add backwards-compatible `base_cycle_fold` assignment mode and strict cycle provenance validation.
- Modify `tests/test_detector_dataset.py`: cover both folds, train-only synthetic samples, holdout exclusion, background authority, and old-run compatibility.
- Create `src/bakery_scanner/detector_cycle.py`: strict experiment config, pretrained preparation, fold preparation/training orchestration, OOF threshold aggregation, CPU benchmark, and immutable summary.
- Create `src/bakery_scanner/detector_cycle_cli.py`: `prepare-weights`, `prepare`, `run`, `summarize`, and `validate-summary` commands.
- Create `tests/test_detector_cycle.py`: matrix, orchestration, threshold, ranking, hash, CPU, and fail-closed tests with fake backends.
- Create `tests/test_detector_cycle_cli.py`: CLI routing, filters, JSON, and error exit tests.
- Create `configs/detector_cycle/base_v2.yaml`: checked-in three-model/two-fold/three-seed experiment matrix.
- Modify `pyproject.toml`: register `bakery-detector-cycle` only after its CLI exists.
- Modify `README.md`: document only implemented detector-cycle commands, selection gates, and the fact that cycle holdout is not evaluated here.

---

### Task 1: Shared Lexical-First Training Path Safety

**Files:**
- Modify: `src/bakery_scanner/safety.py`
- Modify: `tests/test_safety.py`

**Interfaces:**
- Consumes: arbitrary configured paths and a physical dataset root.
- Produces: unchanged public `assert_training_paths_safe(paths, dataset_root) -> tuple[Path, ...]`, now guaranteed not to touch evaluation subtrees and to reject link/junction escapes.

- [ ] **Step 1: Write evaluation-access guard tests**

Add helpers that wrap `Path.resolve`, `Path.stat`, `Path.lstat`, `Path.iterdir`, and `Path.open`; they delegate for safe paths and raise `AssertionError` if normalized lexical parts contain `datasets/base/test` or `datasets/incremental/test`.

```python
def test_safe_training_paths_never_touch_evaluation_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset_tree(tmp_path)
    touched: list[str] = []
    _guard_evaluation_access(monkeypatch, touched)

    resolved = assert_training_paths_safe(
        ["base/val/instances_val.json", "../runs/detector_cycle"], root
    )

    assert resolved[0] == (root / "base/val/instances_val.json").resolve()
    assert touched == []


@pytest.mark.parametrize(
    "value",
    [
        "base/test",
        "base/x/../test/instances.json",
        "incremental/./test",
        "datasets/base/test",
    ],
)
def test_training_paths_reject_lexical_test_alias_before_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    root = _dataset_tree(tmp_path)
    touched: list[str] = []
    _guard_evaluation_access(monkeypatch, touched)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        assert_training_paths_safe([value], root)
    assert touched == []
```

Add a supported symlink/junction test whose safe-looking configured component points to `base/test`; the guard must reject the link via `lstat`/reparse metadata without calling target-following `stat` or `resolve`.

- [ ] **Step 2: Run the new tests and verify red**

Run:

```powershell
python -m pytest tests/test_safety.py -q
```

Expected: the successful-call guard observes the current forbidden-root `resolve` calls, and the dot-segment/link cases fail.

- [ ] **Step 3: Replace the guard implementation**

Keep the public signature `assert_training_paths_safe(paths, dataset_root) -> tuple[Path, ...]` and implement this exact order:

1. Convert `dataset_root` to an absolute path without calling `resolve()`.
2. For each configured value, retain its original-case path components for path construction and build a second case-folded component tuple only for comparisons.
3. Lexically collapse `.` and `..`; reject a `..` that would escape the configured lexical base.
4. If the value is dataset-relative, reject the case-insensitive prefixes `base/test` and `incremental/test` before any `stat`, `lstat`, `exists`, `is_*`, or `resolve` call on that candidate.
5. If the value is absolute and lies lexically under `dataset_root`, perform the same relative-prefix rejection before filesystem access.
6. Walk only the already approved candidate component chain with `lstat`. Reject symlinks and Windows reparse points using `getattr(metadata, "st_file_attributes", 0)`; do not follow their targets.
7. Resolve the approved candidate once, then repeat the relative-prefix check on the resolved result to defend against platform-specific aliases. Cache a single approved `dataset_root.resolve()` value; never construct or resolve either forbidden subtree.
8. Preserve support for physical output/model paths outside `dataset_root`, including `../runs` and `../models`; validate their component chains against their filesystem anchor and never interpret them as dataset-relative inputs.
9. Return normalized resolved paths in input order and continue rejecting an empty input iterable.

Add private helpers with names that state whether they operate on original or comparison components; do not return a case-folded `Path`. External `runs/` and `models/` paths are allowed only when their existing chain is physical.

- [ ] **Step 4: Run focused safety and regression tests**

Run:

```powershell
python -m pytest tests/test_safety.py tests/test_detector_training.py tests/test_synthetic.py -q
```

Expected: all pass and no evaluation-access guard fires.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/bakery_scanner/safety.py tests/test_safety.py
git commit -m "fix(data): 학습 경로를 접근 전에 차단한다"
```

---

### Task 2: Base-Cycle Development-Background Synthetic Runs

**Files:**
- Modify: `src/bakery_scanner/synthetic.py`
- Modify: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `validate_base_cycle(repository_root, cycle_run)` and the development background records from its manifest.
- Produces: `generate_synthetic_dataset_from_backgrounds(dataset_root, background_paths, run_name, config) -> SyntheticGenerationReport`, with the existing manifest/replay format unchanged.

- [ ] **Step 1: Write explicit-background tests**

```python
def test_explicit_background_generation_uses_only_declared_files(
    synthetic_inputs: Path,
) -> None:
    root = synthetic_inputs
    allowed = (
        root / "collected/backgrounds/dev_a.png",
        root / "collected/backgrounds/dev_b.png",
    )
    forbidden = root / "collected/backgrounds/holdout.png"

    report = generate_synthetic_dataset_from_backgrounds(
        root,
        allowed,
        "base_v2_s42",
        SyntheticConfig(seed=42, scene_count=12, objects_per_scene=2),
    )
    payload = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    used = {
        _resolve_manifest_path(item["background_path"], report.output_dir, "background")
        for item in payload["scenes"]
    }
    assert used <= {path.resolve() for path in allowed}
    assert forbidden.resolve() not in used
    assert validate_synthetic_dataset(root, "base_v2_s42").image_count == 12


```

Add three separate tests after the concrete happy-path test above:

- `test_explicit_background_generation_rejects_duplicate_physical_paths`: pass the same physical file directly and through a `.` alias; assert `DataValidationError` before output creation.
- `test_explicit_background_generation_rejects_linked_background`: point a safe-looking development filename at a forbidden fixture; monkeypatch image decode and assert it is never called.
- `test_explicit_background_generation_refuses_existing_run`: pre-create the run directory; assert `DataValidationError` and unchanged directory contents.

- [ ] **Step 2: Verify red**

Run:

```powershell
python -m pytest tests/test_synthetic.py -k "explicit_background" -q
```

Expected: import failure because the explicit-background entry point does not exist.

- [ ] **Step 3: Factor the common generator**

Keep the existing public function and CLI behavior. Factor its body after background discovery into a private function, then add:

```python
def generate_synthetic_dataset_from_backgrounds(
    dataset_root: str | Path,
    background_paths: Sequence[str | Path],
    run_name: str,
    config: SyntheticConfig,
) -> SyntheticGenerationReport:
    root = Path(dataset_root).resolve(strict=False)
    checked = assert_training_paths_safe(background_paths, root)
    if len(checked) != len(background_paths) or len(set(map(_normalized_path_key, checked))) != len(checked):
        raise DataValidationError("explicit backgrounds must be unique physical files")
    for path in checked:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise DataValidationError(f"invalid explicit background image: {path}")
    return _generate_synthetic_from_resolved_backgrounds(
        root, tuple(sorted(checked, key=lambda item: item.as_posix())), run_name, config
    )
```

`_generate_synthetic_from_resolved_backgrounds` contains the current atomic staging, generation, manifest write, replay validation, and immutable publish behavior. The manifest continues to record each original background path and replay hash; no copied background directory is introduced.

- [ ] **Step 4: Run synthetic tests**

Run:

```powershell
python -m pytest tests/test_synthetic.py tests/test_synthetic_cli.py -q
```

Expected: all existing directory-based and new explicit-background tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/bakery_scanner/synthetic.py tests/test_synthetic.py
git commit -m "feat(data): cycle 개발 배경으로 합성 장면을 만든다"
```

---

### Task 3: Two-Fold Base-Cycle Detector Datasets

**Files:**
- Modify: `src/bakery_scanner/detector_dataset.py`
- Modify: `tests/test_detector_dataset.py`

**Interfaces:**
- Consumes: Base-cycle manifest, a replay-valid synthetic run, and existing `_Sample`/COCO writer logic.
- Produces: backwards-compatible `DetectorDatasetConfig` plus cycle mode, still consumed by `build_detector_dataset()` and `validate_detector_dataset()` and therefore by the existing YOLO converter/trainer.

- [ ] **Step 1: Extend the configuration contract with a backwards-compatible mode**

Add fields with defaults so existing callers remain unchanged:

```python
@dataclass(frozen=True, slots=True)
class DetectorDatasetConfig:
    seed: int = 42
    validation_fraction: float = 0.2
    real_coco_path: str = "base/val/instances_val.json"
    assignment_mode: str = "origin_fraction"
    cycle_run: str | None = None
    validation_scene_id: str | None = None

    def validate(self) -> None:
        # Preserve existing numeric/path validation.
        if self.assignment_mode not in {"origin_fraction", "base_cycle_fold"}:
            raise DataValidationError("unsupported detector assignment_mode")
        cycle_fields = (self.cycle_run, self.validation_scene_id)
        if self.assignment_mode == "origin_fraction":
            if any(value is not None for value in cycle_fields):
                raise DataValidationError("fraction mode must not declare cycle fields")
        elif not all(isinstance(value, str) and value for value in cycle_fields):
            raise DataValidationError("base_cycle_fold requires cycle_run and validation_scene_id")
```

- [ ] **Step 2: Write fold assignment and holdout-exclusion tests**

```python
@pytest.mark.parametrize(
    ("validation_id", "training_id"), [("0503", "0509"), ("0509", "0503")]
)
def test_base_cycle_fold_assigns_exact_real_groups_and_synthetic_train_only(
    cycle_detector_inputs: Path, validation_id: str, training_id: str
) -> None:
    report = build_detector_dataset(
        cycle_detector_inputs,
        "base_v2_s42",
        f"base_v2_s42_val{validation_id}",
        DetectorDatasetConfig(
            seed=42,
            validation_fraction=0.5,
            assignment_mode="base_cycle_fold",
            cycle_run="base_v2",
            validation_scene_id=validation_id,
        ),
    )
    payload = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    real = [item for item in payload["samples"] if item["origin"] == "real"]
    synthetic = [item for item in payload["samples"] if item["origin"] == "synthetic"]
    assert {item["provenance"]["scene_id"] for item in real} == {"0503", "0509"}
    assert {item["split"] for item in synthetic} == {"train"}
    assert {
        item["provenance"]["scene_id"] for item in real if item["split"] == "validation"
    } == {validation_id}
    assert {
        item["provenance"]["scene_id"] for item in real if item["split"] == "train"
    } == {training_id}
    assert report.validation_image_count == 3
    assert _validation_gt_count(report.output_dir) == 15
```

Add fail-closed tests for validation ID `0510`, any holdout scene path in output, any synthetic scene whose background path/hash is not one of the two development records, changed cycle-manifest SHA, changed fold assignment, missing e/m/h image, synthetic validation sample, and source/manifest link escape. Add a legacy fixture proving a `1.0.0` origin-fraction run still validates.

- [ ] **Step 3: Verify red**

Run:

```powershell
python -m pytest tests/test_detector_dataset.py -k "base_cycle or legacy" -q
```

Expected: configuration keyword errors or incorrect split assignments.

- [ ] **Step 4: Implement cycle authority and assignment**

Use a versioned manifest contract:

```python
LEGACY_BUILDER_VERSION = "1.0.0"
BUILDER_VERSION = "1.1.0"


def _base_cycle_assignments(
    samples: list[_Sample], development_ids: tuple[str, str], validation_id: str
) -> dict[str, str]:
    if validation_id not in development_ids:
        raise DataValidationError("validation scene must be a development scene")
    assignments: dict[str, str] = {}
    seen_real: dict[str, set[str]] = {scene_id: set() for scene_id in development_ids}
    for sample in samples:
        if sample.origin == "synthetic":
            assignments[sample.key] = "train"
            continue
        scene_id = str(sample.provenance["scene_id"])
        if scene_id not in development_ids:
            raise DataValidationError("detector cycle contains non-development real scene")
        match = SCENE_PATTERN.fullmatch(sample.source_path.name)
        if match is None:
            raise DataValidationError("invalid cycle scene filename")
        seen_real[scene_id].add(match.group("difficulty"))
        assignments[sample.key] = "validation" if scene_id == validation_id else "train"
    if any(value != {"e", "m", "h"} for value in seen_real.values()):
        raise DataValidationError("each development scene must contain e/m/h exactly once")
    return assignments
```

In cycle mode:

1. call `validate_base_cycle(repository_root, cycle_run)`;
2. bind its manifest path/hash into detector `inputs.base_cycle`;
3. filter real `_Sample` values to the two development IDs before any output copy;
4. read the synthetic manifest and require every scene background resolved path and SHA to equal a development background record;
5. call `_base_cycle_assignments` instead of `_split_samples`;
6. publish with builder `1.1.0` and normalized config containing `assignment_mode`, `cycle_run`, and `validation_scene_id`.

The validator branches on builder version. `1.0.0` retains the exact old schema and recalculation. `1.1.0` validates Base-cycle manifest/hash, development background authority, exact filtered inventory, and `_base_cycle_assignments` field-for-field. Both return the existing `DetectorValidationReport` interface so `build_yolo_dataset` remains unchanged.

- [ ] **Step 5: Run detector dataset and YOLO regression tests**

Run:

```powershell
python -m pytest tests/test_detector_dataset.py tests/test_yolo_dataset.py -q
```

Expected: both fold modes and all legacy/YOLO conversion tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/bakery_scanner/detector_dataset.py tests/test_detector_dataset.py
git commit -m "feat(data): Base cycle detector fold를 고정한다"
```

---

### Task 4: Detector Experiment Matrix, OOF Selection, and CPU Timing

**Files:**
- Create: `src/bakery_scanner/detector_cycle.py`
- Create: `tests/test_detector_cycle.py`
- Modify: `src/bakery_scanner/detector_training.py`
- Modify: `tests/test_detector_training.py`

**Interfaces:**
- Consumes: cycle-aware detector/YOLO datasets and existing `train_detector()` backend.
- Produces: `load_detector_cycle_config`, `prepare_detector_cycle`, `run_detector_cycle`, `summarize_detector_cycle`, and `validate_detector_cycle_summary`.

- [ ] **Step 1: Define the strict matrix types**

```python
@dataclass(frozen=True, slots=True)
class DetectorCandidate:
    name: str
    checkpoint: str


@dataclass(frozen=True, slots=True)
class DetectorCycleConfig:
    repository_root: str
    cycle_run: str
    models: tuple[DetectorCandidate, ...]
    seeds: tuple[int, int, int]
    validation_scene_ids: tuple[str, str]
    synthetic_scene_count: int
    objects_per_scene: int
    image_size: int
    epochs: int
    batch_size: int
    patience: int
    workers: int
    device: str
    confidence_floor: float
    nms_iou: float
    matching_iou: float
    cpu_warmups: int
    cpu_repetitions: int
    output_root: str
    experiment_name: str
```

`load_detector_cycle_config()` rejects unknown/missing keys, non-exact model order/names, checkpoint paths outside `models/pretrained`, seeds other than `[42,43,44]`, validation IDs other than `0503/0509`, `device != "0"`, output outside `runs/detector_cycle`, test/dot-segment/link paths, duplicate physical checkpoints, and an invalid run name.

- [ ] **Step 2: Write preparation/orchestration tests with fakes**

```python
def test_prepare_builds_three_synthetic_and_six_fold_datasets(
    detector_cycle_fixture: Path,
) -> None:
    report = prepare_detector_cycle(_config_path(detector_cycle_fixture))
    assert report.synthetic_runs == (
        "base_v2_s42", "base_v2_s43", "base_v2_s44"
    )
    assert len(report.detector_runs) == 6
    assert all(validate_detector_dataset(report.dataset_root, name) for name in report.detector_runs)


def test_run_filter_trains_only_requested_model_seed_and_both_folds(
    prepared_cycle: Path, fake_training_backend: FakeDetectorBackend
) -> None:
    report = run_detector_cycle(
        _config_path(prepared_cycle),
        model_filter="yolo11n",
        seed_filter=42,
        backend=fake_training_backend,
    )
    assert [(item.model, item.seed, item.validation_scene_id) for item in report.runs] == [
        ("yolo11n", 42, "0503"),
        ("yolo11n", 42, "0509"),
    ]
```

Add tests that fail if a pretrained file is missing/changed, a fold manifest/hash changes, a completed training run is silently overwritten, a requested filter is outside the matrix, CUDA is unavailable, output class differs from `bread`, or any holdout/test path reaches the backend.

- [ ] **Step 3: Implement preparation and training orchestration**

`prepare_detector_cycle` validates the cycle, obtains the two development background paths from its manifest, generates `base_v2_s{seed}` with `SyntheticConfig(scene_count=100, objects_per_scene=5)`, builds both fold detector runs, and invokes `build_yolo_dataset` for each.

`run_detector_cycle` constructs existing `DetectorTrainingConfig` instances:

```python
def _training_config(
    config: DetectorCycleConfig,
    model: DetectorCandidate,
    seed: int,
    validation_id: str,
) -> DetectorTrainingConfig:
    detector_run = f"base_v2_s{seed}_val{validation_id}"
    return DetectorTrainingConfig(
        dataset_root=str(Path(config.repository_root) / "datasets"),
        source_detector_run=detector_run,
        yolo_run_name=f"{detector_run}_yolo",
        output_root=str(Path(config.repository_root) / config.output_root),
        run_name=f"{config.experiment_name}_{model.name}_s{seed}_val{validation_id}",
        model=str(Path(config.repository_root) / model.checkpoint),
        image_size=config.image_size,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=seed,
        device=config.device,
        patience=config.patience,
        workers=config.workers,
        thresholds=EvaluationThresholds(
            confidence_floor=config.confidence_floor,
            operating_confidence=config.confidence_floor,
            nms_iou=config.nms_iou,
            matching_iou=config.matching_iou,
        ),
    )
```

Publish an experiment `runs.json` only after both requested folds complete and validate each detector metadata/model/dataset hash. Existing complete runs may be reused only after strict validation proves the same config/checkpoint/source hashes; partial or mismatched runs are errors, not resumed or overwritten.

- [ ] **Step 4: Write OOF threshold tests**

```python
def test_threshold_selection_uses_highest_value_with_thirty_of_thirty_recall() -> None:
    images, predictions = _thirty_truth_fixture(
        true_positive_scores=[0.91] * 29 + [0.37],
        false_positive_scores=[0.80, 0.20],
    )
    selected = select_oof_threshold(images, predictions, confidence_floor=0.001)
    assert selected.threshold == pytest.approx(0.37)
    assert selected.metrics["global"]["recall"] == 1.0
    assert selected.metrics["global"]["ground_truth_count"] == 30
    assert selected.metrics["global"]["false_positive_count"] == 1


```

Add three separate fixture-driven tests after the concrete threshold test:

- `test_threshold_selection_rejects_missing_oof_image_or_gt_count`: parameterize five images, 29 GT, duplicate image IDs, and a missing fold; every case must raise `DataValidationError`.
- `test_threshold_selection_marks_seed_ineligible_when_no_threshold_hits_thirty_of_thirty`: omit one matched prediction at every threshold; assert `eligible is False`, `threshold is None`, and the miss count is one.
- `test_model_ranking_requires_all_seeds_and_uses_fixed_tie_breaks`: reject a two-seed model, then compare complete models to prove ordering by total FP, development-background FP, worst AP50, CPU P95, and finally declared model order.

- [ ] **Step 5: Implement aggregation and ranking**

```python
@dataclass(frozen=True, slots=True)
class OofSelection:
    threshold: float | None
    eligible: bool
    metrics: dict[str, Any]


def select_oof_threshold(
    images: Sequence[EvaluationImage],
    predictions: Mapping[str, Sequence[Detection]],
    confidence_floor: float,
    matching_iou: float = 0.5,
) -> OofSelection:
    if len(images) != 6 or sum(len(item.objects) for item in images) != 30:
        raise DataValidationError("OOF detector selection requires six images and 30 GT")
    if set(predictions) != {item.image_id for item in images}:
        raise DataValidationError("OOF predictions must cover every image exactly once")
    scores = sorted(
        {
            detection.confidence
            for detections in predictions.values()
            for detection in detections
            if detection.confidence >= confidence_floor
        }
        | {confidence_floor},
        reverse=True,
    )
    for threshold in scores:
        metrics = evaluate_detector_predictions(
            images,
            predictions,
            EvaluationThresholds(confidence_floor, threshold, 0.7, matching_iou),
        )
        if metrics["global"]["true_positive_count"] == 30:
            return OofSelection(threshold, True, metrics)
    metrics = evaluate_detector_predictions(
        images,
        predictions,
        EvaluationThresholds(confidence_floor, confidence_floor, 0.7, matching_iou),
    )
    return OofSelection(None, False, metrics)
```

Aggregation reads both fold `predictions.json` files, proves distinct validation IDs and checkpoint/source hashes, reconstructs the exact real validation inputs from the two YOLO manifests, and emits one seed record. A model is eligible only with three eligible seed records. Rank using the Global Constraints tuple.

- [ ] **Step 6: Write CPU benchmark tests and implementation**

Use a fake clock/backend to prove warm-ups are excluded, model/input device is CPU, two folds contribute equal repetition counts, and mean/P50/P95 are correct.

```python
def benchmark_oof_detector_cpu(
    fold_checkpoints: Mapping[str, Path],
    fold_images: Mapping[str, tuple[Path, ...]],
    *,
    image_size: int,
    confidence_floor: float,
    nms_iou: float,
    warmups: int,
    repetitions: int,
    backend: DetectorBackend | None = None,
) -> dict[str, Any]:
    # Load each checkpoint outside timed regions.
    # Call predict(..., device="cpu") for warmups, then repetitions.
    # Return per-fold samples and combined mean/P50/P95 in milliseconds.
```

Record `platform.processor`, logical CPU count, `torch.get_num_threads()`, input size, scene batch size 3, warm-ups, repetitions, backend, dependencies, and checkpoint hashes. Do not claim POS-device performance.

- [ ] **Step 7: Publish and validate the immutable summary**

`summarize_detector_cycle` writes `runs/detector_cycle/<experiment_name>/summary.json` through a temp file after all requested model/seed/fold artifacts validate. Top-level fields are exactly version, created_at, config/config_sha256, cycle manifest record, pretrained records, per-run records, per-seed OOF selections, CPU benchmarks, ranking, and selected detector. `validate_detector_cycle_summary` recomputes every hash/metric/threshold/ranking and rejects any change or unknown/missing field.

- [ ] **Step 8: Run focused tests and commit Task 4**

Run:

```powershell
python -m pytest tests/test_detector_cycle.py tests/test_detector_training.py -q
git diff --check
```

Expected: all pass.

```powershell
git add src/bakery_scanner/detector_cycle.py src/bakery_scanner/detector_training.py tests/test_detector_cycle.py tests/test_detector_training.py
git commit -m "feat(experiment): Base detector OOF 비교를 구현한다"
```

---

### Task 5: CLI, Exact Matrix, Documentation, and Real E0/E2 Execution

**Files:**
- Create: `src/bakery_scanner/detector_cycle_cli.py`
- Create: `tests/test_detector_cycle_cli.py`
- Create: `configs/detector_cycle/base_v2.yaml`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 4 public experiment functions.
- Produces: `bakery-detector-cycle` and real ignored experiment artifacts/results.

- [ ] **Step 1: Write CLI tests**

Cover:

Write concrete CLI tests with a temporary config and monkeypatched Task 4 functions:

- `test_prepare_json_routes_config`: invoke `main(["prepare", "--config", path, "--json"])`; assert exit 0, one routed config path, and JSON status `ok`.
- `test_run_allows_exact_model_and_seed_filters`: invoke a valid `yolo11n`/`42` filter; assert only that pair reaches the runner and both folds remain selected internally.
- `test_run_rejects_unknown_filter`: invoke an unknown model and seed; assert argparse exit 2 before runner access.
- `test_summarize_and_validate_summary_json`: assert both commands route to distinct functions and serialize their returned reports.
- `test_cli_returns_one_for_data_validation_error`: make the routed function raise `DataValidationError`; assert exit 1 and a concise stderr message.
- `test_cli_has_no_holdout_or_test_evaluation_option`: inspect parser actions and assert there is no scene, split, holdout, test, threshold, or checkpoint-selection override.

The parser has exactly these commands:

```text
prepare-weights --config PATH [--json]
prepare --config PATH [--json]
run --config PATH [--model {yolo11n,yolo26s,yolo26m}] [--seed {42,43,44}] [--json]
summarize --config PATH [--json]
validate-summary --config PATH [--json]
```

`prepare-weights` uses Ultralytics only for the allowlisted three official filenames, copies the resolved downloaded checkpoint into the exact checked-in path under `models/pretrained`, and writes an ignored SHA manifest. Existing files are hash-reported and never overwritten.

- [ ] **Step 2: Implement CLI and entry point**

Register only after CLI tests exist:

```toml
bakery-detector-cycle = "bakery_scanner.detector_cycle_cli:main"
```

All commands catch `DataValidationError`, print a concise error to stderr, and return 1. `--json` prints `report.to_dict()` with `status`.

- [ ] **Step 3: Add the exact checked-in configuration**

Create `configs/detector_cycle/base_v2.yaml`:

```yaml
repository_root: .
cycle_run: base_v2
models:
  - name: yolo11n
    checkpoint: models/pretrained/yolo11n.pt
  - name: yolo26s
    checkpoint: models/pretrained/yolo26s.pt
  - name: yolo26m
    checkpoint: models/pretrained/yolo26m.pt
seeds: [42, 43, 44]
validation_scene_ids: ["0503", "0509"]
synthetic_scene_count: 100
objects_per_scene: 5
image_size: 640
epochs: 50
batch_size: 16
patience: 10
workers: 8
device: "0"
confidence_floor: 0.001
nms_iou: 0.7
matching_iou: 0.5
cpu_warmups: 5
cpu_repetitions: 20
output_root: runs/detector_cycle
experiment_name: base_v2
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_safety.py tests/test_synthetic.py tests/test_detector_dataset.py tests/test_yolo_dataset.py tests/test_detector_training.py tests/test_detector_cycle.py tests/test_detector_cycle_cli.py -q
python -m pytest -q
git diff --check
```

Expected: zero failures.

- [ ] **Step 5: Execute the fastest E0 evidence path first**

Run:

```powershell
bakery-detector-cycle prepare-weights --config configs/detector_cycle/base_v2.yaml --json
bakery-detector-cycle prepare --config configs/detector_cycle/base_v2.yaml --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo11n --seed 42 --json
```

Expected: three replay-valid synthetic runs and six fold datasets are prepared; the filtered run trains exactly `yolo11n`, seed 42, both folds. Inspect the resulting OOF record before starting the remaining matrix, but do not change config/threshold/model choices from this result.

- [ ] **Step 6: Execute the full fixed matrix**

Run the remaining fixed filters. Completed identical runs are strictly validated and reported rather than overwritten:

```powershell
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo11n --seed 43 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo11n --seed 44 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo26s --seed 42 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo26s --seed 43 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo26s --seed 44 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo26m --seed 42 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo26m --seed 43 --json
bakery-detector-cycle run --config configs/detector_cycle/base_v2.yaml --model yolo26m --seed 44 --json
bakery-detector-cycle summarize --config configs/detector_cycle/base_v2.yaml --json
bakery-detector-cycle validate-summary --config configs/detector_cycle/base_v2.yaml --json
```

Expected summary: 18 immutable training runs, nine six-image/30-GT OOF seed records, three CPU benchmark records, and either one selected detector or an explicit no-eligible-detector result. No cycle-holdout prediction or metric exists.

- [ ] **Step 7: Document only implemented commands and measured results**

After commands pass, add a README “Base v2 detector cycle” section with the five CLI commands, exact folds/seeds/models, 30/30 Recall eligibility rule, CPU benchmark context, selected detector (if any), and explicit statement that `0510`, the holdout background, and both test trees were not evaluated or used for selection.

- [ ] **Step 8: Verify scope and commit Task 5**

Run:

```powershell
rg -n "0510|tray_wood_white_surface|base/test|incremental/test" runs/detector_cycle/base_v2 datasets/derived/detector/base_v2_* datasets/derived/synthetic/base_v2_*
git status --short
git diff --check
```

The `rg` check is interpreted structurally: config/manifest assignment records may name the holdout only inside the immutable Base-cycle authority record; no synthetic scene, fold sample, prediction, metric, threshold, or benchmark input may reference it. Add a small audit command/test that parses artifact roles instead of relying only on string absence.

```powershell
git add pyproject.toml configs/detector_cycle/base_v2.yaml src/bakery_scanner/detector_cycle_cli.py tests/test_detector_cycle_cli.py README.md
git commit -m "feat(cli): Base detector cycle 실행 경로를 제공한다"
```

---

## Plan Self-Review

- **Spec coverage:** Tasks 1-3 create development-only, cycle-bound synthetic/fold inputs; Task 4 trains and aggregates the required three models, two folds, and three seeds and measures CPU latency; Task 5 runs the fixed matrix and records the selection. Holdout scoring, verifier training, classifier replacement, cascade integration, and final one-shot evaluation remain separate later plans.
- **No test leakage:** Every new data/model path passes lexical-first safety; artifact tests prove no holdout/test sample reaches the backend. Existing `base_seed42` and project checkpoints are not used as new-cycle inputs.
- **Model scope:** Official Ultralytics identifiers are exactly `yolo11n.pt`, `yolo26s.pt`, and `yolo26m.pt`; YOLO26s/m training support is documented by Ultralytics. No DINO/SAM/ensemble is introduced in this phase.
- **Metric scope:** Selection uses combined OOF 30/30 Recall and FP, not prior test results. The cycle holdout stays unopened semantically until the final frozen cascade.
- **Type consistency:** Existing detector/Yolo reports and training config remain public compatibility boundaries; new cycle APIs use immutable dataclasses and strict JSON/YAML schemas.
- **Placeholder scan:** The plan contains no `TODO`, `TBD`, ellipsis-based test body, or unspecified public command. Each test case states its fixture mutation and required assertion.
- **Time control:** `--model/--seed` filters expose YOLO11n seed-42 E0 evidence first without changing the pre-frozen full matrix. The final comparison still requires all 18 runs before selecting a detector.
