# Detector Training Baseline Design

## Goal

Build the first reproducible class-agnostic bread-detector baseline with an
Ultralytics YOLO11n pretrained model. The completed unit includes a validated
YOLO training dataset, a GPU training command, reusable train-side evaluation,
checkpoints, predictions, metrics, and environment metadata.

This unit uses only training-authorized data. It does not read
`datasets/base/test`, `datasets/incremental/test`, or test results for training,
early stopping, threshold selection, augmentation selection, checkpoint
selection, or model selection.

## Scope

The implementation will:

- convert one independently validated detector COCO run into a reproducible
  YOLO dataset;
- train a single-class YOLO11n detector from `yolo11n.pt` on CUDA;
- evaluate the selected checkpoint on the detector run's train-side validation
  split;
- report AP50, Recall at IoU 0.5, miss rate, difficulty-group Recall, and
  Base/Incremental phase-group Recall;
- preserve the exact configuration, source hashes, checkpoint hashes,
  predictions, metrics, dependency versions, and hardware metadata;
- expose matching CLI commands and testable Python interfaces.

Classifier training, end-to-end inference, test-set evaluation, ONNX export,
and CPU benchmarking remain outside this unit.

## Baseline Configuration

The checked-in baseline configuration uses:

- source detector run: `base_seed42_detector_origin_aware`;
- pretrained model: `yolo11n.pt`;
- input size: 640 pixels;
- epochs: 50;
- batch size: 16;
- seed: 42;
- training device: CUDA device `0`;
- early-stopping patience: 10 epochs;
- prediction confidence floor for AP data: 0.001;
- operating confidence threshold: 0.25;
- NMS IoU threshold: 0.7;
- matching IoU threshold: 0.5.

Early stopping and checkpoint selection use only the train-side validation
split. These values establish the first reproducible baseline; no test result
may be used to revise them.

## Component Boundaries

### YOLO dataset conversion

`src/bakery_scanner/yolo_dataset.py` owns conversion and validation. It first
calls the existing independent detector-run validator. It then converts every
COCO `[x, y, width, height]` box to normalized YOLO
`class x_center y_center width height` form with class index `0` for `bread`.
Images without annotations receive an empty label file.

The converter writes a staging directory, verifies its complete inventory and
content, then publishes it atomically under
`datasets/derived/yolo/<run-name>/`. The run contains copied images, label
files, `data.yaml`, and `manifest.json`. The manifest records the source
detector run and manifest hash, output hashes, split counts, dimensions, and
per-image provenance. Existing runs require explicit replacement, and failed
replacement must preserve the prior valid run.

Independent validation rejects missing, extra, altered, or undecodable files;
invalid normalized coordinates; class indices other than zero; manifest/schema
mismatches; source detector mutations; and train/validation inventory drift.

### Training orchestration

`src/bakery_scanner/detector_training.py` owns configuration validation,
environment capture, backend invocation, and publication of a completed model
run. The Ultralytics-specific call is isolated behind a small backend interface
so orchestration behavior can be tested without running a GPU job.

The trainer validates every configured path with
`assert_training_paths_safe`, validates the YOLO run, prints the selected
splits and paths, verifies CUDA availability for device `0`, and refuses an
existing output run. It trains in a staging directory under
`runs/detector/` and publishes the named run only after checkpoint and result
validation succeeds.

A successful run contains at least:

```text
runs/detector/<run-name>/
  config.yaml
  metadata.json
  checkpoints/
    best.pt
    last.pt
  predictions.json
  metrics.json
```

`metadata.json` records the source YOLO manifest path and hash, pretrained
checkpoint name and hash, final checkpoint hashes, Python and dependency
versions, OS, CPU, GPU, CUDA, seed, input size, batch size, worker count, and
the exact backend arguments. Checkpoints and generated run outputs are local
artifacts and are excluded from Git.

### Evaluation

`src/bakery_scanner/detector_evaluation.py` owns the backend-independent metric
contract. Predictions use image-relative xyxy boxes, confidence scores, and
the single `bread` class. An image with no detections has an empty prediction
list.

At IoU 0.5, predictions are sorted by confidence and greedily matched to one
unmatched ground-truth box per image. AP50 is computed from the full prediction
set collected at confidence floor 0.001. Recall and miss rate use confidence
0.25 after NMS at IoU 0.7. Miss rate is `1 - recall` over annotated objects.
Normal images without annotations remain valid and do not add false ground
truth objects.

Difficulty groups derive from real scene file names: `scene_e_*` is `easy`,
`scene_m_*` is `medium`, and `scene_h_*` is `hard`. Phase groups derive each
ground-truth annotation's original COCO category through
`datasets/class_registry.json`. Group records always include a sample count and
ground-truth object count. A group with no ground-truth objects reports metric
values as JSON `null`, never as zero or a fabricated score.

`metrics.json` includes metric version, timestamp, source split, checkpoint
hash, thresholds, counts, global metrics, difficulty metrics, phase metrics,
configuration, dependency versions, and hardware metadata.

### CLI

`src/bakery_scanner/detector_train_cli.py` exposes:

```text
bakery-detector train --config configs/detector/yolo11n_base.yaml
bakery-detector evaluate --config configs/detector/yolo11n_base.yaml --checkpoint <path>
```

Both commands support human-readable output and `--json`. Before training or
evaluation they print the concrete dataset root, source run, train split,
validation split, model, and output path. Data-validation and configuration
errors return exit code 1 with a concise message.

## Data Flow

1. Validate `datasets/derived/detector/base_seed42_detector_origin_aware` and
   its synthetic source independently.
2. Convert the validated train and validation COCO files into an atomic YOLO
   run and independently validate the result.
3. Load the checked-in baseline YAML and reject any evaluation-only path.
4. Load `yolo11n.pt`, record its SHA-256, and train on CUDA device 0.
5. Select `best.pt` using only the train-side validation split.
6. Run predictions for the validation images at confidence floor 0.001.
7. Normalize backend predictions into `predictions.json`.
8. Compute and store global, difficulty, and phase metrics.
9. Validate required checkpoints, hashes, predictions, metrics, and metadata.
10. Atomically publish the completed model run.

## Error Handling

The following conditions fail the command rather than emit a warning:

- an input, configuration, manifest, or resolved path points to Base or
  Incremental test data;
- the source detector run or converted YOLO run fails independent validation;
- a bbox or YOLO label is malformed, non-positive, out of bounds, non-finite,
  or uses a class other than zero;
- an image or label is missing, extra, altered, or undecodable;
- the model output is not the single `bread` class;
- CUDA device 0 is configured but unavailable;
- the pretrained model or evaluation checkpoint is unavailable;
- an output run already exists;
- a completed run lacks a checkpoint, prediction, metric, environment, split,
  dependency, or hardware field required by this design.

Training failures leave no completed run. Temporary staging data created by
the failed invocation may be cleaned without touching inputs or a previously
completed run.

## Testing

Implementation follows test-driven development. Automated tests cover:

- exact COCO-to-YOLO bbox conversion and empty-label handling;
- deterministic conversion, manifest provenance, and output hashes;
- rejection of test paths, invalid boxes, invalid classes, tampering, missing
  files, extra files, and output-name collisions;
- atomic publication and preservation of a prior valid run on failed
  replacement;
- configuration range validation and concrete split reporting;
- exact seed, device, image size, batch, epoch, patience, confidence, and NMS
  arguments passed to the backend;
- IoU matching, duplicate detections, false positives, empty detections, AP50,
  Recall, and miss-rate calculations;
- easy/medium/hard and Base/Incremental aggregation;
- `null` metrics for groups without ground-truth objects;
- required prediction, metric, environment, and checkpoint metadata;
- CLI human and JSON output and nonzero failure exits;
- failed training not publishing a completed run.

Unit tests use tiny generated fixtures and a test backend. They do not perform
a real GPU training job. After focused and full test suites pass, the repository
dataset is converted, validated, trained on the RTX 5080, independently
evaluated, and checked for a clean Git worktree. The actual validation metrics
become the initial reproducible detector baseline; they are not a predetermined
pass/fail target.

## Documentation and Dependencies

`pyproject.toml` will declare the runtime packages used directly by this unit,
including a compatible Ultralytics 8.x range and PyYAML 6.x range. Exact local
versions are captured in every completed run.

`README.md` will document only commands that exist after implementation. It
will distinguish train-side validation from evaluation-only test data and will
state that the resulting latency or accuracy numbers do not describe a
specific POS device.

The project design document and `AGENTS.md` policies remain authoritative. Any
implementation detail that conflicts with their test-isolation, mapping,
reproducibility, or CPU-benchmark rules must fail closed.
