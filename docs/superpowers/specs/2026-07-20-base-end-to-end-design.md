# Base End-to-End Inference and Evaluation Design

## Goal

Combine the frozen YOLO11n bread detector and the Base ResNet18 classifier into
the first scene-level train-side validation pipeline. The pipeline returns one
`bbox`, `model_index`, and confidence per detected bread and records class-aware
mAP50, mAP50:95, and per-class exact-count accuracy.

No Base or Incremental test path, annotation, prediction, metric, or threshold
may influence this unit.

## Evaluation Split

The input is the validation COCO of the already fixed
`base_seed42_detector_origin_aware` detector dataset. That manifest is rebuilt
or replay-validated from train-side data with seed 42; model result files are
not used to choose images. Scene e/m/h variants remain grouped. The current
validation group is scene ID `0509`, containing three images and fifteen Base
objects.

## Inference Flow

1. Validate all configured paths before reading images, annotations, metadata,
   or checkpoints.
2. Replay-validate the detector dataset and Base classifier dataset.
3. Verify detector and classifier checkpoint hashes against their run metadata.
4. Run the class-agnostic detector at the already fixed confidence and NMS
   thresholds.
5. Clip each predicted box to image bounds and reject zero-area crops.
6. Transform all crops from one scene image and classify them as one batch.
7. Emit the detector box, classifier `model_index`, detector confidence,
   classifier confidence, and their product as the ranking score.

The detector remains class-agnostic and frozen. The classifier checkpoint must
match the current manifest SHA-256, registry SHA-256/model-index mapping, and
full Base configuration.

## Metrics

Class-aware matching requires both equal `model_index` and IoU at or above the
current threshold. Predictions are ranked by combined score. AP uses 101-point
interpolated precision over recall.

- `mAP50`: macro AP at IoU 0.50 over classes with ground-truth support.
- `mAP50:95`: macro AP over IoU 0.50 through 0.95 in steps of 0.05 and classes
  with ground-truth support.
- Per-class exact-count accuracy: fraction of validation images whose predicted
  count exactly equals the ground-truth count for that `model_index`.
- Macro exact-count accuracy is recorded for all output classes and separately
  for classes with ground-truth support.

Unsupported AP classes are JSON `null` and excluded from mAP. Empty detection
images produce an empty prediction list and valid zero-recall metrics.

## Artifacts

The ignored output `runs/e2e/<run-name>/` contains:

```text
config.yaml
metadata.json
predictions.json
metrics.json
```

Metadata records source manifest and checkpoint hashes, registry mapping,
thresholds, dependency/hardware information, image sizes, batch behavior, and
the fact that detector weights were not modified.

## Interface and Verification

`bakery-e2e evaluate --config configs/e2e/base_resnet18.yaml` validates and
prints concrete train-side inputs before inference. Unit tests first cover pure
metrics, invalid boxes/classes/scores, path safety, checkpoint hash binding,
empty detections, scene-batched classifier calls, atomic publication, and CLI
output. The final gate executes all three validation scenes on CUDA, replay
validates stored metrics, runs the full repository suite, and receives an
independent review before squash merge.
