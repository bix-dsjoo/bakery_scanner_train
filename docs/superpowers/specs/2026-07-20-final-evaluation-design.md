# Frozen Final Test Evaluation Design

**Date:** 2026-07-20
**Status:** Frozen before any final-test access

## Goal

Evaluate the approved frozen detector, Base classifier, and Incremental
classifier exactly once on the Base and Incremental test splits. Publish all
required detector, classifier, end-to-end, retention, and new-class metrics
without using test results to alter any configuration or implementation.

## Freeze boundary

`configs/final_evaluation/frozen_v1.yaml` is the sole final-evaluation
configuration. Before test access it fixes:

- registry, detector config/checkpoint, Base classifier config/checkpoint, and
  Incremental classifier config/checkpoint SHA-256 values;
- detector input 640, confidence floor 0.001, operating confidence 0.25, NMS
  IoU 0.7, and matching IoU 0.5;
- classifier input 224 and output dimensions 15/20;
- CUDA device 0 for accuracy inference;
- one classifier batch per scene and the combined score formula;
- exact Base and Incremental COCO paths and the final output location.

The test directories and COCO files are not read while designing,
implementing, testing, reviewing, or merging the evaluator.

## Two-PR execution boundary

The evaluator and frozen config are implemented, fully tested with synthetic
fixtures, independently reviewed, and merged first. Only the merged evaluator
on a fresh branch from the resulting `main` may execute the final test run.

The run creates `runs/final_evaluation/.frozen_v1.started.json` atomically
before it opens either test COCO. If that lock already exists, execution fails.
The lock remains after failure or success, preventing a second run. Once test
results exist, only result documentation may change; code, models, thresholds,
configs, and checkpoint selection remain frozen.

## Preflight and provenance

`bakery-final-eval preflight --config ...` validates only non-test inputs. It
must finish without statting, listing, reading, or decoding either test path.
It verifies:

- the frozen config schema and freeze declarations;
- registry/config/checkpoint hashes;
- checkpoint-adjacent metadata and selected run provenance;
- detector class name `bread` and classifier architecture/output/context;
- detector checkpoint referenced by the Incremental classifier is unchanged;
- CUDA device availability;
- output and lock paths are outside the dataset root and absent.

`bakery-final-eval run --config ...` repeats preflight, creates the one-shot
lock, then validates both COCO datasets and their decoded images before
inference.

## Test inputs

The Base test COCO must declare only the registry's 15 Base categories. The
Incremental test COCO must declare only the five Incremental categories. COCO
image/category references, decoded sizes, duplicate IDs, and bbox bounds are
fatal errors.

Each test image is represented in three synchronized forms:

- class-agnostic detector truth with difficulty parsed from `scene_e/m/h`;
- model-index end-to-end truth;
- ground-truth bbox crops for classifier-only evaluation.

COCO `category_id` is always converted through the registry to `model_index`.

## Inference

The frozen YOLO detector runs once per split at confidence floor 0.001. Its
predictions are reused for detector metrics and all relevant end-to-end model
variants. Operating metrics and count accuracy filter those detections at 0.25.

For every scene, all valid detector crops are transformed and classified in one
batch for each relevant classifier. Empty detector output produces an empty
prediction list. Ground-truth bbox crops are also classified in a single batch
per scene for classifier-only metrics.

The evaluated combinations are:

1. Base classifier on Base test ground-truth crops.
2. Incremental classifier on Base test ground-truth crops.
3. Incremental classifier on Incremental test ground-truth crops.
4. Base detector + Base classifier on Base test.
5. Base detector + Incremental classifier on Base test.
6. Base detector + Incremental classifier on Incremental test.

The Base classifier is not run on Incremental test because it has no outputs for
model indices 15–19.

## Metrics

Detector, separately for Base and Incremental test:

- AP50 using the fixed confidence floor;
- Recall@IoU 0.5 and miss rate at operating confidence;
- easy/medium/hard Recall;
- Base/Incremental class-group Recall.

Classifier on ground-truth crops:

- Top-1 accuracy;
- Macro F1 over supported classes;
- per-class Precision, Recall, F1, support, and prediction count;
- Base and Incremental group metrics for the 20-output model.

End-to-end:

- mAP50;
- mAP50:95;
- per-class exact quantity accuracy and supported macro exact-count accuracy;
- raw and operating prediction counts.

The summary records Incremental-minus-Base deltas on Base test for classifier
Top-1/Macro F1 and end-to-end mAP50/mAP50:95/supported exact-count accuracy.
Incremental test metrics are reported separately for the five new classes.

## Output

The ignored run directory contains:

- frozen config copy and its SHA-256;
- metadata with timestamps, environment, versions, hardware, CUDA device,
  metric definitions, thresholds, input/checkpoint/config hashes, and
  before/after checkpoint hashes;
- detector/classifier/end-to-end metric JSON files;
- raw detector, classifier, and end-to-end predictions;
- machine-readable summary JSON and Korean Markdown report.

All output except the already-created start lock is built in a staging
directory and published atomically. The lock is updated to `completed` or
`failed` but never removed.

## Acceptance

- The preflight is proven not to access test paths.
- The evaluator PR is independently reviewed and merged before the one-shot
  run.
- The final run starts from that merged `main` with no config/code changes.
- All required metrics and environment metadata are present and independently
  recalculated from raw predictions.
- All model checkpoints remain byte-identical.
- The result-report PR changes documentation only.
- No setting is selected or altered after observing test results.
