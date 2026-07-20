# Incremental 20-Class Classifier Design

## Goal

Extend the approved Base ResNet18 classifier from 15 to 20 registry-ordered
outputs while preserving the Base checkpoint lineage, reusing Base train-side
data, correcting the Base/Incremental sample imbalance, and proving that the
class-agnostic detector remains unchanged.

This is a train-side development stage. Neither `datasets/base/test` nor
`datasets/incremental/test` may be read, used for model selection, or used to
change any setting.

## Chosen approach

The default experiment fine-tunes the full classifier with replay and a
class-balanced cross-entropy loss:

1. Strict-load the approved 15-output Base classifier checkpoint.
2. Construct a 20-output ResNet18 with the same image size.
3. Copy every backbone parameter and the first 15 classifier rows and biases
   exactly.
4. Keep the seeded initialization for output rows 15 through 19.
5. Train on the deterministic Incremental classifier manifest, which contains
   both Base and Incremental train-side samples.
6. Weight each present class inversely to its train sample count.
7. Select the checkpoint only by the combined train-side validation loss.

Head-only training is deferred as an ablation because it cannot adapt the
feature extractor to visually novel breads. Knowledge distillation is also an
ablation because Base replay is permitted and already available; adding a
teacher objective to the default experiment would introduce an unnecessary
loss-weight hyperparameter.

## Configuration compatibility

The existing `ClassifierTrainingConfig` and Base YAML remain byte-for-byte
compatible with the approved Base checkpoint context. A separate
`IncrementalClassifierTrainingConfig` uses an explicit `phase: incremental`
and these artifact fields:

- `base_checkpoint`: approved Base classifier `best.pt`.
- `frozen_detector_checkpoint`: approved detector `best.pt` whose bytes must
  remain identical before and after classifier training and evaluation.

The remaining optimizer, device, image-size, seed, and output settings follow
the Base config. The existing `bakery-classifier train` and `evaluate`
commands dispatch to the Base or Incremental schema from the config fields;
no fabricated command is added.

## Checkpoint expansion contract

The Base checkpoint must have the existing strict schema, architecture
`resnet18`, output dimension 15, and the configured image size. Its registry
SHA-256 must equal the Incremental manifest registry SHA-256. Its ordered
mapping must exactly equal Incremental mapping entries 0 through 14.

Expansion fails on a missing key, unexpected key, output mismatch, mapping
mismatch, registry mismatch, image-size mismatch, or malformed tensor. Tests
compare every copied backbone tensor and the first 15 head rows and biases for
exact equality. Output rows 15 through 19 must exist and must not alias Base
rows.

The Incremental checkpoint retains the existing checkpoint schema and records
the complete 20-entry mapping and Incremental config in its context. Standard
checkpoint evaluation therefore rejects a different manifest, registry, or
config.

## Dataset and imbalance handling

The source classifier run must validate as phase `incremental`, output
dimension 20, and contain both non-empty train and validation splits. The
manifest remains authoritative for paths, `model_index`, split membership,
and registry hash.

The training run records, for every model index, the train sample count and
the exact loss weight used by the backend. For each present class:

`weight[index] = train_sample_count / (20 * class_sample_count[index])`

All 20 classes must be present in train. Incremental validation must contain
all five new classes; Base validation support follows the fixed scene group
and can cover a subset of the 15 Base classes. A missing train class is an
error rather than a zero weight or warning.

The classifier dataset builder must perform path-safety validation before it
reads the scene COCO payload. This closes the existing pre-read safety gap and
protects Incremental generation from a redirected evaluation-only source.

## Frozen detector proof

The Incremental config names the detector checkpoint explicitly. Before any
training data is loaded, its path and adjacent metadata path are checked by
the evaluation-only path guard. The checkpoint must exist and its SHA-256
must match adjacent detector metadata `model.best_sha256` with class names
exactly `["bread"]`.

The classifier orchestrator hashes the detector checkpoint before backend
training and again after best-checkpoint evaluation. Any byte change fails the
run and removes the staging directory. Metadata records both hashes and
`detector_unchanged: true`.

The detector is never passed to the classifier training backend, so the
default Incremental experiment cannot update it accidentally.

## Metrics

The validation artifact keeps the existing overall Top-1, Macro F1, and
per-class precision/recall/F1/support. For a 20-output Incremental run it also
records:

- `phase.base`: targets with `model_index` 0 through 14.
- `phase.incremental`: targets with `model_index` 15 through 19.

Each group reports sample count, Top-1, and the macro mean of the existing
per-class F1 values in that group. All values are computed only from the
fixed train-side validation manifest.

## Artifacts and atomicity

The run preserves the existing atomic staging layout:

- `config.yaml`
- `history.json`
- `metadata.json`
- `metrics.json`
- `predictions.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`

Incremental metadata adds Base checkpoint path/hash, copied-head proof,
per-class train counts and weights, and frozen detector path/before/after
hashes. A failed validation, backend call, hash check, evaluation, or publish
removes staging output and never overwrites an existing run.

## Real run and acceptance

The default run uses `datasets/derived/classifier/incremental_seed42`, the
approved Base classifier checkpoint, the approved frozen detector checkpoint,
seed 42, CUDA device 0, and the same preprocessing as Base.

Acceptance requires:

- Incremental dataset replay validation succeeds with 20 contiguous outputs.
- Base checkpoint expansion passes exact tensor-copy tests.
- All 20 train classes and all five Incremental validation classes have support.
- Full repository tests, compileall, and diff-check pass.
- The real run publishes overall, Base-group, and Incremental-group metrics.
- Detector SHA-256 is identical before and after the real run.
- No test split is read.
- A fresh independent review of the Ready PR reports no actionable finding
  before a separate merge agent performs the squash merge.
