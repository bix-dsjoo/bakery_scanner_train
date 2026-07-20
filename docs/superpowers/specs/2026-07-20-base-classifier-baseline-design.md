# Base Classifier Baseline Design

## Goal

Build the first reproducible 15-class bread-classifier baseline from the
independently validated `base_seed42` classifier dataset. The unit trains a
torchvision ResNet18 on CUDA, selects a checkpoint using train-side validation
only, evaluates ground-truth bbox crops, and records checkpoints, predictions,
metrics, source hashes, dependencies, and hardware metadata.

No Base or Incremental test image, annotation, result, path, threshold, or
checkpoint may influence this unit.

## Baseline Configuration

The checked-in baseline uses:

- source classifier run: `base_seed42`;
- architecture: torchvision ResNet18;
- official ImageNet-1K weights: `models/pretrained/resnet18-f37072fd.pth`;
- input crop: 224 pixels;
- epochs: 30;
- batch size: 64;
- seed: 42;
- device: CUDA device `0`;
- workers: 8;
- optimizer: AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- early-stopping patience: 5 epochs.

The pretrained file is a local ignored input. Its SHA-256 is recorded in every
run. Training never silently downloads a model; a missing file is an error.

## Dataset and Transforms

`ClassifierManifestDataset` consumes only a run that passes
`validate_classifier_dataset`. It loads samples from manifest `output_path`
and uses manifest `model_index` as the target. A Base run must declare phase
`base`, output dimension 15, and model indices `0..14`.

Training transforms are deterministic per worker and epoch seed: random resized
crop to 224, horizontal flip, and mild color jitter, followed by ImageNet mean
and standard deviation. Validation uses resize to 256, center crop to 224, and
the same normalization. Transform settings are serialized in config and
metadata. Test paths are rejected before any image is opened.

## Model and Training

The backend creates `torchvision.models.resnet18(weights=None)`, loads the
official ImageNet state dictionary strictly, replaces `fc` with a 15-output
linear layer, and trains all parameters with cross-entropy. Seeds are applied
to Python and PyTorch; deterministic algorithm settings and cuDNN flags are
recorded.

Each epoch records training loss and train-side validation loss, Top-1, Macro
F1, and supported-class count. `best.pt` is selected by lower validation loss;
ties prefer the earlier epoch. `last.pt` always records the final attempted
epoch. Early stopping uses validation loss only.

Checkpoint payloads include schema version, architecture, output dimension,
registry model-index mapping, source manifest hash, epoch, model state,
optimizer state where applicable, and configuration.

## Evaluation Contract

Predictions contain sample output path, ground-truth model index, predicted
model index, and confidence. Metrics include:

- Top-1 accuracy;
- Macro F1 over classes with at least one validation ground-truth sample;
- per-class precision, recall, F1, support, predicted count, and true positives;
- sample count, evaluated-class count, checkpoint hash, source split, and
  metric version.

For a class with zero ground-truth support, recall and F1 are JSON `null`. If
the class also has zero predictions, precision is `null`; otherwise precision
is `0.0`. This avoids fabricating performance for absent validation classes.

## Run Layout and Metadata

```text
runs/classifier/<run-name>/
  config.yaml
  metadata.json
  history.json
  predictions.json
  metrics.json
  checkpoints/
    best.pt
    last.pt
```

`metadata.json` records source classifier manifest path/hash, registry hash,
pretrained weight path/hash, final checkpoint hashes, Python, OS, CPU, GPU,
CUDA, torch, torchvision, Pillow and PyYAML versions, seed, transform settings,
input size, batch size, worker count, optimizer settings, deterministic flags,
and exact backend arguments. Local runs are excluded from Git.

Generation occurs under a staging directory in `runs/classifier/`; a completed
run is validated before atomic publication. Existing runs are rejected. A
training failure publishes no completed run.

## Interfaces

`src/bakery_scanner/classifier_evaluation.py` provides pure metric computation.
`src/bakery_scanner/classifier_training.py` provides strict config loading,
manifest datasets, a backend protocol, the torchvision backend, training and
checkpoint reevaluation. `src/bakery_scanner/classifier_train_cli.py` exposes:

```text
bakery-classifier train --config configs/classifier/resnet18_base.yaml
bakery-classifier evaluate --config configs/classifier/resnet18_base.yaml --checkpoint <path>
```

Both commands print the concrete dataset root, source run, train split,
validation split, pretrained model, and output path before backend work. Both
support `--json` and return exit code 1 for validation errors.

## Verification

Unit tests use small manifests and a recording backend; they do not require GPU
training. Tests cover strict config schema/ranges, test-path rejection, 15-class
mapping, absent-class metrics, duplicate predictions, exact backend arguments,
checkpoint and metadata requirements, failed-run atomicity, CLI output, and
validation-only evaluation. The final gate runs the repository Base dataset on
the RTX 5080, independently reevaluates `best.pt`, and records the first
train-side classifier baseline without a predetermined accuracy target.
