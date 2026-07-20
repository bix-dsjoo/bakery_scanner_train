# Classifier Data Foundation Design

## Goal

Create a reproducible, independently validated train-side dataset for the Base
15-class and Incremental 20-class bread classifiers. The dataset maps every
label through `datasets/class_registry.json` `model_index`, never reads either
test split, preserves source provenance, and provides deterministic train and
validation partitions suitable for model and checkpoint selection.

## Scope

This unit builds and validates classifier data; it does not train a classifier.
It consumes the registered Base and Incremental single-object directories and
the Base train-side scene COCO file. It publishes derived files only under
`datasets/derived/classifier/<run-name>/` and never edits source images, COCO
files, or the class registry.

Two phases are supported:

- `base`: model indices `0..14`, all Base single-object images for training,
  Base scene crops split by physical scene group for train and validation;
- `incremental`: model indices `0..19`, Base data as above, six of seven
  Incremental single-object images per class for training, and one per class
  for validation using a deterministic seed-controlled split.

The default repository contract requires exactly 84 Base images and 7
Incremental images in every registered class directory. These expected counts
are explicit reproducibility settings recorded in the manifest; count drift is
an error rather than an implicit change to the train/validation split.

Base single-object images are kept in training because their filenames contain
no independently verifiable physical-item grouping metadata. Base validation is
therefore based on real scene crops. Incremental validation uses held-out
single-object images until train-authorized Incremental scenes are collected;
the manifest marks this source domain explicitly so results cannot be confused
with scene-level generalization.

## Output Layout

```text
datasets/derived/classifier/<run-name>/
  manifest.json
  train/
    <model-index>/
      <deterministic-name>.<ext>
  validation/
    <model-index>/
      <deterministic-name>.<ext>
```

Single-object files are copied byte-for-byte. Scene annotations are clipped to
the validated image bounds and cropped as RGB PNG files. Empty normal scenes
produce no classifier samples. Each output name is derived from source kind,
source image identity, annotation identity, and content hash so collisions are
errors rather than silent overwrites.

## Manifest Contract

`manifest.json` records:

- manifest and builder versions;
- Python, platform, and Pillow versions used to encode and replay crop files;
- phase, seed, validation fraction, dataset root, and output run name;
- registry path and SHA-256;
- every source COCO, source image, and single-object file path and SHA-256;
- each sample's split, output path and SHA-256, source kind, source path,
  source SHA-256, optional annotation ID and bbox, category ID, model index,
  class phase, and validation domain;
- every Base scene image path, SHA-256, dimensions, scene ID, and split,
  including normal images with no annotations;
- per-split, per-class, per-phase, and per-source-kind counts.

All stored paths are portable, dataset-root-relative POSIX paths. The validator
rejects absolute paths, `..` traversal, test paths, unknown fields, duplicate
records, non-contiguous output dimensions, or a category/model mapping that
differs from the current registry.

## Split Rules

Base scene crops reuse `split_scene_paths` with the configured seed and
validation fraction, so `scene_e_*`, `scene_m_*`, and `scene_h_*` variants with
the same scene ID remain together. All annotations from one scene image inherit
that image split.

For Incremental single-object data, each class is independently shuffled with
a stable seed derived from the master seed and class identity. The validation
count is `max(1, round(image_count * validation_fraction))` and must leave at
least one training image. Base single-object files remain in training for both
phases.

## Atomicity and Validation

Generation occurs in a staging directory below
`datasets/derived/classifier/`. The complete tree and manifest are validated
before atomic publication. Existing runs require `overwrite=True`; failed
replacement before the staging rename restores the previous valid run. The
staging rename is the commit point; cleanup of the hidden previous-run backup
is retried and is best-effort, so a cleanup-only failure cannot report the
operation as failed after a valid new run has already been committed.

Independent validation reopens every image, recomputes every hash, revalidates
source COCO and registry data, regenerates every scene crop, and compares crop
bytes and dimensions. Missing, extra, altered, undecodable, or mislabeled files
are errors. Validation also replays the configured split and rejects any
manifest disagreement. Resolved scene paths are checked against evaluation-only
roots before any scene image is decoded. The recorded Pillow version must match
the validation runtime because exact PNG bytes are part of the replay contract.

## Interfaces

`src/bakery_scanner/classifier_dataset.py` provides:

```python
@dataclass(frozen=True)
class ClassifierDatasetConfig:
    dataset_root: Path
    run_name: str
    phase: Literal["base", "incremental"]
    seed: int = 42
    validation_fraction: float = 0.2
    expected_base_images_per_class: int = 84
    expected_incremental_images_per_class: int = 7

def build_classifier_dataset(
    config: ClassifierDatasetConfig,
    *,
    overwrite: bool = False,
) -> ClassifierDatasetReport: ...

def validate_classifier_dataset(
    dataset_root: str | Path,
    run_name: str,
) -> ClassifierValidationReport: ...
```

`src/bakery_scanner/classifier_data_cli.py` exposes `generate` and `validate`
subcommands through the `bakery-classifier-data` console script. Both support
human-readable and `--json` output and return exit code 1 for validation errors.

## Verification

Automated tests cover exact registry mapping, deterministic split replay,
scene-group isolation, byte-preserving single-object copies, exact crop pixels,
Base 15-class and Incremental 20-class output dimensions, test-path rejection,
tampering, missing/extra files, registry and source mutation, unsafe names,
atomic overwrite restoration, and CLI success/failure behavior.
