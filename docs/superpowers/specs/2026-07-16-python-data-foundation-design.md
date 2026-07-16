# Python Data Foundation Design

## Scope

This foundation validates the existing dataset before model training. It does not implement detector or classifier training, inference, synthetic data generation, or evaluation.

## Architecture

The project uses a `src` layout and a small `bakery_scanner` package. Runtime code uses the Python standard library; Pillow is an explicit dependency only for verifying that decoded image dimensions match COCO metadata. The console script `bakery-audit` calls the same public validation functions that future training entry points will reuse.

Modules have narrow responsibilities:

- `errors.py`: one `DataValidationError` type for fail-fast validation failures.
- `registry.py`: load and validate `class_registry.json`; preserve the distinction between COCO `category_id` and model `model_index`.
- `coco.py`: validate COCO structure, unique identifiers, file references, decoded image sizes, category references, and bbox bounds.
- `safety.py`: reject any training input under `datasets/base/test` or `datasets/incremental/test`, including normalized relative paths and descendants.
- `splits.py`: extract scene IDs and deterministically split complete `scene_e_*`, `scene_m_*`, `scene_h_*` groups.
- `audit.py`: discover the fixed project dataset layout, combine validations, and produce JSON-serializable statistics without modifying data.
- `cli.py`: expose the implemented audit command with text or JSON output and a non-zero exit code on validation errors.

## Validation Rules

Registry validation requires version `1`, exactly 20 entries, unique category IDs, model indices, canonical names, and folder names, continuous model indices `0..19`, valid phases, and exactly 15 Base plus 5 Incremental classes. Dataset class directories must match the registry phase and folder mapping. Counts are reported rather than hard-coded so future authorized data additions do not require code changes.

COCO validation requires list-shaped `images`, `annotations`, and `categories`; unique image, annotation, and category IDs; one-to-one correspondence between declared image files and supported image files in the annotation directory; positive declared and decoded dimensions; matching decoded dimensions; valid image/category references; four finite numeric bbox values; positive width and height; and bbox coordinates fully inside the declared image. COCO categories must match the registry by `category_id` and canonical name and must belong to the expected Base or Incremental phase.

The audit is read-only and may inspect evaluation splits to establish integrity. This does not grant training access. Any future training entry point must call `assert_training_paths_safe` before consuming paths.

## Scene Split

`scene_[emh]_<digits>` filenames share the trailing numeric scene ID. Splitting operates on those IDs, not files. A local seeded random generator shuffles sorted group IDs; validation receives a rounded fraction with at least one group and at least one training group. Invalid names, duplicate paths, an invalid fraction, or fewer than two groups are errors. The function returns sorted train and validation path tuples and verifies no group overlap.

## CLI and Output

`bakery-audit --dataset-root datasets` runs the complete registry, class-directory, and COCO audit. `--json` emits machine-readable results. `--seed` and `--validation-fraction` control a proposed train-side split of `datasets/base/val`; they never write split files. The report includes registry phase counts, per-class single-object counts, per-COCO-split image/annotation/category counts, and scene train/validation IDs.

## Testing

Pytest tests create temporary registries, images, and COCO files. Each production behavior is introduced with a failing test, observed failing for the intended missing behavior, then minimally implemented. Tests cover valid data and the required failures: duplicate/non-continuous mappings, phase mismatch, missing/extra images, invalid references, decoded size mismatch, invalid/out-of-bounds bboxes, forbidden training paths, grouped deterministic splits, CLI success, JSON output, and CLI failure.

The final verification runs the complete pytest suite and the CLI against the real `datasets` directory. No dataset file is modified, and no commit is created.
