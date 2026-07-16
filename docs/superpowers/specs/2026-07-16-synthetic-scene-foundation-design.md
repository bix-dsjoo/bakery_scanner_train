# Synthetic Scene Foundation Design

## Scope

This change adds deterministic synthetic scene generation and validation for future class-agnostic detector training. It does not add detector or classifier training, inference, evaluation, threshold selection, or model configuration.

Generation may read only registered single-object class directories and explicitly supplied background images. Evaluation-only paths are rejected before any image is opened. Existing dataset images, COCO files, and `class_registry.json` remain read-only. Every generated file is placed under `datasets/derived/synthetic/<run-name>/`.

## Architecture

`bakery_scanner.synthetic` owns generation, manifest replay, and strict validation. A `SyntheticConfig` records the seed, scene and object counts, class phase, foreground threshold, and transform ranges. `generate_synthetic_dataset` discovers source objects through the class registry, checks all source and background paths with the training-path safety guard, creates a named run, and returns a `SyntheticGenerationReport`. `validate_synthetic_dataset` loads the manifest, validates its schema and paths, replays every scene from the recorded source files and transforms, and returns a `SyntheticValidationReport` only if the replayed PNG bytes, image hash, dimensions, categories, and bboxes all agree.

`bakery_scanner.synthetic_cli` exposes `generate` and `validate` subcommands through the `bakery-synthetic` console script and `python -m bakery_scanner.synthetic_cli`. Validation failures use the existing `DataValidationError` and exit code `1`.

## Object Extraction and Transforms

The current single-object JPEGs have white backgrounds and no masks. The generator derives an alpha mask by treating pixels whose RGB channels are all at or above the configured foreground threshold as background. A source with an empty mask or a mask covering the entire image is rejected. The foreground is cropped to the mask bounds before transformation.

For each object, a seeded local random generator selects a source and records:

- the registry `category_id` (never `model_index`),
- target size as a fraction of the background's shorter side,
- the resulting resize scale and pixel size,
- rotation in degrees,
- brightness and contrast factors,
- integer placement coordinates,
- the final `[x, y, width, height]` bbox.

RGBA resize and rotation use fixed Pillow resampling modes. The final bbox is calculated from the transformed alpha mask after rotation, then translated to scene coordinates. Placement keeps the transformed mask inside the background and rejects rectangular bbox overlap so every label describes a fully visible object. Failure to place the requested object count within a fixed attempt limit is an error, not a partial result.

## Determinism and Output

Input paths are sorted before sampling. A master `random.Random(seed)` produces a recorded seed for each scene, and each scene is generated only from its local seed. Images are RGB PNG files to avoid lossy and platform-dependent JPEG output. JSON is written with stable key ordering and formatting.

The run manifest has schema version `1` and records the generator version, Pillow version, master seed, complete config, and scenes. Each scene records its seed, output filename, SHA-256, dimensions, background path, and objects. Each object records its source path, `category_id`, transform values, and bbox. Source and background paths are stored relative to the manifest when possible and resolved relative to it during replay.

The run directory must be a direct child of `datasets/derived/synthetic/`. A new run refuses to overwrite an existing directory unless `overwrite=True`. Overwrite is limited to that validated run directory and never touches source data.

## Validation

Validation fails for a missing or malformed manifest, unsupported schema or generator version, unsafe run location, evaluation-only source/background references, missing or undecodable inputs, unknown `category_id`, invalid transforms, missing/extra output images, out-of-bounds or non-positive bboxes, hash mismatch, bbox mismatch, dimension mismatch, or replayed bytes that differ from the stored image.

Replay is the source of truth for image/bbox consistency: it reconstructs each scene from the manifest without random sampling, calculates bboxes from transformed alpha masks, and encodes the expected PNG. This checks more than manifest self-consistency and detects tampering with either image pixels or annotations.

## Testing and Documentation

Pytest uses temporary registered cutouts and backgrounds. Tests are written and observed failing before implementation. They cover required manifest fields, category mapping, deterministic regeneration, bbox/pixel agreement, output confinement, test-path rejection, overwrite behavior, tampered images and bboxes, missing files, and both CLI subcommands.

The README documents only commands implemented and exercised in this change. Real empty-tray background images are stored under `datasets/collected/backgrounds/` and can be passed directly to the generation CLI. Initial implementation smoke verification used a temporary plain background before those assets were added.
