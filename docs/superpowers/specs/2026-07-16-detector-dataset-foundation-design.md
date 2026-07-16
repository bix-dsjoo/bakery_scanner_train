# Detector Dataset Foundation Design

## Scope

Build and validate a class-agnostic detector dataset from the train-side real scenes in `datasets/base/val` and one named synthetic run in `datasets/derived/synthetic`. This work stops at COCO dataset assembly; detector training and inference remain out of scope.

## Public interface and layout

The Python API exposes `DetectorDatasetConfig`, `build_detector_dataset(...)`, and `validate_detector_dataset(...)`. The `bakery-detector-data` command provides matching `generate` and `validate` subcommands.

Every completed run is a direct child of `datasets/derived/detector/`:

```text
<run>/
  manifest.json
  train/
    images/
    instances.json
  validation/
    images/
    instances.json
```

The builder writes a sibling staging directory, validates it completely, then renames it into place. Existing runs require `--overwrite`; replacement uses a backup and rollback so a failed build does not damage the prior valid run.

## Inputs and detector conversion

The real input is `datasets/base/val/instances_val.json`. It is validated against `class_registry.json` as Base data before use. The synthetic input is one named run and is replay-validated with the existing synthetic validator before use. Evaluation-only paths are rejected by the existing training-path safety API.

All output annotations use exactly one COCO category: `{"id": 1, "name": "bread"}`. Original COCO `category_id` values are stored only in each sample's provenance. Detector output never uses or records `model_index` as a category identifier. Images with no annotations are valid and produce an empty annotation list.

## Leakage-safe deterministic split

Each image is a node. Resource keys connect nodes that must stay together:

- real images share a `real-scene:<scene_id>` key across `scene_e`, `scene_m`, and `scene_h`;
- synthetic images share resolved source-object paths and source-object SHA-256 values;
- synthetic images share resolved background paths and background SHA-256 values;
- byte-identical input images share an image SHA-256 key.

Connected components are indivisible split groups. Components are ordered with a seed-controlled deterministic shuffle. A subset-sum search chooses a non-empty validation subset whose image count is closest to `validation_fraction * total_images`; train must also remain non-empty. If fewer than two leakage components exist, a safe train/validation split is impossible and generation fails without output.

## Manifest and validation

Manifest version 1 records the builder version, seed, requested validation fraction, input COCO and synthetic manifest paths and hashes, split summaries, and one record per image. A sample record contains origin, source/output paths and SHA-256, dimensions, split, original annotations, and either real scene ID or synthetic background/object provenance. Synthetic object provenance preserves each original `category_id`, source path/hash, transform, and bbox.

Validation is independent of the generation result object and checks:

- exact manifest schema and supported versions;
- all configured input paths remain training-safe;
- source COCO, synthetic manifest, source assets, output images, and output COCO hashes;
- exact file inventory, decodable image dimensions, COCO references, positive in-bounds bboxes, and the single `bread` category;
- manifest-to-COCO agreement, including empty scenes;
- real scene grouping and synthetic source/background path-and-hash disjointness across splits;
- image hash disjointness across splits.

Missing, extra, altered, or undecodable files are errors. No original image, source COCO, registry, or synthetic run is modified.

## Testing and actual-data check

Tests are written first and observed failing before implementation. Unit and CLI tests cover conversion, determinism, leakage grouping, provenance, empty scenes, invalid bbox/test paths, tampering, missing files, impossible splits, atomic failure, and JSON CLI output. After unit tests pass, a small valid synthetic input derived from the repository's actual source images and backgrounds is combined with the actual `datasets/base/val` scenes in an isolated temporary dataset, then validated through a separate CLI invocation. A persistent repository detector run is produced only when a named repository synthetic run exists.
