# Model Artifact Layout Design

## Goal

Separate downloaded model inputs from generated experiment outputs while preserving the completed detector baseline byte-for-byte.

The repository will use `models/pretrained/` for downloaded pretrained weights and `runs/detector/` for complete detector training runs. The existing `artifacts/` directory will no longer be used.

## Scope

This change will:

- move `artifacts/pretrained/yolo11n.pt` to `models/pretrained/yolo11n.pt`;
- move `artifacts/pretrained/yolo26n.pt` to `models/pretrained/yolo26n.pt`;
- move the complete `artifacts/detector/yolo11n_base_seed42/` directory to `runs/detector/yolo11n_base_seed42/`;
- update the checked-in YOLO11n baseline configuration to load `models/pretrained/yolo11n.pt`;
- update `.gitignore` and `README.md` so the documented layout matches the implementation;
- remove the empty `artifacts/` directory after every move and verification succeeds.

The change will not retrain the detector, regenerate predictions, alter metrics, or edit any dataset file.

## Target Layout

```text
models/
  pretrained/
    yolo11n.pt
    yolo26n.pt
runs/
  detector/
    yolo11n_base_seed42/
      config.yaml
      metadata.json
      metrics.json
      predictions.json
      checkpoints/
        best.pt
        last.pt
      backend/
      evaluation/
```

`models/pretrained/` contains reusable inputs that were obtained independently of a project experiment. `runs/detector/<run-name>/` contains one self-contained experiment result, including checkpoints, backend diagnostics, evaluation output, metrics, and provenance.

## Move Semantics

All source and destination paths are resolved under the repository root before any move. The destination paths must not already exist. Files are moved on the same filesystem without rewriting their contents.

Before moving, SHA-256 hashes are recorded for:

- both pretrained checkpoints;
- `checkpoints/best.pt` and `checkpoints/last.pt`;
- the duplicate backend copies under `backend/backend/weights/`;
- `config.yaml`, `metadata.json`, `metrics.json`, and `predictions.json`.

After moving, every recorded hash must match. The project checkpoint and backend checkpoint copies must remain identical. The detector checkpoint must still load as an Ultralytics detection model with the single class mapping `{0: "bread"}`.

If a destination collision or verification failure occurs, the process stops. A source directory is removed only after all of its expected contents have been moved and verified. The repository-level `artifacts/` directory is removed only when empty.

## Configuration and Documentation

`configs/detector/yolo11n_base.yaml` changes only its pretrained model path:

```yaml
model: models/pretrained/yolo11n.pt
```

The run name, detector dataset, seed, device, thresholds, and all other training settings remain unchanged.

`.gitignore` will explicitly ignore `models/pretrained/` and continue to ignore `runs/detector/`. `README.md` will document the two directories and use the new pretrained path when describing the detector baseline. No unimplemented command will be added.

## Provenance

The completed run's JSON and YAML files are historical records of the original training execution. Absolute paths inside `metadata.json` may refer to the former training worktree; these values will not be rewritten because doing so would falsify recorded provenance. The run remains identifiable through its checkpoint and source-manifest hashes.

The existing validation result is train-side only: AP50 and Recall are both `1.0` over 15 ground-truth objects in three validation images. It is not a test-set result and must not be used to tune test-dependent settings.

## Verification

The completed change must pass all of the following checks:

1. Every source path under `artifacts/` is absent and every target path exists.
2. All pre-move and post-move SHA-256 hashes match.
3. `best.pt` loads with Ultralytics as task `detect` and class names `{0: "bread"}`.
4. The baseline configuration resolves `models/pretrained/yolo11n.pt` to an existing file.
5. `git check-ignore` confirms pretrained weights and detector run outputs are ignored.
6. The full automated test suite passes.
7. `git status` contains only the intended tracked configuration, documentation, ignore-rule, design, and plan changes.

## Git Workflow

Tracked changes are made on `codex/chore-organize-model-artifacts`, created from the latest `main`. The implementation is committed with Korean imperative commit messages and published through a PR. The local weight and run files remain ignored and are not added to Git.
