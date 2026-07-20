# Model Artifact Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move downloaded pretrained weights and the completed detector run into their documented long-term locations without changing any model or result bytes.

**Architecture:** Reusable downloaded weights live under `models/pretrained/`, while every file belonging to the completed detector experiment stays together under `runs/detector/yolo11n_base_seed42/`. A checked-in configuration contract test fixes the pretrained path, and pre/post SHA-256 comparison plus an Ultralytics load smoke test verifies the ignored local artifacts.

**Tech Stack:** PowerShell, Python 3.11, PyYAML, pytest, Ultralytics YOLO

## Global Constraints

- Do not read, move, modify, or evaluate `datasets/base/test` or `datasets/incremental/test`.
- Do not modify existing dataset images, COCO JSON, or `datasets/class_registry.json`.
- Do not retrain the detector, regenerate predictions, or rewrite historical run metadata.
- Preserve every moved file byte-for-byte and fail on any destination collision.
- Keep pretrained weights and detector runs ignored by Git.
- Run all filesystem moves in PowerShell with resolved absolute paths under `C:\workspace\bakery_scanner_train`.
- Make tracked changes only on `codex/chore-organize-model-artifacts` and use a Korean imperative commit message.

---

## File Structure

- Create locally: `models/pretrained/yolo11n.pt` — YOLO11n pretrained detector input, ignored by Git.
- Create locally: `models/pretrained/yolo26n.pt` — preserved alternate pretrained detector input, ignored by Git.
- Create locally: `runs/detector/yolo11n_base_seed42/` — the complete existing detector run, ignored by Git.
- Modify: `configs/detector/yolo11n_base.yaml` — point the baseline at the new pretrained path.
- Modify: `.gitignore` — explicitly ignore `models/pretrained/` and retain `runs/detector/`.
- Modify: `README.md` — document the new input/output layout and pretrained path.
- Modify: `tests/test_detector_training.py` — lock the checked-in baseline configuration to the new path.
- Preserve unchanged: every file below the existing `artifacts/` directory, including JSON/YAML provenance, checkpoints, backend diagnostics, and evaluation output.

### Task 1: Relocate and Verify All Model Artifacts

**Files:**
- Move: `artifacts/pretrained/yolo11n.pt` → `models/pretrained/yolo11n.pt`
- Move: `artifacts/pretrained/yolo26n.pt` → `models/pretrained/yolo26n.pt`
- Move: `artifacts/detector/yolo11n_base_seed42/` → `runs/detector/yolo11n_base_seed42/`
- Modify: `configs/detector/yolo11n_base.yaml`
- Modify: `.gitignore`
- Modify: `README.md`
- Test: `tests/test_detector_training.py`

**Interfaces:**
- Consumes: the approved layout in `docs/superpowers/specs/2026-07-20-model-artifact-layout-design.md` and the existing `DetectorTrainingConfig` returned by `load_detector_training_config(path: str | Path)`.
- Produces: `DetectorTrainingConfig.model == "models/pretrained/yolo11n.pt"`, a loadable single-class detector at `runs/detector/yolo11n_base_seed42/checkpoints/best.pt`, and an empty/removed `artifacts/` path.

- [ ] **Step 1: Add the failing checked-in configuration contract test**

Add this test immediately after `test_load_detector_training_config_accepts_baseline` in `tests/test_detector_training.py`:

```python
def test_checked_in_baseline_uses_pretrained_model_directory() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    config = load_detector_training_config(
        repository_root / "configs" / "detector" / "yolo11n_base.yaml"
    )

    assert config.model == "models/pretrained/yolo11n.pt"
```

- [ ] **Step 2: Run the contract test and verify the current path fails**

Run:

```powershell
python -m pytest tests/test_detector_training.py::test_checked_in_baseline_uses_pretrained_model_directory -q
```

Expected: FAIL because the checked-in configuration currently returns `yolo11n.pt`.

- [ ] **Step 3: Preflight every path, record all hashes in memory, and move the ignored local artifacts**

Run this as one PowerShell command from `C:\workspace\bakery_scanner_train`. It validates exact absolute paths and all destination collisions before the first move, records every source-file hash, moves the two source directories on the same filesystem, verifies every target hash, and removes only empty source parents.

```powershell
$repoRoot = (Resolve-Path -LiteralPath '.').Path
$expectedRoot = 'C:\workspace\bakery_scanner_train'
if (-not [StringComparer]::OrdinalIgnoreCase.Equals($repoRoot, $expectedRoot)) {
    throw "Unexpected repository root: $repoRoot"
}

$sourcePretrained = Join-Path $repoRoot 'artifacts\pretrained'
$sourceRun = Join-Path $repoRoot 'artifacts\detector\yolo11n_base_seed42'
$targetModelsRoot = Join-Path $repoRoot 'models'
$targetPretrained = Join-Path $targetModelsRoot 'pretrained'
$targetRunsRoot = Join-Path $repoRoot 'runs\detector'
$targetRun = Join-Path $targetRunsRoot 'yolo11n_base_seed42'

foreach ($source in @($sourcePretrained, $sourceRun)) {
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Missing source directory: $source"
    }
    if (-not [IO.Path]::GetFullPath($source).StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source escapes repository root: $source"
    }
}
foreach ($target in @($targetPretrained, $targetRun)) {
    if (Test-Path -LiteralPath $target) {
        throw "Destination already exists: $target"
    }
    if (-not [IO.Path]::GetFullPath($target).StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Destination escapes repository root: $target"
    }
}

$records = @(
    Get-ChildItem -LiteralPath $sourcePretrained -Recurse -File | ForEach-Object {
        [pscustomobject]@{
            Source = $_.FullName
            Target = Join-Path $targetPretrained ([IO.Path]::GetRelativePath($sourcePretrained, $_.FullName))
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }
    Get-ChildItem -LiteralPath $sourceRun -Recurse -File | ForEach-Object {
        [pscustomobject]@{
            Source = $_.FullName
            Target = Join-Path $targetRun ([IO.Path]::GetRelativePath($sourceRun, $_.FullName))
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }
)
if ($records.Count -ne 30) {
    throw "Expected 30 artifact files, found $($records.Count)"
}

New-Item -ItemType Directory -Path $targetModelsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $targetRunsRoot -Force | Out-Null
Move-Item -LiteralPath $sourcePretrained -Destination $targetPretrained
Move-Item -LiteralPath $sourceRun -Destination $targetRun

foreach ($record in $records) {
    if (-not (Test-Path -LiteralPath $record.Target -PathType Leaf)) {
        throw "Moved file is missing: $($record.Target)"
    }
    $actualHash = (Get-FileHash -LiteralPath $record.Target -Algorithm SHA256).Hash
    if ($actualHash -ne $record.SHA256) {
        throw "Hash mismatch after move: $($record.Target)"
    }
}

$sourceDetectorParent = Join-Path $repoRoot 'artifacts\detector'
$sourceArtifacts = Join-Path $repoRoot 'artifacts'
if ((Get-ChildItem -LiteralPath $sourceDetectorParent -Force).Count -ne 0) {
    throw "Source detector parent is not empty: $sourceDetectorParent"
}
Remove-Item -LiteralPath $sourceDetectorParent
if ((Get-ChildItem -LiteralPath $sourceArtifacts -Force).Count -ne 0) {
    throw "Source artifacts directory is not empty: $sourceArtifacts"
}
Remove-Item -LiteralPath $sourceArtifacts

$records | Select-Object Target, SHA256 | Format-Table -AutoSize
```

Expected: 30 target files are printed with their SHA-256 hashes; `artifacts/` no longer exists. If the count differs before any move, inspect the new inventory and amend the approved design instead of changing the expected count ad hoc.

- [ ] **Step 4: Update the checked-in layout contract**

Change only this line in `configs/detector/yolo11n_base.yaml`:

```yaml
model: models/pretrained/yolo11n.pt
```

Replace the model-artifact portion of `.gitignore` with:

```gitignore
# Downloaded pretrained weights and completed detector runs are local artifacts.
models/pretrained/
runs/detector/
*.pt
```

In the `README.md` project structure block, place these entries after `configs/detector/`:

```text
models/pretrained/                 다운로드한 사전학습 모델 가중치
runs/detector/                    로컬 checkpoint, 예측, metric과 환경 metadata
```

Replace the baseline description sentence with:

```markdown
기본 기준선은 `configs/detector/yolo11n_base.yaml`에 고정되어 있습니다. `models/pretrained/yolo11n.pt`, 입력 크기 640, epoch 50, batch 16, seed 42, CUDA device 0과 train-side early stopping을 사용합니다. 다운로드한 사전학습 가중치는 `models/pretrained/`에 두며 Git에 포함하지 않습니다.
```

- [ ] **Step 5: Run the contract test and verify it passes**

Run:

```powershell
python -m pytest tests/test_detector_training.py::test_checked_in_baseline_uses_pretrained_model_directory -q
```

Expected: `1 passed`.

- [ ] **Step 6: Verify paths, hashes, ignore rules, configuration resolution, and detector loading**

Run:

```powershell
$required = @(
    'models\pretrained\yolo11n.pt',
    'models\pretrained\yolo26n.pt',
    'runs\detector\yolo11n_base_seed42\checkpoints\best.pt',
    'runs\detector\yolo11n_base_seed42\checkpoints\last.pt',
    'runs\detector\yolo11n_base_seed42\backend\backend\weights\best.pt',
    'runs\detector\yolo11n_base_seed42\backend\backend\weights\last.pt'
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required relocated artifact is missing: $path"
    }
}
if (Test-Path -LiteralPath 'artifacts') {
    throw 'Legacy artifacts directory still exists'
}

$best = (Get-FileHash -LiteralPath 'runs\detector\yolo11n_base_seed42\checkpoints\best.pt' -Algorithm SHA256).Hash
$backendBest = (Get-FileHash -LiteralPath 'runs\detector\yolo11n_base_seed42\backend\backend\weights\best.pt' -Algorithm SHA256).Hash
$last = (Get-FileHash -LiteralPath 'runs\detector\yolo11n_base_seed42\checkpoints\last.pt' -Algorithm SHA256).Hash
$backendLast = (Get-FileHash -LiteralPath 'runs\detector\yolo11n_base_seed42\backend\backend\weights\last.pt' -Algorithm SHA256).Hash
if ($best -ne 'CA109B8A3CEBB92C31A11D0B82DD532E9943E59A0E009095BFAADA106C0E151B' -or $best -ne $backendBest) {
    throw 'best.pt hash verification failed'
}
if ($last -ne $backendLast) {
    throw 'last.pt backend copy differs from the published checkpoint'
}

git check-ignore -v -- 'models/pretrained/yolo11n.pt' 'models/pretrained/yolo26n.pt' 'runs/detector/yolo11n_base_seed42/checkpoints/best.pt'
python -c "from pathlib import Path; from bakery_scanner.detector_training import load_detector_training_config; c=load_detector_training_config('configs/detector/yolo11n_base.yaml'); p=Path(c.model); assert p.is_file(), p; print({'model': c.model, 'resolved': str(p.resolve())})"
python -c "from ultralytics import YOLO; p=r'runs/detector/yolo11n_base_seed42/checkpoints/best.pt'; m=YOLO(p); assert m.task == 'detect'; assert m.names == {0: 'bread'}; print({'task': m.task, 'names': m.names, 'checkpoint': p})"
```

Expected: Git reports `models/pretrained/` and `runs/detector/` ignore rules; the configuration resolves an existing pretrained checkpoint; Ultralytics reports task `detect` and names `{0: 'bread'}`.

- [ ] **Step 7: Run the full automated test suite**

Run:

```powershell
python -m pytest -q
```

Expected: 126 tests pass, including the new configuration contract test.

- [ ] **Step 8: Inspect the tracked diff and confirm ignored model files are not staged**

Run:

```powershell
git diff --check
git diff -- .gitignore README.md configs/detector/yolo11n_base.yaml tests/test_detector_training.py
git status --short
```

Expected: no whitespace errors; only `.gitignore`, `README.md`, `configs/detector/yolo11n_base.yaml`, and `tests/test_detector_training.py` are modified. Neither `models/pretrained/` nor `runs/detector/` appears in status.

- [ ] **Step 9: Commit the tracked layout change**

Run:

```powershell
git add -- .gitignore README.md configs/detector/yolo11n_base.yaml tests/test_detector_training.py
git commit -m "chore(artifact): 모델 산출물 경로를 정리한다"
```

Expected: one commit containing only the layout contract, configuration, documentation, and test changes. The ignored local artifacts are not part of the commit.
