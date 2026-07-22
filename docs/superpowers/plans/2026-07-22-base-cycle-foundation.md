# Base Cycle Isolation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and independently validate the Base redesign cycle's two development scene groups, one cycle-holdout scene group, two development backgrounds, one holdout background, and three experiment seeds without reading any evaluation-only data.

**Architecture:** A new `base_cycle` module owns a strict YAML contract and an atomic, hash-bound manifest under `datasets/derived/base_cycle/<run-name>/`. The manifest inventories the existing Base scene COCO, registry, every real scene image, every allowed background, the development/holdout assignment, and seeds; downstream detector, verifier, classifier, cascade, and benchmark plans must consume this artifact instead of duplicating split constants.

**Tech Stack:** Python 3.11, pathlib, dataclasses, PyYAML 6.x, existing COCO/registry/safety validators, pytest 9.x.

## Global Constraints

- Do not read, list, stat, decode, train on, mine from, calibrate on, or select with `datasets/base/test` or `datasets/incremental/test`.
- Do not modify original images, existing COCO JSON, `datasets/class_registry.json`, or an existing derived run.
- Treat `datasets/base/val` as train-authorized scene data and keep `scene_e_*`, `scene_m_*`, and `scene_h_*` with the same numeric scene ID in one group.
- Use `datasets/class_registry.json` as the sole class authority; never substitute COCO `category_id` for `model_index`.
- Require exactly two development scene IDs, one holdout scene ID, two development backgrounds, one holdout background, and three distinct non-negative seeds.
- Publish through a staging directory and validate before atomic rename; any failure leaves no completed run and preserves an existing completed run.
- Record resolved portable paths and SHA-256 for the config, registry, COCO, every scene image, and every background.
- The cycle holdout is not a pristine never-observed dataset; metadata must label it `cycle_holdout`, never `test` or `independent_test`.
- No README command may be documented before the corresponding entry point exists and its CLI tests pass.
- This plan document remains on `codex/docs-base-inference-redesign-v2`. After the spec and plan are squash-merged, implementation starts from updated `main` on a new `codex/feat-base-cycle-foundation` branch.
- All implementation commits use `<type>[optional scope]: <imperative Korean summary>` on that single-author execution branch; never commit implementation directly to `main` or the documentation branch.

---

## Scope Decomposition

This plan intentionally implements only the common isolation contract. The approved design is executed through the following separately reviewable plans, generated after this foundation's interfaces are merged:

1. detector fold dataset and YOLO11n/YOLO26s/YOLO26m ablation;
2. out-of-fold proposal mining and breadness verifier;
3. ResNet18/YOLO26m-cls/ConvNeXt-Tiny classifier comparison;
4. single-detector/single-verifier/single-classifier cascade and CPU benchmark;
5. cycle-holdout one-shot evaluation and `frozen_v2` reporting.

None of those model-training or holdout-access actions belongs to this plan.

## File Map

- Create `src/bakery_scanner/base_cycle.py`: strict config parsing, scene/background inventory, atomic freeze, independent validation, and report contract.
- Create `src/bakery_scanner/base_cycle_cli.py`: `freeze` and `validate` commands with human and JSON output.
- Create `tests/test_base_cycle.py`: config, leakage, grouping, hash, atomicity, tamper, and relocation tests.
- Create `tests/test_base_cycle_cli.py`: CLI argument, output, and failure-exit tests.
- Create `configs/base_cycle/base_v2.yaml`: checked-in performance-independent assignment and seeds.
- Modify `pyproject.toml`: register only the implemented `bakery-base-cycle` entry point.
- Modify `README.md`: document only the implemented freeze/validate commands and the non-pristine cycle-holdout limitation.

### Public interfaces

The public module exports `BaseCycleConfig`, `BaseCycleReport`,
`load_base_cycle_config(path)`, `freeze_base_cycle(config_path, *, overwrite=False)`,
and `validate_base_cycle(dataset_root, output_root, run_name)`. Their exact types and
method bodies are defined in Tasks 1 and 2 and must not be renamed by later plans.

The manifest schema is version 1 and has exactly these top-level keys:

The manifest has exactly these top-level fields: `manifest_version` integer `1`,
`cycle_version` string `1.0.0`, UTC ISO-8601 `created_at`, `run_name`, normalized
`config`, lowercase-hex `config_sha256`, path/hash records `registry` and
`real_coco`, lists `scenes` and `backgrounds`, and integer list `seeds` equal to
`[42, 43, 44]` for the checked-in run.

Each scene record has exactly `scene_id`, `difficulty`, `split`, `path`, `sha256`, `width`, and `height`. Each background record has exactly `split`, `path`, and `sha256`. `split` is only `development` or `cycle_holdout`.

---

### Task 1: Strict Base Cycle Configuration and Inventory

**Files:**
- Create: `src/bakery_scanner/base_cycle.py`
- Create: `tests/test_base_cycle.py`

**Interfaces:**
- Consumes: `validate_coco(path, registry, "base")`, `load_class_registry(path)`, `assert_training_paths_safe(paths, dataset_root)`, `SCENE_PATTERN`.
- Produces: `BaseCycleConfig`, `BaseCycleReport`, `load_base_cycle_config()` and the internal `_prepare_inventory()` used by Tasks 2 and 3.

- [ ] **Step 1: Write strict config-loader tests**

Create a fixture that writes a 15-class Base registry, three complete scene groups (`0503`, `0509`, `0510`), a valid Base COCO, and three RGB backgrounds. Add these tests:

```python
def test_load_base_cycle_config_accepts_exact_schema(cycle_fixture) -> None:
    root, config_path = cycle_fixture

    config = load_base_cycle_config(config_path)

    assert config.dataset_root == str(root)
    assert config.output_root == "derived/base_cycle"
    assert config.run_name == "base_v2"
    assert config.development_scene_ids == ("0503", "0509")
    assert config.holdout_scene_id == "0510"
    assert config.development_backgrounds == (
        "collected/backgrounds/tray_white_square.png",
        "collected/backgrounds/tray_wood_black_surface.png",
    )
    assert config.holdout_background == (
        "collected/backgrounds/tray_wood_white_surface.png"
    )
    assert config.seeds == (42, 43, 44)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"development_scene_ids": ["0503", "0503"]}, "scene IDs"),
        ({"development_scene_ids": ["0503"]}, "two development scene IDs"),
        ({"holdout_scene_id": "0503"}, "scene IDs"),
        ({"development_backgrounds": ["same.png", "same.png"]}, "backgrounds"),
        ({"holdout_background": "collected/backgrounds/tray_white_square.png"}, "backgrounds"),
        ({"seeds": [42, 42, 44]}, "seeds"),
        ({"seeds": [42, -1, 44]}, "seeds"),
    ],
)
def test_load_base_cycle_config_rejects_invalid_assignment(
    cycle_fixture, mutation: dict[str, object], message: str
) -> None:
    _, config_path = cycle_fixture
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload.update(mutation)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match=message):
        load_base_cycle_config(config_path)


def test_load_base_cycle_config_rejects_unknown_field(cycle_fixture) -> None:
    _, config_path = cycle_fixture
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["surprise"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="fields"):
        load_base_cycle_config(config_path)
```

- [ ] **Step 2: Run the loader tests and verify red**

Run:

```powershell
python -m pytest tests/test_base_cycle.py -k "load_base_cycle_config" -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'bakery_scanner.base_cycle'`.

- [ ] **Step 3: Implement the config contract**

Start `src/bakery_scanner/base_cycle.py` with these exact constants and dataclasses:

```python
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from .coco import validate_coco
from .errors import DataValidationError
from .registry import load_class_registry
from .safety import assert_training_paths_safe
from .splits import SCENE_PATTERN

CYCLE_VERSION = "1.0.0"
MANIFEST_VERSION = 1
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_FIELDS = {
    "dataset_root",
    "output_root",
    "run_name",
    "real_coco_path",
    "development_scene_ids",
    "holdout_scene_id",
    "development_backgrounds",
    "holdout_background",
    "seeds",
}


@dataclass(frozen=True, slots=True)
class BaseCycleConfig:
    dataset_root: str
    output_root: str
    run_name: str
    real_coco_path: str
    development_scene_ids: tuple[str, str]
    holdout_scene_id: str
    development_backgrounds: tuple[str, str]
    holdout_background: str
    seeds: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BaseCycleReport:
    output_dir: Path
    manifest_path: Path
    development_scene_ids: tuple[str, str]
    holdout_scene_id: str
    development_image_count: int
    holdout_image_count: int
    development_background_count: int
    holdout_background_count: int
    seeds: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "development_scene_ids": list(self.development_scene_ids),
            "holdout_scene_id": self.holdout_scene_id,
            "development_image_count": self.development_image_count,
            "holdout_image_count": self.holdout_image_count,
            "development_background_count": self.development_background_count,
            "holdout_background_count": self.holdout_background_count,
            "seeds": list(self.seeds),
        }
```

Implement `_strict_object`, `_text`, `_text_tuple`, `_seed_tuple`, and `load_base_cycle_config`. `_text_tuple` requires exact length, non-empty strings, and unique values. `_seed_tuple` rejects bools, negative values, duplicates, and any length other than three. Reject overlap across development/holdout scene IDs and backgrounds. Require `_RUN_NAME.fullmatch(run_name)`.

- [ ] **Step 4: Write inventory failure tests**

Add tests that call the internal preparation through `freeze_base_cycle` after declaring its import:

```python
def test_prepare_inventory_rejects_test_path_before_coco_read(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    config = replace(
        load_base_cycle_config(config_path),
        real_coco_path="base/test/annotations.json",
    )
    touched = False

    def forbidden_read(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("test COCO must not be read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(DataValidationError, match="evaluation-only"):
        _prepare_inventory(config_path, config)
    assert touched is False


def test_prepare_inventory_requires_three_complete_declared_scene_groups(
    cycle_fixture,
) -> None:
    root, config_path = cycle_fixture
    (root / "base/val/scene_h_0510.jpg").unlink()

    with pytest.raises(DataValidationError, match="COCO image files"):
        _prepare_inventory(config_path, load_base_cycle_config(config_path))


def test_prepare_inventory_rejects_undeclared_scene_group(cycle_fixture) -> None:
    root, config_path = cycle_fixture
    payload = json.loads((root / "base/val/instances_val.json").read_text())
    payload["images"].extend(_scene_image_records("0511", start_id=100))
    _write_scene_images(root / "base/val", "0511")
    (root / "base/val/instances_val.json").write_text(json.dumps(payload))

    with pytest.raises(DataValidationError, match="exactly partition"):
        _prepare_inventory(config_path, load_base_cycle_config(config_path))


def test_prepare_inventory_rejects_background_outside_dataset(
    cycle_fixture, tmp_path
) -> None:
    _, config_path = cycle_fixture
    external = tmp_path / "external.png"
    Image.new("RGB", (32, 24), "white").save(external)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["holdout_background"] = str(external)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="dataset root"):
        _prepare_inventory(config_path, load_base_cycle_config(config_path))
```

- [ ] **Step 5: Implement inventory preparation**

Implement private helpers `_sha256(path)`, `_portable(path, root)`,
`_configured_path(value, root)`, `_decode_size(path, label)`,
`_scene_inventory(coco_path, development_ids, holdout_id, root)`,
`_background_inventory(development, holdout, root)`, and
`_prepare_inventory(config_path, config)`. Return types are respectively `str`,
`str`, `Path`, `tuple[int, int]`, `list[dict[str, object]]`,
`list[dict[str, str]]`, and `tuple[Path, dict[str, object]]`.

`_prepare_inventory` must:

1. resolve `dataset_root` relative to the config directory only when it is not already a valid process-relative path;
2. call `assert_training_paths_safe` on COCO and all backgrounds before any of them are read;
3. require every resolved input to remain inside the resolved dataset root;
4. load and validate the registry, then call `validate_coco(coco_path, registry, "base")`;
5. parse every COCO image filename with `SCENE_PATTERN` and require exactly the declared three IDs, each with e/m/h exactly once;
6. decode every background, require RGB-convertible positive dimensions, and reject symlinks/junction escapes;
7. return the resolved root and a deterministic inventory dictionary without `created_at`.

- [ ] **Step 6: Run Task 1 tests**

Run:

```powershell
python -m pytest tests/test_base_cycle.py -q
```

Expected: config and inventory tests pass; publication tests added in Task 2 do not exist yet.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/bakery_scanner/base_cycle.py tests/test_base_cycle.py
git commit -m "feat(data): Base cycle 격리 계약을 추가한다"
```

---

### Task 2: Atomic Freeze and Independent Replay Validation

**Files:**
- Modify: `src/bakery_scanner/base_cycle.py`
- Modify: `tests/test_base_cycle.py`

**Interfaces:**
- Consumes: `BaseCycleConfig` and deterministic inventory from Task 1.
- Produces: completed `freeze_base_cycle()` and `validate_base_cycle()` with `BaseCycleReport`.

- [ ] **Step 1: Write publication and replay tests**

```python
def test_freeze_publishes_hash_bound_manifest_atomically(cycle_fixture) -> None:
    root, config_path = cycle_fixture

    report = freeze_base_cycle(config_path)

    assert report.output_dir == (root / "derived/base_cycle/base_v2").resolve()
    assert report.development_scene_ids == ("0503", "0509")
    assert report.holdout_scene_id == "0510"
    assert report.development_image_count == 6
    assert report.holdout_image_count == 3
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "manifest_version",
        "cycle_version",
        "created_at",
        "run_name",
        "config",
        "config_sha256",
        "registry",
        "real_coco",
        "scenes",
        "backgrounds",
        "seeds",
    }
    assert {record["split"] for record in manifest["scenes"]} == {
        "development",
        "cycle_holdout",
    }
    assert not list(report.output_dir.parent.glob(".base_v2.tmp-*"))


def test_validate_replays_every_input_hash(cycle_fixture) -> None:
    root, config_path = cycle_fixture
    freeze_base_cycle(config_path)

    report = validate_base_cycle(root, "derived/base_cycle", "base_v2")

    assert report.development_image_count == 6
    assert report.holdout_image_count == 3


@pytest.mark.parametrize("target", ["scene", "background", "coco", "registry"])
def test_validate_rejects_tampered_source(cycle_fixture, target: str) -> None:
    root, config_path = cycle_fixture
    freeze_base_cycle(config_path)
    targets = {
        "scene": root / "base/val/scene_e_0503.jpg",
        "background": root / "collected/backgrounds/tray_white_square.png",
        "coco": root / "base/val/instances_val.json",
        "registry": root / "class_registry.json",
    }
    targets[target].write_bytes(b"tampered")

    with pytest.raises(DataValidationError, match="SHA-256"):
        validate_base_cycle(root, "derived/base_cycle", "base_v2")


def test_freeze_refuses_existing_run_without_overwrite(cycle_fixture) -> None:
    _, config_path = cycle_fixture
    first = freeze_base_cycle(config_path)

    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)

    assert first.manifest_path.exists()


def test_failed_overwrite_preserves_prior_valid_run(cycle_fixture, monkeypatch) -> None:
    _, config_path = cycle_fixture
    first = freeze_base_cycle(config_path)
    before = first.manifest_path.read_bytes()
    monkeypatch.setattr("bakery_scanner.base_cycle._validate_run_dir", lambda *_: (_ for _ in ()).throw(DataValidationError("forced")))

    with pytest.raises(DataValidationError, match="forced"):
        freeze_base_cycle(config_path, overwrite=True)

    assert first.manifest_path.read_bytes() == before
```

- [ ] **Step 2: Run publication tests and verify red**

Run:

```powershell
python -m pytest tests/test_base_cycle.py -k "publishes or replays or tampered or overwrite" -q
```

Expected: FAIL because freeze publication and `validate_base_cycle` are not implemented.

- [ ] **Step 3: Implement manifest publication**

Implement:

```python
def freeze_base_cycle(
    config_path: str | Path, *, overwrite: bool = False
) -> BaseCycleReport:
    source = Path(config_path).resolve(strict=False)
    config = load_base_cycle_config(source)
    root, inventory = _prepare_inventory(source, config)
    output_root = _configured_path(config.output_root, root)
    output_dir = output_root / config.run_name
    if output_dir.parent != output_root:
        raise DataValidationError("run_name must select a direct cycle directory")
    if output_dir.exists() and not overwrite:
        raise DataValidationError(f"base cycle run already exists: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{config.run_name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    backup: Path | None = None
    try:
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "cycle_version": CYCLE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": config.run_name,
            **inventory,
        }
        (staging / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _validate_run_dir(root, staging)
        if output_dir.exists():
            backup = output_root / f".{config.run_name}.backup-{uuid.uuid4().hex}"
            output_dir.rename(backup)
        staging.rename(output_dir)
        if backup is not None:
            shutil.rmtree(backup)
        return _validate_run_dir(root, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
```

Before writing, add normalized `config = asdict(config)` to the inventory. Compute
`config_sha256` from `json.dumps(config, ensure_ascii=False, sort_keys=True,
separators=(",", ":")).encode("utf-8")` so independent validation can recompute it
without depending on the config file's physical location. Store data paths relative to
dataset root with forward slashes.

- [ ] **Step 4: Implement strict independent validation**

Implement `_load_json_object`, `_validate_sha_record`, `_validate_run_dir`, and:

```python
def validate_base_cycle(
    dataset_root: str | Path, output_root: str | Path, run_name: str
) -> BaseCycleReport:
    root = Path(dataset_root).resolve(strict=False)
    if not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError("run_name is invalid")
    resolved_output_root = _configured_path(str(output_root), root)
    return _validate_run_dir(root, resolved_output_root / run_name)
```

`_validate_run_dir` must reject unknown/missing keys, versions other than 1/`1.0.0`, non-UTC or invalid `created_at`, duplicate paths, wrong split names, missing e/m/h variants, any scene/background overlap across splits, wrong counts, path traversal, symlinks/junctions, missing files, and SHA drift. It must rerun registry and COCO validation and reconstruct `BaseCycleReport` only from validated manifest data.

- [ ] **Step 5: Run the focused and full foundation tests**

Run:

```powershell
python -m pytest tests/test_base_cycle.py -q
```

Expected: all Base cycle unit tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/bakery_scanner/base_cycle.py tests/test_base_cycle.py
git commit -m "feat(data): Base cycle manifest를 원자적으로 동결한다"
```

---

### Task 3: CLI and Checked-in Cycle Assignment

**Files:**
- Create: `src/bakery_scanner/base_cycle_cli.py`
- Create: `tests/test_base_cycle_cli.py`
- Create: `configs/base_cycle/base_v2.yaml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `freeze_base_cycle()` and `validate_base_cycle()` from Task 2.
- Produces: `bakery-base-cycle freeze|validate` and the exact approved `base_v2` assignment.

- [ ] **Step 1: Write CLI tests**

```python
def test_freeze_cli_prints_concrete_assignment(monkeypatch, capsys) -> None:
    report = BaseCycleReport(
        output_dir=Path("datasets/derived/base_cycle/base_v2"),
        manifest_path=Path("datasets/derived/base_cycle/base_v2/manifest.json"),
        development_scene_ids=("0503", "0509"),
        holdout_scene_id="0510",
        development_image_count=6,
        holdout_image_count=3,
        development_background_count=2,
        holdout_background_count=1,
        seeds=(42, 43, 44),
    )
    monkeypatch.setattr(base_cycle_cli, "freeze_base_cycle", lambda *_args, **_kwargs: report)

    assert base_cycle_cli.main(["freeze", "--config", "configs/base_cycle/base_v2.yaml"]) == 0
    output = capsys.readouterr().out
    assert "development scene IDs: 0503, 0509" in output
    assert "cycle holdout scene ID: 0510" in output
    assert "seeds: 42, 43, 44" in output


def test_validate_cli_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(base_cycle_cli, "validate_base_cycle", lambda *_args: _report())

    assert base_cycle_cli.main([
        "validate",
        "--dataset-root", "datasets",
        "--output-root", "derived/base_cycle",
        "--run-name", "base_v2",
        "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["holdout_scene_id"] == "0510"


def test_cli_returns_one_for_data_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        base_cycle_cli,
        "freeze_base_cycle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataValidationError("unsafe")),
    )

    assert base_cycle_cli.main(["freeze", "--config", "bad.yaml"]) == 1
    assert "unsafe" in capsys.readouterr().err
```

- [ ] **Step 2: Run CLI tests and verify red**

Run:

```powershell
python -m pytest tests/test_base_cycle_cli.py -q
```

Expected: collection fails because `base_cycle_cli` does not exist.

- [ ] **Step 3: Implement the CLI**

Create `src/bakery_scanner/base_cycle_cli.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .base_cycle import freeze_base_cycle, validate_base_cycle
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-base-cycle",
        description="Freeze and validate the Base redesign cycle split.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--overwrite", action="store_true")
    freeze.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset-root", default="datasets")
    validate.add_argument("--output-root", default="derived/base_cycle")
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--json", action="store_true")
    return parser


def _print_human(payload: dict[str, object]) -> None:
    print(f"Base cycle validation passed: {payload['output_dir']}")
    print("development scene IDs: " + ", ".join(payload["development_scene_ids"]))
    print(f"cycle holdout scene ID: {payload['holdout_scene_id']}")
    print("seeds: " + ", ".join(str(seed) for seed in payload["seeds"]))
    print(f"manifest: {payload['manifest_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            report = freeze_base_cycle(args.config, overwrite=args.overwrite)
        else:
            report = validate_base_cycle(
                args.dataset_root, args.output_root, args.run_name
            )
    except DataValidationError as exc:
        print(f"Base cycle command failed: {exc}", file=sys.stderr)
        return 1
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the entry point and exact config**

Add to `[project.scripts]` in `pyproject.toml`:

```toml
bakery-base-cycle = "bakery_scanner.base_cycle_cli:main"
```

Create `configs/base_cycle/base_v2.yaml`:

```yaml
dataset_root: datasets
output_root: derived/base_cycle
run_name: base_v2
real_coco_path: base/val/instances_val.json
development_scene_ids:
  - "0503"
  - "0509"
holdout_scene_id: "0510"
development_backgrounds:
  - collected/backgrounds/tray_white_square.png
  - collected/backgrounds/tray_wood_black_surface.png
holdout_background: collected/backgrounds/tray_wood_white_surface.png
seeds:
  - 42
  - 43
  - 44
```

The choice is performance-independent: `0510` is the lexicographically greater of the two scene IDs that were in the prior detector training split, and `tray_wood_white_surface.png` is the lexicographically greatest background filename. The document must still state that both were used in prior project artifacts and are only isolated for the new cycle.

- [ ] **Step 5: Run CLI and config tests**

Run:

```powershell
python -m pytest tests/test_base_cycle.py tests/test_base_cycle_cli.py -q
```

Expected: all Base cycle tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add pyproject.toml configs/base_cycle/base_v2.yaml src/bakery_scanner/base_cycle_cli.py tests/test_base_cycle_cli.py
git commit -m "feat(cli): Base cycle 동결 명령을 제공한다"
```

---

### Task 4: Documentation, Real Preflight, and Repository Verification

**Files:**
- Modify: `README.md`
- Verify: `AGENTS.md`
- Verify: `docs/superpowers/specs/2026-07-16-bakery-scanner-design.md`
- Verify: `docs/superpowers/specs/2026-07-22-base-inference-redesign-design.md`

**Interfaces:**
- Consumes: implemented `bakery-base-cycle` entry point and checked-in `base_v2.yaml`.
- Produces: documented, replay-validated local Base cycle manifest; no model checkpoint or holdout metric.

- [ ] **Step 1: Add README documentation only for implemented commands**

Add a “Base v2 cycle 격리” subsection describing:

```powershell
bakery-base-cycle freeze --config configs/base_cycle/base_v2.yaml
bakery-base-cycle validate --dataset-root datasets --output-root derived/base_cycle --run-name base_v2
```

State explicitly:

- development IDs are `0503` and `0509`;
- cycle-holdout ID is `0510`;
- no model-training command in the redesign exists yet;
- the holdout is not pristine because every current scene/background influenced prior artifacts;
- freezing the manifest does not authorize reading test data or holdout metrics;
- later plans must validate this manifest and use generic pretrained weights, not existing project checkpoints that saw the holdout inputs.

- [ ] **Step 2: Run focused tests**

```powershell
python -m pytest tests/test_base_cycle.py tests/test_base_cycle_cli.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the full repository suite**

```powershell
python -m pytest -q
```

Expected: all repository tests pass with zero failures.

- [ ] **Step 4: Execute the real freeze command**

```powershell
bakery-base-cycle freeze --config configs/base_cycle/base_v2.yaml
```

Expected human output includes:

```text
development scene IDs: 0503, 0509
cycle holdout scene ID: 0510
seeds: 42, 43, 44
```

Do not pass `--overwrite` if a valid run already exists. Do not open or score the holdout images; this command inventories and hashes them only.

- [ ] **Step 5: Independently replay-validate the real artifact**

```powershell
bakery-base-cycle validate --dataset-root datasets --output-root derived/base_cycle --run-name base_v2 --json
```

Expected JSON contains `status: "ok"`, six development images, three holdout images, two development backgrounds, one holdout background, and seeds `[42, 43, 44]`.

- [ ] **Step 6: Verify no test path was recorded and inspect Git scope**

```powershell
rg -n "base/test|incremental/test" datasets/derived/base_cycle/base_v2/manifest.json
git status --short
git diff --check
```

Expected: `rg` exits 1 with no matches; Git shows only the planned source, tests, config, dependency metadata, and README changes. The generated `datasets/derived/` artifact remains ignored.

- [ ] **Step 7: Commit Task 4**

```powershell
git add README.md
git commit -m "docs(data): Base cycle 격리 절차를 설명한다"
```

- [ ] **Step 8: Independent review gate**

Request an independent agent instance to review the ready diff against:

- `AGENTS.md` test-isolation and scene-group rules;
- `docs/superpowers/specs/2026-07-22-base-inference-redesign-design.md` Sections 5, 6, 8, and 9;
- this plan's public interfaces and manifest schema.

If the diff changes, rerun focused tests, the full suite, real replay validation, and independent review before PR readiness.

---

## Plan Self-Review Checklist

- Spec coverage: this phase covers holdout/split/background/seed freeze, manifest provenance, path safety, atomic publication, validation, CLI, and documentation.
- Deliberate exclusions: detector training, hard-negative mining, verifier/classifier training, cascade inference, CPU timing, holdout scoring, and test evaluation belong to the five named follow-up plans.
- Type consistency: all tasks use the same `BaseCycleConfig`, `BaseCycleReport`, `freeze_base_cycle`, and `validate_base_cycle` signatures.
- No test-result dependency: the checked-in assignment is derived from filenames and prior split membership, not from accuracy, FP location, confidence, or test output.
- No unimplemented README command: documentation is added only after the CLI entry point and tests exist in Task 3.
