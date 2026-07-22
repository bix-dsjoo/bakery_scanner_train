# Base Cycle Isolation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and independently validate the Base redesign cycle's two development scene groups, one cycle-holdout scene group, two development backgrounds, one holdout background, and three experiment seeds without reading any evaluation-only data.

**Architecture:** A new `base_cycle` module owns a strict YAML contract and an atomic, hash-bound manifest under `datasets/derived/base_cycle/<run-name>/`. The manifest inventories the existing Base scene COCO, registry, every real scene image, every allowed background, the development/holdout assignment, and seeds; downstream detector, verifier, classifier, cascade, and benchmark plans must consume this artifact instead of duplicating split constants.

**Tech Stack:** Python 3.11, pathlib, dataclasses, PyYAML 6.x, existing COCO/registry/safety validators, pytest 9.x.

## Global Constraints

- Do not read, list, stat, decode, train on, mine from, calibrate on, or select with `datasets/base/test` or `datasets/incremental/test`.
- Lexically reject a config path or repository-root argument under either evaluation subtree before any resolve, stat, list, open, or decode.
- Do not modify original images, existing COCO JSON, `datasets/class_registry.json`, or an existing derived run.
- Treat `datasets/base/val` as train-authorized scene data and keep `scene_e_*`, `scene_m_*`, and `scene_h_*` with the same numeric scene ID in one group.
- Use `datasets/class_registry.json` as the sole class authority; never substitute COCO `category_id` for `model_index`.
- Require exactly two development scene IDs, one holdout scene ID, two development backgrounds, one holdout background, and three distinct non-negative seeds.
- Publish an immutable assignment lock directory before holdout byte access, then publish the completed manifest through a temporary file and atomic replace. A run name is never overwritten or reused after either success or failure.
- `output_root` is the exact normalized value `derived/base_cycle`; each `run_name` selects one direct child below `datasets/derived/base_cycle/` and no other output location is accepted.
- Record the normalized semantic config and its SHA-256 plus portable paths and SHA-256 for the registry, COCO, every scene image, and every background; never hash an absolute config location into the portable contract.
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

None of those model-training or semantic holdout-access actions belongs to this plan. This foundation may perform automated integrity-only stat/hash/decode after publishing the assignment lock; it never emits pixels, predictions, scores, or metrics.

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
`load_base_cycle_config(path)`, `freeze_base_cycle(config_path)`,
and `validate_base_cycle(repository_root, run_name)`. Their exact types and
method bodies are defined in Tasks 1 and 2 and must not be renamed by later plans.

The manifest schema is version 1 and has exactly these top-level keys:

Before the manifest exists, `assignment.lock.json` has exactly `lock_version`,
`run_name`, normalized `config`, `config_sha256`, `created_at`, and `state`. It is
published atomically with `state: "integrity_pending"` before any holdout stat/hash/decode
and is never replaced. A failed integrity phase leaves this lock in place and emits no
completed manifest, forcing a new run name after correction.

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
- Consumes: `validate_coco(path, registry, "base")`, `load_class_registry(path)`, and `SCENE_PATTERN`; it deliberately does not call the existing `assert_training_paths_safe`, because that helper resolves both evaluation directories even for a safe input.
- Produces: `BaseCycleConfig`, `BaseCycleReport`, `load_base_cycle_config()` and the internal `_prepare_inventory()` used by Tasks 2 and 3.

- [ ] **Step 1: Write strict config-loader tests**

Create a relocatable miniature repository fixture containing `pyproject.toml` and
`AGENTS.md`. It writes the full registry (15 Base entries with `model_index` 0-14
plus 5 Incremental entries with `model_index` 15-19), three complete scene groups
(`0503`, `0509`, `0510`), a valid Base COCO, and three RGB backgrounds. Add these
tests:

```python
def test_load_base_cycle_config_accepts_exact_schema(cycle_fixture) -> None:
    root, config_path = cycle_fixture

    config = load_base_cycle_config(config_path)

    assert config.dataset_root == "datasets"
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


@pytest.mark.parametrize("missing", sorted(_CONFIG_FIELDS))
def test_load_base_cycle_config_rejects_every_missing_field(
    cycle_fixture, missing: str
) -> None:
    _, config_path = cycle_fixture
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload[missing]
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="missing"):
        load_base_cycle_config(config_path)


@pytest.mark.parametrize(
    "output_root",
    ["derived", "derived/base_cycle/extra", "../outside", "/absolute/output"],
)
def test_load_base_cycle_config_accepts_only_exact_output_root(
    cycle_fixture, output_root: str
) -> None:
    _, config_path = cycle_fixture
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["output_root"] = output_root
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="derived/base_cycle"):
        load_base_cycle_config(config_path)


@pytest.mark.parametrize(
    "dataset_root",
    [".", "../datasets", "datasets/base/test", "/absolute/datasets"],
)
def test_load_base_cycle_config_accepts_only_repository_datasets(
    cycle_fixture, dataset_root: str
) -> None:
    _, config_path = cycle_fixture
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["dataset_root"] = dataset_root
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="dataset_root"):
        load_base_cycle_config(config_path)


def test_config_path_under_test_is_rejected_before_filesystem_access(
    cycle_fixture, monkeypatch
) -> None:
    root, _ = cycle_fixture
    forbidden = root / "base/test/config.yaml"
    touched: list[str] = []

    def forbidden_read(*_args, **_kwargs):
        touched.append("read")
        raise AssertionError("config bytes must not be read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        load_base_cycle_config(forbidden)
    assert touched == []
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

Implement `_strict_object`, `_text`, `_text_tuple`, `_seed_tuple`, and
`load_base_cycle_config`. `_strict_object` rejects every unknown or missing key.
`_text_tuple` requires exact length, non-empty strings, and unique values.
`_seed_tuple` rejects bools, negative values, duplicates, and any length other than
three. Reject overlap across development/holdout scene IDs and backgrounds. Require
`_RUN_NAME.fullmatch(run_name)` and require the normalized `output_root` to equal
exactly `derived/base_cycle`. Require normalized `dataset_root` to equal exactly
`datasets`; absolute, parent-relative, and evaluation-subtree roots are invalid.
`load_base_cycle_config` must call `_reject_evaluation_path` on the raw argument before
constructing any resolved path or reading bytes, then use `_resolve_config_context_safely`
and read only the returned non-link path. Implement those two helpers in this step so the
red config-path test passes before inventory work begins.

- [ ] **Step 4: Write inventory failure tests**

Add tests for lexical rejection and for the lock-gated internal inventory preparation:

```python
def test_configured_test_path_is_rejected_before_any_filesystem_access(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    config = replace(
        load_base_cycle_config(config_path),
        real_coco_path="base/test/annotations.json",
    )
    touched: list[str] = []

    def forbidden_filesystem_access(*_args, **_kwargs):
        touched.append("filesystem")
        raise AssertionError("test path must be rejected lexically")

    monkeypatch.setattr(
        "bakery_scanner.base_cycle._resolve_configured_input",
        forbidden_filesystem_access,
    )
    with pytest.raises(DataValidationError, match="evaluation-only"):
        _validate_config_paths_lexically(config)
    assert touched == []


def test_coco_image_test_path_is_rejected_before_image_filesystem_access(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    _inject_coco_image_filename(root, "base/test/forbidden.jpg")
    lock_path = _publish_assignment_lock_for_test(config_path)
    touched: list[str] = []

    def forbidden_list(*_args, **_kwargs):
        touched.append("iterdir")
        raise AssertionError("COCO directory must not be listed")

    def forbidden_decode(*_args, **_kwargs):
        touched.append("decode")
        raise AssertionError("COCO image must not be decoded")

    monkeypatch.setattr(Path, "iterdir", forbidden_list)
    monkeypatch.setattr(Image, "open", forbidden_decode)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        _prepare_inventory(
            config_path, load_base_cycle_config(config_path), lock_path
        )
    assert touched == []


def test_prepare_inventory_requires_three_complete_declared_scene_groups(
    cycle_fixture,
) -> None:
    root, config_path = cycle_fixture
    (root / "base/val/scene_h_0510.jpg").unlink()

    with pytest.raises(DataValidationError, match="COCO image files"):
        _prepare_inventory(
            config_path,
            load_base_cycle_config(config_path),
            _publish_assignment_lock_for_test(config_path),
        )


def test_prepare_inventory_rejects_undeclared_scene_group(cycle_fixture) -> None:
    root, config_path = cycle_fixture
    payload = json.loads((root / "base/val/instances_val.json").read_text())
    payload["images"].extend(_scene_image_records("0511", start_id=100))
    _write_scene_images(root / "base/val", "0511")
    (root / "base/val/instances_val.json").write_text(json.dumps(payload))

    with pytest.raises(DataValidationError, match="exactly partition"):
        _prepare_inventory(
            config_path,
            load_base_cycle_config(config_path),
            _publish_assignment_lock_for_test(config_path),
        )


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
        _prepare_inventory(
            config_path,
            load_base_cycle_config(config_path),
            _publish_assignment_lock_for_test(config_path),
        )


def test_relative_dataset_root_is_resolved_from_repository_not_cwd(
    cycle_fixture, tmp_path, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    monkeypatch.chdir(tmp_path)

    prepared_root, _ = _prepare_inventory(
        config_path,
        load_base_cycle_config(config_path),
        _publish_assignment_lock_for_test(config_path),
    )

    assert prepared_root == root.resolve()


def test_relocated_repository_produces_same_portable_inventory(
    cycle_fixture, tmp_path
) -> None:
    root, config_path = cycle_fixture
    relocated_repo = tmp_path / "relocated-repo"
    shutil.copytree(root.parent, relocated_repo)
    relocated_config = relocated_repo / config_path.relative_to(root.parent)
    _, first = _prepare_inventory(
        config_path,
        load_base_cycle_config(config_path),
        _publish_assignment_lock_for_test(config_path),
    )

    _, second = _prepare_inventory(
        relocated_config,
        load_base_cycle_config(relocated_config),
        _publish_assignment_lock_for_test(relocated_config),
    )

    assert second == first
```

- [ ] **Step 5: Implement inventory preparation**

Implement private helpers `_sha256(path)`, `_portable(path, root)`,
`_reject_evaluation_path(value)`, `_resolve_config_context_safely(config_path)`,
`_resolve_repository_root_safely(repository_root)`,
`_resolve_configured_input(value, root)`, `_decode_size(path, label)`,
`_load_and_screen_coco_filenames(coco_path)`,
`_assert_cycle_paths_safe_lexically(values)`,
`_scene_inventory(coco_path, development_ids, holdout_id, root)`,
`_background_inventory(development, holdout, root)`,
`_validate_assignment_lock(lock_path, config)`, and
`_prepare_inventory(config_path, config, lock_path)`. Use explicit return annotations;
the final helper returns `tuple[Path, dict[str, object]]`.

`_prepare_inventory` must:

1. validate the immutable assignment lock before any configured input is resolved,
   statted, listed, opened, hashed, or decoded;
2. use `_resolve_config_context_safely` to reject the raw config path lexically before
   filesystem access, convert it to an absolute path without following links, reject any
   symlink/junction component with `lstat`, require it below the repository's
   `configs/base_cycle/` directory, and only then resolve/read it; identify the repository
   by `pyproject.toml` and `AGENTS.md`, and fail if either sentinel is absent;
3. always resolve the exact relative `dataset_root: datasets` from that repository root,
   never from process CWD or file existence heuristics;
4. lexically reject any normalized configured path or COCO image filename containing
   `base/test` or
   `incremental/test` before `resolve`, `stat`, directory listing, open, or decode;
5. call the new pure `_assert_cycle_paths_safe_lexically` on COCO, every background,
   and every screened COCO filename; it operates on normalized path parts only and must
   not call `resolve`, `stat`, `lstat`, listing, open, or decode;
6. require every resolved input to remain inside the resolved dataset root and reject
   symlink/junction escapes;
7. read the already-approved COCO JSON once, require every `images[*].file_name` to be
   a plain basename with no absolute path, separator, `.`/`..`, traversal, or evaluation
   subtree, and complete this lexical pass before `validate_coco` can list or decode;
8. load and validate the full registry, assert Base indices are exactly 0-14, then call
   `validate_coco(coco_path, registry, "base")`;
9. parse every screened COCO image filename with `SCENE_PATTERN` and require exactly the
   declared three IDs, each with e/m/h exactly once;
10. decode every scene and background, require RGB-convertible positive dimensions;
11. return the resolved root and a deterministic, portable inventory dictionary without
   `created_at`; relocation alone must not alter this dictionary or `config_sha256`.

`_publish_assignment_lock_for_test` is test-only fixture glue: it uses the same pure
`_assignment_lock(config)` serializer and atomic direct-child publication contract that
Task 2 wires into `freeze_base_cycle`. It does not inspect any configured input.

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

- [ ] **Step 1: Write lock, publication, and replay tests**

```python
def test_freeze_publishes_lock_before_holdout_byte_access(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    original = base_cycle._sha256

    def guarded_sha256(path: Path) -> str:
        if "0510" in path.name or path.name == "tray_wood_white_surface.png":
            lock = root / "derived/base_cycle/base_v2/assignment.lock.json"
            assert lock.exists()
            payload = json.loads(lock.read_text(encoding="utf-8"))
            assert payload["state"] == "integrity_pending"
        return original(path)

    monkeypatch.setattr(base_cycle, "_sha256", guarded_sha256)
    freeze_base_cycle(config_path)


def test_successful_freeze_never_touches_evaluation_subtrees(
    cycle_fixture, monkeypatch
) -> None:
    _, config_path = cycle_fixture
    touched: list[str] = []
    _install_evaluation_access_guards(
        monkeypatch,
        touched,
        methods=(
            Path.resolve, Path.stat, Path.lstat, Path.iterdir, Path.open, Image.open,
        ),
    )

    report = freeze_base_cycle(config_path)

    assert report.manifest_path.exists()
    assert touched == []


def test_freeze_publishes_hash_bound_manifest_atomically(cycle_fixture) -> None:
    root, config_path = cycle_fixture

    report = freeze_base_cycle(config_path)

    assert report.output_dir == (root / "derived/base_cycle/base_v2").resolve()
    assert report.development_scene_ids == ("0503", "0509")
    assert report.holdout_scene_id == "0510"
    assert report.development_image_count == 6
    assert report.holdout_image_count == 3
    assert (report.output_dir / "assignment.lock.json").exists()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "manifest_version", "cycle_version", "created_at", "run_name",
        "config", "config_sha256", "registry", "real_coco", "scenes",
        "backgrounds", "seeds",
    }
    assert {record["split"] for record in manifest["scenes"]} == {
        "development", "cycle_holdout",
    }
    assert not list(report.output_dir.glob("*.tmp-*"))


def test_validate_replays_every_input_hash(cycle_fixture) -> None:
    root, config_path = cycle_fixture
    freeze_base_cycle(config_path)

    report = validate_base_cycle(root.parent, "base_v2")

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
        validate_base_cycle(root.parent, "base_v2")


def test_freeze_never_reuses_existing_run(cycle_fixture) -> None:
    _, config_path = cycle_fixture
    first = freeze_base_cycle(config_path)

    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)

    assert first.manifest_path.exists()


def test_failed_integrity_keeps_lock_and_no_manifest(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    monkeypatch.setattr(
        base_cycle, "_prepare_inventory",
        lambda *_: (_ for _ in ()).throw(DataValidationError("forced integrity")),
    )

    with pytest.raises(DataValidationError, match="forced integrity"):
        freeze_base_cycle(config_path)

    run = root / "derived/base_cycle/base_v2"
    assert (run / "assignment.lock.json").exists()
    assert not (run / "manifest.json").exists()
    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)


def test_post_publish_validation_failure_never_reuses_run(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    monkeypatch.setattr(
        base_cycle, "_validate_run_dir",
        lambda *_: (_ for _ in ()).throw(DataValidationError("post publish")),
    )

    with pytest.raises(DataValidationError, match="post publish"):
        freeze_base_cycle(config_path)

    run = root / "derived/base_cycle/base_v2"
    assert (run / "assignment.lock.json").exists()
    assert (run / "manifest.json").exists()
    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)


@pytest.mark.parametrize("link_level", ["derived", "base_cycle"])
def test_freeze_rejects_output_symlink_or_junction_escape(
    cycle_fixture, tmp_path, link_level: str
) -> None:
    root, config_path = cycle_fixture
    external = tmp_path / "external"
    external.mkdir()
    _install_supported_directory_link(root, external, link_level)

    with pytest.raises(DataValidationError, match="output.*link|junction"):
        freeze_base_cycle(config_path)

    assert list(external.iterdir()) == []


def test_validate_rejects_test_repository_argument_before_resolve(
    cycle_fixture, monkeypatch
) -> None:
    root, _ = cycle_fixture
    touched: list[str] = []

    def forbidden_resolve(*_args, **_kwargs):
        touched.append("resolve")
        raise AssertionError("evaluation root must be rejected lexically")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        validate_base_cycle(root / "base/test", "base_v2")
    assert touched == []
```

`_install_evaluation_access_guards` wraps each real method and delegates for every safe
path. If an argument's normalized lexical parts contain `datasets/base/test` or
`datasets/incremental/test`, it appends the operation name and raises `AssertionError`.
This makes a successful freeze fail if any legacy helper merely resolves or stats an
evaluation directory, even when no evaluation file is opened.

- [ ] **Step 2: Run publication tests and verify red**

Run:

```powershell
python -m pytest tests/test_base_cycle.py -k "lock or publishes or replays or tampered or existing or integrity" -q
```

Expected: FAIL because freeze publication and `validate_base_cycle` are not implemented.

- [ ] **Step 3: Implement manifest publication**

Implement the following sequence; no overwrite parameter or recovery path exists:

```python
def freeze_base_cycle(config_path: str | Path) -> BaseCycleReport:
    source, repository_root = _resolve_config_context_safely(config_path)
    config = load_base_cycle_config(source)
    root = _resolve_dataset_root_safely(config.dataset_root, repository_root)
    output_root = _resolve_cycle_output_root_safely(root, create=True)
    output_dir = output_root / config.run_name
    if output_dir.parent != output_root:
        raise DataValidationError("run_name must select a direct cycle directory")
    if output_dir.exists():
        raise DataValidationError(f"base cycle run already exists: {output_dir}")
    lock_staging = output_root / f".{config.run_name}.lock-{uuid.uuid4().hex}"
    lock_staging.mkdir()
    _write_json(lock_staging / "assignment.lock.json", _assignment_lock(config))
    lock_staging.rename(output_dir)

    # Only now may _prepare_inventory stat/hash/decode holdout inputs.
    lock_path = output_dir / "assignment.lock.json"
    manifest_tmp: Path | None = None
    try:
        _, inventory = _prepare_inventory(source, config, lock_path)
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "cycle_version": CYCLE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": config.run_name,
            **inventory,
        }
        manifest_tmp = output_dir / f"manifest.json.tmp-{uuid.uuid4().hex}"
        _write_json(manifest_tmp, payload)
        _validate_manifest_payload(root, output_dir, payload)
        os.replace(manifest_tmp, output_dir / "manifest.json")
        return _validate_run_dir(root, output_dir)
    except Exception:
        if manifest_tmp is not None:
            manifest_tmp.unlink(missing_ok=True)
        raise
```

Before writing, add normalized `config = asdict(config)` to the inventory. Compute
`config_sha256` from `json.dumps(config, ensure_ascii=False, sort_keys=True,
separators=(",", ":")).encode("utf-8")` so independent validation can recompute it
without depending on the config file's physical location. Store data paths relative to
dataset root with forward slashes. `_assignment_lock(config)` contains only lexical
assignment/config data and cannot stat, hash, list, open, or decode configured inputs.
After the lock directory is renamed into place, every exception intentionally leaves the
immutable lock and removes any manifest temp file; callers must select a new `run_name`.

`_resolve_cycle_output_root_safely` constructs only `<physical repository>/datasets/
derived/base_cycle`. Before creating anything it `lstat`s every existing component from
the physical dataset root through `derived/base_cycle`, rejects symlinks and Windows
reparse-point/junction attributes, resolves the physical parent, and proves containment
under the resolved dataset root. It repeats the no-link and containment checks after
directory creation. Tests cover links at both `derived` and `base_cycle` levels.

- [ ] **Step 4: Write strict independent-validation tests and verify red**

Add one red test for every validator promise, using a helper that freezes a valid run,
mutates `manifest.json`, and calls `validate_base_cycle(root.parent, "base_v2")`:

```python
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown_top_level", "fields"),
        ("missing_top_level", "fields"),
        ("unknown_scene_field", "scene fields"),
        ("missing_scene_field", "scene fields"),
        ("unknown_background_field", "background fields"),
        ("missing_background_field", "background fields"),
        ("unknown_sha_record_field", "record fields"),
        ("missing_sha_record_field", "record fields"),
        ("manifest_version", "manifest_version"),
        ("cycle_version", "cycle_version"),
        ("manifest_run_name_mismatch", "run_name"),
        ("invalid_timestamp", "created_at"),
        ("non_utc_timestamp", "created_at"),
        ("duplicate_path", "duplicate"),
        ("invalid_split", "split"),
        ("missing_difficulty", "e/m/h"),
        ("scene_split_overlap", "overlap"),
        ("background_split_overlap", "overlap"),
        ("wrong_scene_count", "count"),
        ("wrong_background_count", "count"),
        ("swap_background_assignments", "assignment"),
        ("scene_id_path_remap", "assignment"),
        ("real_coco_authority_mismatch", "assignment"),
        ("registry_authority_mismatch", "assignment"),
        ("path_traversal", "path"),
        ("config_hash_mismatch", "config_sha256"),
        ("config_payload_mismatch", "config"),
        ("seeds_mismatch", "seeds"),
        ("invalid_sha_length", "SHA-256"),
        ("invalid_sha_case", "SHA-256"),
        ("invalid_sha_character", "SHA-256"),
    ],
)
def test_validate_rejects_each_manifest_contract_violation(
    cycle_fixture, case: str, message: str
) -> None:
    root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    _mutate_manifest_case(manifest, case)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataValidationError, match=message):
        validate_base_cycle(root.parent, "base_v2")


@pytest.mark.parametrize(
    "case",
    [
        "unknown_lock_key", "missing_lock_key", "invalid_lock_version",
        "invalid_lock_state", "invalid_lock_timestamp",
        "non_utc_lock_timestamp", "lock_run_name_mismatch",
        "lock_config_mismatch", "lock_hash_mismatch",
    ],
)
def test_validate_rejects_each_assignment_lock_violation(
    cycle_fixture, case: str
) -> None:
    root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    lock_path = run / "assignment.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _mutate_lock_case(lock, case)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(DataValidationError, match="assignment lock"):
        validate_base_cycle(root.parent, "base_v2")


@pytest.mark.parametrize("case", ["symlink_escape", "junction_escape"])
def test_validate_rejects_link_escape(cycle_fixture, case: str) -> None:
    # Create the platform-supported link type or skip explicitly when unavailable.
    root, config_path = cycle_fixture
    _replace_manifest_source_with_external_link(root, config_path, case)
    with pytest.raises(DataValidationError, match="escape"):
        validate_base_cycle(root.parent, "base_v2")


@pytest.mark.parametrize(
    ("artifact", "link_kind"),
    [
        ("run_dir", "symlink"),
        ("run_dir", "junction"),
        ("assignment.lock.json", "symlink"),
        ("manifest.json", "symlink"),
    ],
)
def test_validate_rejects_linked_output_artifact(
    cycle_fixture, tmp_path, artifact: str, link_kind: str
) -> None:
    root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    _replace_output_artifact_with_external_link(run, tmp_path, artifact, link_kind)

    with pytest.raises(DataValidationError, match="output artifact.*link|junction"):
        validate_base_cycle(root.parent, "base_v2")


def test_validate_rejects_missing_source(cycle_fixture) -> None:
    root, config_path = cycle_fixture
    freeze_base_cycle(config_path)
    (root / "base/val/scene_e_0503.jpg").unlink()
    with pytest.raises(DataValidationError, match="missing"):
        validate_base_cycle(root.parent, "base_v2")


def test_validate_screens_coco_before_listing_or_decode(
    cycle_fixture, monkeypatch
) -> None:
    root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    coco = root / "base/val/instances_val.json"
    _inject_coco_image_filename(root, "base/test/forbidden.jpg")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["real_coco"]["sha256"] = _sha256(coco)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    touched: list[str] = []
    _install_listing_and_decode_guards(monkeypatch, touched)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        validate_base_cycle(root.parent, "base_v2")
    assert touched == []


def test_relocated_completed_run_validates_with_same_config_hash(
    cycle_fixture, tmp_path
) -> None:
    root, config_path = cycle_fixture
    first = freeze_base_cycle(config_path)
    before = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    relocated_repo = tmp_path / "relocated-repo"
    shutil.copytree(root.parent, relocated_repo)
    report = validate_base_cycle(relocated_repo, "base_v2")
    after = json.loads(report.manifest_path.read_text(encoding="utf-8"))

    assert after["config_sha256"] == before["config_sha256"]
```

The mutation helper itself is table-tested so each case changes exactly the intended
field. Assignment mutations swap complete valid records or point to fixture decoys with
matching SHA values, so counts, split vocabulary, and ordinary hash validation still pass;
only lock/config authority can reject them. Run all tests above and confirm they fail for
the intended reason before writing the validator.

- [ ] **Step 5: Implement strict independent validation**

Implement `_load_json_object`, `_validate_sha_record`, `_validate_manifest_payload`,
`_validate_run_dir`, and:

```python
def validate_base_cycle(
    repository_root: str | Path, run_name: str
) -> BaseCycleReport:
    repository = _resolve_repository_root_safely(repository_root)
    root = _resolve_dataset_root_safely("datasets", repository)
    if not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError("run_name is invalid")
    output_root = _resolve_cycle_output_root_safely(root, create=False)
    run_dir = _resolve_run_dir_safely(output_root, run_name)
    return _validate_run_dir(root, run_dir)
```

`_validate_run_dir` first validates the immutable lock and proves its normalized config
and `config_sha256` match the manifest. It rejects unknown/missing keys, versions other
than 1/`1.0.0`, non-UTC or invalid `created_at`, duplicate paths, wrong split names,
missing e/m/h variants, scene/background overlap across splits, wrong counts, path
traversal, symlink/junction escapes, missing files, and SHA drift. It reruns registry and
COCO validation and reconstructs `BaseCycleReport` only from validated manifest data.
Immediately before every replay call to `validate_coco`, `_validate_run_dir` must call
`_load_and_screen_coco_filenames` on the hash-verified COCO and reject unsafe basenames;
no directory listing or image decode may occur between screening and `validate_coco`.
The lock must have `lock_version == 1` and `state == "integrity_pending"`; manifest seeds
must exactly match both lock/config seeds and the checked-in `[42, 43, 44]`. Every SHA-256
field is exactly 64 lowercase hexadecimal characters before any hash comparison.
From the validated lock/config, reconstruct the only authorized registry path
(`class_registry.json`), COCO path, development/holdout background path-to-split mapping,
and scene ID-to-split mapping. From the screened COCO basenames, reconstruct the exact
scene ID/difficulty/path tuples. Manifest `registry`, `real_coco`, `backgrounds`, and
`scenes` must equal those authoritative records field-for-field (apart from their
independently verified hashes/dimensions); matching counts alone is never sufficient.
Neither the public API nor the CLI accepts an alternate output root.
`_resolve_repository_root_safely` lexically rejects an evaluation-subtree argument before
`resolve` or any other filesystem call, converts a safe argument to an absolute path
without following links, rejects a symlink/junction root, requires `pyproject.toml` and
`AGENTS.md`, then derives the sole dataset and output roots. The test above proves the
early rejection by making `Path.resolve` observable.
`_resolve_run_dir_safely` requires the run to be a physical direct child directory, uses
`lstat`/Windows reparse attributes to reject a symlink or junction before resolution, and
proves physical containment. `_load_json_object` similarly requires
`assignment.lock.json` and `manifest.json` to be regular non-link files inside that
physical run directory before reading either one. The linked-artifact tests cover the run
directory, lock file, and manifest file paths.

- [ ] **Step 6: Run the focused and full foundation tests**

Run:

```powershell
python -m pytest tests/test_base_cycle.py -q
```

Expected: all Base cycle unit tests pass.

- [ ] **Step 7: Commit Task 2**

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
        "--repository-root", ".",
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
    freeze.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--repository-root", default=".")
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
            report = freeze_base_cycle(args.config)
        else:
            report = validate_base_cycle(args.repository_root, args.run_name)
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
bakery-base-cycle validate --repository-root . --run-name base_v2
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

The command has no overwrite mode. It publishes the assignment lock first, then performs
automated integrity-only hashing and decoding; it does not display or score holdout
images. Any failure leaves the run name consumed, so retry with a new run name after
correcting the cause.

- [ ] **Step 5: Independently replay-validate the real artifact**

```powershell
bakery-base-cycle validate --repository-root . --run-name base_v2 --json
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
