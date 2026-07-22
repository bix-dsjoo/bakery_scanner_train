from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from PIL import Image

import bakery_scanner.base_cycle as base_cycle
from bakery_scanner.base_cycle import (
    _CONFIG_FIELDS,
    _assert_existing_components_are_physical,
    _assignment_lock,
    _prepare_inventory,
    _publish_assignment_lock,
    _reject_evaluation_path,
    freeze_base_cycle,
    _validate_config_paths_lexically,
    validate_base_cycle,
    load_base_cycle_config,
)
from bakery_scanner.errors import DataValidationError


def _registry_records() -> list[dict[str, object]]:
    return [
        {
            "category_id": model_index + 1,
            "model_index": model_index,
            "canonical_name": f"Bread {model_index + 1}",
            "folder_name": f"bread_{model_index + 1:02d}_item",
            "phase": "base" if model_index < 15 else "incremental",
        }
        for model_index in range(20)
    ]


def _scene_image_records(scene_id: str, start_id: int) -> list[dict[str, object]]:
    return [
        {
            "id": start_id + offset,
            "file_name": f"scene_{difficulty}_{scene_id}.jpg",
            "width": 40,
            "height": 30,
        }
        for offset, difficulty in enumerate(("e", "m", "h"))
    ]


def _write_scene_images(directory: Path, scene_id: str) -> None:
    for difficulty in ("e", "m", "h"):
        Image.new("RGB", (40, 30), "white").save(
            directory / f"scene_{difficulty}_{scene_id}.jpg"
        )


def _config_payload() -> dict[str, object]:
    return {
        "dataset_root": "datasets",
        "output_root": "derived/base_cycle",
        "run_name": "base_v2",
        "real_coco_path": "base/val/instances_val.json",
        "development_scene_ids": ["0503", "0509"],
        "holdout_scene_id": "0510",
        "development_backgrounds": [
            "collected/backgrounds/tray_white_square.png",
            "collected/backgrounds/tray_wood_black_surface.png",
        ],
        "holdout_background": "collected/backgrounds/tray_wood_white_surface.png",
        "seeds": [42, 43, 44],
    }


@pytest.fixture
def cycle_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "mini-repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (repository / "AGENTS.md").write_text("fixture\n", encoding="utf-8")

    dataset_root = repository / "datasets"
    scene_dir = dataset_root / "base" / "val"
    scene_dir.mkdir(parents=True)
    records = _registry_records()
    (dataset_root / "class_registry.json").write_text(
        json.dumps({"version": 1, "classes": records}), encoding="utf-8"
    )

    images: list[dict[str, object]] = []
    for start_id, scene_id in zip((1, 4, 7), ("0503", "0509", "0510"), strict=True):
        images.extend(_scene_image_records(scene_id, start_id))
        _write_scene_images(scene_dir, scene_id)
    categories = [
        {"id": record["category_id"], "name": record["canonical_name"]}
        for record in records
        if record["phase"] == "base"
    ]
    annotations = [
        {
            "id": image["id"],
            "image_id": image["id"],
            "category_id": categories[(int(image["id"]) - 1) % len(categories)]["id"],
            "bbox": [1, 2, 10, 8],
        }
        for image in images
    ]
    (scene_dir / "instances_val.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
        ),
        encoding="utf-8",
    )

    backgrounds = dataset_root / "collected" / "backgrounds"
    backgrounds.mkdir(parents=True)
    for name, color in (
        ("tray_white_square.png", "white"),
        ("tray_wood_black_surface.png", "black"),
        ("tray_wood_white_surface.png", "beige"),
    ):
        Image.new("RGB", (32, 24), color).save(backgrounds / name)

    config_path = repository / "configs" / "base_cycle" / "base_v2.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(_config_payload(), sort_keys=False), encoding="utf-8"
    )
    return dataset_root, config_path


def _publish_assignment_lock_for_test(config_path: Path) -> Path:
    config = load_base_cycle_config(config_path)
    return _publish_assignment_lock(config_path, config)


def _inject_coco_image_filename(dataset_root: Path, file_name: str) -> None:
    coco_path = dataset_root / "base" / "val" / "instances_val.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = file_name
    coco_path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_base_cycle_config_accepts_exact_schema(cycle_fixture) -> None:
    _, config_path = cycle_fixture

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
        (
            {"holdout_background": "collected/backgrounds/tray_white_square.png"},
            "backgrounds",
        ),
        (
            {
                "holdout_background": (
                    "collected/backgrounds/./tray_white_square.png"
                )
            },
            "backgrounds",
        ),
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


def test_load_base_cycle_config_normalizes_paths_before_semantic_hash(
    cycle_fixture,
) -> None:
    _, config_path = cycle_fixture
    original = load_base_cycle_config(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["development_backgrounds"][0] = (
        "collected/backgrounds/./tray_white_square.png"
    )
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    normalized = load_base_cycle_config(config_path)

    assert normalized == original
    assert _assignment_lock(normalized)["config_sha256"] == _assignment_lock(original)[
        "config_sha256"
    ]


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
    dataset_root, _ = cycle_fixture
    forbidden = dataset_root / "base" / "test" / "config.yaml"
    touched: list[str] = []

    def forbidden_read(*_args, **_kwargs):
        touched.append("read")
        raise AssertionError("config bytes must not be read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        load_base_cycle_config(forbidden)
    assert touched == []


@pytest.mark.parametrize("operation", ["load", "freeze"])
def test_dotdot_config_test_path_is_rejected_before_any_filesystem_access(
    cycle_fixture, monkeypatch, operation: str
) -> None:
    dataset_root, _ = cycle_fixture
    forbidden = dataset_root / "base" / "x" / ".." / "test" / "config.yaml"
    touched: list[str] = []

    def forbidden_access(*_args, **_kwargs):
        touched.append("filesystem")
        raise AssertionError("normalized evaluation path must be rejected lexically")

    with monkeypatch.context() as guard:
        guard.setattr(base_cycle.os.path, "abspath", forbidden_access)
        guard.setattr(Path, "lstat", forbidden_access)
        guard.setattr(Path, "stat", forbidden_access)
        guard.setattr(Path, "exists", forbidden_access)
        guard.setattr(Path, "resolve", forbidden_access)
        with pytest.raises(DataValidationError, match="evaluation-only"):
            if operation == "load":
                load_base_cycle_config(forbidden)
            else:
                freeze_base_cycle(forbidden)

    assert touched == []


def test_dotdot_test_repository_is_rejected_before_any_filesystem_access(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, _ = cycle_fixture
    forbidden = dataset_root / "base" / "x" / ".." / "test"
    touched: list[str] = []

    def forbidden_access(*_args, **_kwargs):
        touched.append("filesystem")
        raise AssertionError("normalized evaluation root must be rejected lexically")

    with monkeypatch.context() as guard:
        guard.setattr(base_cycle.os.path, "abspath", forbidden_access)
        guard.setattr(Path, "lstat", forbidden_access)
        guard.setattr(Path, "stat", forbidden_access)
        guard.setattr(Path, "exists", forbidden_access)
        guard.setattr(Path, "resolve", forbidden_access)
        with pytest.raises(DataValidationError, match="evaluation-only"):
            validate_base_cycle(forbidden, "base_v2")

    assert touched == []


@pytest.mark.parametrize("value", ["../safe", "/../safe", "C:/../safe"])
def test_lexical_path_rejects_parent_traversal_beyond_root(value: str) -> None:
    with pytest.raises(DataValidationError, match="parent traversal"):
        _reject_evaluation_path(value)


def test_physical_component_check_rejects_link_without_stat_or_resolve(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, config_path = cycle_fixture
    target = dataset_root / "base" / "test"
    target.mkdir(parents=True)
    (target / "forbidden.yaml").write_text("forbidden\n", encoding="utf-8")
    safe_link = config_path.parent / "linked"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(safe_link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        safe_link.symlink_to(target, target_is_directory=True)

    touched: list[str] = []
    original_stat = Path.stat

    def forbidden_stat(path: Path, *args, **kwargs):
        if kwargs.get("follow_symlinks") is False:
            return original_stat(path, *args, **kwargs)
        touched.append("stat")
        raise AssertionError("physical check must not follow the link target")

    def forbidden_resolve(*_args, **_kwargs):
        touched.append("resolve")
        raise AssertionError("physical check must not resolve the link target")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "stat", forbidden_stat)
        guard.setattr(Path, "resolve", forbidden_resolve)
        with pytest.raises(DataValidationError, match="symlink|junction"):
            _assert_existing_components_are_physical(safe_link)

    assert touched == []


def test_configured_test_path_is_rejected_before_any_filesystem_access(
    cycle_fixture, monkeypatch
) -> None:
    _, config_path = cycle_fixture
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


def test_prepare_inventory_rejects_valid_lock_outside_cycle_direct_child(
    cycle_fixture, tmp_path
) -> None:
    _, config_path = cycle_fixture
    config = load_base_cycle_config(config_path)
    unrelated = tmp_path / "unrelated" / config.run_name / "assignment.lock.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text(
        json.dumps(_assignment_lock(config), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="direct child"):
        _prepare_inventory(config_path, config, unrelated)


def test_coco_image_test_path_is_rejected_before_image_filesystem_access(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, config_path = cycle_fixture
    _inject_coco_image_filename(dataset_root, "base/test/forbidden.jpg")
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
    dataset_root, config_path = cycle_fixture
    (dataset_root / "base" / "val" / "scene_h_0510.jpg").unlink()

    with pytest.raises(DataValidationError, match="COCO image files"):
        _prepare_inventory(
            config_path,
            load_base_cycle_config(config_path),
            _publish_assignment_lock_for_test(config_path),
        )


def test_prepare_inventory_rejects_undeclared_scene_group(cycle_fixture) -> None:
    dataset_root, config_path = cycle_fixture
    coco_path = dataset_root / "base" / "val" / "instances_val.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    payload["images"].extend(_scene_image_records("0511", start_id=100))
    _write_scene_images(coco_path.parent, "0511")
    coco_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="exactly partition"):
        _prepare_inventory(
            config_path,
            load_base_cycle_config(config_path),
            _publish_assignment_lock_for_test(config_path),
        )


def test_prepare_inventory_rejects_duplicate_scene_difficulty(cycle_fixture) -> None:
    dataset_root, config_path = cycle_fixture
    coco_path = dataset_root / "base" / "val" / "instances_val.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    payload["images"].append(
        {
            "id": 100,
            "file_name": "scene_e_0503.png",
            "width": 40,
            "height": 30,
        }
    )
    Image.new("RGB", (40, 30), "white").save(coco_path.parent / "scene_e_0503.png")
    coco_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="exactly once"):
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
    dataset_root, config_path = cycle_fixture
    monkeypatch.chdir(tmp_path)

    prepared_root, _ = _prepare_inventory(
        config_path,
        load_base_cycle_config(config_path),
        _publish_assignment_lock_for_test(config_path),
    )

    assert prepared_root == dataset_root.resolve()


def test_relocated_repository_produces_same_portable_inventory(
    cycle_fixture, tmp_path
) -> None:
    dataset_root, config_path = cycle_fixture
    repository = dataset_root.parent
    relocated_repo = tmp_path / "relocated-repo"
    shutil.copytree(repository, relocated_repo)
    relocated_config = relocated_repo / config_path.relative_to(repository)
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


def test_freeze_publishes_lock_before_holdout_byte_access(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, config_path = cycle_fixture
    original = base_cycle._sha256

    def guarded_sha256(path: Path) -> str:
        if "0510" in path.name or path.name == "tray_wood_white_surface.png":
            lock = (
                dataset_root
                / "derived"
                / "base_cycle"
                / "base_v2"
                / "assignment.lock.json"
            )
            assert lock.exists()
            assert json.loads(lock.read_text(encoding="utf-8"))["state"] == (
                "integrity_pending"
            )
        return original(path)

    monkeypatch.setattr(base_cycle, "_sha256", guarded_sha256)

    freeze_base_cycle(config_path)


def test_freeze_publishes_hash_bound_manifest_atomically(cycle_fixture) -> None:
    dataset_root, config_path = cycle_fixture

    report = freeze_base_cycle(config_path)

    expected = (dataset_root / "derived/base_cycle/base_v2").resolve()
    assert report.output_dir == expected
    assert report.development_scene_ids == ("0503", "0509")
    assert report.holdout_scene_id == "0510"
    assert report.development_image_count == 6
    assert report.holdout_image_count == 3
    assert (expected / "assignment.lock.json").exists()
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
    assert not list(expected.glob("*.tmp-*"))


def test_validate_replays_every_input_hash(cycle_fixture) -> None:
    dataset_root, config_path = cycle_fixture
    freeze_base_cycle(config_path)

    report = validate_base_cycle(dataset_root.parent, "base_v2")

    assert report.development_image_count == 6
    assert report.holdout_image_count == 3


@pytest.mark.parametrize("target", ["scene", "background", "coco", "registry"])
def test_validate_rejects_tampered_source(cycle_fixture, target: str) -> None:
    dataset_root, config_path = cycle_fixture
    freeze_base_cycle(config_path)
    targets = {
        "scene": dataset_root / "base/val/scene_e_0503.jpg",
        "background": dataset_root / "collected/backgrounds/tray_white_square.png",
        "coco": dataset_root / "base/val/instances_val.json",
        "registry": dataset_root / "class_registry.json",
    }
    targets[target].write_bytes(b"tampered")

    with pytest.raises(DataValidationError, match="SHA-256"):
        validate_base_cycle(dataset_root.parent, "base_v2")


def test_freeze_never_reuses_existing_run(cycle_fixture) -> None:
    _, config_path = cycle_fixture
    first = freeze_base_cycle(config_path)

    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)

    assert first.manifest_path.exists()


def test_failed_integrity_keeps_lock_and_no_manifest(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, config_path = cycle_fixture

    def fail_integrity(*_args, **_kwargs):
        raise DataValidationError("forced integrity")

    monkeypatch.setattr(base_cycle, "_prepare_inventory", fail_integrity)

    with pytest.raises(DataValidationError, match="forced integrity"):
        freeze_base_cycle(config_path)

    run = dataset_root / "derived/base_cycle/base_v2"
    assert (run / "assignment.lock.json").exists()
    assert not (run / "manifest.json").exists()
    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)


def test_post_publish_validation_failure_never_reuses_run(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, config_path = cycle_fixture

    def fail_validation(*_args, **_kwargs):
        raise DataValidationError("post publish")

    monkeypatch.setattr(base_cycle, "_validate_run_dir", fail_validation)

    with pytest.raises(DataValidationError, match="post publish"):
        freeze_base_cycle(config_path)

    run = dataset_root / "derived/base_cycle/base_v2"
    assert (run / "assignment.lock.json").exists()
    assert (run / "manifest.json").exists()
    with pytest.raises(DataValidationError, match="already exists"):
        freeze_base_cycle(config_path)


def test_validate_rejects_test_repository_argument_before_resolve(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, _ = cycle_fixture
    touched: list[str] = []

    def forbidden_resolve(*_args, **_kwargs):
        touched.append("resolve")
        raise AssertionError("evaluation root must be rejected lexically")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        validate_base_cycle(dataset_root / "base/test", "base_v2")
    assert touched == []


def _mutate_manifest_case(manifest: dict[str, object], case: str) -> None:
    scenes = manifest["scenes"]
    backgrounds = manifest["backgrounds"]
    assert isinstance(scenes, list) and isinstance(backgrounds, list)
    scene = scenes[0]
    background = backgrounds[0]
    assert isinstance(scene, dict) and isinstance(background, dict)
    if case == "unknown_top_level":
        manifest["surprise"] = True
    elif case == "missing_top_level":
        manifest.pop("cycle_version")
    elif case == "unknown_scene_field":
        scene["surprise"] = True
    elif case == "missing_scene_field":
        scene.pop("height")
    elif case == "unknown_background_field":
        background["surprise"] = True
    elif case == "missing_background_field":
        background.pop("sha256")
    elif case == "unknown_sha_record_field":
        manifest["registry"]["surprise"] = True  # type: ignore[index]
    elif case == "missing_sha_record_field":
        manifest["real_coco"].pop("sha256")  # type: ignore[union-attr]
    elif case == "manifest_version":
        manifest["manifest_version"] = 2
    elif case == "manifest_version_type":
        manifest["manifest_version"] = 1.0
    elif case == "cycle_version":
        manifest["cycle_version"] = "2.0.0"
    elif case == "manifest_run_name_mismatch":
        manifest["run_name"] = "other"
    elif case == "invalid_timestamp":
        manifest["created_at"] = "invalid"
    elif case == "non_utc_timestamp":
        manifest["created_at"] = "2026-01-01T00:00:00+09:00"
    elif case == "duplicate_path":
        scenes[1]["path"] = scene["path"]  # type: ignore[index]
    elif case == "invalid_split":
        scene["split"] = "test"
    elif case == "scene_split_overlap":
        next(item for item in scenes if item["split"] == "cycle_holdout")[  # type: ignore[index]
            "split"
        ] = "development"
    elif case == "background_split_overlap":
        next(item for item in backgrounds if item["split"] == "cycle_holdout")[  # type: ignore[index]
            "split"
        ] = "development"
    elif case == "wrong_scene_count":
        scenes.pop()
    elif case == "wrong_background_count":
        backgrounds.pop()
    elif case == "swap_background_assignments":
        backgrounds[0]["split"], backgrounds[-1]["split"] = (  # type: ignore[index]
            backgrounds[-1]["split"],  # type: ignore[index]
            backgrounds[0]["split"],  # type: ignore[index]
        )
    elif case == "scene_id_path_remap":
        scene["scene_id"] = "0510"
    elif case == "real_coco_authority_mismatch":
        manifest["real_coco"] = dict(manifest["registry"])  # type: ignore[arg-type]
    elif case == "registry_authority_mismatch":
        manifest["registry"] = dict(manifest["real_coco"])  # type: ignore[arg-type]
    elif case == "registry_path_whitespace":
        manifest["registry"]["path"] = " class_registry.json "  # type: ignore[index]
    elif case == "real_coco_path_whitespace":
        manifest["real_coco"]["path"] = (  # type: ignore[index]
            " base/val/instances_val.json "
        )
    elif case == "path_traversal":
        scene["path"] = "../escape.jpg"
    elif case == "config_hash_mismatch":
        manifest["config_sha256"] = "0" * 64
    elif case == "config_payload_mismatch":
        manifest["config"]["holdout_scene_id"] = "0503"  # type: ignore[index]
    elif case == "seeds_mismatch":
        manifest["seeds"] = [42, 43, 45]
    elif case == "invalid_sha_length":
        scene["sha256"] = "0" * 63
    elif case == "invalid_sha_case":
        scene["sha256"] = "A" * 64
    elif case == "invalid_sha_character":
        scene["sha256"] = "g" * 64
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_top_level",
        "missing_top_level",
        "unknown_scene_field",
        "missing_scene_field",
        "unknown_background_field",
        "missing_background_field",
        "unknown_sha_record_field",
        "missing_sha_record_field",
        "manifest_version",
        "manifest_version_type",
        "cycle_version",
        "manifest_run_name_mismatch",
        "invalid_timestamp",
        "non_utc_timestamp",
        "duplicate_path",
        "invalid_split",
        "scene_split_overlap",
        "background_split_overlap",
        "wrong_scene_count",
        "wrong_background_count",
        "swap_background_assignments",
        "scene_id_path_remap",
        "real_coco_authority_mismatch",
        "registry_authority_mismatch",
        "registry_path_whitespace",
        "real_coco_path_whitespace",
        "path_traversal",
        "config_hash_mismatch",
        "config_payload_mismatch",
        "seeds_mismatch",
        "invalid_sha_length",
        "invalid_sha_case",
        "invalid_sha_character",
    ],
)
def test_validate_rejects_each_manifest_contract_violation(
    cycle_fixture, case: str
) -> None:
    dataset_root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _mutate_manifest_case(manifest, case)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataValidationError):
        validate_base_cycle(dataset_root.parent, "base_v2")


def _mutate_lock_case(lock: dict[str, object], case: str) -> None:
    if case == "unknown_lock_key":
        lock["surprise"] = True
    elif case == "missing_lock_key":
        lock.pop("state")
    elif case == "invalid_lock_version":
        lock["lock_version"] = 2
    elif case == "invalid_lock_version_type":
        lock["lock_version"] = 1.0
    elif case == "invalid_lock_state":
        lock["state"] = "done"
    elif case == "invalid_lock_timestamp":
        lock["created_at"] = "invalid"
    elif case == "non_utc_lock_timestamp":
        lock["created_at"] = "2026-01-01T00:00:00+09:00"
    elif case == "lock_run_name_mismatch":
        lock["run_name"] = "other"
    elif case == "lock_config_mismatch":
        lock["config"]["holdout_scene_id"] = "0503"  # type: ignore[index]
    elif case == "lock_hash_mismatch":
        lock["config_sha256"] = "0" * 64
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_lock_key",
        "missing_lock_key",
        "invalid_lock_version",
        "invalid_lock_version_type",
        "invalid_lock_state",
        "invalid_lock_timestamp",
        "non_utc_lock_timestamp",
        "lock_run_name_mismatch",
        "lock_config_mismatch",
        "lock_hash_mismatch",
    ],
)
def test_validate_rejects_each_assignment_lock_violation(
    cycle_fixture, case: str
) -> None:
    dataset_root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    lock_path = run / "assignment.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _mutate_lock_case(lock, case)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(DataValidationError, match="assignment lock"):
        validate_base_cycle(dataset_root.parent, "base_v2")


def _make_directory_link(alias: Path, target: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        alias.symlink_to(target, target_is_directory=True)


def test_freeze_rejects_output_junction_escape(cycle_fixture, tmp_path) -> None:
    dataset_root, config_path = cycle_fixture
    external = tmp_path / "external"
    external.mkdir()
    _make_directory_link(dataset_root / "derived", external)

    with pytest.raises(DataValidationError, match="symlink|junction"):
        freeze_base_cycle(config_path)
    assert list(external.iterdir()) == []


def test_validate_rejects_source_junction_escape(cycle_fixture, tmp_path) -> None:
    dataset_root, config_path = cycle_fixture
    freeze_base_cycle(config_path)
    backgrounds = dataset_root / "collected/backgrounds"
    external = tmp_path / "external-backgrounds"
    backgrounds.rename(external)
    _make_directory_link(backgrounds, external)

    with pytest.raises(DataValidationError, match="symlink|junction"):
        validate_base_cycle(dataset_root.parent, "base_v2")


def test_validate_rejects_linked_run_directory(cycle_fixture, tmp_path) -> None:
    dataset_root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    external = tmp_path / "external-run"
    run.rename(external)
    _make_directory_link(run, external)

    with pytest.raises(DataValidationError, match="symlink|junction"):
        validate_base_cycle(dataset_root.parent, "base_v2")


def test_validate_rejects_missing_source(cycle_fixture) -> None:
    dataset_root, config_path = cycle_fixture
    freeze_base_cycle(config_path)
    (dataset_root / "base/val/scene_e_0503.jpg").unlink()

    with pytest.raises(DataValidationError, match="missing"):
        validate_base_cycle(dataset_root.parent, "base_v2")


def test_validate_screens_coco_before_listing_or_decode(
    cycle_fixture, monkeypatch
) -> None:
    dataset_root, config_path = cycle_fixture
    run = freeze_base_cycle(config_path).output_dir
    coco = dataset_root / "base/val/instances_val.json"
    _inject_coco_image_filename(dataset_root, "base/test/forbidden.jpg")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["real_coco"]["sha256"] = base_cycle._sha256(coco)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    touched: list[str] = []

    def forbidden_list(*_args, **_kwargs):
        touched.append("iterdir")
        raise AssertionError("unsafe COCO must be screened before listing")

    def forbidden_decode(*_args, **_kwargs):
        touched.append("decode")
        raise AssertionError("unsafe COCO must be screened before decode")

    monkeypatch.setattr(Path, "iterdir", forbidden_list)
    monkeypatch.setattr(Image, "open", forbidden_decode)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        validate_base_cycle(dataset_root.parent, "base_v2")
    assert touched == []


def test_relocated_completed_run_validates_with_same_config_hash(
    cycle_fixture, tmp_path
) -> None:
    dataset_root, config_path = cycle_fixture
    first = freeze_base_cycle(config_path)
    before = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    relocated = tmp_path / "relocated-repo"
    shutil.copytree(dataset_root.parent, relocated)

    report = validate_base_cycle(relocated, "base_v2")
    after = json.loads(report.manifest_path.read_text(encoding="utf-8"))

    assert after["config_sha256"] == before["config_sha256"]


def test_manifest_authority_uses_screened_coco_filename_extension(
    cycle_fixture,
) -> None:
    dataset_root, config_path = cycle_fixture
    coco_path = dataset_root / "base/val/instances_val.json"
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    source = coco_path.parent / "scene_e_0503.jpg"
    replacement = source.with_suffix(".png")
    source.rename(replacement)
    next(
        image
        for image in payload["images"]
        if image["file_name"] == "scene_e_0503.jpg"
    )["file_name"] = "scene_e_0503.png"
    coco_path.write_text(json.dumps(payload), encoding="utf-8")

    report = freeze_base_cycle(config_path)

    assert validate_base_cycle(dataset_root.parent, "base_v2") == report
