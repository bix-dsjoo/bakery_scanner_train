from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from PIL import Image

from bakery_scanner.base_cycle import (
    _CONFIG_FIELDS,
    _assignment_lock,
    _prepare_inventory,
    _validate_config_paths_lexically,
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
    repository = config_path.parents[2]
    run_dir = repository / config.dataset_root / config.output_root / config.run_name
    run_dir.mkdir(parents=True)
    lock_path = run_dir / "assignment.lock.json"
    lock_path.write_text(
        json.dumps(_assignment_lock(config), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return lock_path


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
