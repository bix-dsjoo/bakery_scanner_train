import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

from bakery_scanner.coco import validate_coco
from bakery_scanner.errors import DataValidationError
from bakery_scanner.registry import ClassRegistry


def _valid_payload(registry: ClassRegistry) -> dict[str, object]:
    categories = [
        {"id": record.category_id, "name": record.canonical_name}
        for record in registry.classes
        if record.phase == "base"
    ]
    return {
        "images": [{"id": 10, "file_name": "scene_e_0001.jpg", "width": 40, "height": 30}],
        "annotations": [
            {"id": 100, "image_id": 10, "category_id": categories[0]["id"], "bbox": [2, 3, 20, 10]}
        ],
        "categories": categories,
    }


def _write_coco(directory: Path, payload: dict[str, object]) -> Path:
    path = directory / "instances.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_valid_dataset(directory: Path, registry: ClassRegistry) -> tuple[Path, dict[str, object]]:
    directory.mkdir()
    Image.new("RGB", (40, 30), "white").save(directory / "scene_e_0001.jpg")
    payload = _valid_payload(registry)
    return _write_coco(directory, payload), payload


def test_validate_coco_returns_counts(
    tmp_path: Path, registry_factory: Callable[[], ClassRegistry]
) -> None:
    registry = registry_factory()
    annotation_path, _ = _make_valid_dataset(tmp_path / "scene", registry)

    stats = validate_coco(annotation_path, registry, expected_phase="base")

    assert stats.image_count == 1
    assert stats.annotation_count == 1
    assert stats.category_count == 15


def test_validate_coco_rejects_unknown_expected_phase(
    tmp_path: Path, registry_factory: Callable[[], ClassRegistry]
) -> None:
    registry = registry_factory()
    directory = tmp_path / "scene"
    directory.mkdir()
    annotation_path = _write_coco(
        directory, {"images": [], "annotations": [], "categories": []}
    )

    with pytest.raises(DataValidationError, match="expected_phase"):
        validate_coco(annotation_path, registry, expected_phase="typo")


def test_validate_coco_rejects_missing_declared_image(
    tmp_path: Path, registry_factory: Callable[[], ClassRegistry]
) -> None:
    registry = registry_factory()
    directory = tmp_path / "scene"
    directory.mkdir()
    annotation_path = _write_coco(directory, _valid_payload(registry))

    with pytest.raises(DataValidationError, match="image files do not match"):
        validate_coco(annotation_path, registry, expected_phase="base")


def test_validate_coco_rejects_undeclared_image(
    tmp_path: Path, registry_factory: Callable[[], ClassRegistry]
) -> None:
    registry = registry_factory()
    annotation_path, _ = _make_valid_dataset(tmp_path / "scene", registry)
    Image.new("RGB", (10, 10)).save(annotation_path.parent / "extra.jpg")

    with pytest.raises(DataValidationError, match="image files do not match"):
        validate_coco(annotation_path, registry, expected_phase="base")


@pytest.mark.parametrize(
    ("section", "message"),
    [("images", "duplicate image id"), ("annotations", "duplicate annotation id"), ("categories", "duplicate category id")],
)
def test_validate_coco_rejects_duplicate_ids(
    tmp_path: Path,
    registry_factory: Callable[[], ClassRegistry],
    section: str,
    message: str,
) -> None:
    registry = registry_factory()
    annotation_path, payload = _make_valid_dataset(tmp_path / "scene", registry)
    items = payload[section]
    assert isinstance(items, list)
    items.append(deepcopy(items[0]))
    _write_coco(annotation_path.parent, payload)

    with pytest.raises(DataValidationError, match=message):
        validate_coco(annotation_path, registry, expected_phase="base")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("image_id", 999, "unknown image_id"), ("category_id", 999, "unknown category_id")],
)
def test_validate_coco_rejects_invalid_annotation_references(
    tmp_path: Path,
    registry_factory: Callable[[], ClassRegistry],
    field: str,
    value: int,
    message: str,
) -> None:
    registry = registry_factory()
    annotation_path, payload = _make_valid_dataset(tmp_path / "scene", registry)
    payload["annotations"][0][field] = value  # type: ignore[index]
    _write_coco(annotation_path.parent, payload)

    with pytest.raises(DataValidationError, match=message):
        validate_coco(annotation_path, registry, expected_phase="base")


def test_validate_coco_rejects_registry_category_name_mismatch(
    tmp_path: Path, registry_factory: Callable[[], ClassRegistry]
) -> None:
    registry = registry_factory()
    annotation_path, payload = _make_valid_dataset(tmp_path / "scene", registry)
    payload["categories"][0]["name"] = "Wrong Bread"  # type: ignore[index]
    _write_coco(annotation_path.parent, payload)

    with pytest.raises(DataValidationError, match="categories do not match registry"):
        validate_coco(annotation_path, registry, expected_phase="base")


def test_validate_coco_rejects_decoded_size_mismatch(
    tmp_path: Path, registry_factory: Callable[[], ClassRegistry]
) -> None:
    registry = registry_factory()
    annotation_path, payload = _make_valid_dataset(tmp_path / "scene", registry)
    payload["images"][0]["width"] = 41  # type: ignore[index]
    _write_coco(annotation_path.parent, payload)

    with pytest.raises(DataValidationError, match="decoded size"):
        validate_coco(annotation_path, registry, expected_phase="base")


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ([1, 2, 0, 10], "positive width and height"),
        ([-1, 2, 10, 10], "outside image bounds"),
        ([30, 2, 11, 10], "outside image bounds"),
        ([1, 25, 10, 6], "outside image bounds"),
        ([1, 2, float("nan"), 6], "finite numeric values"),
    ],
)
def test_validate_coco_rejects_invalid_bbox(
    tmp_path: Path,
    registry_factory: Callable[[], ClassRegistry],
    bbox: list[float],
    message: str,
) -> None:
    registry = registry_factory()
    annotation_path, payload = _make_valid_dataset(tmp_path / "scene", registry)
    payload["annotations"][0]["bbox"] = bbox  # type: ignore[index]
    _write_coco(annotation_path.parent, payload)

    with pytest.raises(DataValidationError, match=message):
        validate_coco(annotation_path, registry, expected_phase="base")
