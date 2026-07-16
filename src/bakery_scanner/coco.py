from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .errors import DataValidationError
from .registry import IMAGE_SUFFIXES, PHASE_COUNTS, ClassRegistry


@dataclass(frozen=True, slots=True)
class CocoStats:
    image_count: int
    annotation_count: int
    category_count: int


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    parsed = _integer(value, label)
    if parsed <= 0:
        raise DataValidationError(f"{label} must be positive")
    return parsed


def _objects(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise DataValidationError(f"COCO {key} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise DataValidationError(f"every COCO {key} entry must be an object")
    return value  # type: ignore[return-value]


def _unique_ids(items: list[dict[str, object]], kind: str) -> list[int]:
    ids = [_integer(item.get("id"), f"{kind} id") for item in items]
    if len(ids) != len(set(ids)):
        raise DataValidationError(f"duplicate {kind} id in COCO file")
    return ids


def _load_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load COCO annotation {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("COCO root must be an object")
    return payload


def validate_coco(
    annotation_path: str | Path,
    registry: ClassRegistry,
    expected_phase: str,
) -> CocoStats:
    if expected_phase not in PHASE_COUNTS:
        raise DataValidationError(f"invalid expected_phase: {expected_phase}")
    path = Path(annotation_path)
    payload = _load_payload(path)
    images = _objects(payload, "images")
    annotations = _objects(payload, "annotations")
    categories = _objects(payload, "categories")

    image_ids = _unique_ids(images, "image")
    _unique_ids(annotations, "annotation")
    category_ids = _unique_ids(categories, "category")

    image_by_id: dict[int, tuple[str, int, int]] = {}
    declared_files: list[str] = []
    for item, image_id in zip(images, image_ids, strict=True):
        file_name = item.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise DataValidationError("image file_name must be a non-empty string")
        width = _positive_integer(item.get("width"), f"image {image_id} width")
        height = _positive_integer(item.get("height"), f"image {image_id} height")
        declared_files.append(file_name)
        image_by_id[image_id] = (file_name, width, height)

    actual_files = {
        item.name
        for item in path.parent.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    }
    if set(declared_files) != actual_files or len(declared_files) != len(set(declared_files)):
        missing = sorted(set(declared_files) - actual_files)
        extra = sorted(actual_files - set(declared_files))
        raise DataValidationError(
            f"COCO image files do not match directory: missing={missing}, extra={extra}"
        )

    for image_id, (file_name, width, height) in image_by_id.items():
        try:
            with Image.open(path.parent / file_name) as decoded:
                decoded.load()
                decoded_size = decoded.size
        except (OSError, UnidentifiedImageError) as exc:
            raise DataValidationError(f"cannot decode image {file_name}: {exc}") from exc
        if decoded_size != (width, height):
            raise DataValidationError(
                f"image {image_id} decoded size {decoded_size} does not match COCO {(width, height)}"
            )

    expected_categories = {
        record.category_id: record.canonical_name
        for record in registry.classes
        if record.phase == expected_phase
    }
    actual_categories: dict[int, str] = {}
    for item, category_id in zip(categories, category_ids, strict=True):
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise DataValidationError("category name must be a non-empty string")
        actual_categories[category_id] = name
    if actual_categories != expected_categories:
        raise DataValidationError(
            "COCO categories do not match registry for phase " + expected_phase
        )

    for annotation in annotations:
        annotation_id = _integer(annotation.get("id"), "annotation id")
        image_id = _integer(annotation.get("image_id"), f"annotation {annotation_id} image_id")
        if image_id not in image_by_id:
            raise DataValidationError(
                f"annotation {annotation_id} references unknown image_id {image_id}"
            )
        category_id = _integer(
            annotation.get("category_id"), f"annotation {annotation_id} category_id"
        )
        if category_id not in actual_categories:
            raise DataValidationError(
                f"annotation {annotation_id} references unknown category_id {category_id}"
            )

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in bbox
        ):
            raise DataValidationError(
                f"annotation {annotation_id} bbox must contain four finite numeric values"
            )
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            raise DataValidationError(
                f"annotation {annotation_id} bbox must have positive width and height"
            )
        _, image_width, image_height = image_by_id[image_id]
        if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
            raise DataValidationError(
                f"annotation {annotation_id} bbox is outside image bounds"
            )

    return CocoStats(
        image_count=len(images),
        annotation_count=len(annotations),
        category_count=len(categories),
    )
