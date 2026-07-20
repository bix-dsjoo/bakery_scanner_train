from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from .coco import validate_coco
from .errors import DataValidationError
from .registry import ClassRecord, load_class_registry, validate_class_directories
from .safety import assert_training_paths_safe
from .splits import scene_id_from_path, split_scene_paths

BUILDER_VERSION = "1.0.0"
MANIFEST_VERSION = 1
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SPLITS = ("train", "validation")
_CLEANUP_ATTEMPTS = 20
_CLEANUP_DELAY_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class ClassifierDatasetConfig:
    dataset_root: Path
    run_name: str
    phase: Literal["base", "incremental"]
    seed: int = 42
    validation_fraction: float = 0.2
    expected_base_images_per_class: int = 84
    expected_incremental_images_per_class: int = 7

    def validate(self) -> None:
        if not isinstance(self.dataset_root, Path):
            raise DataValidationError("dataset_root must be a pathlib.Path")
        if not isinstance(self.run_name, str) or not _RUN_NAME.fullmatch(self.run_name):
            raise DataValidationError(
                "run_name must contain only letters, digits, dot, underscore, or hyphen"
            )
        if self.phase not in {"base", "incremental"}:
            raise DataValidationError("phase must be base or incremental")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DataValidationError("seed must be an integer")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 1
        ):
            raise DataValidationError("validation_fraction must be between 0 and 1")
        for label, value in (
            ("expected_base_images_per_class", self.expected_base_images_per_class),
            (
                "expected_incremental_images_per_class",
                self.expected_incremental_images_per_class,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DataValidationError(f"{label} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ClassifierDatasetReport:
    output_dir: Path
    manifest_path: Path
    phase: str
    output_dimension: int
    sample_count: int
    train_sample_count: int
    validation_sample_count: int
    builder_version: str = BUILDER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "builder_version": self.builder_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "phase": self.phase,
            "output_dimension": self.output_dimension,
            "sample_count": self.sample_count,
            "train_sample_count": self.train_sample_count,
            "validation_sample_count": self.validation_sample_count,
        }


@dataclass(frozen=True, slots=True)
class ClassifierValidationReport(ClassifierDatasetReport):
    pass


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DataValidationError(f"cannot hash file {path}: {exc}") from exc


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} root must be an object")
    return value


def _expect_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise DataValidationError(
            f"{label} fields do not match schema: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _classifier_root(dataset_root: Path) -> Path:
    return (dataset_root / "derived" / "classifier").resolve(strict=False)


def _run_dir(dataset_root: Path, run_name: str) -> Path:
    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(
            "run_name must contain only letters, digits, dot, underscore, or hyphen"
        )
    parent = _classifier_root(dataset_root)
    output = parent / run_name
    if output.parent != parent:
        raise DataValidationError("run_name must select a direct classifier run directory")
    if output.exists() and output.is_symlink():
        raise DataValidationError(f"classifier run path must not be a link: {output}")
    return output


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise DataValidationError(f"{label} must stay within dataset root: {path}") from exc


def _resolve_relative(value: object, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DataValidationError(f"{label} must be dataset-root-relative: {value}")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DataValidationError(f"{label} escapes dataset root: {value}") from exc
    return resolved


def _image_files(directory: Path) -> tuple[Path, ...]:
    try:
        files = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                ),
                key=lambda path: path.name.casefold(),
            )
        )
    except OSError as exc:
        raise DataValidationError(f"cannot enumerate classifier source {directory}: {exc}") from exc
    if not files:
        raise DataValidationError(f"classifier source contains no images: {directory}")
    return files


def _validate_class_image_counts(
    counts: dict[str, dict[str, int]],
    registry,
    config: ClassifierDatasetConfig,
) -> None:
    included_phases = {"base"} if config.phase == "base" else {"base", "incremental"}
    for record in registry.classes:
        if record.phase not in included_phases:
            continue
        expected = (
            config.expected_base_images_per_class
            if record.phase == "base"
            else config.expected_incremental_images_per_class
        )
        actual = counts[record.phase].get(record.folder_name)
        if actual != expected:
            raise DataValidationError(
                f"{record.folder_name} must contain exactly {expected} images, found {actual}"
            )


def _incremental_partition(
    files: tuple[Path, ...],
    *,
    seed: int,
    validation_fraction: float,
    record: ClassRecord,
) -> dict[Path, str]:
    if len(files) < 2:
        raise DataValidationError(
            f"incremental class requires at least two images: {record.folder_name}"
        )
    digest = hashlib.sha256(
        f"{seed}:{record.category_id}:{record.model_index}:{record.folder_name}".encode(
            "utf-8"
        )
    ).digest()
    class_seed = int.from_bytes(digest[:8], "big")
    shuffled = list(files)
    random.Random(class_seed).shuffle(shuffled)
    validation_count = max(
        1,
        min(len(files) - 1, int(len(files) * validation_fraction + 0.5)),
    )
    validation = set(shuffled[:validation_count])
    return {path: "validation" if path in validation else "train" for path in files}


def _decode_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise DataValidationError(f"cannot decode classifier image {path}: {exc}") from exc


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pixel_bbox(value: object, annotation_id: int) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"annotation {annotation_id} bbox must contain four values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise DataValidationError(f"annotation {annotation_id} bbox must be finite numeric values")
    x, y, width, height = (float(item) for item in value)
    left = math.floor(x)
    top = math.floor(y)
    right = math.ceil(x + width)
    bottom = math.ceil(y + height)
    if right <= left or bottom <= top:
        raise DataValidationError(f"annotation {annotation_id} has empty pixel crop")
    return [left, top, right - left, bottom - top]


def _output_relative(
    run_name: str, split: str, model_index: int, file_name: str
) -> Path:
    return Path("derived") / "classifier" / run_name / split / str(model_index) / file_name


def _write_single_sample(
    *,
    root: Path,
    staging: Path,
    run_name: str,
    source: Path,
    record: ClassRecord,
    split: str,
    validation_domain: str,
) -> dict[str, Any]:
    source_hash = _sha256(source)
    file_name = f"single_{source.stem}_{source_hash[:12]}{source.suffix.lower()}"
    relative_output = _output_relative(run_name, split, record.model_index, file_name)
    staged_output = staging / split / str(record.model_index) / file_name
    staged_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(source, staged_output)
    except OSError as exc:
        raise DataValidationError(f"cannot copy classifier source {source}: {exc}") from exc
    return {
        "split": split,
        "output_path": relative_output.as_posix(),
        "output_sha256": _sha256(staged_output),
        "source_kind": "single_object",
        "source_path": _relative(source, root, "single-object source"),
        "source_sha256": source_hash,
        "annotation_id": None,
        "bbox": None,
        "pixel_bbox": None,
        "scene_id": None,
        "category_id": record.category_id,
        "model_index": record.model_index,
        "class_phase": record.phase,
        "validation_domain": validation_domain,
    }


def _scene_payload(coco_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _json_object(coco_path, "scene COCO")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
        raise DataValidationError("scene COCO images must be a list of objects")
    if not isinstance(annotations, list) or not all(
        isinstance(item, dict) for item in annotations
    ):
        raise DataValidationError("scene COCO annotations must be a list of objects")
    return images, annotations


def _scene_paths(coco_path: Path, images: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for index, item in enumerate(images):
        file_name = item.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise DataValidationError(f"scene image {index} has invalid file_name")
        paths.append(coco_path.parent / file_name)
    return paths


def _write_scene_samples(
    *,
    root: Path,
    staging: Path,
    run_name: str,
    registry,
    coco_path: Path,
    seed: int,
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images, annotations = _scene_payload(coco_path)
    paths = _scene_paths(coco_path, images)
    assert_training_paths_safe(paths, root)
    validate_coco(coco_path, registry, expected_phase="base")
    image_by_id: dict[int, dict[str, Any]] = {}
    for item in images:
        image_id = item.get("id")
        file_name = item.get("file_name")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise DataValidationError("scene image id must be an integer")
        if not isinstance(file_name, str) or not file_name:
            raise DataValidationError(f"scene image {image_id} has invalid file_name")
        path = coco_path.parent / file_name
        image_by_id[image_id] = {**item, "path": path}
    split = split_scene_paths(paths, validation_fraction, seed)
    split_by_path = {path.resolve(): "train" for path in split.train_paths}
    split_by_path.update({path.resolve(): "validation" for path in split.validation_paths})

    scene_sources: list[dict[str, Any]] = []
    for item in images:
        source = coco_path.parent / str(item["file_name"])
        source_split = split_by_path[source.resolve()]
        scene_sources.append(
            {
                "path": _relative(source, root, "scene source"),
                "sha256": _sha256(source),
                "scene_id": scene_id_from_path(source),
                "split": source_split,
                "width": item["width"],
                "height": item["height"],
            }
        )

    samples: list[dict[str, Any]] = []
    seen_names: set[Path] = set()
    for annotation in sorted(annotations, key=lambda item: int(item["id"])):
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
            raise DataValidationError("scene annotation id must be an integer")
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id not in image_by_id:
            raise DataValidationError(f"annotation {annotation_id} has invalid image_id")
        if (
            isinstance(category_id, bool)
            or not isinstance(category_id, int)
            or category_id not in registry.by_category_id
        ):
            raise DataValidationError(f"annotation {annotation_id} has invalid category_id")
        class_record = registry.by_category_id[category_id]
        image_info = image_by_id[image_id]
        source = Path(image_info["path"])
        sample_split = split_by_path[source.resolve()]
        pixels = _pixel_bbox(annotation.get("bbox"), annotation_id)
        left, top, width, height = pixels
        decoded = _decode_rgb(source)
        crop = decoded.crop((left, top, left + width, top + height))
        source_hash = _sha256(source)
        file_name = f"scene_{source.stem}_ann{annotation_id}_{source_hash[:12]}.png"
        relative_output = _output_relative(
            run_name, sample_split, class_record.model_index, file_name
        )
        if relative_output in seen_names:
            raise DataValidationError(f"classifier output collision: {relative_output}")
        seen_names.add(relative_output)
        staged_output = staging / sample_split / str(class_record.model_index) / file_name
        staged_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            staged_output.write_bytes(_png_bytes(crop))
        except OSError as exc:
            raise DataValidationError(f"cannot write classifier crop {staged_output}: {exc}") from exc
        samples.append(
            {
                "split": sample_split,
                "output_path": relative_output.as_posix(),
                "output_sha256": _sha256(staged_output),
                "source_kind": "scene_crop",
                "source_path": _relative(source, root, "scene source"),
                "source_sha256": source_hash,
                "annotation_id": annotation_id,
                "bbox": annotation["bbox"],
                "pixel_bbox": pixels,
                "scene_id": scene_id_from_path(source),
                "category_id": category_id,
                "model_index": class_record.model_index,
                "class_phase": class_record.phase,
                "validation_domain": "scene",
            }
        )
    return samples, sorted(scene_sources, key=lambda item: str(item["path"]))


def _counts(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {split: 0 for split in _SPLITS}
    by_split_class: dict[str, dict[str, int]] = {split: {} for split in _SPLITS}
    by_class: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    by_source_kind: dict[str, int] = {}
    for sample in samples:
        by_split[str(sample["split"])] += 1
        class_key = str(sample["model_index"])
        split_classes = by_split_class[str(sample["split"])]
        split_classes[class_key] = split_classes.get(class_key, 0) + 1
        by_class[class_key] = by_class.get(class_key, 0) + 1
        phase = str(sample["class_phase"])
        by_phase[phase] = by_phase.get(phase, 0) + 1
        kind = str(sample["source_kind"])
        by_source_kind[kind] = by_source_kind.get(kind, 0) + 1
    return {
        "total": len(samples),
        "by_split": dict(sorted(by_split.items())),
        "by_split_class": {
            split: dict(sorted(counts.items(), key=lambda item: int(item[0])))
            for split, counts in sorted(by_split_class.items())
        },
        "by_class": dict(sorted(by_class.items(), key=lambda item: int(item[0]))),
        "by_phase": dict(sorted(by_phase.items())),
        "by_source_kind": dict(sorted(by_source_kind.items())),
    }


def _manifest(
    *,
    config: ClassifierDatasetConfig,
    root: Path,
    registry_path: Path,
    output_dimension: int,
    scene_coco: Path,
    scene_sources: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "builder_version": BUILDER_VERSION,
        "config": {
            "run_name": config.run_name,
            "phase": config.phase,
            "seed": config.seed,
            "validation_fraction": config.validation_fraction,
            "expected_base_images_per_class": config.expected_base_images_per_class,
            "expected_incremental_images_per_class": (
                config.expected_incremental_images_per_class
            ),
        },
        "registry": {
            "path": _relative(registry_path, root, "registry"),
            "sha256": _sha256(registry_path),
            "output_dimension": output_dimension,
        },
        "scene_coco": {
            "path": _relative(scene_coco, root, "scene COCO"),
            "sha256": _sha256(scene_coco),
        },
        "scene_sources": scene_sources,
        "samples": sorted(
            samples,
            key=lambda item: (
                str(item["split"]),
                int(item["model_index"]),
                str(item["output_path"]),
            ),
        ),
        "counts": _counts(samples),
    }


def _sample_keys(sample: object, index: int) -> dict[str, Any]:
    return _expect_keys(
        sample,
        {
            "split",
            "output_path",
            "output_sha256",
            "source_kind",
            "source_path",
            "source_sha256",
            "annotation_id",
            "bbox",
            "pixel_bbox",
            "scene_id",
            "category_id",
            "model_index",
            "class_phase",
            "validation_domain",
        },
        f"sample {index}",
    )


def _validate_run_dir(root: Path, run_name: str, run_dir: Path) -> ClassifierValidationReport:
    manifest_path = run_dir / "manifest.json"
    payload = _expect_keys(
        _json_object(manifest_path, "classifier manifest"),
        {
            "manifest_version",
            "builder_version",
            "config",
            "registry",
            "scene_coco",
            "scene_sources",
            "samples",
            "counts",
        },
        "classifier manifest",
    )
    if payload["manifest_version"] != MANIFEST_VERSION:
        raise DataValidationError("unsupported classifier manifest version")
    if payload["builder_version"] != BUILDER_VERSION:
        raise DataValidationError("unsupported classifier builder version")
    config_payload = _expect_keys(
        payload["config"],
        {
            "run_name",
            "phase",
            "seed",
            "validation_fraction",
            "expected_base_images_per_class",
            "expected_incremental_images_per_class",
        },
        "classifier manifest config",
    )
    config = ClassifierDatasetConfig(
        dataset_root=root,
        run_name=config_payload["run_name"],
        phase=config_payload["phase"],
        seed=config_payload["seed"],
        validation_fraction=config_payload["validation_fraction"],
        expected_base_images_per_class=config_payload[
            "expected_base_images_per_class"
        ],
        expected_incremental_images_per_class=config_payload[
            "expected_incremental_images_per_class"
        ],
    )
    config.validate()
    if config.run_name != run_name:
        raise DataValidationError("classifier manifest run_name does not match directory")

    registry_payload = _expect_keys(
        payload["registry"], {"path", "sha256", "output_dimension"}, "registry record"
    )
    registry_path = _resolve_relative(registry_payload["path"], root, "registry path")
    assert_training_paths_safe([registry_path], root)
    if _sha256(registry_path) != registry_payload["sha256"]:
        raise DataValidationError("classifier registry hash changed")
    registry = load_class_registry(registry_path)
    directory_counts = validate_class_directories(root, registry)
    _validate_class_image_counts(directory_counts, registry, config)
    expected_dimension = 15 if config.phase == "base" else 20
    if registry_payload["output_dimension"] != expected_dimension:
        raise DataValidationError("classifier output dimension does not match phase")

    coco_payload = _expect_keys(
        payload["scene_coco"], {"path", "sha256"}, "scene COCO record"
    )
    coco_path = _resolve_relative(coco_payload["path"], root, "scene COCO path")
    assert_training_paths_safe([coco_path], root)
    if _sha256(coco_path) != coco_payload["sha256"]:
        raise DataValidationError("classifier scene COCO hash changed")
    images, annotations = _scene_payload(coco_path)
    scene_paths = _scene_paths(coco_path, images)
    assert_training_paths_safe(scene_paths, root)
    validate_coco(coco_path, registry, expected_phase="base")
    scene_split = split_scene_paths(scene_paths, config.validation_fraction, config.seed)
    split_by_scene = {scene_id_from_path(path): "train" for path in scene_split.train_paths}
    split_by_scene.update(
        {scene_id_from_path(path): "validation" for path in scene_split.validation_paths}
    )
    split_by_path = {path.resolve(): "train" for path in scene_split.train_paths}
    split_by_path.update(
        {path.resolve(): "validation" for path in scene_split.validation_paths}
    )
    expected_scene_sources = {
        _relative(path, root, "scene source"): {
            "path": _relative(path, root, "scene source"),
            "scene_id": scene_id_from_path(path),
            "split": split_by_path[path.resolve()],
            "width": next(
                int(item["width"])
                for item in images
                if str(item["file_name"]) == path.name
            ),
            "height": next(
                int(item["height"])
                for item in images
                if str(item["file_name"]) == path.name
            ),
        }
        for path in scene_paths
    }
    raw_scene_sources = payload["scene_sources"]
    if not isinstance(raw_scene_sources, list):
        raise DataValidationError("scene_sources must be a list")
    actual_scene_sources: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw_scene_sources):
        source_record = _expect_keys(
            value,
            {"path", "sha256", "scene_id", "split", "width", "height"},
            f"scene source {index}",
        )
        source_path = source_record["path"]
        if not isinstance(source_path, str) or source_path in actual_scene_sources:
            raise DataValidationError("scene source paths must be unique strings")
        expected = expected_scene_sources.get(source_path)
        if expected is None or any(
            source_record[field] != expected[field]
            for field in ("path", "scene_id", "split", "width", "height")
        ):
            raise DataValidationError(f"scene source {index} disagrees with COCO split")
        resolved_source = _resolve_relative(source_path, root, f"scene source {index}")
        assert_training_paths_safe([resolved_source], root)
        if _sha256(resolved_source) != source_record["sha256"]:
            raise DataValidationError(f"scene source hash changed: {source_path}")
        actual_scene_sources[source_path] = source_record
    if set(actual_scene_sources) != set(expected_scene_sources):
        raise DataValidationError("scene source inventory differs from COCO")
    image_by_id = {int(item["id"]): item for item in images}
    expected_scene: dict[tuple[str, str, object], dict[str, Any]] = {}
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        image = image_by_id[int(annotation["image_id"])]
        source = coco_path.parent / str(image["file_name"])
        identity = (
            "scene_crop",
            _relative(source, root, "scene source"),
            annotation_id,
        )
        expected_scene[identity] = {
            "split": split_by_scene[scene_id_from_path(source)],
            "scene_id": scene_id_from_path(source),
            "category_id": int(annotation["category_id"]),
            "bbox": annotation["bbox"],
            "pixel_bbox": _pixel_bbox(annotation["bbox"], annotation_id),
        }

    included_phases = {"base"} if config.phase == "base" else {"base", "incremental"}
    expected_single: dict[str, str] = {}
    for class_record in registry.classes:
        if class_record.phase not in included_phases:
            continue
        files = _image_files(root / class_record.phase / class_record.folder_name)
        if class_record.phase == "incremental":
            partition = _incremental_partition(
                files,
                seed=config.seed,
                validation_fraction=config.validation_fraction,
                record=class_record,
            )
        else:
            partition = {path: "train" for path in files}
        for path, expected_split in partition.items():
            expected_single[_relative(path, root, "single-object source")] = expected_split

    raw_samples = payload["samples"]
    if not isinstance(raw_samples, list):
        raise DataValidationError("classifier samples must be a list")
    samples = [_sample_keys(item, index) for index, item in enumerate(raw_samples)]
    expected_files: set[Path] = set()
    identities: set[tuple[str, str, object]] = set()
    for index, sample in enumerate(samples):
        split_name = sample["split"]
        if split_name not in _SPLITS:
            raise DataValidationError(f"sample {index} has invalid split")
        model_index = sample["model_index"]
        category_id = sample["category_id"]
        if (
            isinstance(model_index, bool)
            or not isinstance(model_index, int)
            or model_index not in registry.by_model_index
        ):
            raise DataValidationError(f"sample {index} has invalid model_index")
        class_record = registry.by_model_index[model_index]
        if category_id != class_record.category_id or sample["class_phase"] != class_record.phase:
            raise DataValidationError(f"sample {index} label mapping disagrees with registry")
        if model_index >= expected_dimension:
            raise DataValidationError(f"sample {index} exceeds classifier output dimension")

        source = _resolve_relative(sample["source_path"], root, f"sample {index} source")
        assert_training_paths_safe([source], root)
        if _sha256(source) != sample["source_sha256"]:
            raise DataValidationError(f"sample {index} source hash changed")
        output = _resolve_relative(sample["output_path"], root, f"sample {index} output")
        expected_parent = run_dir / str(split_name) / str(model_index)
        try:
            relative_within_run = output.relative_to(_run_dir(root, run_name))
        except ValueError as exc:
            raise DataValidationError(f"sample {index} output is outside classifier run") from exc
        actual_output = run_dir / relative_within_run
        if actual_output.parent != expected_parent:
            raise DataValidationError(f"sample {index} output directory does not match label")
        if actual_output in expected_files:
            raise DataValidationError(f"duplicate classifier output: {actual_output}")
        expected_files.add(actual_output)
        if not actual_output.is_file() or _sha256(actual_output) != sample["output_sha256"]:
            raise DataValidationError(f"sample {index} output is missing or altered")
        _decode_rgb(actual_output)

        source_kind = sample["source_kind"]
        if source_kind == "single_object":
            if sample["annotation_id"] is not None or sample["bbox"] is not None:
                raise DataValidationError(f"sample {index} single-object metadata is invalid")
            source_key = str(sample["source_path"])
            if source_key not in expected_single or expected_single[source_key] != split_name:
                raise DataValidationError(
                    f"sample {index} single-object split disagrees with replay"
                )
            if sample["validation_domain"] != "single_object":
                raise DataValidationError(f"sample {index} has invalid validation domain")
            if actual_output.read_bytes() != source.read_bytes():
                raise DataValidationError(f"sample {index} single-object copy changed bytes")
            identity = (source_kind, sample["source_path"], None)
        elif source_kind == "scene_crop":
            annotation_id = sample["annotation_id"]
            pixel_bbox = sample["pixel_bbox"]
            scene_id = sample["scene_id"]
            if (
                isinstance(annotation_id, bool)
                or not isinstance(annotation_id, int)
                or not isinstance(pixel_bbox, list)
                or len(pixel_bbox) != 4
                or scene_id not in split_by_scene
            ):
                raise DataValidationError(f"sample {index} scene metadata is invalid")
            if split_by_scene[scene_id] != split_name or sample["validation_domain"] != "scene":
                raise DataValidationError(f"sample {index} scene split disagrees with replay")
            identity = (source_kind, sample["source_path"], annotation_id)
            expected = expected_scene.get(identity)
            if expected is None or any(
                sample[field] != expected[field]
                for field in ("split", "scene_id", "category_id", "bbox", "pixel_bbox")
            ):
                raise DataValidationError(f"sample {index} scene metadata disagrees with COCO")
            left, top, width, height = pixel_bbox
            if any(isinstance(item, bool) or not isinstance(item, int) for item in pixel_bbox):
                raise DataValidationError(f"sample {index} pixel bbox must contain integers")
            replay = _decode_rgb(source).crop((left, top, left + width, top + height))
            if _png_bytes(replay) != actual_output.read_bytes():
                raise DataValidationError(f"sample {index} crop replay differs")
        else:
            raise DataValidationError(f"sample {index} has invalid source_kind")
        if identity in identities:
            raise DataValidationError(f"duplicate classifier source record: {identity}")
        identities.add(identity)

    expected_identities = {
        ("single_object", source_path, None) for source_path in expected_single
    } | set(expected_scene)
    if identities != expected_identities:
        raise DataValidationError(
            "classifier source inventory differs from replay: "
            f"missing={sorted(str(item) for item in expected_identities - identities)}, "
            f"extra={sorted(str(item) for item in identities - expected_identities)}"
        )

    actual_files = {
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != expected_files:
        raise DataValidationError(
            "classifier output inventory differs: "
            f"missing={sorted(str(path) for path in expected_files - actual_files)}, "
            f"extra={sorted(str(path) for path in actual_files - expected_files)}"
        )
    if payload["counts"] != _counts(samples):
        raise DataValidationError("classifier manifest counts do not match samples")
    present_indices = {int(sample["model_index"]) for sample in samples}
    if present_indices != set(range(expected_dimension)):
        raise DataValidationError("classifier samples do not cover every output index")
    counts = _counts(samples)
    return ClassifierValidationReport(
        output_dir=_run_dir(root, run_name),
        manifest_path=_run_dir(root, run_name) / "manifest.json",
        phase=config.phase,
        output_dimension=expected_dimension,
        sample_count=counts["total"],
        train_sample_count=counts["by_split"]["train"],
        validation_sample_count=counts["by_split"]["validation"],
    )


def _remove_tree_after_commit(path: Path) -> None:
    for attempt in range(_CLEANUP_ATTEMPTS):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == _CLEANUP_ATTEMPTS - 1:
                return
            time.sleep(_CLEANUP_DELAY_SECONDS)


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise DataValidationError(f"classifier run already exists: {output}")
    backup: Path | None = None
    try:
        if output.exists():
            backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
            output.rename(backup)
        staging.rename(output)
    except OSError as exc:
        if not output.exists() and backup is not None and backup.exists():
            backup.rename(output)
        raise DataValidationError(f"cannot publish classifier run {output}: {exc}") from exc
    if backup is not None:
        _remove_tree_after_commit(backup)


def build_classifier_dataset(
    config: ClassifierDatasetConfig,
    *,
    overwrite: bool = False,
) -> ClassifierDatasetReport:
    config.validate()
    root = config.dataset_root.resolve(strict=False)
    output = _run_dir(root, config.run_name)
    registry_path = root / "class_registry.json"
    scene_coco = root / "base" / "val" / "instances_val.json"
    assert_training_paths_safe([registry_path, scene_coco], root)
    registry = load_class_registry(registry_path)
    directory_counts = validate_class_directories(root, registry)
    _validate_class_image_counts(directory_counts, registry, config)
    included_phases = {"base"} if config.phase == "base" else {"base", "incremental"}
    class_records = [record for record in registry.classes if record.phase in included_phases]
    source_paths = [scene_coco, registry_path]
    scene_images, _ = _scene_payload(scene_coco)
    source_paths.extend(_scene_paths(scene_coco, scene_images))
    for record in class_records:
        source_paths.extend(_image_files(root / record.phase / record.folder_name))
    assert_training_paths_safe(source_paths, root)
    if output.exists() and not overwrite:
        raise DataValidationError(f"classifier run already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        samples: list[dict[str, Any]] = []
        for record in class_records:
            files = _image_files(root / record.phase / record.folder_name)
            if record.phase == "incremental":
                partition = _incremental_partition(
                    files,
                    seed=config.seed,
                    validation_fraction=config.validation_fraction,
                    record=record,
                )
            else:
                partition = {path: "train" for path in files}
            for source in files:
                samples.append(
                    _write_single_sample(
                        root=root,
                        staging=staging,
                        run_name=config.run_name,
                        source=source,
                        record=record,
                        split=partition[source],
                        validation_domain="single_object",
                    )
                )
        scene_samples, scene_sources = _write_scene_samples(
            root=root,
            staging=staging,
            run_name=config.run_name,
            registry=registry,
            coco_path=scene_coco,
            seed=config.seed,
            validation_fraction=config.validation_fraction,
        )
        samples.extend(scene_samples)
        payload = _manifest(
            config=config,
            root=root,
            registry_path=registry_path,
            output_dimension=15 if config.phase == "base" else 20,
            scene_coco=scene_coco,
            scene_sources=scene_sources,
            samples=samples,
        )
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _validate_run_dir(root, config.run_name, staging)
        _publish(staging, output, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_classifier_dataset(root, config.run_name)


def validate_classifier_dataset(
    dataset_root: str | Path,
    run_name: str,
) -> ClassifierValidationReport:
    root = Path(dataset_root).resolve(strict=False)
    output = _run_dir(root, run_name)
    if not output.is_dir():
        raise DataValidationError(f"classifier run does not exist: {output}")
    return _validate_run_dir(root, run_name, output)
