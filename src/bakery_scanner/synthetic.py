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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageEnhance, UnidentifiedImageError, __version__ as PILLOW_VERSION

from .errors import DataValidationError
from .registry import IMAGE_SUFFIXES, ClassRecord, load_class_registry
from .safety import assert_training_paths_safe

GENERATOR_VERSION = "1.0.0"
MANIFEST_VERSION = 1
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SCENE_NAME = re.compile(r"^scene_\d{6}\.png$")
_PLACEMENT_ATTEMPTS = 500
_RENAME_ATTEMPTS = 20
_RENAME_DELAY_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    seed: int = 42
    scene_count: int = 10
    objects_per_scene: int = 5
    phase: str = "base"
    size_fraction_range: tuple[float, float] = (0.12, 0.28)
    rotation_range: tuple[float, float] = (-25.0, 25.0)
    brightness_range: tuple[float, float] = (0.85, 1.15)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    foreground_threshold: int = 245

    def validate(self) -> None:
        for label, value in (
            ("seed", self.seed),
            ("scene_count", self.scene_count),
            ("objects_per_scene", self.objects_per_scene),
            ("foreground_threshold", self.foreground_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise DataValidationError(f"{label} must be an integer")
        if self.scene_count <= 0 or self.objects_per_scene <= 0:
            raise DataValidationError("scene_count and objects_per_scene must be positive")
        if self.phase not in {"base", "incremental", "all"}:
            raise DataValidationError("phase must be base, incremental, or all")
        if not 0 <= self.foreground_threshold <= 255:
            raise DataValidationError("foreground_threshold must be from 0 through 255")
        _validate_range("size_fraction_range", self.size_fraction_range, positive=True)
        _validate_range("rotation_range", self.rotation_range)
        _validate_range("brightness_range", self.brightness_range, positive=True)
        _validate_range("contrast_range", self.contrast_range, positive=True)
        if self.size_fraction_range[1] >= 1:
            raise DataValidationError("size_fraction_range maximum must be less than 1")


@dataclass(frozen=True, slots=True)
class SyntheticGenerationReport:
    output_dir: Path
    manifest_path: Path
    image_count: int
    object_count: int
    generator_version: str = GENERATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generator_version": self.generator_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "image_count": self.image_count,
            "object_count": self.object_count,
        }


@dataclass(frozen=True, slots=True)
class SyntheticValidationReport:
    output_dir: Path
    manifest_path: Path
    image_count: int
    object_count: int
    generator_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generator_version": self.generator_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "image_count": self.image_count,
            "object_count": self.object_count,
        }


def _validate_range(
    label: str, values: object, *, positive: bool = False
) -> tuple[float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise DataValidationError(f"{label} must contain exactly two values")
    low, high = values
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (low, high)
    ):
        raise DataValidationError(f"{label} must contain finite numeric values")
    parsed = (float(low), float(high))
    if parsed[0] > parsed[1]:
        raise DataValidationError(f"{label} minimum must not exceed maximum")
    if positive and parsed[0] <= 0:
        raise DataValidationError(f"{label} values must be positive")
    return parsed


def _run_dir(dataset_root: Path, run_name: str) -> Path:
    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(
            "run_name must contain only letters, digits, dot, underscore, or hyphen"
        )
    synthetic_root = (dataset_root / "derived" / "synthetic").resolve(strict=False)
    output_dir = synthetic_root / run_name
    if output_dir.parent != synthetic_root:
        raise DataValidationError("run_name must select a direct synthetic run directory")
    if output_dir.exists() and os.path.normcase(str(output_dir.resolve())) != os.path.normcase(
        str(output_dir)
    ):
        raise DataValidationError(
            f"synthetic run path must not be a link or junction: {output_dir}"
        )
    return output_dir


def _image_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise DataValidationError(f"image directory does not exist: {directory}")
    paths = tuple(
        sorted(
            (
                path.resolve()
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.as_posix(),
        )
    )
    if not paths:
        raise DataValidationError(f"image directory contains no supported images: {directory}")
    return paths


def _source_records(
    dataset_root: Path, phase: str
) -> tuple[tuple[Path, ClassRecord], ...]:
    registry = load_class_registry(dataset_root / "class_registry.json")
    selected = tuple(
        record
        for record in registry.classes
        if phase == "all" or record.phase == phase
    )
    records: list[tuple[Path, ClassRecord]] = []
    for record in selected:
        directory = dataset_root / record.phase / record.folder_name
        assert_training_paths_safe([directory], dataset_root)
        records.extend((path, record) for path in _image_paths(directory))
    if not records:
        raise DataValidationError(f"no registered single-object images for phase {phase}")
    return tuple(sorted(records, key=lambda item: item[0].as_posix()))


def _open_rgb(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as decoded:
            decoded.load()
            return decoded.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise DataValidationError(f"cannot decode {label} image {path}: {exc}") from exc


def _extract_foreground(path: Path, threshold: int) -> Image.Image:
    image = _open_rgb(path, "source")
    channels = image.split()
    foreground_channels = tuple(
        channel.point([255 if value < threshold else 0 for value in range(256)])
        for channel in channels
    )
    mask = ImageChops.lighter(
        ImageChops.lighter(foreground_channels[0], foreground_channels[1]),
        foreground_channels[2],
    )
    bounds = mask.getbbox()
    if bounds is None:
        raise DataValidationError(f"source foreground mask is empty: {path}")
    if mask.getextrema() == (255, 255):
        raise DataValidationError(f"source foreground mask covers the entire image: {path}")
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba.crop(bounds)


def _manifest_path(path: Path, manifest_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path, manifest_dir)).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_manifest_path(value: object, manifest_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{label} must be a non-empty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    return candidate.resolve(strict=False)


def _transform_foreground(
    foreground: Image.Image,
    size: tuple[int, int],
    rotation: float,
    brightness: float,
    contrast: float,
) -> Image.Image:
    alpha = foreground.getchannel("A")
    rgb = foreground.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb.putalpha(alpha)
    transformed = rgb.resize(size, Image.Resampling.LANCZOS)
    transformed = transformed.rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )
    bounds = transformed.getchannel("A").getbbox()
    if bounds is None:
        raise DataValidationError("transformed foreground mask is empty")
    return transformed.crop(bounds)


def _overlaps(candidate: list[int], occupied: list[list[int]]) -> bool:
    x, y, width, height = candidate
    return any(
        x < other_x + other_width
        and x + width > other_x
        and y < other_y + other_height
        and y + height > other_y
        for other_x, other_y, other_width, other_height in occupied
    )


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=9, optimize=False)
    return buffer.getvalue()


def _rename_with_retry(source: Path, target: Path) -> Path:
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            return source.rename(target)
        except PermissionError:
            if attempt == _RENAME_ATTEMPTS - 1:
                raise
            time.sleep(_RENAME_DELAY_SECONDS)
    raise RuntimeError("unreachable rename retry state")


def _config_payload(config: SyntheticConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "size_fraction_range",
        "rotation_range",
        "brightness_range",
        "contrast_range",
    ):
        payload[key] = list(payload[key])
    return payload


def _generate_scene(
    *,
    scene_seed: int,
    background_path: Path,
    sources: tuple[tuple[Path, ClassRecord], ...],
    config: SyntheticConfig,
    manifest_dir: Path,
    foreground_cache: dict[Path, Image.Image],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    rng = random.Random(scene_seed)
    scene = _open_rgb(background_path, "background")
    occupied: list[list[int]] = []
    objects: list[dict[str, Any]] = []
    for object_index in range(config.objects_per_scene):
        source_path, record = rng.choice(sources)
        foreground = foreground_cache.get(source_path)
        if foreground is None:
            foreground = _extract_foreground(source_path, config.foreground_threshold)
            foreground_cache[source_path] = foreground
        target_fraction = rng.uniform(*config.size_fraction_range)
        target_side = max(1, round(min(scene.size) * target_fraction))
        scale = target_side / max(foreground.size)
        size = (
            max(1, round(foreground.width * scale)),
            max(1, round(foreground.height * scale)),
        )
        rotation = rng.uniform(*config.rotation_range)
        brightness = rng.uniform(*config.brightness_range)
        contrast = rng.uniform(*config.contrast_range)
        transformed = _transform_foreground(
            foreground, size, rotation, brightness, contrast
        )
        if transformed.width > scene.width or transformed.height > scene.height:
            raise DataValidationError(
                f"transformed object does not fit background: {source_path}"
            )
        bbox: list[int] | None = None
        for _ in range(_PLACEMENT_ATTEMPTS):
            x = rng.randint(0, scene.width - transformed.width)
            y = rng.randint(0, scene.height - transformed.height)
            candidate = [x, y, transformed.width, transformed.height]
            if not _overlaps(candidate, occupied):
                bbox = candidate
                break
        if bbox is None:
            raise DataValidationError(
                f"cannot place object {object_index} without overlap after "
                f"{_PLACEMENT_ATTEMPTS} attempts"
            )
        x, y, _, _ = bbox
        scene.paste(transformed, (x, y), transformed)
        occupied.append(bbox)
        objects.append(
            {
                "source_path": _manifest_path(source_path, manifest_dir),
                "category_id": record.category_id,
                "transform": {
                    "target_size_fraction": target_fraction,
                    "scale": scale,
                    "size": list(size),
                    "rotation_degrees": rotation,
                    "brightness": brightness,
                    "contrast": contrast,
                    "position": [x, y],
                },
                "bbox": bbox,
            }
        )
    return scene, objects


def generate_synthetic_dataset(
    dataset_root: str | Path,
    background_dir: str | Path,
    run_name: str,
    config: SyntheticConfig,
    overwrite: bool = False,
) -> SyntheticGenerationReport:
    if not isinstance(config, SyntheticConfig):
        raise DataValidationError("config must be a SyntheticConfig")
    config.validate()
    root = Path(dataset_root).resolve(strict=False)
    backgrounds_root = Path(background_dir).resolve(strict=False)
    assert_training_paths_safe([backgrounds_root], root)
    backgrounds = _image_paths(backgrounds_root)
    sources = _source_records(root, config.phase)
    output_dir = _run_dir(root, run_name)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise DataValidationError(
                f"synthetic run path must be a directory: {output_dir}"
            )
        if not overwrite:
            raise DataValidationError(f"synthetic run already exists: {output_dir}")
    synthetic_root = output_dir.parent
    synthetic_root.mkdir(parents=True, exist_ok=True)
    staging_dir = synthetic_root / f".{run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()

    master_rng = random.Random(config.seed)
    foreground_cache: dict[Path, Image.Image] = {}
    scenes: list[dict[str, Any]] = []
    backup_dir: Path | None = None
    try:
        for scene_index in range(config.scene_count):
            scene_seed = master_rng.randrange(0, 2**63)
            scene_rng = random.Random(scene_seed)
            background_path = scene_rng.choice(backgrounds)
            image, objects = _generate_scene(
                scene_seed=scene_seed,
                background_path=background_path,
                sources=sources,
                config=config,
                manifest_dir=staging_dir,
                foreground_cache=foreground_cache,
            )
            file_name = f"scene_{scene_index:06d}.png"
            encoded = _png_bytes(image)
            (staging_dir / file_name).write_bytes(encoded)
            scenes.append(
                {
                    "file_name": file_name,
                    "seed": scene_seed,
                    "width": image.width,
                    "height": image.height,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "background_path": _manifest_path(background_path, staging_dir),
                    "objects": objects,
                }
            )
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "generator_version": GENERATOR_VERSION,
            "pillow_version": PILLOW_VERSION,
            "seed": config.seed,
            "config": _config_payload(config),
            "scenes": scenes,
        }
        staging_manifest_path = staging_dir / "manifest.json"
        staging_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            backup_dir = synthetic_root / f".{run_name}.backup-{uuid.uuid4().hex}"
            _rename_with_retry(output_dir, backup_dir)
        try:
            _rename_with_retry(staging_dir, output_dir)
        except OSError:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                _rename_with_retry(backup_dir, output_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
            backup_dir = None
    except Exception as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            _rename_with_retry(backup_dir, output_dir)
        if isinstance(exc, OSError):
            raise DataValidationError(
                f"cannot write synthetic run {output_dir}: {exc}"
            ) from exc
        raise

    return SyntheticGenerationReport(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        image_count=len(scenes),
        object_count=sum(len(scene["objects"]) for scene in scenes),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load synthetic manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("synthetic manifest root must be an object")
    return payload


def _config_from_payload(value: object) -> SyntheticConfig:
    if not isinstance(value, dict):
        raise DataValidationError("manifest config must be an object")
    required = {
        "seed",
        "scene_count",
        "objects_per_scene",
        "phase",
        "size_fraction_range",
        "rotation_range",
        "brightness_range",
        "contrast_range",
        "foreground_threshold",
    }
    if set(value) != required:
        raise DataValidationError("manifest config fields do not match schema")
    try:
        config = SyntheticConfig(
            seed=value["seed"],
            scene_count=value["scene_count"],
            objects_per_scene=value["objects_per_scene"],
            phase=value["phase"],
            size_fraction_range=tuple(value["size_fraction_range"]),
            rotation_range=tuple(value["rotation_range"]),
            brightness_range=tuple(value["brightness_range"]),
            contrast_range=tuple(value["contrast_range"]),
            foreground_threshold=value["foreground_threshold"],
        )
    except (KeyError, TypeError) as exc:
        raise DataValidationError(f"invalid manifest config: {exc}") from exc
    config.validate()
    return config


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if positive and value <= 0:
        raise DataValidationError(f"{label} must be positive")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DataValidationError(f"{label} must be a finite number")
    parsed = float(value)
    if positive and parsed <= 0:
        raise DataValidationError(f"{label} must be positive")
    return parsed


def _require_in_range(
    value: float, configured_range: tuple[float, float], label: str
) -> None:
    if not configured_range[0] <= value <= configured_range[1]:
        raise DataValidationError(f"{label} is outside configured range")


def _int_pair(value: object, label: str, *, positive: bool = False) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise DataValidationError(f"{label} must contain two integers")
    return (
        _integer(value[0], f"{label}[0]", positive=positive),
        _integer(value[1], f"{label}[1]", positive=positive),
    )


def _bbox(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"{label} bbox must contain four integers")
    parsed = [_integer(item, f"{label} bbox") for item in value]
    if parsed[0] < 0 or parsed[1] < 0 or parsed[2] <= 0 or parsed[3] <= 0:
        raise DataValidationError(f"{label} bbox is invalid")
    return parsed


def _replay_scene(
    *,
    scene_record: dict[str, Any],
    manifest_dir: Path,
    dataset_root: Path,
    config: SyntheticConfig,
    category_sources: dict[int, Path],
    foreground_cache: dict[Path, Image.Image],
) -> tuple[Image.Image, list[list[int]]]:
    background_path = _resolve_manifest_path(
        scene_record.get("background_path"), manifest_dir, "background_path"
    )
    assert_training_paths_safe([background_path], dataset_root)
    scene = _open_rgb(background_path, "background")
    declared_width = _integer(scene_record.get("width"), "scene width", positive=True)
    declared_height = _integer(scene_record.get("height"), "scene height", positive=True)
    if scene.size != (declared_width, declared_height):
        raise DataValidationError("scene dimensions do not match background")
    raw_objects = scene_record.get("objects")
    if not isinstance(raw_objects, list):
        raise DataValidationError("scene objects must be a list")
    if len(raw_objects) != config.objects_per_scene:
        raise DataValidationError("scene object count does not match config")
    calculated_bboxes: list[list[int]] = []
    occupied: list[list[int]] = []
    for index, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, dict):
            raise DataValidationError(f"object {index} must be an object")
        if set(raw_object) != {"source_path", "category_id", "transform", "bbox"}:
            raise DataValidationError(f"object {index} fields do not match schema")
        category_id = _integer(raw_object.get("category_id"), f"object {index} category_id")
        expected_parent = category_sources.get(category_id)
        if expected_parent is None:
            raise DataValidationError(f"object {index} has unknown category_id {category_id}")
        source_path = _resolve_manifest_path(
            raw_object.get("source_path"), manifest_dir, f"object {index} source_path"
        )
        assert_training_paths_safe([source_path], dataset_root)
        if source_path.parent != expected_parent or source_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise DataValidationError(
                f"object {index} source_path does not match category_id {category_id}"
            )
        transform = raw_object.get("transform")
        if not isinstance(transform, dict):
            raise DataValidationError(f"object {index} transform must be an object")
        required_transform = {
            "target_size_fraction",
            "scale",
            "size",
            "rotation_degrees",
            "brightness",
            "contrast",
            "position",
        }
        if set(transform) != required_transform:
            raise DataValidationError(f"object {index} transform fields do not match schema")
        target_fraction = _number(
            transform["target_size_fraction"],
            "target_size_fraction",
            positive=True,
        )
        _require_in_range(
            target_fraction, config.size_fraction_range, "target_size_fraction"
        )
        scale = _number(transform["scale"], "scale", positive=True)
        size = _int_pair(transform["size"], "size", positive=True)
        rotation = _number(transform["rotation_degrees"], "rotation_degrees")
        brightness = _number(transform["brightness"], "brightness", positive=True)
        contrast = _number(transform["contrast"], "contrast", positive=True)
        _require_in_range(rotation, config.rotation_range, "rotation_degrees")
        _require_in_range(brightness, config.brightness_range, "brightness")
        _require_in_range(contrast, config.contrast_range, "contrast")
        position = _int_pair(transform["position"], "position")
        if position[0] < 0 or position[1] < 0:
            raise DataValidationError(f"object {index} position must be non-negative")
        foreground = foreground_cache.get(source_path)
        if foreground is None:
            foreground = _extract_foreground(source_path, config.foreground_threshold)
            foreground_cache[source_path] = foreground
        target_side = max(1, round(min(scene.size) * target_fraction))
        expected_scale = target_side / max(foreground.size)
        expected_size = (
            max(1, round(foreground.width * expected_scale)),
            max(1, round(foreground.height * expected_scale)),
        )
        if not math.isclose(scale, expected_scale, rel_tol=1e-12, abs_tol=1e-12):
            raise DataValidationError(f"object {index} scale does not match replay")
        if size != expected_size:
            raise DataValidationError(f"object {index} size does not match scale")
        transformed = _transform_foreground(
            foreground, size, rotation, brightness, contrast
        )
        calculated = [position[0], position[1], transformed.width, transformed.height]
        declared = _bbox(raw_object.get("bbox"), f"object {index}")
        if declared != calculated:
            raise DataValidationError(f"object {index} bbox does not match replay")
        if (
            calculated[0] + calculated[2] > scene.width
            or calculated[1] + calculated[3] > scene.height
        ):
            raise DataValidationError(f"object {index} bbox is outside image bounds")
        if _overlaps(calculated, occupied):
            raise DataValidationError(f"object {index} bbox overlaps another object")
        scene.paste(transformed, position, transformed)
        occupied.append(calculated)
        calculated_bboxes.append(calculated)
    return scene, calculated_bboxes


def validate_synthetic_dataset(
    dataset_root: str | Path, run_name: str
) -> SyntheticValidationReport:
    root = Path(dataset_root).resolve(strict=False)
    output_dir = _run_dir(root, run_name)
    manifest_path = output_dir / "manifest.json"
    payload = _load_manifest(manifest_path)
    if set(payload) != {
        "manifest_version",
        "generator_version",
        "pillow_version",
        "seed",
        "config",
        "scenes",
    }:
        raise DataValidationError("synthetic manifest fields do not match schema")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise DataValidationError("unsupported synthetic manifest version")
    if payload.get("generator_version") != GENERATOR_VERSION:
        raise DataValidationError("unsupported synthetic generator version")
    if payload.get("pillow_version") != PILLOW_VERSION:
        raise DataValidationError("manifest Pillow version does not match runtime")
    config = _config_from_payload(payload.get("config"))
    if payload.get("seed") != config.seed:
        raise DataValidationError("manifest seed does not match config seed")
    raw_scenes = payload.get("scenes")
    if not isinstance(raw_scenes, list):
        raise DataValidationError("manifest scenes must be a list")
    if len(raw_scenes) != config.scene_count:
        raise DataValidationError("manifest scene count does not match config")

    registry = load_class_registry(root / "class_registry.json")
    selected = tuple(
        record
        for record in registry.classes
        if config.phase == "all" or record.phase == config.phase
    )
    category_sources = {
        record.category_id: (root / record.phase / record.folder_name).resolve()
        for record in selected
    }
    declared_files: list[str] = []
    for index, scene in enumerate(raw_scenes):
        if not isinstance(scene, dict):
            raise DataValidationError(f"scene {index} must be an object")
        if set(scene) != {
            "file_name",
            "seed",
            "width",
            "height",
            "sha256",
            "background_path",
            "objects",
        }:
            raise DataValidationError(f"scene {index} fields do not match schema")
        file_name = scene.get("file_name")
        if not isinstance(file_name, str) or not _SCENE_NAME.fullmatch(file_name):
            raise DataValidationError(f"scene {index} file_name is invalid")
        declared_files.append(file_name)
    if len(declared_files) != len(set(declared_files)):
        raise DataValidationError("duplicate output image file_name in manifest")
    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual_files != set(declared_files):
        raise DataValidationError("manifest and output images do not match")

    seed_rng = random.Random(config.seed)
    expected_scene_seeds = [
        seed_rng.randrange(0, 2**63) for _ in range(config.scene_count)
    ]
    foreground_cache: dict[Path, Image.Image] = {}
    object_count = 0
    for index, scene in enumerate(raw_scenes):
        scene_seed = _integer(scene.get("seed"), f"scene {index} seed")
        if scene_seed != expected_scene_seeds[index]:
            raise DataValidationError(
                f"scene seed {scene_seed} is not derived from master seed"
            )
        expected, calculated_bboxes = _replay_scene(
            scene_record=scene,
            manifest_dir=output_dir,
            dataset_root=root,
            config=config,
            category_sources=category_sources,
            foreground_cache=foreground_cache,
        )
        expected_bytes = _png_bytes(expected)
        image_path = output_dir / declared_files[index]
        actual_bytes = image_path.read_bytes()
        actual_sha256 = hashlib.sha256(actual_bytes).hexdigest()
        declared_sha256 = scene.get("sha256")
        if not isinstance(declared_sha256, str) or actual_sha256 != declared_sha256:
            raise DataValidationError(f"scene {index} sha256 does not match manifest")
        if actual_bytes != expected_bytes:
            raise DataValidationError(f"scene {index} image does not match replay")
        object_count += len(calculated_bboxes)

    return SyntheticValidationReport(
        output_dir=output_dir,
        manifest_path=manifest_path,
        image_count=len(raw_scenes),
        object_count=object_count,
        generator_version=GENERATOR_VERSION,
    )
