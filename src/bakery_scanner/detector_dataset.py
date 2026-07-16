from __future__ import annotations

import hashlib
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
from typing import Any

from PIL import Image, UnidentifiedImageError

from .coco import validate_coco
from .errors import DataValidationError
from .registry import load_class_registry
from .safety import assert_training_paths_safe
from .splits import SCENE_PATTERN, scene_id_from_path
from .synthetic import validate_synthetic_dataset

BUILDER_VERSION = "1.0.0"
MANIFEST_VERSION = 1
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SPLITS = ("train", "validation")
_CLEANUP_ATTEMPTS = 20
_CLEANUP_DELAY_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class DetectorDatasetConfig:
    seed: int = 42
    validation_fraction: float = 0.2
    real_coco_path: str = "base/val/instances_val.json"

    def validate(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DataValidationError("seed must be an integer")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 1
        ):
            raise DataValidationError("validation_fraction must be between 0 and 1")
        if not isinstance(self.real_coco_path, str) or not self.real_coco_path:
            raise DataValidationError("real_coco_path must be a non-empty string")


@dataclass(frozen=True, slots=True)
class DetectorBuildReport:
    output_dir: Path
    manifest_path: Path
    image_count: int
    annotation_count: int
    train_image_count: int
    validation_image_count: int
    builder_version: str = BUILDER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "builder_version": self.builder_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "train_image_count": self.train_image_count,
            "validation_image_count": self.validation_image_count,
        }


@dataclass(frozen=True, slots=True)
class DetectorValidationReport:
    output_dir: Path
    manifest_path: Path
    image_count: int
    annotation_count: int
    train_image_count: int
    validation_image_count: int
    builder_version: str = BUILDER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "builder_version": self.builder_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "train_image_count": self.train_image_count,
            "validation_image_count": self.validation_image_count,
        }


@dataclass(frozen=True, slots=True)
class _Sample:
    key: str
    origin: str
    source_path: Path
    source_sha256: str
    width: int
    height: int
    original_annotations: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    resources: frozenset[str]


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DataValidationError(f"cannot hash file {path}: {exc}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} root must be an object")
    return payload


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


def _detector_run_dir(dataset_root: Path, run_name: str) -> Path:
    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(
            "run_name must contain only letters, digits, dot, underscore, or hyphen"
        )
    detector_root = (dataset_root / "derived" / "detector").resolve(strict=False)
    output_dir = detector_root / run_name
    if output_dir.parent != detector_root:
        raise DataValidationError("run_name must select a direct detector run directory")
    if output_dir.exists() and os.path.normcase(str(output_dir.resolve())) != os.path.normcase(
        str(output_dir)
    ):
        raise DataValidationError(
            f"detector run path must not be a link or junction: {output_dir}"
        )
    return output_dir


def _require_list(payload: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise DataValidationError(f"{label} {key} must be a list of objects")
    return value


def _normalized_path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _load_real_samples(root: Path, coco_path: Path) -> list[_Sample]:
    registry = load_class_registry(root / "class_registry.json")
    validate_coco(coco_path, registry, expected_phase="base")
    payload = _load_json(coco_path, "real COCO")
    images = _require_list(payload, "images", "real COCO")
    annotations = _require_list(payload, "annotations", "real COCO")
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    samples: list[_Sample] = []
    scene_difficulties: dict[str, set[str]] = {}
    for image in images:
        source_path = (coco_path.parent / image["file_name"]).resolve()
        source_sha256 = _sha256(source_path)
        scene_id = scene_id_from_path(source_path)
        match = SCENE_PATTERN.fullmatch(source_path.name)
        if match is None:
            raise DataValidationError(f"invalid real scene filename: {source_path.name}")
        difficulty = match.group("difficulty")
        group = scene_difficulties.setdefault(scene_id, set())
        if difficulty in group:
            raise DataValidationError(
                f"scene {scene_id} contains duplicate {difficulty} difficulty"
            )
        group.add(difficulty)
        originals = tuple(
            {
                "category_id": annotation["category_id"],
                "bbox": list(annotation["bbox"]),
                "area": annotation.get(
                    "area", annotation["bbox"][2] * annotation["bbox"][3]
                ),
                "iscrowd": annotation.get("iscrowd", 0),
            }
            for annotation in sorted(
                annotations_by_image.get(image["id"], []), key=lambda item: item["id"]
            )
        )
        samples.append(
            _Sample(
                key=f"real:{image['file_name']}",
                origin="real",
                source_path=source_path,
                source_sha256=source_sha256,
                width=image["width"],
                height=image["height"],
                original_annotations=originals,
                provenance={"scene_id": scene_id},
                resources=frozenset(
                    {f"real-scene:{scene_id}", f"image-sha256:{source_sha256}"}
                ),
            )
        )
    for scene_id, difficulties in scene_difficulties.items():
        if difficulties != {"e", "m", "h"}:
            raise DataValidationError(
                f"scene {scene_id} must contain exactly e, m, and h images"
            )
    return samples


def _load_synthetic_samples(root: Path, synthetic_run: str) -> list[_Sample]:
    validated = validate_synthetic_dataset(root, synthetic_run)
    payload = _load_json(validated.manifest_path, "synthetic manifest")
    scenes = _require_list(payload, "scenes", "synthetic manifest")
    samples: list[_Sample] = []
    for scene in scenes:
        source_path = (validated.output_dir / scene["file_name"]).resolve()
        source_sha256 = _sha256(source_path)
        background_path = _resolve_manifest_path(
            scene["background_path"], validated.output_dir, "background_path"
        )
        background_sha256 = _sha256(background_path)
        objects: list[dict[str, Any]] = []
        resources = {
            f"image-sha256:{source_sha256}",
            f"background-path:{_normalized_path_key(background_path)}",
            f"background-sha256:{background_sha256}",
        }
        for raw_object in scene["objects"]:
            object_path = _resolve_manifest_path(
                raw_object["source_path"], validated.output_dir, "source_path"
            )
            object_sha256 = _sha256(object_path)
            resources.update(
                {
                    f"object-path:{_normalized_path_key(object_path)}",
                    f"object-sha256:{object_sha256}",
                }
            )
            objects.append(
                {
                    "source_path": str(object_path),
                    "source_sha256": object_sha256,
                    "category_id": raw_object["category_id"],
                    "transform": raw_object["transform"],
                    "bbox": list(raw_object["bbox"]),
                }
            )
        originals = tuple(
            {
                "category_id": item["category_id"],
                "bbox": list(item["bbox"]),
                "area": item["bbox"][2] * item["bbox"][3],
                "iscrowd": 0,
            }
            for item in objects
        )
        samples.append(
            _Sample(
                key=f"synthetic:{synthetic_run}:{scene['file_name']}",
                origin="synthetic",
                source_path=source_path,
                source_sha256=source_sha256,
                width=scene["width"],
                height=scene["height"],
                original_annotations=originals,
                provenance={
                    "synthetic_run": synthetic_run,
                    "background_path": str(background_path),
                    "background_sha256": background_sha256,
                    "objects": objects,
                },
                resources=frozenset(resources),
            )
        )
    return samples


def _split_samples(
    samples: list[_Sample], validation_fraction: float, seed: int
) -> dict[str, str]:
    if len(samples) < 2:
        raise DataValidationError("safe train/validation split requires at least two images")
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    resource_owner: dict[str, int] = {}
    for index, sample in enumerate(samples):
        for resource in sorted(sample.resources):
            owner = resource_owner.setdefault(resource, index)
            union(index, owner)

    components_by_root: dict[int, list[int]] = {}
    for index in range(len(samples)):
        components_by_root.setdefault(find(index), []).append(index)
    components = [
        sorted(group, key=lambda index: samples[index].key)
        for group in components_by_root.values()
    ]
    if len(components) < 2:
        raise DataValidationError(
            "safe train/validation split is impossible because all images share leakage resources"
        )
    components.sort(key=lambda group: tuple(samples[index].key for index in group))
    random.Random(seed).shuffle(components)

    choices: dict[int, tuple[int, ...]] = {0: ()}
    for component_index, component in enumerate(components):
        for count, selected in list(choices.items()):
            choices.setdefault(count + len(component), (*selected, component_index))
    target = len(samples) * validation_fraction
    valid_counts = [count for count in choices if 0 < count < len(samples)]
    if not valid_counts:
        raise DataValidationError("safe train/validation split is impossible")
    validation_count = min(valid_counts, key=lambda count: (abs(count - target), count))
    selected_components = set(choices[validation_count])
    validation_indices = {
        index
        for component_index in selected_components
        for index in components[component_index]
    }
    return {
        sample.key: ("validation" if index in validation_indices else "train")
        for index, sample in enumerate(samples)
    }


def _output_name(sample: _Sample, synthetic_run: str) -> str:
    if sample.origin == "real":
        return f"real__{sample.source_path.name}"
    return f"synthetic__{synthetic_run}__{sample.source_path.name}"


def _write_split(
    staging_dir: Path,
    split: str,
    samples: list[_Sample],
    assignments: dict[str, str],
    synthetic_run: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_dir = staging_dir / split
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    manifest_samples: list[dict[str, Any]] = []
    annotation_id = 1
    selected = [sample for sample in samples if assignments[sample.key] == split]
    for image_id, sample in enumerate(sorted(selected, key=lambda item: item.key), start=1):
        output_name = _output_name(sample, synthetic_run)
        output_path = image_dir / output_name
        shutil.copyfile(sample.source_path, output_path)
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{output_name}",
                "width": sample.width,
                "height": sample.height,
            }
        )
        for original in sample.original_annotations:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": list(original["bbox"]),
                    "area": original["area"],
                    "iscrowd": original["iscrowd"],
                }
            )
            annotation_id += 1
        manifest_samples.append(
            {
                "sample_id": sample.key,
                "origin": sample.origin,
                "split": split,
                "source_path": _manifest_path(sample.source_path, staging_dir),
                "source_sha256": sample.source_sha256,
                "output_path": f"{split}/images/{output_name}",
                "output_sha256": _sha256(output_path),
                "width": sample.width,
                "height": sample.height,
                "original_annotations": list(sample.original_annotations),
                "provenance": _portable_provenance(sample.provenance, staging_dir),
            }
        )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "bread"}],
    }
    coco_path = split_dir / "instances.json"
    coco_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "coco_path": f"{split}/instances.json",
        "coco_sha256": _sha256(coco_path),
    }
    return manifest_samples, summary


def _portable_provenance(provenance: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    if "scene_id" in provenance:
        return dict(provenance)
    portable = dict(provenance)
    portable["background_path"] = _manifest_path(
        Path(portable["background_path"]), manifest_dir
    )
    portable["objects"] = [dict(item) for item in portable["objects"]]
    for item in portable["objects"]:
        item["source_path"] = _manifest_path(Path(item["source_path"]), manifest_dir)
    return portable


def _remove_tree_after_commit(path: Path) -> None:
    for attempt in range(_CLEANUP_ATTEMPTS):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt == _CLEANUP_ATTEMPTS - 1:
                return
            time.sleep(_CLEANUP_DELAY_SECONDS)


def _validate_bbox(bbox: object, width: int, height: int, label: str) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in bbox
    ):
        raise DataValidationError(f"{label} bbox must contain four finite numeric values")
    x, y, box_width, box_height = bbox
    if box_width <= 0 or box_height <= 0:
        raise DataValidationError(f"{label} bbox must have positive width and height")
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise DataValidationError(f"{label} bbox is outside image bounds")
    return bbox


def _expect_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise DataValidationError(f"{label} fields do not match schema")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if positive and value <= 0:
        raise DataValidationError(f"{label} must be positive")
    return value


def _sha_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataValidationError(f"{label} must be a lowercase SHA-256 value")
    return value


def _manifest_resources(sample: dict[str, Any], output_dir: Path) -> set[str]:
    split = sample.get("split")
    if split not in _SPLITS:
        raise DataValidationError("manifest sample split is invalid")
    resources = {f"image-sha256:{_sha_text(sample.get('source_sha256'), 'source_sha256')}"}
    provenance = sample.get("provenance")
    if sample.get("origin") == "real":
        provenance = _expect_keys(provenance, {"scene_id"}, "real provenance")
        scene_id = provenance.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise DataValidationError("real provenance scene_id must be non-empty")
        resources.add(f"real-scene:{scene_id}")
        return resources
    if sample.get("origin") != "synthetic":
        raise DataValidationError("manifest sample origin must be real or synthetic")
    provenance = _expect_keys(
        provenance,
        {"synthetic_run", "background_path", "background_sha256", "objects"},
        "synthetic provenance",
    )
    background_path = _resolve_manifest_path(
        provenance.get("background_path"), output_dir, "background_path"
    )
    resources.update(
        {
            f"background-path:{_normalized_path_key(background_path)}",
            "background-sha256:"
            + _sha_text(provenance.get("background_sha256"), "background_sha256"),
        }
    )
    objects = provenance.get("objects")
    if not isinstance(objects, list):
        raise DataValidationError("synthetic provenance objects must be a list")
    for index, item in enumerate(objects):
        item = _expect_keys(
            item,
            {"source_path", "source_sha256", "category_id", "transform", "bbox"},
            f"synthetic provenance object {index}",
        )
        source_path = _resolve_manifest_path(
            item.get("source_path"), output_dir, f"object {index} source_path"
        )
        resources.update(
            {
                f"object-path:{_normalized_path_key(source_path)}",
                "object-sha256:"
                + _sha_text(
                    item.get("source_sha256"), f"object {index} source_sha256"
                ),
            }
        )
    return resources


def _validate_no_split_leakage(samples: list[dict[str, Any]], output_dir: Path) -> None:
    resource_splits: dict[str, str] = {}
    for sample in samples:
        split = sample["split"]
        for resource in _manifest_resources(sample, output_dir):
            previous = resource_splits.setdefault(resource, split)
            if previous != split:
                raise DataValidationError(
                    f"split leakage detected for resource {resource}"
                )


def _validate_source_inputs(
    root: Path,
    output_dir: Path,
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, _Sample]:
    real_input = _expect_keys(inputs.get("real_coco"), {"path", "sha256"}, "real COCO input")
    synthetic_input = _expect_keys(
        inputs.get("synthetic_manifest"), {"path", "sha256"}, "synthetic manifest input"
    )
    real_coco_path = _resolve_manifest_path(
        real_input.get("path"), output_dir, "real COCO input path"
    )
    configured_real_path = _resolve_manifest_path(
        config.get("real_coco_path"), output_dir, "configured real COCO path"
    )
    if configured_real_path != real_coco_path:
        raise DataValidationError("configured and recorded real COCO paths do not match")
    assert_training_paths_safe([real_coco_path], root)
    if _sha256(real_coco_path) != _sha_text(real_input.get("sha256"), "real COCO sha256"):
        raise DataValidationError("source real COCO sha256 does not match manifest")

    synthetic_run = config.get("synthetic_run")
    if not isinstance(synthetic_run, str) or not synthetic_run:
        raise DataValidationError("configured synthetic_run must be non-empty")
    synthetic_manifest_path = _resolve_manifest_path(
        synthetic_input.get("path"), output_dir, "synthetic manifest input path"
    )
    assert_training_paths_safe([synthetic_manifest_path], root)
    expected_synthetic_manifest = (
        root / "derived" / "synthetic" / synthetic_run / "manifest.json"
    ).resolve(strict=False)
    if synthetic_manifest_path != expected_synthetic_manifest:
        raise DataValidationError("synthetic manifest path does not match configured run")
    if _sha256(synthetic_manifest_path) != _sha_text(
        synthetic_input.get("sha256"), "synthetic manifest sha256"
    ):
        raise DataValidationError("synthetic manifest sha256 does not match manifest")

    expected = _load_real_samples(root, real_coco_path)
    expected.extend(_load_synthetic_samples(root, synthetic_run))
    return {sample.key: sample for sample in expected}


def _validate_manifest_sample(
    sample: dict[str, Any], expected: _Sample, output_dir: Path, synthetic_run: str
) -> None:
    if sample.get("origin") != expected.origin:
        raise DataValidationError("manifest sample origin does not match source")
    source_path = _resolve_manifest_path(
        sample.get("source_path"), output_dir, "sample source_path"
    )
    if source_path != expected.source_path:
        raise DataValidationError("manifest sample source_path does not match source")
    if sample.get("source_sha256") != expected.source_sha256:
        raise DataValidationError("manifest sample source sha256 does not match source")
    if sample.get("width") != expected.width or sample.get("height") != expected.height:
        raise DataValidationError("manifest sample dimensions do not match source")
    if sample.get("original_annotations") != list(expected.original_annotations):
        raise DataValidationError("manifest annotation does not match source COCO provenance")
    portable_provenance = _portable_provenance(expected.provenance, output_dir)
    if sample.get("provenance") != portable_provenance:
        raise DataValidationError("manifest provenance does not match source")
    split = sample["split"]
    expected_output = f"{split}/images/{_output_name(expected, synthetic_run)}"
    if sample.get("output_path") != expected_output:
        raise DataValidationError("manifest output path does not match source sample")
    if sample.get("output_sha256") != expected.source_sha256:
        raise DataValidationError("manifest output sha256 does not match copied source")


def _validate_detector_coco(
    output_dir: Path,
    split: str,
    summary: dict[str, Any],
    samples_by_output: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    coco_path = output_dir / split / "instances.json"
    if _sha256(coco_path) != _sha_text(summary.get("coco_sha256"), f"{split} COCO sha256"):
        raise DataValidationError(f"{split} COCO sha256 does not match manifest")
    coco = _load_json(coco_path, f"{split} detector COCO")
    if set(coco) != {"images", "annotations", "categories"}:
        raise DataValidationError(f"{split} detector COCO fields do not match schema")
    if coco.get("categories") != [{"id": 1, "name": "bread"}]:
        raise DataValidationError(f"{split} COCO must contain only the bread category")
    images = _require_list(coco, "images", f"{split} detector COCO")
    annotations = _require_list(coco, "annotations", f"{split} detector COCO")
    image_ids: set[int] = set()
    image_paths: set[str] = set()
    images_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        image = _expect_keys(image, {"id", "file_name", "width", "height"}, "COCO image")
        image_id = _integer(image.get("id"), "COCO image id", positive=True)
        if image_id in image_ids:
            raise DataValidationError("duplicate image id in detector COCO")
        image_ids.add(image_id)
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name.startswith("images/"):
            raise DataValidationError("detector COCO file_name must be below images/")
        image_path = (output_dir / split / file_name).resolve(strict=False)
        try:
            output_key = image_path.relative_to(output_dir).as_posix()
        except ValueError as exc:
            raise DataValidationError("detector COCO image path escapes run") from exc
        if output_key in image_paths:
            raise DataValidationError("duplicate file_name in detector COCO")
        image_paths.add(output_key)
        sample = samples_by_output.get(output_key)
        if sample is None:
            raise DataValidationError(f"unmanifested output image: {output_key}")
        if sample.get("split") != split:
            raise DataValidationError("manifest sample split disagrees with output path")
        actual_sha = _sha256(image_path)
        if actual_sha != sample.get("output_sha256"):
            raise DataValidationError(f"output image sha256 does not match: {output_key}")
        try:
            with Image.open(image_path) as decoded:
                decoded.load()
                decoded_size = decoded.size
        except (OSError, UnidentifiedImageError) as exc:
            raise DataValidationError(f"cannot decode detector image {image_path}: {exc}") from exc
        width = _integer(image.get("width"), "COCO image width", positive=True)
        height = _integer(image.get("height"), "COCO image height", positive=True)
        if decoded_size != (width, height) or (width, height) != (
            sample.get("width"),
            sample.get("height"),
        ):
            raise DataValidationError(f"detector image dimensions do not match: {output_key}")
        images_by_id[image_id] = image

    annotation_ids: set[int] = set()
    actual_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        annotation = _expect_keys(
            annotation,
            {"id", "image_id", "category_id", "bbox", "area", "iscrowd"},
            "COCO annotation",
        )
        annotation_id = _integer(annotation.get("id"), "COCO annotation id", positive=True)
        if annotation_id in annotation_ids:
            raise DataValidationError("duplicate annotation id in detector COCO")
        annotation_ids.add(annotation_id)
        if annotation.get("category_id") != 1:
            raise DataValidationError("detector annotation category_id must be 1")
        image_id = _integer(annotation.get("image_id"), "COCO annotation image_id", positive=True)
        image = images_by_id.get(image_id)
        if image is None:
            raise DataValidationError("detector annotation references unknown image_id")
        bbox = _validate_bbox(
            annotation.get("bbox"), image["width"], image["height"], "detector annotation"
        )
        area = annotation.get("area")
        if (
            isinstance(area, bool)
            or not isinstance(area, (int, float))
            or not math.isfinite(area)
            or area <= 0
        ):
            raise DataValidationError("detector annotation area must be positive and finite")
        if not math.isclose(area, bbox[2] * bbox[3], rel_tol=1e-9, abs_tol=1e-9):
            raise DataValidationError("detector annotation area does not match bbox")
        if annotation.get("iscrowd") not in {0, 1}:
            raise DataValidationError("detector annotation iscrowd must be 0 or 1")
        actual_by_image.setdefault(image_id, []).append(
            {
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
                "area": annotation["area"],
                "iscrowd": annotation["iscrowd"],
            }
        )

    for image_id, image in images_by_id.items():
        output_key = (output_dir / split / image["file_name"]).relative_to(output_dir).as_posix()
        sample = samples_by_output[output_key]
        expected_annotations = [
            {
                "category_id": 1,
                "bbox": item["bbox"],
                "area": item["area"],
                "iscrowd": item["iscrowd"],
            }
            for item in sample["original_annotations"]
        ]
        if actual_by_image.get(image_id, []) != expected_annotations:
            raise DataValidationError("manifest annotations do not match detector COCO")
    if len(images) != summary.get("image_count") or len(annotations) != summary.get(
        "annotation_count"
    ):
        raise DataValidationError(f"{split} counts do not match manifest")
    return len(images), len(annotations)


def _validate_run_dir(root: Path, output_dir: Path) -> DetectorValidationReport:
    manifest_path = output_dir / "manifest.json"
    manifest = _load_json(manifest_path, "detector manifest")
    if set(manifest) != {
        "manifest_version",
        "builder_version",
        "config",
        "inputs",
        "splits",
        "samples",
    }:
        raise DataValidationError("detector manifest fields do not match schema")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise DataValidationError("unsupported detector manifest version")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise DataValidationError("unsupported detector builder version")
    config = _expect_keys(
        manifest.get("config"),
        {"seed", "validation_fraction", "real_coco_path", "synthetic_run"},
        "detector config",
    )
    DetectorDatasetConfig(
        seed=config.get("seed"),
        validation_fraction=config.get("validation_fraction"),
        real_coco_path=config.get("real_coco_path"),
    ).validate()
    inputs = _expect_keys(
        manifest.get("inputs"), {"real_coco", "synthetic_manifest"}, "detector inputs"
    )
    samples = _require_list(manifest, "samples", "detector manifest")
    summaries = _expect_keys(manifest.get("splits"), set(_SPLITS), "detector splits")
    for split in _SPLITS:
        _expect_keys(
            summaries.get(split),
            {"image_count", "annotation_count", "coco_path", "coco_sha256"},
            f"{split} split summary",
        )
        expected_coco_path = f"{split}/instances.json"
        if summaries[split].get("coco_path") != expected_coco_path:
            raise DataValidationError(f"{split} COCO path does not match schema")

    sample_keys = {
        "sample_id",
        "origin",
        "split",
        "source_path",
        "source_sha256",
        "output_path",
        "output_sha256",
        "width",
        "height",
        "original_annotations",
        "provenance",
    }
    for sample in samples:
        _expect_keys(sample, sample_keys, "detector sample")
    sample_ids = [sample.get("sample_id") for sample in samples]
    output_paths = [sample.get("output_path") for sample in samples]
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise DataValidationError("sample_id must be a non-empty string")
    if len(sample_ids) != len(set(sample_ids)):
        raise DataValidationError("duplicate sample_id in detector manifest")
    if any(not isinstance(value, str) or not value for value in output_paths):
        raise DataValidationError("output_path must be a non-empty string")
    if len(output_paths) != len(set(output_paths)):
        raise DataValidationError("duplicate output_path in detector manifest")

    _validate_no_split_leakage(samples, output_dir)
    expected_sources = _validate_source_inputs(root, output_dir, config, inputs)
    if set(sample_ids) != set(expected_sources):
        raise DataValidationError("manifest samples do not match source samples")
    expected_assignments = _split_samples(
        list(expected_sources.values()),
        config["validation_fraction"],
        config["seed"],
    )
    actual_assignments = {
        sample["sample_id"]: sample["split"] for sample in samples
    }
    if actual_assignments != expected_assignments:
        raise DataValidationError("manifest split does not match recorded config")
    synthetic_run = config["synthetic_run"]
    for sample in samples:
        _validate_manifest_sample(
            sample, expected_sources[sample["sample_id"]], output_dir, synthetic_run
        )

    sample_by_output = {sample["output_path"]: sample for sample in samples}
    annotation_total = 0
    split_counts: dict[str, int] = {}
    for split in _SPLITS:
        summary = summaries.get(split)
        image_count, annotation_count = _validate_detector_coco(
            output_dir, split, summary, sample_by_output
        )
        split_counts[split] = image_count
        annotation_total += annotation_count

    expected_files = {
        "manifest.json",
        "train/instances.json",
        "validation/instances.json",
        *sample_by_output,
    }
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise DataValidationError(
            "detector run file inventory does not match manifest: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    expected_dirs = {"train", "train/images", "validation", "validation/images"}
    actual_dirs = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_dir()
    }
    if actual_dirs != expected_dirs:
        raise DataValidationError("detector run directory inventory does not match schema")
    return DetectorValidationReport(
        output_dir=output_dir,
        manifest_path=manifest_path,
        image_count=len(samples),
        annotation_count=annotation_total,
        train_image_count=split_counts["train"],
        validation_image_count=split_counts["validation"],
    )


def build_detector_dataset(
    dataset_root: str | Path,
    synthetic_run: str,
    run_name: str,
    config: DetectorDatasetConfig,
    overwrite: bool = False,
) -> DetectorBuildReport:
    if not isinstance(config, DetectorDatasetConfig):
        raise DataValidationError("config must be a DetectorDatasetConfig")
    config.validate()
    root = Path(dataset_root).resolve(strict=False)
    real_coco_path = assert_training_paths_safe([config.real_coco_path], root)[0]
    synthetic_dir = (root / "derived" / "synthetic" / synthetic_run).resolve(strict=False)
    assert_training_paths_safe([synthetic_dir], root)
    output_dir = _detector_run_dir(root, run_name)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise DataValidationError(f"detector run path must be a directory: {output_dir}")
        if not overwrite:
            raise DataValidationError(f"detector run already exists: {output_dir}")

    samples = _load_real_samples(root, real_coco_path)
    samples.extend(_load_synthetic_samples(root, synthetic_run))
    assignments = _split_samples(samples, config.validation_fraction, config.seed)

    detector_root = output_dir.parent
    detector_root.mkdir(parents=True, exist_ok=True)
    staging_dir = detector_root / f".{run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    backup_dir: Path | None = None
    try:
        manifest_samples: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}
        for split in _SPLITS:
            split_samples, summary = _write_split(
                staging_dir, split, samples, assignments, synthetic_run
            )
            manifest_samples.extend(split_samples)
            summaries[split] = summary
        manifest_samples.sort(key=lambda item: item["sample_id"])
        synthetic_manifest = root / "derived" / "synthetic" / synthetic_run / "manifest.json"
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "builder_version": BUILDER_VERSION,
            "config": {
                "seed": config.seed,
                "validation_fraction": config.validation_fraction,
                "real_coco_path": _manifest_path(real_coco_path, staging_dir),
                "synthetic_run": synthetic_run,
            },
            "inputs": {
                "real_coco": {
                    "path": _manifest_path(real_coco_path, staging_dir),
                    "sha256": _sha256(real_coco_path),
                },
                "synthetic_manifest": {
                    "path": _manifest_path(synthetic_manifest, staging_dir),
                    "sha256": _sha256(synthetic_manifest),
                },
            },
            "splits": summaries,
            "samples": manifest_samples,
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validated = _validate_run_dir(root, staging_dir)
        if output_dir.exists():
            if not overwrite:
                raise DataValidationError(
                    f"detector run appeared during generation: {output_dir}"
                )
            _detector_run_dir(root, run_name)
            if not output_dir.is_dir():
                raise DataValidationError(
                    f"detector run path must be a directory: {output_dir}"
                )
            backup_dir = detector_root / f".{run_name}.backup-{uuid.uuid4().hex}"
            output_dir.rename(backup_dir)
        try:
            staging_dir.rename(output_dir)
        except OSError:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                backup_dir.rename(output_dir)
            raise
        if backup_dir is not None:
            _remove_tree_after_commit(backup_dir)
            backup_dir = None
        return DetectorBuildReport(
            output_dir=output_dir,
            manifest_path=output_dir / "manifest.json",
            image_count=validated.image_count,
            annotation_count=validated.annotation_count,
            train_image_count=validated.train_image_count,
            validation_image_count=validated.validation_image_count,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise


def validate_detector_dataset(
    dataset_root: str | Path, run_name: str
) -> DetectorValidationReport:
    root = Path(dataset_root).resolve(strict=False)
    output_dir = _detector_run_dir(root, run_name)
    return _validate_run_dir(root, output_dir)
