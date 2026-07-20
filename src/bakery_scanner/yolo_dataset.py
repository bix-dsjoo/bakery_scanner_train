from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from .detector_dataset import validate_detector_dataset
from .errors import DataValidationError
from .safety import assert_training_paths_safe

CONVERTER_VERSION = "1.0.0"
MANIFEST_VERSION = 1
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SPLITS = ("train", "validation")


@dataclass(frozen=True, slots=True)
class YoloDatasetReport:
    output_dir: Path
    manifest_path: Path
    image_count: int
    annotation_count: int
    train_image_count: int
    validation_image_count: int
    converter_version: str = CONVERTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "converter_version": self.converter_version,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "train_image_count": self.train_image_count,
            "validation_image_count": self.validation_image_count,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dir(dataset_root: Path, run_name: str) -> Path:
    if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(f"invalid YOLO run name: {run_name!r}")
    root = (dataset_root / "derived" / "yolo").resolve(strict=False)
    output = (root / run_name).resolve(strict=False)
    if output.parent != root:
        raise DataValidationError(f"YOLO run escapes output root: {run_name}")
    return output


def _label_line(annotation: dict[str, Any], width: int, height: int) -> str:
    x, y, box_width, box_height = annotation["bbox"]
    values = (
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )
    return "0 " + " ".join(f"{value:.10f}" for value in values)


def _expect_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise DataValidationError(
            f"{label} fields are invalid: expected={sorted(expected)}, actual={actual}"
        )
    return value


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return value


def _relative_file(run_dir: Path, value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{label} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataValidationError(f"{label} must stay inside the YOLO run")
    resolved = (run_dir / relative).resolve(strict=False)
    try:
        common = os.path.commonpath([str(run_dir.resolve()), str(resolved)])
    except ValueError as exc:
        raise DataValidationError(f"{label} escapes the YOLO run") from exc
    if os.path.normcase(common) != os.path.normcase(str(run_dir.resolve())):
        raise DataValidationError(f"{label} escapes the YOLO run")
    return resolved, relative.as_posix()


def _validate_label(
    path: Path,
    annotations: list[dict[str, Any]],
    width: int,
    height: int,
    label: str,
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"cannot read {label}: {exc}") from exc
    if len(lines) != len(annotations):
        raise DataValidationError(
            f"{label} annotation count mismatch: {len(lines)} != {len(annotations)}"
        )
    for index, (line, annotation) in enumerate(zip(lines, annotations, strict=True)):
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise DataValidationError(f"{label} line {index} must use bread class 0")
        try:
            values = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise DataValidationError(f"{label} line {index} has invalid numbers") from exc
        if not all(math.isfinite(value) for value in values):
            raise DataValidationError(f"{label} line {index} must be finite")
        x_center, y_center, box_width, box_height = values
        if (
            box_width <= 0
            or box_height <= 0
            or x_center - box_width / 2 < -1e-9
            or y_center - box_height / 2 < -1e-9
            or x_center + box_width / 2 > 1 + 1e-9
            or y_center + box_height / 2 > 1 + 1e-9
        ):
            raise DataValidationError(f"{label} line {index} is outside normalized bounds")
        expected = tuple(
            float(value)
            for value in _label_line(annotation, width, height).split()[1:]
        )
        if any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9)
            for actual, wanted in zip(values, expected, strict=True)
        ):
            raise DataValidationError(f"{label} line {index} disagrees with provenance")


def _validate_run_dir(
    dataset_root: Path,
    run_dir: Path,
    *,
    data_root: Path | None = None,
) -> YoloDatasetReport:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load YOLO manifest {manifest_path}: {exc}") from exc
    manifest = _expect_keys(
        manifest,
        {"manifest_version", "converter_version", "source", "splits", "samples"},
        "YOLO manifest",
    )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise DataValidationError(
            f"unsupported YOLO manifest version: {manifest['manifest_version']}"
        )
    if manifest["converter_version"] != CONVERTER_VERSION:
        raise DataValidationError(
            f"unsupported YOLO converter version: {manifest['converter_version']}"
        )

    source = _expect_keys(
        manifest["source"],
        {"run_name", "manifest_path", "manifest_sha256"},
        "YOLO source",
    )
    if not isinstance(source["run_name"], str) or not source["run_name"]:
        raise DataValidationError("YOLO source run_name must be non-empty")
    source_report = validate_detector_dataset(dataset_root, source["run_name"])
    assert_training_paths_safe([source_report.output_dir], dataset_root)
    recorded_source_path = Path(source["manifest_path"]).resolve(strict=False)
    if recorded_source_path != source_report.manifest_path.resolve(strict=False):
        raise DataValidationError("YOLO source manifest path changed")
    if source["manifest_sha256"] != _sha256(source_report.manifest_path):
        raise DataValidationError("YOLO source manifest hash changed")
    source_manifest = json.loads(source_report.manifest_path.read_text(encoding="utf-8"))
    source_samples = {item["sample_id"]: item for item in source_manifest["samples"]}

    splits = _expect_keys(manifest["splits"], set(_SPLITS), "YOLO splits")
    split_counts: dict[str, dict[str, int]] = {}
    for split in _SPLITS:
        summary = _expect_keys(
            splits[split], {"images", "annotations"}, f"YOLO split {split}"
        )
        split_counts[split] = {
            "images": _positive_int(
                summary["images"], f"YOLO split {split} images", allow_zero=True
            ),
            "annotations": _positive_int(
                summary["annotations"],
                f"YOLO split {split} annotations",
                allow_zero=True,
            ),
        }

    samples = manifest["samples"]
    if not isinstance(samples, list):
        raise DataValidationError("YOLO samples must be a list")
    expected_files = {"manifest.json", "data.yaml"}
    expected_dirs = {
        "train",
        "train/images",
        "train/labels",
        "validation",
        "validation/images",
        "validation/labels",
    }
    actual_counts = {split: {"images": 0, "annotations": 0} for split in _SPLITS}
    seen_sample_ids: set[str] = set()
    sample_fields = {
        "sample_id",
        "source_sample_id",
        "split",
        "width",
        "height",
        "annotation_count",
        "original_annotations",
        "image_path",
        "image_sha256",
        "label_path",
        "label_sha256",
    }
    for index, raw_sample in enumerate(samples):
        sample = _expect_keys(raw_sample, sample_fields, f"YOLO sample {index}")
        sample_id = sample["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_sample_ids:
            raise DataValidationError(f"YOLO sample {index} has invalid sample_id")
        seen_sample_ids.add(sample_id)
        source_sample_id = sample["source_sample_id"]
        if source_sample_id not in source_samples:
            raise DataValidationError(f"YOLO sample {sample_id} has unknown source sample")
        source_sample = source_samples[source_sample_id]
        split = sample["split"]
        if split not in _SPLITS or split != source_sample["split"]:
            raise DataValidationError(f"YOLO sample {sample_id} has invalid split")
        width = _positive_int(sample["width"], f"YOLO sample {sample_id} width")
        height = _positive_int(sample["height"], f"YOLO sample {sample_id} height")
        if (width, height) != (source_sample["width"], source_sample["height"]):
            raise DataValidationError(f"YOLO sample {sample_id} dimensions changed")
        annotations = sample["original_annotations"]
        if annotations != source_sample["original_annotations"]:
            raise DataValidationError(f"YOLO sample {sample_id} provenance changed")
        annotation_count = _positive_int(
            sample["annotation_count"],
            f"YOLO sample {sample_id} annotation_count",
            allow_zero=True,
        )
        if not isinstance(annotations, list) or annotation_count != len(annotations):
            raise DataValidationError(f"YOLO sample {sample_id} annotation count changed")

        image_path, image_relative = _relative_file(
            run_dir, sample["image_path"], f"YOLO sample {sample_id} image_path"
        )
        label_path, label_relative = _relative_file(
            run_dir, sample["label_path"], f"YOLO sample {sample_id} label_path"
        )
        if not image_relative.startswith(f"{split}/images/") or not label_relative.startswith(
            f"{split}/labels/"
        ):
            raise DataValidationError(f"YOLO sample {sample_id} output path has wrong split")
        if not image_path.is_file() or not label_path.is_file():
            raise DataValidationError(f"YOLO sample {sample_id} output file is missing")
        if sample["image_sha256"] != _sha256(image_path):
            raise DataValidationError(f"YOLO sample {sample_id} image hash changed")
        if sample["label_sha256"] != _sha256(label_path):
            raise DataValidationError(f"YOLO sample {sample_id} label hash changed")
        try:
            with Image.open(image_path) as image:
                image.load()
                decoded_size = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise DataValidationError(
                f"YOLO sample {sample_id} image cannot be decoded: {exc}"
            ) from exc
        if decoded_size != (width, height):
            raise DataValidationError(f"YOLO sample {sample_id} decoded size changed")
        _validate_label(label_path, annotations, width, height, f"YOLO sample {sample_id} label")
        expected_files.update({image_relative, label_relative})
        actual_counts[split]["images"] += 1
        actual_counts[split]["annotations"] += annotation_count

    if set(source_samples) != seen_sample_ids:
        raise DataValidationError("YOLO samples do not match source detector inventory")
    if actual_counts != split_counts:
        raise DataValidationError("YOLO split counts disagree with samples")

    actual_files = {
        item.relative_to(run_dir).as_posix()
        for item in run_dir.rglob("*")
        if item.is_file()
    }
    actual_dirs = {
        item.relative_to(run_dir).as_posix()
        for item in run_dir.rglob("*")
        if item.is_dir()
    }
    runtime_cache_files = {
        "train/labels.cache",
        "validation/labels.cache",
    }
    inventory_files = actual_files - runtime_cache_files
    if inventory_files != expected_files or actual_dirs != expected_dirs:
        raise DataValidationError(
            "YOLO output inventory mismatch: "
            f"missing={sorted(expected_files - inventory_files)}, "
            f"extra={sorted(inventory_files - expected_files)}"
        )

    try:
        data = yaml.safe_load((run_dir / "data.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot load YOLO data.yaml: {exc}") from exc
    data = _expect_keys(data, {"path", "train", "val", "names"}, "YOLO data.yaml")
    expected_data_root = (data_root or run_dir).resolve(strict=False)
    if data != {
        "path": str(expected_data_root),
        "train": "train/images",
        "val": "validation/images",
        "names": {0: "bread"},
    }:
        raise DataValidationError("YOLO data.yaml disagrees with run layout")

    return YoloDatasetReport(
        output_dir=run_dir,
        manifest_path=manifest_path,
        image_count=sum(actual_counts[split]["images"] for split in _SPLITS),
        annotation_count=sum(
            actual_counts[split]["annotations"] for split in _SPLITS
        ),
        train_image_count=actual_counts["train"]["images"],
        validation_image_count=actual_counts["validation"]["images"],
    )


def build_yolo_dataset(
    dataset_root: str | Path,
    source_run: str,
    run_name: str,
    overwrite: bool = False,
) -> YoloDatasetReport:
    root = Path(dataset_root).resolve(strict=False)
    source_report = validate_detector_dataset(root, source_run)
    source_dir = source_report.output_dir
    source_manifest_path = source_report.manifest_path
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_dir = _run_dir(root, run_name)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise DataValidationError(f"YOLO run path must be a directory: {output_dir}")
        if not overwrite:
            raise DataValidationError(f"YOLO run already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    backup_dir: Path | None = None
    try:
        samples: list[dict[str, Any]] = []
        split_counts = {split: {"images": 0, "annotations": 0} for split in _SPLITS}
        for split in _SPLITS:
            (staging_dir / split / "images").mkdir(parents=True)
            (staging_dir / split / "labels").mkdir()

        for source_sample in source_manifest["samples"]:
            split = source_sample["split"]
            source_image = source_dir / source_sample["output_path"]
            image_name = Path(source_sample["output_path"]).name
            output_image = staging_dir / split / "images" / image_name
            output_label = staging_dir / split / "labels" / f"{Path(image_name).stem}.txt"
            shutil.copy2(source_image, output_image)
            annotations = source_sample["original_annotations"]
            label_text = "".join(
                _label_line(annotation, source_sample["width"], source_sample["height"])
                + "\n"
                for annotation in annotations
            )
            output_label.write_text(label_text, encoding="utf-8")
            split_counts[split]["images"] += 1
            split_counts[split]["annotations"] += len(annotations)
            samples.append(
                {
                    "sample_id": source_sample["sample_id"],
                    "source_sample_id": source_sample["sample_id"],
                    "split": split,
                    "width": source_sample["width"],
                    "height": source_sample["height"],
                    "annotation_count": len(annotations),
                    "original_annotations": annotations,
                    "image_path": output_image.relative_to(staging_dir).as_posix(),
                    "image_sha256": _sha256(output_image),
                    "label_path": output_label.relative_to(staging_dir).as_posix(),
                    "label_sha256": _sha256(output_label),
                }
            )

        samples.sort(key=lambda item: item["sample_id"])
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "converter_version": CONVERTER_VERSION,
            "source": {
                "run_name": source_run,
                "manifest_path": str(source_manifest_path),
                "manifest_sha256": _sha256(source_manifest_path),
            },
            "splits": split_counts,
            "samples": samples,
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(output_dir),
                    "train": "train/images",
                    "val": "validation/images",
                    "names": {0: "bread"},
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _validate_run_dir(root, staging_dir, data_root=output_dir)
        if output_dir.exists():
            if not overwrite:
                raise DataValidationError(f"YOLO run appeared during generation: {output_dir}")
            backup_dir = output_dir.parent / f".{run_name}.backup-{uuid.uuid4().hex}"
            output_dir.rename(backup_dir)
        try:
            staging_dir.rename(output_dir)
        except OSError:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                backup_dir.rename(output_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
            backup_dir = None
        return validate_yolo_dataset(root, run_name)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise


def validate_yolo_dataset(
    dataset_root: str | Path,
    run_name: str,
) -> YoloDatasetReport:
    root = Path(dataset_root).resolve(strict=False)
    output_dir = _run_dir(root, run_name)
    return _validate_run_dir(root, output_dir)
