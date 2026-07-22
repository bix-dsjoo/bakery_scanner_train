from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .detector_evaluation import Detection, evaluate_detector_predictions
from .detector_postprocess import DetectorPostprocessConfig, filter_detections
from .detector_training import (
    DetectorBackend,
    DetectorTrainingConfig,
    UltralyticsBackend,
    _environment_metadata,
    _evaluation_inputs,
    load_detector_training_config,
)
from .errors import DataValidationError
from .safety import assert_training_paths_safe
from .yolo_dataset import validate_yolo_dataset

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_FIELDS = {
    "dataset_root",
    "output_root",
    "run_name",
    "members",
    "cpu_threads",
    "cpu_warmups",
    "cpu_repetitions",
}
_MEMBER_FIELDS = {
    "config_path",
    "config_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
}


@dataclass(frozen=True, slots=True)
class DetectorEnsembleMember:
    config_path: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class DetectorEnsembleConfig:
    dataset_root: str
    output_root: str
    run_name: str
    members: tuple[DetectorEnsembleMember, DetectorEnsembleMember]
    cpu_threads: int
    cpu_warmups: int
    cpu_repetitions: int


@dataclass(frozen=True, slots=True)
class DetectorEnsembleReport:
    output_dir: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path
    benchmark_path: Path | None = None
    split: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        result = {
            "status": "ok",
            "split": self.split,
            "output_dir": str(self.output_dir),
            "metadata_path": str(self.metadata_path),
            "predictions_path": str(self.predictions_path),
            "metrics_path": str(self.metrics_path),
            "metrics": metrics["metrics"],
        }
        if self.benchmark_path is not None:
            result["benchmark_path"] = str(self.benchmark_path)
        return result


def _strict_object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise DataValidationError(
            f"{label} fields are invalid: expected={sorted(fields)}, actual={actual}"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    result = _text(value, label)
    if not _SHA256.fullmatch(result):
        raise DataValidationError(f"{label} must be a lowercase SHA-256")
    return result


def _integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return value


def load_detector_ensemble_config(path: str | Path) -> DetectorEnsembleConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot load detector ensemble config {source}: {exc}") from exc
    payload = _strict_object(payload, _CONFIG_FIELDS, "detector ensemble config")
    raw_members = payload["members"]
    if not isinstance(raw_members, list) or len(raw_members) != 2:
        raise DataValidationError("detector ensemble requires exactly two members")
    members: list[DetectorEnsembleMember] = []
    for index, value in enumerate(raw_members):
        member = _strict_object(value, _MEMBER_FIELDS, f"ensemble member {index}")
        members.append(
            DetectorEnsembleMember(
                config_path=_text(member["config_path"], "member config_path"),
                config_sha256=_sha(member["config_sha256"], "member config SHA-256"),
                checkpoint_path=_text(
                    member["checkpoint_path"], "member checkpoint_path"
                ),
                checkpoint_sha256=_sha(
                    member["checkpoint_sha256"], "member checkpoint SHA-256"
                ),
            )
        )
    if len({member.config_path for member in members}) != 2 or len(
        {member.checkpoint_path for member in members}
    ) != 2:
        raise DataValidationError("detector ensemble member paths must be unique")
    run_name = _text(payload["run_name"], "run_name")
    if not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(f"run_name is invalid: {run_name!r}")
    return DetectorEnsembleConfig(
        dataset_root=_text(payload["dataset_root"], "dataset_root"),
        output_root=_text(payload["output_root"], "output_root"),
        run_name=run_name,
        members=(members[0], members[1]),
        cpu_threads=_integer(payload["cpu_threads"], "cpu_threads"),
        cpu_warmups=_integer(payload["cpu_warmups"], "cpu_warmups", allow_zero=True),
        cpu_repetitions=_integer(payload["cpu_repetitions"], "cpu_repetitions"),
    )


def merge_member_predictions(
    member_predictions: Sequence[Mapping[str, Sequence[Detection]]],
    image_ids: Sequence[str],
    postprocess: DetectorPostprocessConfig,
) -> dict[str, tuple[Detection, ...]]:
    if len(member_predictions) != 2:
        raise DataValidationError("detector ensemble requires exactly two prediction sets")
    expected_order = tuple(image_ids)
    if len(expected_order) != len(set(expected_order)):
        raise DataValidationError("detector ensemble image IDs must be unique")
    expected = set(expected_order)
    for index, predictions in enumerate(member_predictions):
        if set(predictions) != expected:
            raise DataValidationError(
                f"ensemble member {index} image IDs do not match evaluation images"
            )
    merged: dict[str, tuple[Detection, ...]] = {}
    for image_id in expected_order:
        candidates = tuple(
            detection
            for predictions in member_predictions
            for detection in predictions[image_id]
        )
        merged[image_id] = filter_detections(candidates, postprocess)
    return merged


@dataclass(frozen=True, slots=True)
class _PreparedMember:
    declared: DetectorEnsembleMember
    config_path: Path
    checkpoint_path: Path
    config: DetectorTrainingConfig
    yolo_dir: Path
    validation_signature: tuple[tuple[object, ...], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve(strict=False)


def _validation_signature(manifest_path: Path) -> tuple[tuple[object, ...], ...]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load YOLO manifest {manifest_path}: {exc}") from exc
    samples = []
    for sample in payload.get("samples", []):
        if sample.get("split") != "validation":
            continue
        samples.append(
            (
                Path(sample["image_path"]).name,
                sample["image_sha256"],
                Path(sample["label_path"]).name,
                sample["label_sha256"],
                sample["width"],
                sample["height"],
                sample["annotation_count"],
                json.dumps(sample["original_annotations"], sort_keys=True),
            )
        )
    if not samples:
        raise DataValidationError("ensemble validation split must not be empty")
    return tuple(sorted(samples))


def _prepare_members(config: DetectorEnsembleConfig) -> tuple[_PreparedMember, ...]:
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    project_root = dataset_root.parent
    prepared = []
    for index, declared in enumerate(config.members):
        config_path = _resolve_project_path(declared.config_path, project_root)
        checkpoint_path = _resolve_project_path(declared.checkpoint_path, project_root)
        assert_training_paths_safe([config_path, checkpoint_path], dataset_root)
        if not config_path.is_file():
            raise DataValidationError(f"ensemble member {index} config is missing: {config_path}")
        if not checkpoint_path.is_file():
            raise DataValidationError(
                f"ensemble member {index} checkpoint is missing: {checkpoint_path}"
            )
        if _sha256(config_path) != declared.config_sha256:
            raise DataValidationError(f"ensemble member {index} config SHA-256 mismatch")
        if _sha256(checkpoint_path) != declared.checkpoint_sha256:
            raise DataValidationError(
                f"ensemble member {index} checkpoint SHA-256 mismatch"
            )
        member_config = load_detector_training_config(config_path)
        if Path(member_config.dataset_root).resolve(strict=False) != dataset_root:
            raise DataValidationError(
                f"ensemble member {index} dataset_root does not match ensemble config"
            )
        yolo_report = validate_yolo_dataset(dataset_root, member_config.yolo_run_name)
        prepared.append(
            _PreparedMember(
                declared,
                config_path,
                checkpoint_path,
                member_config,
                yolo_report.output_dir,
                _validation_signature(yolo_report.manifest_path),
            )
        )
    return tuple(prepared)


def _inference_arguments(config: DetectorTrainingConfig) -> tuple[object, ...]:
    return (
        config.image_size,
        config.device,
        config.thresholds.confidence_floor,
        config.thresholds.operating_confidence,
        config.thresholds.nms_iou,
        config.thresholds.matching_iou,
        config.thresholds.max_symmetric_aspect_ratio,
    )


def _validate_member_compatibility(members: Sequence[_PreparedMember]) -> None:
    first = members[0]
    expected_arguments = _inference_arguments(first.config)
    expected_signature = first.validation_signature
    for index, member in enumerate(members[1:], start=1):
        if _inference_arguments(member.config) != expected_arguments:
            raise DataValidationError(
                f"ensemble member {index} inference arguments do not match"
            )
        if member.validation_signature != expected_signature:
            raise DataValidationError(
                f"ensemble member {index} validation samples do not match"
            )
    if first.config.thresholds.max_symmetric_aspect_ratio is None:
        raise DataValidationError("detector ensemble requires an aspect-ratio limit")


def _canonical_config(config: DetectorEnsembleConfig) -> dict[str, object]:
    return {
        "dataset_root": config.dataset_root,
        "output_root": config.output_root,
        "run_name": config.run_name,
        "members": [
            {
                "config_path": item.config_path,
                "config_sha256": item.config_sha256,
                "checkpoint_path": item.checkpoint_path,
                "checkpoint_sha256": item.checkpoint_sha256,
            }
            for item in config.members
        ],
        "cpu_threads": config.cpu_threads,
        "cpu_warmups": config.cpu_warmups,
        "cpu_repetitions": config.cpu_repetitions,
    }


def _prediction_payload(
    predictions: Mapping[str, Sequence[Detection]], members: Sequence[_PreparedMember]
) -> dict[str, object]:
    return {
        "prediction_version": 1,
        "split": "validation",
        "member_checkpoint_sha256": [
            member.declared.checkpoint_sha256 for member in members
        ],
        "images": [
            {
                "image_id": image_id,
                "detections": [
                    {
                        "bbox_xyxy": list(detection.bbox_xyxy),
                        "confidence": detection.confidence,
                        "class_index": detection.class_index,
                    }
                    for detection in detections
                ],
            }
            for image_id, detections in sorted(predictions.items())
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_detector_ensemble(
    config: DetectorEnsembleConfig,
    backend: DetectorBackend | None = None,
) -> DetectorEnsembleReport:
    if not isinstance(config, DetectorEnsembleConfig):
        raise DataValidationError("config must be DetectorEnsembleConfig")
    selected_backend = backend or UltralyticsBackend()
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    output_root = Path(config.output_root).resolve(strict=False)
    output_dir = output_root / config.run_name
    assert_training_paths_safe([output_root, output_dir], dataset_root)
    if output_dir.exists():
        raise DataValidationError(f"detector ensemble run already exists: {output_dir}")
    members = _prepare_members(config)
    _validate_member_compatibility(members)
    device = members[0].config.device
    if not selected_backend.cuda_available(device):
        raise DataValidationError(f"CUDA device {device} is unavailable")
    member_images = [_evaluation_inputs(dataset_root, item.yolo_dir) for item in members]
    images, image_paths = member_images[0]
    if any(item[0] != images for item in member_images[1:]):
        raise DataValidationError("ensemble evaluation truth records do not match")
    predictions = []
    for member, (_images, paths) in zip(members, member_images, strict=True):
        predictions.append(
            selected_backend.predict(
                checkpoint=member.checkpoint_path,
                image_paths=paths,
                confidence_floor=member.config.thresholds.confidence_floor,
                nms_iou=member.config.thresholds.nms_iou,
                image_size=member.config.image_size,
                device=member.config.device,
            )
        )
    thresholds = members[0].config.thresholds
    assert thresholds.max_symmetric_aspect_ratio is not None
    merged = merge_member_predictions(
        predictions,
        tuple(path.name for path in image_paths),
        DetectorPostprocessConfig(
            thresholds.confidence_floor,
            thresholds.nms_iou,
            thresholds.max_symmetric_aspect_ratio,
        ),
    )
    metrics = evaluate_detector_predictions(images, merged, thresholds)
    for index, member in enumerate(members):
        if _sha256(member.config_path) != member.declared.config_sha256:
            raise DataValidationError(
                f"ensemble member {index} config changed during inference"
            )
        if _sha256(member.checkpoint_path) != member.declared.checkpoint_sha256:
            raise DataValidationError(
                f"ensemble member {index} checkpoint changed during inference"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{config.run_name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        metadata = {
            "ensemble_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": "validation",
            "members": [
                {
                    "order": index,
                    "config_path": str(member.config_path),
                    "config_sha256": member.declared.config_sha256,
                    "checkpoint_path": str(member.checkpoint_path),
                    "checkpoint_sha256": member.declared.checkpoint_sha256,
                    "yolo_run_name": member.config.yolo_run_name,
                }
                for index, member in enumerate(members)
            ],
            "environment": _environment_metadata(),
        }
        _write_json(staging / "config.json", _canonical_config(config))
        _write_json(staging / "metadata.json", metadata)
        _write_json(staging / "predictions.json", _prediction_payload(merged, members))
        _write_json(
            staging / "metrics.json",
            {
                "metric_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split": "validation",
                "thresholds": {
                    "confidence_floor": thresholds.confidence_floor,
                    "operating_confidence": thresholds.operating_confidence,
                    "nms_iou": thresholds.nms_iou,
                    "matching_iou": thresholds.matching_iou,
                    "max_symmetric_aspect_ratio": (
                        thresholds.max_symmetric_aspect_ratio
                    ),
                },
                "metrics": metrics,
            },
        )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DetectorEnsembleReport(
        output_dir,
        output_dir / "metadata.json",
        output_dir / "predictions.json",
        output_dir / "metrics.json",
    )
