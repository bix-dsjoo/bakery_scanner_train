from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

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
_FOLD_TOKEN = re.compile(r"(?:^|_)val(\d{4})(?:_|$)")
_APPROVED_DEVELOPMENT_FOLDS = frozenset({"0503", "0509"})


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
    source_path: Path | None = None


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


class DetectorEnsembleCpuBackend(Protocol):
    def execution_provider(self) -> str: ...

    def prepare(self, checkpoints: Sequence[Path], threads: int) -> None: ...

    def predict(
        self,
        *,
        checkpoint: Path,
        image_path: Path,
        confidence_floor: float,
        nms_iou: float,
        image_size: int,
        device: str,
    ) -> Sequence[Detection]: ...


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
    source = Path(path).resolve(strict=False)
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
        source_path=source,
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


def _repository_context(config: DetectorEnsembleConfig) -> tuple[Path, Path]:
    if config.source_path is None:
        raise DataValidationError(
            "detector ensemble config must be loaded from a repository config file"
        )
    source = config.source_path.resolve(strict=False)
    repository_root = next(
        (
            parent
            for parent in source.parents
            if (parent / "pyproject.toml").is_file()
            and (parent / "datasets").is_dir()
        ),
        None,
    )
    if repository_root is None:
        raise DataValidationError(
            "detector ensemble config is not inside a project with datasets"
        )
    config_root = (repository_root / "configs").resolve(strict=False)
    try:
        source.relative_to(config_root)
    except ValueError as exc:
        raise DataValidationError(
            "detector ensemble config must be stored under project configs"
        ) from exc
    expected_dataset_root = (repository_root / "datasets").resolve(strict=False)
    configured_dataset_root = _resolve_project_path(
        config.dataset_root, repository_root
    )
    if configured_dataset_root != expected_dataset_root:
        raise DataValidationError(
            "dataset_root must be the canonical project datasets directory"
        )
    return repository_root, expected_dataset_root


def _assert_approved_development_fold(
    config: DetectorTrainingConfig, index: int
) -> None:
    values = (config.source_detector_run, config.yolo_run_name)
    matches = tuple(
        tuple(match.group(1) for match in _FOLD_TOKEN.finditer(value))
        for value in values
    )
    if (
        any(len(tokens) != 1 for tokens in matches)
        or matches[0][0] not in _APPROVED_DEVELOPMENT_FOLDS
        or matches[0] != matches[1]
    ):
        raise DataValidationError(
            f"ensemble member {index} source and YOLO run must use the same "
            "approved development fold; approved development folds are 0503/0509"
        )


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
                Path(sample["image_path"]).as_posix(),
                sample["image_sha256"],
                Path(sample["label_path"]).as_posix(),
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


def _prepare_members(
    config: DetectorEnsembleConfig, dataset_root: Path, project_root: Path
) -> tuple[_PreparedMember, ...]:
    resolved = tuple(
        (
            _resolve_project_path(member.config_path, project_root),
            _resolve_project_path(member.checkpoint_path, project_root),
        )
        for member in config.members
    )
    if len({item[0] for item in resolved}) != 2 or len(
        {item[1] for item in resolved}
    ) != 2:
        raise DataValidationError("resolved member paths must be unique")
    prepared = []
    for index, (declared, paths) in enumerate(zip(config.members, resolved, strict=True)):
        config_path, checkpoint_path = paths
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
        if _resolve_project_path(member_config.dataset_root, project_root) != dataset_root:
            raise DataValidationError(
                f"ensemble member {index} dataset_root does not match ensemble config"
            )
        _assert_approved_development_fold(member_config, index)
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


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object")
    return payload


def _signature_sha256(signature: tuple[tuple[object, ...], ...]) -> str:
    encoded = json.dumps(
        signature, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_members(members: Sequence[_PreparedMember]) -> list[dict[str, object]]:
    return [
        {
            "order": index,
            "config_path": str(member.config_path),
            "config_sha256": member.declared.config_sha256,
            "checkpoint_path": str(member.checkpoint_path),
            "checkpoint_sha256": member.declared.checkpoint_sha256,
            "yolo_run_name": member.config.yolo_run_name,
        }
        for index, member in enumerate(members)
    ]


def _validate_completed_evaluation_binding(
    output_dir: Path,
    config: DetectorEnsembleConfig,
    members: Sequence[_PreparedMember],
) -> None:
    required = {
        "config.json",
        "metadata.json",
        "predictions.json",
        "metrics.json",
    }
    if not output_dir.is_dir() or any(
        not (output_dir / name).is_file() for name in required
    ):
        raise DataValidationError("ensemble evaluation must complete before CPU benchmark")
    recorded_config = _json_object(output_dir / "config.json", "ensemble config")
    if recorded_config != _canonical_config(config):
        raise DataValidationError(
            "benchmark config does not match completed evaluation"
        )
    metadata = _json_object(output_dir / "metadata.json", "ensemble metadata")
    if (
        metadata.get("ensemble_version") != 1
        or metadata.get("split") != "validation"
        or metadata.get("members") != _metadata_members(members)
        or metadata.get("validation_signature_sha256")
        != _signature_sha256(members[0].validation_signature)
    ):
        raise DataValidationError(
            "benchmark metadata does not match completed evaluation"
        )
    predictions = _json_object(
        output_dir / "predictions.json", "ensemble predictions"
    )
    if predictions.get("split") != "validation" or predictions.get(
        "member_checkpoint_sha256"
    ) != [member.declared.checkpoint_sha256 for member in members]:
        raise DataValidationError(
            "benchmark predictions do not match completed evaluation"
        )
    metrics = _json_object(output_dir / "metrics.json", "ensemble metrics")
    if metrics.get("split") != "validation":
        raise DataValidationError("benchmark metrics do not match completed evaluation")


def evaluate_detector_ensemble(
    config: DetectorEnsembleConfig,
    backend: DetectorBackend | None = None,
) -> DetectorEnsembleReport:
    if not isinstance(config, DetectorEnsembleConfig):
        raise DataValidationError("config must be DetectorEnsembleConfig")
    selected_backend = backend or UltralyticsBackend()
    project_root, dataset_root = _repository_context(config)
    output_root = _resolve_project_path(config.output_root, project_root)
    output_dir = output_root / config.run_name
    assert_training_paths_safe([output_root, output_dir], dataset_root)
    if output_dir.exists():
        raise DataValidationError(f"detector ensemble run already exists: {output_dir}")
    members = _prepare_members(config, dataset_root, project_root)
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
            "members": _metadata_members(members),
            "validation_signature_sha256": _signature_sha256(
                members[0].validation_signature
            ),
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


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise DataValidationError("timing samples must not be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timing_statistics(values: Sequence[float]) -> dict[str, float | int]:
    samples = tuple(float(value) for value in values)
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise DataValidationError("timing samples must be finite non-negative values")
    return {
        "count": len(samples),
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": _percentile(samples, 0.5),
        "p95_ms": _percentile(samples, 0.95),
    }


class UltralyticsEnsembleCpuBackend:
    def __init__(self) -> None:
        self._models: dict[Path, object] = {}

    def execution_provider(self) -> str:
        return "torch-cpu"

    def prepare(self, checkpoints: Sequence[Path], threads: int) -> None:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)
        for checkpoint in checkpoints:
            model = YOLO(str(checkpoint))
            names = model.names
            if tuple(names[index] for index in sorted(names)) != ("bread",):
                raise DataValidationError("ensemble detector must have one bread class")
            self._models[checkpoint] = model

    def predict(
        self,
        *,
        checkpoint: Path,
        image_path: Path,
        confidence_floor: float,
        nms_iou: float,
        image_size: int,
        device: str,
    ) -> Sequence[Detection]:
        if device != "cpu":
            raise DataValidationError("ensemble CPU backend requires device='cpu'")
        model = self._models.get(checkpoint)
        if model is None:
            raise DataValidationError("ensemble CPU backend was not prepared")
        results = model.predict(
            source=str(image_path),
            conf=confidence_floor,
            iou=nms_iou,
            imgsz=image_size,
            device="cpu",
            verbose=False,
            stream=False,
        )
        if len(results) != 1:
            raise DataValidationError("ensemble CPU backend returned wrong result count")
        boxes = results[0].boxes
        return tuple(
            Detection(
                tuple(float(value) for value in xyxy.tolist()),
                float(confidence),
                int(class_index),
            )
            for xyxy, confidence, class_index in zip(
                boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu(), strict=True
            )
        )


def benchmark_detector_ensemble_cpu(
    config: DetectorEnsembleConfig,
    backend: DetectorEnsembleCpuBackend | None = None,
) -> Path:
    if not isinstance(config, DetectorEnsembleConfig):
        raise DataValidationError("config must be DetectorEnsembleConfig")
    selected_backend = backend or UltralyticsEnsembleCpuBackend()
    if selected_backend.execution_provider() != "torch-cpu":
        raise DataValidationError("ensemble benchmark requires the torch CPU provider")
    project_root, dataset_root = _repository_context(config)
    output_dir = _resolve_project_path(config.output_root, project_root) / config.run_name
    benchmark_path = output_dir / "benchmark.json"
    assert_training_paths_safe([output_dir, benchmark_path], dataset_root)
    if benchmark_path.exists():
        raise DataValidationError(f"ensemble benchmark already exists: {benchmark_path}")
    members = _prepare_members(config, dataset_root, project_root)
    _validate_member_compatibility(members)
    _validate_completed_evaluation_binding(output_dir, config, members)
    member_inputs = [_evaluation_inputs(dataset_root, member.yolo_dir) for member in members]
    images, first_paths = member_inputs[0]
    if any(item[0] != images for item in member_inputs[1:]):
        raise DataValidationError("ensemble benchmark truth records do not match")
    selected_backend.prepare(
        tuple(member.checkpoint_path for member in members), config.cpu_threads
    )
    thresholds = members[0].config.thresholds
    assert thresholds.max_symmetric_aspect_ratio is not None
    postprocess = DetectorPostprocessConfig(
        thresholds.confidence_floor,
        thresholds.nms_iou,
        thresholds.max_symmetric_aspect_ratio,
    )

    def invoke(image_index: int) -> None:
        image_id = first_paths[image_index].name
        predictions = []
        for member, (_records, paths) in zip(members, member_inputs, strict=True):
            predictions.append(
                {
                    image_id: tuple(
                        selected_backend.predict(
                            checkpoint=member.checkpoint_path,
                            image_path=paths[image_index],
                            confidence_floor=thresholds.confidence_floor,
                            nms_iou=thresholds.nms_iou,
                            image_size=member.config.image_size,
                            device="cpu",
                        )
                    )
                }
            )
        merge_member_predictions(predictions, (image_id,), postprocess)

    for _warmup in range(config.cpu_warmups):
        for image_index in range(len(first_paths)):
            invoke(image_index)
    samples = []
    durations = []
    for repetition in range(config.cpu_repetitions):
        for image_index, image_path in enumerate(first_paths):
            started = time.perf_counter()
            invoke(image_index)
            duration_ms = (time.perf_counter() - started) * 1000.0
            durations.append(duration_ms)
            samples.append(
                {
                    "repetition": repetition,
                    "image_id": image_path.name,
                    "duration_ms": duration_ms,
                }
            )
    for index, member in enumerate(members):
        if _sha256(member.config_path) != member.declared.config_sha256:
            raise DataValidationError(
                f"ensemble member {index} config changed during CPU benchmark"
            )
        if _sha256(member.checkpoint_path) != member.declared.checkpoint_sha256:
            raise DataValidationError(
                f"ensemble member {index} checkpoint changed during CPU benchmark"
            )
    payload = {
        "benchmark_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "context": {
            "device": "cpu",
            "execution_provider": selected_backend.execution_provider(),
            "threads": config.cpu_threads,
            "image_size": members[0].config.image_size,
            "image_count": len(first_paths),
            "member_count": len(members),
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "environment": _environment_metadata(),
            "claim": (
                "Current development-PC CPU measurement only; "
                "not a specific POS-device claim."
            ),
        },
        "members": [
            {
                "order": index,
                "config_sha256": member.declared.config_sha256,
                "checkpoint_sha256": member.declared.checkpoint_sha256,
            }
            for index, member in enumerate(members)
        ],
        "warmup_iterations": config.cpu_warmups,
        "warmup_invocation_count": config.cpu_warmups * len(first_paths),
        "repetitions": config.cpu_repetitions,
        "timing": _timing_statistics(durations),
        "samples": samples,
    }
    _validate_completed_evaluation_binding(output_dir, config, members)
    temporary = output_dir / f"benchmark.json.tmp-{uuid.uuid4().hex}"
    try:
        _write_json(temporary, payload)
        temporary.replace(benchmark_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return benchmark_path
