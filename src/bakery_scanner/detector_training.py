from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from .detector_evaluation import (
    Detection,
    EvaluationImage,
    EvaluationObject,
    EvaluationThresholds,
    evaluate_detector_predictions,
)
from .detector_postprocess import DetectorPostprocessConfig, filter_detections
from .errors import DataValidationError
from .registry import load_class_registry
from .safety import assert_training_paths_safe
from .yolo_dataset import build_yolo_dataset, validate_yolo_dataset

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_FIELDS = {
    "dataset_root",
    "source_detector_run",
    "yolo_run_name",
    "output_root",
    "run_name",
    "model",
    "image_size",
    "epochs",
    "batch_size",
    "seed",
    "device",
    "patience",
    "workers",
    "thresholds",
}
_THRESHOLD_FIELDS = {
    "confidence_floor",
    "operating_confidence",
    "nms_iou",
    "matching_iou",
}


@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    dataset_root: str
    source_detector_run: str
    yolo_run_name: str
    output_root: str
    run_name: str
    model: str
    image_size: int
    epochs: int
    batch_size: int
    seed: int
    device: str
    patience: int
    workers: int
    thresholds: EvaluationThresholds


@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    best_checkpoint: Path
    last_checkpoint: Path
    class_names: tuple[str, ...]
    pretrained_checkpoint: Path


class DetectorBackend(Protocol):
    def cuda_available(self, device: str) -> bool: ...

    def train(
        self,
        *,
        model: str,
        data_yaml: Path,
        output_dir: Path,
        arguments: Mapping[str, object],
    ) -> BackendTrainingResult: ...

    def predict(
        self,
        *,
        checkpoint: Path,
        image_paths: Sequence[Path],
        confidence_floor: float,
        nms_iou: float,
        image_size: int,
        device: str,
    ) -> Mapping[str, Sequence[Detection]]: ...


@dataclass(frozen=True, slots=True)
class DetectorTrainingReport:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path
    split: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        metrics_payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        return {
            "status": "ok",
            "split": self.split,
            "output_dir": str(self.output_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint),
            "metadata_path": str(self.metadata_path),
            "predictions_path": str(self.predictions_path),
            "metrics_path": str(self.metrics_path),
            "metrics": metrics_payload["metrics"],
        }


@dataclass(frozen=True, slots=True)
class DetectorEvaluationReport:
    output_dir: Path
    checkpoint: Path
    predictions_path: Path
    metrics_path: Path
    split: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        metrics_payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        return {
            "status": "ok",
            "split": self.split,
            "output_dir": str(self.output_dir),
            "checkpoint": str(self.checkpoint),
            "predictions_path": str(self.predictions_path),
            "metrics_path": str(self.metrics_path),
            "metrics": metrics_payload["metrics"],
        }


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


def _integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if (value < 0) if allow_zero else (value <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DataValidationError(f"{label} must be a finite number")
    return float(value)


def _safe_run_name(value: object, label: str) -> str:
    name = _text(value, label)
    if not _RUN_NAME.fullmatch(name):
        raise DataValidationError(f"{label} is invalid: {name!r}")
    return name


def load_detector_training_config(path: str | Path) -> DetectorTrainingConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot load detector config {config_path}: {exc}") from exc
    payload = _strict_object(payload, _CONFIG_FIELDS, "detector config")
    threshold_payload = payload["thresholds"]
    actual_threshold_fields = (
        set(threshold_payload) if isinstance(threshold_payload, dict) else set()
    )
    allowed_threshold_fields = _THRESHOLD_FIELDS | {
        "max_symmetric_aspect_ratio"
    }
    if (
        not isinstance(threshold_payload, dict)
        or not _THRESHOLD_FIELDS <= actual_threshold_fields
        or not actual_threshold_fields <= allowed_threshold_fields
    ):
        actual = (
            sorted(threshold_payload)
            if isinstance(threshold_payload, dict)
            else type(threshold_payload).__name__
        )
        raise DataValidationError(
            "detector thresholds fields are invalid: "
            f"required={sorted(_THRESHOLD_FIELDS)}, actual={actual}"
        )
    thresholds = EvaluationThresholds(
        confidence_floor=_number(
            threshold_payload["confidence_floor"], "confidence_floor"
        ),
        operating_confidence=_number(
            threshold_payload["operating_confidence"], "operating_confidence"
        ),
        nms_iou=_number(threshold_payload["nms_iou"], "nms_iou"),
        matching_iou=_number(threshold_payload["matching_iou"], "matching_iou"),
        max_symmetric_aspect_ratio=(
            _number(
                threshold_payload["max_symmetric_aspect_ratio"],
                "max_symmetric_aspect_ratio",
            )
            if "max_symmetric_aspect_ratio" in threshold_payload
            else None
        ),
    )
    if not 0 < thresholds.confidence_floor <= thresholds.operating_confidence < 1:
        raise DataValidationError("confidence thresholds are invalid")
    if not 0 < thresholds.nms_iou <= 1 or not 0 < thresholds.matching_iou <= 1:
        raise DataValidationError("IoU thresholds are invalid")
    device = _text(payload["device"], "device")
    if device != "0":
        raise DataValidationError("device must be CUDA device '0' for this baseline")
    return DetectorTrainingConfig(
        dataset_root=_text(payload["dataset_root"], "dataset_root"),
        source_detector_run=_safe_run_name(
            payload["source_detector_run"], "source_detector_run"
        ),
        yolo_run_name=_safe_run_name(payload["yolo_run_name"], "yolo_run_name"),
        output_root=_text(payload["output_root"], "output_root"),
        run_name=_safe_run_name(payload["run_name"], "run_name"),
        model=_text(payload["model"], "model"),
        image_size=_integer(payload["image_size"], "image_size"),
        epochs=_integer(payload["epochs"], "epochs"),
        batch_size=_integer(payload["batch_size"], "batch_size"),
        seed=_integer(payload["seed"], "seed", allow_zero=True),
        device=device,
        patience=_integer(payload["patience"], "patience", allow_zero=True),
        workers=_integer(payload["workers"], "workers", allow_zero=True),
        thresholds=thresholds,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_payload(config: DetectorTrainingConfig) -> dict[str, Any]:
    payload = {
        "dataset_root": config.dataset_root,
        "source_detector_run": config.source_detector_run,
        "yolo_run_name": config.yolo_run_name,
        "output_root": config.output_root,
        "run_name": config.run_name,
        "model": config.model,
        "image_size": config.image_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "device": config.device,
        "patience": config.patience,
        "workers": config.workers,
        "thresholds": {
            "confidence_floor": config.thresholds.confidence_floor,
            "operating_confidence": config.thresholds.operating_confidence,
            "nms_iou": config.thresholds.nms_iou,
            "matching_iou": config.thresholds.matching_iou,
        },
    }
    if config.thresholds.max_symmetric_aspect_ratio is not None:
        payload["thresholds"]["max_symmetric_aspect_ratio"] = (
            config.thresholds.max_symmetric_aspect_ratio
        )
    return payload


def _environment_metadata() -> dict[str, Any]:
    dependency_names = ("Pillow", "PyYAML", "torch", "torchvision", "ultralytics")
    dependencies: dict[str, str | None] = {}
    for name in dependency_names:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    gpu: str | None = None
    cuda: str | None = None
    try:
        import torch

        cuda = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "cuda": cuda,
        "dependencies": dependencies,
    }


def _difficulty(image_name: str) -> str | None:
    match = re.search(r"scene_([emh])_", image_name)
    return {"e": "easy", "m": "medium", "h": "hard"}.get(
        match.group(1) if match else ""
    )


def _evaluation_inputs(
    dataset_root: Path, yolo_dir: Path
) -> tuple[tuple[EvaluationImage, ...], tuple[Path, ...]]:
    manifest = json.loads((yolo_dir / "manifest.json").read_text(encoding="utf-8"))
    registry = load_class_registry(dataset_root / "class_registry.json")
    images: list[EvaluationImage] = []
    image_paths: list[Path] = []
    for sample in manifest["samples"]:
        if sample["split"] != "validation":
            continue
        image_path = yolo_dir / sample["image_path"]
        image_id = image_path.name
        objects: list[EvaluationObject] = []
        for annotation in sample["original_annotations"]:
            category_id = annotation["category_id"]
            record = registry.by_category_id.get(category_id)
            if record is None:
                raise DataValidationError(
                    f"validation annotation uses unknown category_id: {category_id}"
                )
            x, y, width, height = annotation["bbox"]
            objects.append(
                EvaluationObject(
                    (float(x), float(y), float(x + width), float(y + height)),
                    category_id,
                    record.phase,
                )
            )
        images.append(EvaluationImage(image_id, _difficulty(image_id), tuple(objects)))
        image_paths.append(image_path)
    if not images:
        raise DataValidationError("YOLO validation split must contain at least one image")
    return tuple(images), tuple(image_paths)


def _prediction_payload(
    predictions: Mapping[str, Sequence[Detection]], checkpoint_sha256: str
) -> dict[str, Any]:
    return {
        "prediction_version": 1,
        "split": "validation",
        "checkpoint_sha256": checkpoint_sha256,
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


def _write_evaluation(
    *,
    dataset_root: Path,
    yolo_dir: Path,
    checkpoint: Path,
    backend: DetectorBackend,
    config: DetectorTrainingConfig,
    output_dir: Path,
    environment: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    images, image_paths = _evaluation_inputs(dataset_root, yolo_dir)
    predictions = backend.predict(
        checkpoint=checkpoint,
        image_paths=image_paths,
        confidence_floor=config.thresholds.confidence_floor,
        nms_iou=config.thresholds.nms_iou,
        image_size=config.image_size,
        device=config.device,
    )
    if config.thresholds.max_symmetric_aspect_ratio is not None:
        postprocess = DetectorPostprocessConfig(
            confidence_threshold=config.thresholds.confidence_floor,
            nms_iou=config.thresholds.nms_iou,
            max_symmetric_aspect_ratio=(
                config.thresholds.max_symmetric_aspect_ratio
            ),
        )
        predictions = {
            image_id: filter_detections(detections, postprocess)
            for image_id, detections in predictions.items()
        }
    metrics = evaluate_detector_predictions(images, predictions, config.thresholds)
    checkpoint_hash = _sha256(checkpoint)
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            _prediction_payload(predictions, checkpoint_hash),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_payload = {
        "metric_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "checkpoint_sha256": checkpoint_hash,
        "thresholds": _config_payload(config)["thresholds"],
        "configuration": _config_payload(config),
        "environment": environment,
        "metrics": metrics,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return predictions_path, metrics_path, metrics


def _validate_completed_run(output_dir: Path) -> None:
    required = {
        "config.yaml",
        "metadata.json",
        "predictions.json",
        "metrics.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    }
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and not path.relative_to(output_dir).as_posix().startswith("backend/")
    }
    if not required.issubset(actual):
        raise DataValidationError(
            f"completed detector run is missing files: {sorted(required - actual)}"
        )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    for section in ("dataset", "model", "environment", "backend_arguments"):
        if section not in metadata:
            raise DataValidationError(f"detector metadata is missing {section}")


def train_detector(
    config: DetectorTrainingConfig,
    backend: DetectorBackend | None = None,
) -> DetectorTrainingReport:
    if not isinstance(config, DetectorTrainingConfig):
        raise DataValidationError("config must be DetectorTrainingConfig")
    selected_backend = backend or UltralyticsBackend()
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    source_dir = dataset_root / "derived" / "detector" / config.source_detector_run
    output_root = Path(config.output_root).resolve(strict=False)
    assert_training_paths_safe(
        [source_dir, output_root, config.model], dataset_root
    )
    yolo_dir = dataset_root / "derived" / "yolo" / config.yolo_run_name
    if yolo_dir.exists():
        yolo_report = validate_yolo_dataset(dataset_root, config.yolo_run_name)
    else:
        yolo_report = build_yolo_dataset(
            dataset_root, config.source_detector_run, config.yolo_run_name
        )
    assert_training_paths_safe([yolo_report.output_dir], dataset_root)
    if not selected_backend.cuda_available(config.device):
        raise DataValidationError(f"CUDA device {config.device} is unavailable")

    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise DataValidationError(f"detector training run already exists: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".{config.run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        arguments: dict[str, object] = {
            "image_size": config.image_size,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "device": config.device,
            "patience": config.patience,
            "workers": config.workers,
        }
        backend_result = selected_backend.train(
            model=config.model,
            data_yaml=yolo_report.output_dir / "data.yaml",
            output_dir=staging_dir / "backend",
            arguments=arguments,
        )
        if backend_result.class_names != ("bread",):
            raise DataValidationError(
                f"detector output classes must be ('bread',): {backend_result.class_names}"
            )
        for checkpoint in (
            backend_result.pretrained_checkpoint,
            backend_result.best_checkpoint,
            backend_result.last_checkpoint,
        ):
            if not checkpoint.is_file():
                raise DataValidationError(f"detector checkpoint is missing: {checkpoint}")
        checkpoints_dir = staging_dir / "checkpoints"
        checkpoints_dir.mkdir()
        best_checkpoint = checkpoints_dir / "best.pt"
        last_checkpoint = checkpoints_dir / "last.pt"
        shutil.copy2(backend_result.best_checkpoint, best_checkpoint)
        shutil.copy2(backend_result.last_checkpoint, last_checkpoint)
        environment = _environment_metadata()
        predictions_path, metrics_path, _ = _write_evaluation(
            dataset_root=dataset_root,
            yolo_dir=yolo_report.output_dir,
            checkpoint=best_checkpoint,
            backend=selected_backend,
            config=config,
            output_dir=staging_dir,
            environment=environment,
        )
        config_path = staging_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(_config_payload(config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        yolo_manifest = yolo_report.manifest_path
        metadata = {
            "metadata_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": "validation",
            "dataset": {
                "manifest_path": str(yolo_manifest),
                "manifest_sha256": _sha256(yolo_manifest),
                "train_path": str(yolo_report.output_dir / "train" / "images"),
                "validation_path": str(
                    yolo_report.output_dir / "validation" / "images"
                ),
            },
            "model": {
                "pretrained_path": str(backend_result.pretrained_checkpoint),
                "pretrained_sha256": _sha256(backend_result.pretrained_checkpoint),
                "best_sha256": _sha256(best_checkpoint),
                "last_sha256": _sha256(last_checkpoint),
                "class_names": list(backend_result.class_names),
            },
            "environment": environment,
            "backend_arguments": arguments,
        }
        metadata_path = staging_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _validate_completed_run(staging_dir)
        staging_dir.rename(output_dir)
        return DetectorTrainingReport(
            output_dir=output_dir,
            best_checkpoint=output_dir / "checkpoints" / "best.pt",
            last_checkpoint=output_dir / "checkpoints" / "last.pt",
            metadata_path=output_dir / metadata_path.name,
            predictions_path=output_dir / predictions_path.name,
            metrics_path=output_dir / metrics_path.name,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def evaluate_detector_checkpoint(
    config: DetectorTrainingConfig,
    checkpoint: str | Path,
    backend: DetectorBackend | None = None,
    output_dir: str | Path | None = None,
) -> DetectorEvaluationReport:
    if not isinstance(config, DetectorTrainingConfig):
        raise DataValidationError("config must be DetectorTrainingConfig")
    selected_backend = backend or UltralyticsBackend()
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    yolo_report = validate_yolo_dataset(dataset_root, config.yolo_run_name)
    checkpoint_path = Path(checkpoint).resolve(strict=False)
    target_dir = (
        Path(output_dir).resolve(strict=False)
        if output_dir is not None
        else checkpoint_path.parent.parent / "evaluation"
    )
    assert_training_paths_safe(
        [yolo_report.output_dir, checkpoint_path, target_dir], dataset_root
    )
    if not checkpoint_path.is_file():
        raise DataValidationError(f"detector checkpoint is missing: {checkpoint_path}")
    if not selected_backend.cuda_available(config.device):
        raise DataValidationError(f"CUDA device {config.device} is unavailable")
    if target_dir.exists():
        raise DataValidationError(f"detector evaluation output already exists: {target_dir}")
    target_dir.mkdir(parents=True)
    try:
        predictions_path, metrics_path, _ = _write_evaluation(
            dataset_root=dataset_root,
            yolo_dir=yolo_report.output_dir,
            checkpoint=checkpoint_path,
            backend=selected_backend,
            config=config,
            output_dir=target_dir,
            environment=_environment_metadata(),
        )
        return DetectorEvaluationReport(
            output_dir=target_dir,
            checkpoint=checkpoint_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
        )
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise


class UltralyticsBackend:
    def cuda_available(self, device: str) -> bool:
        import torch

        return device == "0" and torch.cuda.is_available()

    def train(
        self,
        *,
        model: str,
        data_yaml: Path,
        output_dir: Path,
        arguments: Mapping[str, object],
    ) -> BackendTrainingResult:
        from ultralytics import YOLO

        detector = YOLO(model)
        pretrained_checkpoint = Path(detector.ckpt_path or model).resolve(strict=False)
        detector.train(
            data=str(data_yaml),
            imgsz=arguments["image_size"],
            epochs=arguments["epochs"],
            batch=arguments["batch_size"],
            seed=arguments["seed"],
            device=arguments["device"],
            patience=arguments["patience"],
            workers=arguments["workers"],
            deterministic=True,
            project=str(output_dir),
            name="backend",
            exist_ok=False,
        )
        if detector.trainer is None:
            raise DataValidationError("Ultralytics training did not create a trainer")
        names = detector.names
        class_names = tuple(names[index] for index in sorted(names))
        return BackendTrainingResult(
            Path(detector.trainer.best),
            Path(detector.trainer.last),
            class_names,
            pretrained_checkpoint,
        )

    def predict(
        self,
        *,
        checkpoint: Path,
        image_paths: Sequence[Path],
        confidence_floor: float,
        nms_iou: float,
        image_size: int,
        device: str,
    ) -> Mapping[str, Sequence[Detection]]:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        results = model.predict(
            source=[str(path) for path in image_paths],
            conf=confidence_floor,
            iou=nms_iou,
            imgsz=image_size,
            device=device,
            verbose=False,
            stream=False,
        )
        predictions: dict[str, tuple[Detection, ...]] = {}
        for path, result in zip(image_paths, results, strict=True):
            boxes = result.boxes
            predictions[path.name] = tuple(
                Detection(
                    tuple(float(value) for value in xyxy.tolist()),
                    float(confidence),
                    int(class_index),
                )
                for xyxy, confidence, class_index in zip(
                    boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu(), strict=True
                )
            )
        return predictions
