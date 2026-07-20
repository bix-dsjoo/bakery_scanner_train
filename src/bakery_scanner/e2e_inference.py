from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from .classifier_dataset import validate_classifier_dataset
from .classifier_training import (
    _checkpoint_context,
    _classifier_transforms,
    _environment_metadata,
    _load_classifier_checkpoint_model,
    _torch_device,
    load_classifier_training_config,
)
from .detector_dataset import validate_detector_dataset
from .detector_training import load_detector_training_config
from .e2e_evaluation import (
    EndToEndImage,
    EndToEndPrediction,
    EndToEndTruth,
    evaluate_end_to_end_predictions,
)
from .errors import DataValidationError
from .registry import load_class_registry
from .safety import assert_training_paths_safe
from .yolo_dataset import validate_yolo_dataset

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_FIELD_ORDER = (
    "dataset_root",
    "detector_config",
    "classifier_config",
    "detector_checkpoint",
    "classifier_checkpoint",
    "output_root",
    "run_name",
)
_CONFIG_FIELDS = set(_CONFIG_FIELD_ORDER)


@dataclass(frozen=True, slots=True)
class EndToEndConfig:
    dataset_root: str
    detector_config: str
    classifier_config: str
    detector_checkpoint: str
    classifier_checkpoint: str
    output_root: str
    run_name: str


@dataclass(frozen=True, slots=True)
class BackendInferenceResult:
    predictions: Mapping[str, Sequence[EndToEndPrediction]]
    classifier_batch_sizes: Mapping[str, int]


class EndToEndBackend(Protocol):
    def cuda_available(self, device: str) -> bool: ...

    def predict(
        self,
        *,
        image_paths: Sequence[Path],
        images: Sequence[EndToEndImage],
        detector_checkpoint: Path,
        classifier_checkpoint: Path,
        classifier_context: Mapping[str, Any],
        output_dimension: int,
        arguments: Mapping[str, object],
    ) -> BackendInferenceResult: ...


@dataclass(frozen=True, slots=True)
class EndToEndReport:
    output_dir: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path
    split: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        metrics = _json_object(self.metrics_path, "end-to-end metrics")
        return {
            "status": "ok",
            "split": self.split,
            "output_dir": str(self.output_dir),
            "metadata_path": str(self.metadata_path),
            "predictions_path": str(self.predictions_path),
            "metrics_path": str(self.metrics_path),
            "metrics": metrics["metrics"],
        }


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string")
    return value


def _run_name(value: object) -> str:
    result = _text(value, "run_name")
    if not _RUN_NAME.fullmatch(result):
        raise DataValidationError(f"run_name is invalid: {result!r}")
    return result


def load_end_to_end_config(path: str | Path) -> EndToEndConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot load end-to-end config {config_path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise DataValidationError(
            f"end-to-end config fields are invalid: expected={sorted(_CONFIG_FIELDS)}, actual={actual}"
        )
    return EndToEndConfig(
        dataset_root=_text(payload["dataset_root"], "dataset_root"),
        detector_config=_text(payload["detector_config"], "detector_config"),
        classifier_config=_text(payload["classifier_config"], "classifier_config"),
        detector_checkpoint=_text(
            payload["detector_checkpoint"], "detector_checkpoint"
        ),
        classifier_checkpoint=_text(
            payload["classifier_checkpoint"], "classifier_checkpoint"
        ),
        output_root=_text(payload["output_root"], "output_root"),
        run_name=_run_name(payload["run_name"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object")
    return payload


def _config_payload(config: EndToEndConfig) -> dict[str, str]:
    return {field: getattr(config, field) for field in _CONFIG_FIELD_ORDER}


def _checkpoint_metadata(checkpoint: Path, label: str) -> tuple[Path, dict[str, Any]]:
    metadata_path = checkpoint.parent.parent / "metadata.json"
    payload = _json_object(metadata_path, f"{label} metadata")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise DataValidationError(f"{label} metadata model must be an object")
    expected_hash = model.get("best_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise DataValidationError(f"{label} metadata best SHA-256 is invalid")
    actual_hash = _sha256(checkpoint)
    if actual_hash != expected_hash:
        raise DataValidationError(f"{label} checkpoint SHA-256 does not match metadata")
    return metadata_path, payload


def _detector_training_arguments(config) -> dict[str, object]:
    return {
        "image_size": config.image_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "device": config.device,
        "patience": config.patience,
        "workers": config.workers,
    }


def _validate_detector_checkpoint_provenance(
    selected_config,
    run_config,
    checkpoint: Path,
    metadata: Mapping[str, Any],
    yolo_manifest_path: Path,
) -> None:
    if replace(run_config, model=selected_config.model) != selected_config:
        raise DataValidationError(
            "detector run config does not match the selected detector config"
        )
    if metadata.get("backend_arguments") != _detector_training_arguments(
        selected_config
    ):
        raise DataValidationError(
            "detector metadata backend arguments do not match the selected config"
        )
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        raise DataValidationError("detector metadata dataset must be an object")
    manifest_path = dataset.get("manifest_path")
    manifest_sha256 = dataset.get("manifest_sha256")
    if not isinstance(manifest_path, str) or not isinstance(manifest_sha256, str):
        raise DataValidationError(
            "detector metadata dataset manifest does not match the selected YOLO run"
        )
    if Path(manifest_path).resolve(strict=False) != yolo_manifest_path.resolve(
        strict=False
    ):
        raise DataValidationError(
            "detector metadata dataset manifest path does not match the validated YOLO run"
        )
    if not yolo_manifest_path.is_file() or _sha256(yolo_manifest_path) != manifest_sha256:
        raise DataValidationError(
            "detector metadata dataset manifest SHA-256 does not match the validated YOLO run"
        )
    model = metadata.get("model")
    if not isinstance(model, dict) or model.get("class_names") != ["bread"]:
        raise DataValidationError("detector metadata must declare one bread class")
    pretrained_sha256 = model.get("pretrained_sha256")
    selected_model = Path(selected_config.model).resolve(strict=False)
    if (
        not isinstance(pretrained_sha256, str)
        or not selected_model.is_file()
        or _sha256(selected_model) != pretrained_sha256
    ):
        raise DataValidationError(
            "detector pretrained model SHA-256 does not match the selected config"
        )
    expected_checkpoint = (
        Path(selected_config.output_root)
        / selected_config.run_name
        / "checkpoints"
        / "best.pt"
    ).resolve(strict=False)
    if checkpoint != expected_checkpoint:
        raise DataValidationError(
            "detector checkpoint path does not match the selected detector run"
        )


def _validate_yolo_source_binding(
    yolo_manifest_path: Path,
    expected_source_run: str,
    detector_manifest_path: Path,
) -> None:
    manifest = _json_object(yolo_manifest_path, "YOLO manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise DataValidationError("YOLO source must be an object")
    recorded_path = source.get("manifest_path")
    if (
        source.get("run_name") != expected_source_run
        or not isinstance(recorded_path, str)
        or Path(recorded_path).resolve(strict=False)
        != detector_manifest_path.resolve(strict=False)
        or source.get("manifest_sha256") != _sha256(detector_manifest_path)
    ):
        raise DataValidationError(
            "YOLO source does not match the selected detector dataset"
        )


def _load_validation_images(
    dataset_root: Path, detector_dir: Path, manifest_path: Path
) -> tuple[tuple[EndToEndImage, ...], tuple[Path, ...]]:
    manifest = _json_object(manifest_path, "detector manifest")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list):
        raise DataValidationError("detector manifest samples must be a list")
    registry = load_class_registry(dataset_root / "class_registry.json")
    images: list[EndToEndImage] = []
    image_paths: list[Path] = []
    for position, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise DataValidationError(f"detector sample {position} must be an object")
        if raw.get("split") != "validation":
            continue
        image_path_value = raw.get("output_path")
        image_id = raw.get("sample_id")
        annotations = raw.get("original_annotations")
        if not isinstance(image_path_value, str) or not image_path_value:
            raise DataValidationError("detector validation output_path is invalid")
        if not isinstance(image_id, str) or not image_id:
            raise DataValidationError("detector validation sample_id is invalid")
        if not isinstance(annotations, list):
            raise DataValidationError("detector validation annotations must be a list")
        image_path = (detector_dir / image_path_value).resolve(strict=False)
        if not image_path.is_file():
            raise DataValidationError(f"detector validation image is missing: {image_path}")
        truths = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise DataValidationError("detector validation annotation must be an object")
            category_id = annotation.get("category_id")
            record = registry.by_category_id.get(category_id)
            if record is None:
                raise DataValidationError(
                    f"detector validation annotation category is unknown: {category_id}"
                )
            bbox = annotation.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(value, (int, float)) for value in bbox)
            ):
                raise DataValidationError("detector validation bbox is invalid")
            x, y, width, height = (float(value) for value in bbox)
            truths.append(
                EndToEndTruth((x, y, x + width, y + height), record.model_index)
            )
        images.append(EndToEndImage(image_id, tuple(truths)))
        image_paths.append(image_path)
    if not images:
        raise DataValidationError("detector validation split must contain images")
    return tuple(images), tuple(image_paths)


def _backend_arguments(detector_config, classifier_config) -> dict[str, object]:
    return {
        "device": detector_config.device,
        "detector_image_size": detector_config.image_size,
        "detector_confidence": detector_config.thresholds.confidence_floor,
        "detector_operating_confidence": (
            detector_config.thresholds.operating_confidence
        ),
        "detector_nms_iou": detector_config.thresholds.nms_iou,
        "classifier_image_size": classifier_config.image_size,
    }


def _validate_completed_run(output_dir: Path) -> None:
    expected = {"config.yaml", "metadata.json", "metrics.json", "predictions.json"}
    actual = {path.name for path in output_dir.iterdir()}
    if actual != expected:
        raise DataValidationError(
            f"completed end-to-end files are invalid: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for filename in ("metadata.json", "metrics.json", "predictions.json"):
        _json_object(output_dir / filename, filename)


def evaluate_end_to_end(
    config: EndToEndConfig,
    backend: EndToEndBackend | None = None,
) -> EndToEndReport:
    if not isinstance(config, EndToEndConfig):
        raise DataValidationError("config must be EndToEndConfig")
    selected_backend = backend or TorchEndToEndBackend()
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    detector_config_path = Path(config.detector_config).resolve(strict=False)
    classifier_config_path = Path(config.classifier_config).resolve(strict=False)
    detector_checkpoint = Path(config.detector_checkpoint).resolve(strict=False)
    classifier_checkpoint = Path(config.classifier_checkpoint).resolve(strict=False)
    detector_metadata_path = detector_checkpoint.parent.parent / "metadata.json"
    detector_run_config_path = detector_checkpoint.parent.parent / "config.yaml"
    classifier_metadata_path = classifier_checkpoint.parent.parent / "metadata.json"
    output_root = Path(config.output_root).resolve(strict=False)
    registry_path = dataset_root / "class_registry.json"
    assert_training_paths_safe(
        [
            registry_path,
            detector_config_path,
            classifier_config_path,
            detector_checkpoint,
            classifier_checkpoint,
            detector_metadata_path,
            detector_run_config_path,
            classifier_metadata_path,
            output_root,
        ],
        dataset_root,
    )

    detector_config = load_detector_training_config(detector_config_path)
    detector_run_config = load_detector_training_config(detector_run_config_path)
    classifier_config = load_classifier_training_config(classifier_config_path)
    if Path(detector_config.dataset_root).resolve(strict=False) != dataset_root:
        raise DataValidationError("detector config dataset_root does not match end-to-end config")
    if Path(classifier_config.dataset_root).resolve(strict=False) != dataset_root:
        raise DataValidationError("classifier config dataset_root does not match end-to-end config")
    detector_model_path = Path(detector_config.model).resolve(strict=False)
    yolo_dir = (
        dataset_root / "derived" / "yolo" / detector_config.yolo_run_name
    ).resolve(strict=False)
    assert_training_paths_safe([detector_model_path, yolo_dir], dataset_root)
    if detector_config.device != classifier_config.device:
        raise DataValidationError("detector and classifier devices must match")
    if not selected_backend.cuda_available(detector_config.device):
        raise DataValidationError(f"CUDA device {detector_config.device} is unavailable")

    detector_report = validate_detector_dataset(
        dataset_root, detector_config.source_detector_run
    )
    yolo_report = validate_yolo_dataset(dataset_root, detector_config.yolo_run_name)
    _validate_yolo_source_binding(
        yolo_report.manifest_path,
        detector_config.source_detector_run,
        detector_report.manifest_path,
    )
    classifier_report = validate_classifier_dataset(
        dataset_root, classifier_config.source_classifier_run
    )
    if classifier_report.phase != "base" or classifier_report.output_dimension != 15:
        raise DataValidationError("Base end-to-end evaluation requires 15 classifier outputs")
    images, image_paths = _load_validation_images(
        dataset_root, detector_report.output_dir, detector_report.manifest_path
    )
    classifier_context = _checkpoint_context(
        classifier_config,
        dataset_root,
        classifier_report.manifest_path,
        classifier_report.output_dimension,
    )
    if not detector_checkpoint.is_file() or not classifier_checkpoint.is_file():
        raise DataValidationError("end-to-end checkpoint is missing")
    detector_metadata_path, detector_metadata = _checkpoint_metadata(
        detector_checkpoint, "detector"
    )
    _validate_detector_checkpoint_provenance(
        detector_config,
        detector_run_config,
        detector_checkpoint,
        detector_metadata,
        yolo_report.manifest_path,
    )
    classifier_metadata_path, _classifier_metadata = _checkpoint_metadata(
        classifier_checkpoint, "classifier"
    )
    detector_hash_before = _sha256(detector_checkpoint)
    classifier_hash_before = _sha256(classifier_checkpoint)
    detector_config_hash = _sha256(detector_config_path)
    classifier_config_hash = _sha256(classifier_config_path)
    arguments = _backend_arguments(detector_config, classifier_config)

    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise DataValidationError(f"end-to-end run already exists: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".{config.run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        result = selected_backend.predict(
            image_paths=image_paths,
            images=images,
            detector_checkpoint=detector_checkpoint,
            classifier_checkpoint=classifier_checkpoint,
            classifier_context=classifier_context,
            output_dimension=classifier_report.output_dimension,
            arguments=arguments,
        )
        expected_image_ids = {image.image_id for image in images}
        if set(result.classifier_batch_sizes) != expected_image_ids:
            raise DataValidationError(
                "classifier batch sizes must exactly match evaluation image IDs"
            )
        for image_id, batch_size in result.classifier_batch_sizes.items():
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size < 0
                or image_id not in result.predictions
                or batch_size != len(result.predictions[image_id])
            ):
                raise DataValidationError(
                    "classifier batch sizes must match per-image prediction counts"
                )
        detector_hash_after = _sha256(detector_checkpoint)
        classifier_hash_after = _sha256(classifier_checkpoint)
        if detector_hash_after != detector_hash_before:
            raise DataValidationError("detector checkpoint changed during frozen inference")
        if classifier_hash_after != classifier_hash_before:
            raise DataValidationError("classifier checkpoint changed during inference")
        metrics = evaluate_end_to_end_predictions(
            images,
            result.predictions,
            output_dimension=classifier_report.output_dimension,
            count_detector_confidence=(
                detector_config.thresholds.operating_confidence
            ),
        )
        predictions_path = staging_dir / "predictions.json"
        predictions_path.write_text(
            json.dumps(
                {
                    "prediction_version": 1,
                    "split": "validation",
                    "images": [
                        {
                            "image_id": image.image_id,
                            "predictions": [
                                prediction.to_dict()
                                for prediction in result.predictions[image.image_id]
                            ],
                        }
                        for image in images
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metrics_path = staging_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metric_version": 1,
                    "split": "validation",
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        config_path = staging_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(_config_payload(config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        metadata_path = staging_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "metadata_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "split": "validation",
                    "configuration": {
                        "detector_config_path": str(detector_config_path),
                        "detector_config_sha256": detector_config_hash,
                        "detector_run_config_path": str(detector_run_config_path),
                        "detector_run_config_sha256": _sha256(
                            detector_run_config_path
                        ),
                        "classifier_config_path": str(classifier_config_path),
                        "classifier_config_sha256": classifier_config_hash,
                    },
                    "dataset": {
                        "detector_manifest_path": str(detector_report.manifest_path),
                        "detector_manifest_sha256": _sha256(detector_report.manifest_path),
                        "detector_training_manifest_path": detector_metadata[
                            "dataset"
                        ]["manifest_path"],
                        "detector_training_manifest_sha256": detector_metadata[
                            "dataset"
                        ]["manifest_sha256"],
                        "classifier_manifest_path": str(classifier_report.manifest_path),
                        "classifier_manifest_sha256": _sha256(classifier_report.manifest_path),
                        "registry_sha256": classifier_context["registry_sha256"],
                        "model_index_mapping": classifier_context["model_index_mapping"],
                    },
                    "model": {
                        "detector_checkpoint": str(detector_checkpoint),
                        "detector_metadata": str(detector_metadata_path),
                        "detector_sha256_before": detector_hash_before,
                        "detector_sha256_after": detector_hash_after,
                        "detector_unchanged": detector_hash_before == detector_hash_after,
                        "classifier_checkpoint": str(classifier_checkpoint),
                        "classifier_metadata": str(classifier_metadata_path),
                        "classifier_sha256_before": classifier_hash_before,
                        "classifier_sha256_after": classifier_hash_after,
                        "classifier_unchanged": (
                            classifier_hash_before == classifier_hash_after
                        ),
                    },
                    "inference": {
                        "arguments": arguments,
                        "classifier_batch_sizes": dict(result.classifier_batch_sizes),
                        "score": "detector_confidence * classifier_confidence",
                    },
                    "environment": _environment_metadata(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_completed_run(staging_dir)
        staging_dir.rename(output_dir)
        return EndToEndReport(
            output_dir=output_dir,
            metadata_path=output_dir / metadata_path.name,
            predictions_path=output_dir / predictions_path.name,
            metrics_path=output_dir / metrics_path.name,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


class TorchEndToEndBackend:
    def cuda_available(self, device: str) -> bool:
        import torch

        return device == "0" and torch.cuda.is_available()

    def predict(
        self,
        *,
        image_paths: Sequence[Path],
        images: Sequence[EndToEndImage],
        detector_checkpoint: Path,
        classifier_checkpoint: Path,
        classifier_context: Mapping[str, Any],
        output_dimension: int,
        arguments: Mapping[str, object],
    ) -> BackendInferenceResult:
        import torch
        from PIL import Image, UnidentifiedImageError
        from ultralytics import YOLO

        device_value = arguments.get("device")
        if device_value != "0":
            raise DataValidationError("end-to-end backend requires CUDA device '0'")
        device = _torch_device(device_value)
        detector_image_size = _positive_integer_argument(
            arguments, "detector_image_size"
        )
        classifier_image_size = _positive_integer_argument(
            arguments, "classifier_image_size"
        )
        detector_confidence = _unit_interval_argument(
            arguments, "detector_confidence"
        )
        detector_nms_iou = _unit_interval_argument(
            arguments, "detector_nms_iou", inclusive_one=True
        )
        detector = YOLO(str(detector_checkpoint))
        names = detector.names
        if tuple(names[index] for index in sorted(names)) != ("bread",):
            raise DataValidationError("end-to-end detector must have one bread class")
        classifier = _load_classifier_checkpoint_model(
            classifier_checkpoint,
            output_dimension=output_dimension,
            checkpoint_context=classifier_context,
            image_size=classifier_image_size,
            device=device,
        )
        _train_transform, validation_transform = _classifier_transforms(
            classifier_image_size
        )
        detector_results = detector.predict(
            source=[str(path) for path in image_paths],
            conf=detector_confidence,
            iou=detector_nms_iou,
            imgsz=detector_image_size,
            device=device_value,
            verbose=False,
            stream=False,
        )
        if len(detector_results) != len(images):
            raise DataValidationError(
                "detector result count does not match end-to-end images"
            )

        predictions: dict[str, tuple[EndToEndPrediction, ...]] = {}
        batch_sizes: dict[str, int] = {}
        for image_record, image_path, result in zip(
            images, image_paths, detector_results, strict=True
        ):
            try:
                with Image.open(image_path) as source:
                    scene = source.convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                raise DataValidationError(
                    f"cannot load end-to-end image {image_path}: {exc}"
                ) from exc
            width, height = scene.size
            crop_tensors = []
            detections: list[tuple[tuple[float, float, float, float], float]] = []
            boxes = result.boxes
            for xyxy, confidence, class_index in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
                strict=True,
            ):
                if int(class_index) != 0:
                    raise DataValidationError("detector emitted a non-bread class")
                x1 = max(0.0, min(float(width), float(xyxy[0])))
                y1 = max(0.0, min(float(height), float(xyxy[1])))
                x2 = max(0.0, min(float(width), float(xyxy[2])))
                y2 = max(0.0, min(float(height), float(xyxy[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = scene.crop(
                    (
                        math.floor(x1),
                        math.floor(y1),
                        math.ceil(x2),
                        math.ceil(y2),
                    )
                )
                crop_tensors.append(validation_transform(crop))
                detections.append(((x1, y1, x2, y2), float(confidence)))
            batch_sizes[image_record.image_id] = len(crop_tensors)
            if not crop_tensors:
                predictions[image_record.image_id] = ()
                continue
            batch = torch.stack(crop_tensors).to(device, non_blocking=True)
            with torch.inference_mode():
                probabilities = classifier(batch).softmax(dim=1)
                classifier_confidences, model_indices = probabilities.max(dim=1)
            image_predictions = []
            for (bbox, detector_score), model_index, classifier_score in zip(
                detections,
                model_indices.cpu().tolist(),
                classifier_confidences.cpu().tolist(),
                strict=True,
            ):
                image_predictions.append(
                    EndToEndPrediction(
                        image_id=image_record.image_id,
                        bbox_xyxy=bbox,
                        model_index=int(model_index),
                        detector_confidence=detector_score,
                        classifier_confidence=float(classifier_score),
                        score=detector_score * float(classifier_score),
                    )
                )
            predictions[image_record.image_id] = tuple(image_predictions)
        return BackendInferenceResult(predictions, batch_sizes)


def _positive_integer_argument(arguments: Mapping[str, object], label: str) -> int:
    value = arguments.get(label)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataValidationError(f"{label} must be a positive integer")
    return value


def _unit_interval_argument(
    arguments: Mapping[str, object],
    label: str,
    *,
    inclusive_one: bool = False,
) -> float:
    value = arguments.get(label)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DataValidationError(f"{label} must be a finite number")
    parsed = float(value)
    upper_valid = parsed <= 1 if inclusive_one else parsed < 1
    if not 0 < parsed or not upper_valid:
        raise DataValidationError(f"{label} must be in the valid unit interval")
    return parsed
