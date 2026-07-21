from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import platform
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import yaml

from .errors import DataValidationError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ROOT_FIELDS = {
    "freeze_version",
    "evaluation_id",
    "frozen_at_utc",
    "selection_status",
    "selection_basis",
    "dataset_root",
    "registry",
    "test_splits",
    "detector",
    "classifiers",
    "inference",
    "output_root",
    "run_name",
}


@dataclass(frozen=True, slots=True)
class FrozenRegistryConfig:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenTestSplitConfig:
    coco_path: str
    expected_phase: str


@dataclass(frozen=True, slots=True)
class FrozenDetectorConfig:
    config_path: str
    config_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    image_size: int
    confidence_floor: float
    operating_confidence: float
    nms_iou: float
    matching_iou: float


@dataclass(frozen=True, slots=True)
class FrozenClassifierConfig:
    config_path: str
    config_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    output_dimension: int
    image_size: int


@dataclass(frozen=True, slots=True)
class FrozenInferenceConfig:
    device: str
    classifier_batch_strategy: str
    combined_score: str


@dataclass(frozen=True, slots=True)
class FinalEvaluationConfig:
    freeze_version: int
    evaluation_id: str
    frozen_at_utc: str
    selection_status: str
    selection_basis: str
    dataset_root: str
    registry: FrozenRegistryConfig
    test_splits: Mapping[str, FrozenTestSplitConfig]
    detector: FrozenDetectorConfig
    classifiers: Mapping[str, FrozenClassifierConfig]
    inference: FrozenInferenceConfig
    output_root: str
    run_name: str


@dataclass(frozen=True, slots=True)
class PreparedFinalEvaluation:
    config_path: Path
    config_sha256: str
    dataset_root: Path
    registry_path: Path
    detector_config_path: Path
    detector_checkpoint: Path
    classifier_config_paths: Mapping[str, Path]
    classifier_checkpoints: Mapping[str, Path]
    classifier_contexts: Mapping[str, Mapping[str, Any]]
    output_dir: Path
    lock_path: Path
    provenance: Mapping[str, Any]
    evaluation_id: str


@dataclass(frozen=True, slots=True)
class FinalEvaluationPreflightReport:
    prepared: PreparedFinalEvaluation
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_path": str(self.prepared.config_path),
            "config_sha256": self.prepared.config_sha256,
            "evaluation_id": self.prepared.evaluation_id,
            "output_dir": str(self.prepared.output_dir),
            "lock_path": str(self.prepared.lock_path),
            "test_data_accessed": False,
        }


@dataclass(frozen=True, slots=True)
class TestObjectRecord:
    sample_id: str
    annotation_id: int
    bbox_xyxy: tuple[float, float, float, float]
    category_id: int
    model_index: int
    phase: str


@dataclass(frozen=True, slots=True)
class TestImageRecord:
    image_id: str
    image_path: Path
    width: int
    height: int
    difficulty: str | None
    objects: tuple[TestObjectRecord, ...]


@dataclass(frozen=True, slots=True)
class LoadedTestSplit:
    name: str
    phase: str
    coco_path: Path
    images: tuple[TestImageRecord, ...]


@dataclass(frozen=True, slots=True)
class FinalSplitInferenceResult:
    detector_predictions: Mapping[str, Sequence[Any]]
    classifier_predictions: Mapping[str, Sequence[Any]]
    e2e_predictions: Mapping[str, Mapping[str, Sequence[Any]]]
    batch_sizes: Mapping[str, Mapping[str, Sequence[int]]]


@dataclass(frozen=True, slots=True)
class FinalInferenceResult:
    splits: Mapping[str, FinalSplitInferenceResult]


@dataclass(frozen=True, slots=True)
class FinalEvaluationReport:
    output_dir: Path
    summary_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "metadata_path": str(self.metadata_path),
            "summary": _coco_payload(self.summary_path),
        }


def _object(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise DataValidationError(
            f"{label} fields are invalid: expected={sorted(expected)}, actual={actual}"
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


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataValidationError(f"{label} must be a positive integer")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DataValidationError(f"{label} must be a finite number")
    return float(value)


def _path(value: object, label: str) -> str:
    result = _text(value, label)
    pure = PurePath(result)
    if ".." in pure.parts:
        raise DataValidationError(f"{label} must not contain parent traversal")
    return result


def load_final_evaluation_config(path: str | Path) -> FinalEvaluationConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(
            f"cannot load final evaluation config {config_path}: {exc}"
        ) from exc
    payload = _object(raw, _ROOT_FIELDS, "final evaluation config")
    if payload["freeze_version"] != 1:
        raise DataValidationError("freeze_version must be 1")
    evaluation_id = _text(payload["evaluation_id"], "evaluation_id")
    frozen_at = _text(payload["frozen_at_utc"], "frozen_at_utc")
    try:
        parsed_time = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataValidationError("frozen_at_utc must be ISO-8601") from exc
    if parsed_time.tzinfo is None:
        raise DataValidationError("frozen_at_utc must include a timezone")
    if payload["selection_status"] != "frozen_before_test_access":
        raise DataValidationError(
            "selection_status must be frozen_before_test_access"
        )
    if payload["selection_basis"] != "train_side_validation_only":
        raise DataValidationError(
            "selection_basis must be train_side_validation_only"
        )
    registry_raw = _object(payload["registry"], {"path", "sha256"}, "registry")
    registry = FrozenRegistryConfig(
        _path(registry_raw["path"], "registry path"),
        _sha(registry_raw["sha256"], "registry SHA-256"),
    )
    splits_raw = _object(
        payload["test_splits"], {"base", "incremental"}, "test_splits"
    )
    test_splits = {}
    for name, expected_phase, expected_fragment in (
        ("base", "base", ("base", "test", "instances_test.json")),
        (
            "incremental",
            "incremental",
            ("incremental", "test", "instances_test.json"),
        ),
    ):
        split_raw = _object(
            splits_raw[name], {"coco_path", "expected_phase"}, f"{name} test"
        )
        if split_raw["expected_phase"] != expected_phase:
            label = "Base test" if name == "base" else "Incremental test"
            raise DataValidationError(f"{label} expected_phase is invalid")
        coco_path = _path(split_raw["coco_path"], f"{name} test COCO path")
        if tuple(part.casefold() for part in PurePath(coco_path).parts[-3:]) != tuple(
            part.casefold() for part in expected_fragment
        ):
            raise DataValidationError(f"{name} test COCO path is not the frozen path")
        test_splits[name] = FrozenTestSplitConfig(coco_path, expected_phase)

    detector_raw = _object(
        payload["detector"],
        {
            "config_path",
            "config_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "image_size",
            "confidence_floor",
            "operating_confidence",
            "nms_iou",
            "matching_iou",
        },
        "detector",
    )
    confidence_floor = _number(
        detector_raw["confidence_floor"], "detector confidence threshold"
    )
    operating_confidence = _number(
        detector_raw["operating_confidence"], "detector operating threshold"
    )
    nms_iou = _number(detector_raw["nms_iou"], "detector NMS threshold")
    matching_iou = _number(
        detector_raw["matching_iou"], "detector matching threshold"
    )
    if not 0 < confidence_floor <= operating_confidence < 1 or not all(
        0 < value <= 1 for value in (nms_iou, matching_iou)
    ):
        raise DataValidationError("detector thresholds are invalid")
    detector = FrozenDetectorConfig(
        _path(detector_raw["config_path"], "detector config path"),
        _sha(detector_raw["config_sha256"], "detector config SHA-256"),
        _path(detector_raw["checkpoint"], "detector checkpoint"),
        _sha(detector_raw["checkpoint_sha256"], "detector checkpoint SHA-256"),
        _positive_integer(detector_raw["image_size"], "detector image_size"),
        confidence_floor,
        operating_confidence,
        nms_iou,
        matching_iou,
    )

    classifiers_raw = _object(
        payload["classifiers"], {"base", "incremental"}, "classifiers"
    )
    classifiers = {}
    classifier_fields = {
        "config_path",
        "config_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "output_dimension",
        "image_size",
    }
    for name, expected_dimension in (("base", 15), ("incremental", 20)):
        item = _object(classifiers_raw[name], classifier_fields, f"{name} classifier")
        dimension = _positive_integer(
            item["output_dimension"], f"{name} classifier output_dimension"
        )
        if dimension != expected_dimension:
            label = "Base classifier" if name == "base" else "Incremental classifier"
            raise DataValidationError(
                f"{label} output_dimension must be {expected_dimension}"
            )
        classifiers[name] = FrozenClassifierConfig(
            _path(item["config_path"], f"{name} classifier config path"),
            _sha(item["config_sha256"], f"{name} classifier config SHA-256"),
            _path(item["checkpoint"], f"{name} classifier checkpoint"),
            _sha(item["checkpoint_sha256"], f"{name} classifier checkpoint SHA-256"),
            dimension,
            _positive_integer(item["image_size"], f"{name} classifier image_size"),
        )

    inference_raw = _object(
        payload["inference"],
        {"device", "classifier_batch_strategy", "combined_score"},
        "inference",
    )
    if inference_raw["device"] != "0":
        raise DataValidationError("final evaluation inference device must be '0'")
    if inference_raw["classifier_batch_strategy"] != "one_batch_per_scene":
        raise DataValidationError("classifier_batch_strategy is invalid")
    if (
        inference_raw["combined_score"]
        != "detector_confidence_times_classifier_confidence"
    ):
        raise DataValidationError("combined_score is invalid")
    inference = FrozenInferenceConfig(
        "0", "one_batch_per_scene", "detector_confidence_times_classifier_confidence"
    )
    run_name = _text(payload["run_name"], "run_name")
    if not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(f"run_name is invalid: {run_name!r}")
    return FinalEvaluationConfig(
        freeze_version=1,
        evaluation_id=evaluation_id,
        frozen_at_utc=frozen_at,
        selection_status="frozen_before_test_access",
        selection_basis="train_side_validation_only",
        dataset_root=_path(payload["dataset_root"], "dataset_root"),
        registry=registry,
        test_splits=test_splits,
        detector=detector,
        classifiers=classifiers,
        inference=inference,
        output_root=_path(payload["output_root"], "output_root"),
        run_name=run_name,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise DataValidationError(f"{label} SHA-256 does not match frozen config")


def _contains(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath(
            [os.path.normcase(str(root)), os.path.normcase(str(candidate))]
        ) == os.path.normcase(str(root))
    except ValueError:
        return False


def _cuda_available(device: str) -> bool:
    import torch

    return device == "0" and torch.cuda.is_available()


def _validate_classifier_metadata(
    *,
    name: str,
    selected_config,
    frozen: FrozenClassifierConfig,
    checkpoint: Path,
    metadata: Mapping[str, Any],
    report,
    context: Mapping[str, Any],
    detector_checkpoint: Path,
    detector_hash: str,
) -> None:
    expected_checkpoint = (
        Path(selected_config.output_root)
        / selected_config.run_name
        / "checkpoints"
        / "best.pt"
    ).resolve(strict=False)
    dataset = metadata.get("dataset")
    model = metadata.get("model")
    if checkpoint != expected_checkpoint:
        raise DataValidationError(
            f"{name} classifier checkpoint path does not match selected run"
        )
    if (
        not isinstance(dataset, dict)
        or dataset.get("phase") != report.phase
        or dataset.get("output_dimension") != frozen.output_dimension
        or Path(str(dataset.get("manifest_path"))).resolve(strict=False)
        != report.manifest_path.resolve(strict=False)
        or dataset.get("manifest_sha256") != _sha256(report.manifest_path)
        or dataset.get("registry_sha256") != context.get("registry_sha256")
        or dataset.get("model_index_mapping") != context.get("model_index_mapping")
    ):
        raise DataValidationError(
            f"{name} classifier metadata does not match selected dataset/context"
        )
    if not isinstance(model, dict) or model.get("architecture") != "resnet18":
        raise DataValidationError(f"{name} classifier metadata must describe ResNet18")
    if name == "incremental":
        frozen_detector = metadata.get("frozen_detector")
        if (
            not isinstance(frozen_detector, dict)
            or Path(str(frozen_detector.get("checkpoint"))).resolve(strict=False)
            != detector_checkpoint
            or frozen_detector.get("sha256_before") != detector_hash
            or frozen_detector.get("sha256_after") != detector_hash
            or frozen_detector.get("detector_unchanged") is not True
        ):
            raise DataValidationError(
                "Incremental classifier frozen detector provenance is invalid"
            )


def _strict_validate_classifier_checkpoint(
    *,
    checkpoint: Path,
    output_dimension: int,
    checkpoint_context: Mapping[str, Any],
    image_size: int,
    loader: Callable[..., Any] | None = None,
) -> None:
    if loader is None:
        from .classifier_training import _load_classifier_checkpoint_model

        loader = _load_classifier_checkpoint_model
    model = loader(
        checkpoint=checkpoint,
        output_dimension=output_dimension,
        checkpoint_context=checkpoint_context,
        image_size=image_size,
        device="cpu",
    )
    del model


def _prepare_non_test_inputs(
    config: FinalEvaluationConfig,
    config_path: Path,
    cuda_available: Callable[[str], bool],
) -> PreparedFinalEvaluation:
    from .classifier_dataset import validate_classifier_dataset
    from .classifier_training import (
        _checkpoint_context,
        load_classifier_experiment_config,
    )
    from .detector_dataset import validate_detector_dataset
    from .detector_training import load_detector_training_config
    from .e2e_inference import (
        _checkpoint_metadata,
        _validate_detector_checkpoint_provenance,
        _validate_yolo_source_binding,
    )
    from .registry import load_class_registry
    from .yolo_dataset import validate_yolo_dataset

    config_path = config_path.resolve(strict=False)
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    registry_path = Path(config.registry.path).resolve(strict=False)
    detector_config_path = Path(config.detector.config_path).resolve(strict=False)
    detector_checkpoint = Path(config.detector.checkpoint).resolve(strict=False)
    output_root = Path(config.output_root).resolve(strict=False)
    output_dir = output_root / config.run_name
    lock_path = output_root / f".{config.run_name}.started.json"
    if _contains(dataset_root, output_root):
        raise DataValidationError("final evaluation output must be outside dataset root")
    if output_dir.exists() or lock_path.exists():
        raise DataValidationError("final evaluation output or one-shot lock already exists")
    if not cuda_available(config.inference.device):
        raise DataValidationError("frozen CUDA device 0 is unavailable")

    _require_hash(registry_path, config.registry.sha256, "registry")
    load_class_registry(registry_path)
    _require_hash(
        detector_config_path, config.detector.config_sha256, "detector config"
    )
    _require_hash(
        detector_checkpoint,
        config.detector.checkpoint_sha256,
        "detector checkpoint",
    )
    detector_config = load_detector_training_config(detector_config_path)
    detector_run_config_path = detector_checkpoint.parent.parent / "config.yaml"
    detector_run_config = load_detector_training_config(detector_run_config_path)
    thresholds = detector_config.thresholds
    if (
        Path(detector_config.dataset_root).resolve(strict=False) != dataset_root
        or detector_config.image_size != config.detector.image_size
        or thresholds.confidence_floor != config.detector.confidence_floor
        or thresholds.operating_confidence != config.detector.operating_confidence
        or thresholds.nms_iou != config.detector.nms_iou
        or thresholds.matching_iou != config.detector.matching_iou
    ):
        raise DataValidationError("detector training config does not match frozen values")
    detector_report = validate_detector_dataset(
        dataset_root, detector_config.source_detector_run
    )
    yolo_report = validate_yolo_dataset(dataset_root, detector_config.yolo_run_name)
    _validate_yolo_source_binding(
        yolo_report.manifest_path,
        detector_config.source_detector_run,
        detector_report.manifest_path,
        project_root=dataset_root.parent,
    )
    detector_metadata_path, detector_metadata = _checkpoint_metadata(
        detector_checkpoint, "detector"
    )
    _validate_detector_checkpoint_provenance(
        detector_config,
        detector_run_config,
        detector_checkpoint,
        detector_metadata,
        yolo_report.manifest_path,
        project_root=dataset_root.parent,
    )

    classifier_config_paths = {}
    classifier_checkpoints = {}
    classifier_contexts = {}
    classifier_metadata_paths = {}
    classifier_manifest_paths = {}
    for name in ("base", "incremental"):
        frozen = config.classifiers[name]
        selected_path = Path(frozen.config_path).resolve(strict=False)
        checkpoint = Path(frozen.checkpoint).resolve(strict=False)
        _require_hash(selected_path, frozen.config_sha256, f"{name} classifier config")
        _require_hash(checkpoint, frozen.checkpoint_sha256, f"{name} classifier checkpoint")
        selected_config = load_classifier_experiment_config(selected_path)
        if (
            Path(selected_config.dataset_root).resolve(strict=False) != dataset_root
            or selected_config.image_size != frozen.image_size
        ):
            raise DataValidationError(
                f"{name} classifier config does not match frozen values"
            )
        report = validate_classifier_dataset(
            dataset_root, selected_config.source_classifier_run
        )
        if report.output_dimension != frozen.output_dimension:
            raise DataValidationError(
                f"{name} classifier dataset output dimension is invalid"
            )
        context = _checkpoint_context(
            selected_config,
            dataset_root,
            report.manifest_path,
            report.output_dimension,
        )
        metadata_path, metadata = _checkpoint_metadata(checkpoint, f"{name} classifier")
        _validate_classifier_metadata(
            name=name,
            selected_config=selected_config,
            frozen=frozen,
            checkpoint=checkpoint,
            metadata=metadata,
            report=report,
            context=context,
            detector_checkpoint=detector_checkpoint,
            detector_hash=config.detector.checkpoint_sha256,
        )
        _strict_validate_classifier_checkpoint(
            checkpoint=checkpoint,
            output_dimension=frozen.output_dimension,
            checkpoint_context=context,
            image_size=frozen.image_size,
        )
        classifier_config_paths[name] = selected_path
        classifier_checkpoints[name] = checkpoint
        classifier_contexts[name] = context
        classifier_metadata_paths[name] = str(metadata_path)
        classifier_manifest_paths[name] = str(report.manifest_path)

    return PreparedFinalEvaluation(
        config_path=config_path,
        config_sha256=_sha256(config_path),
        dataset_root=dataset_root,
        registry_path=registry_path,
        detector_config_path=detector_config_path,
        detector_checkpoint=detector_checkpoint,
        classifier_config_paths=classifier_config_paths,
        classifier_checkpoints=classifier_checkpoints,
        classifier_contexts=classifier_contexts,
        output_dir=output_dir,
        lock_path=lock_path,
        provenance={
            "registry_sha256": config.registry.sha256,
            "detector_config_sha256": config.detector.config_sha256,
            "detector_checkpoint_sha256": config.detector.checkpoint_sha256,
            "detector_metadata_path": str(detector_metadata_path),
            "detector_training_manifest_path": str(yolo_report.manifest_path),
            "detector_training_manifest_sha256": _sha256(yolo_report.manifest_path),
            "classifier_config_sha256": {
                name: config.classifiers[name].config_sha256
                for name in ("base", "incremental")
            },
            "classifier_checkpoint_sha256": {
                name: config.classifiers[name].checkpoint_sha256
                for name in ("base", "incremental")
            },
            "classifier_metadata_paths": classifier_metadata_paths,
            "classifier_manifest_paths": classifier_manifest_paths,
        },
        evaluation_id=config.evaluation_id,
    )


def preflight_final_evaluation(
    config: FinalEvaluationConfig,
    config_path: str | Path,
    *,
    cuda_available: Callable[[str], bool] | None = None,
) -> FinalEvaluationPreflightReport:
    if not isinstance(config, FinalEvaluationConfig):
        raise DataValidationError("config must be FinalEvaluationConfig")
    config_path = Path(config_path)
    if load_final_evaluation_config(config_path) != config:
        raise DataValidationError(
            "final evaluation config object does not match config_path bytes"
        )
    prepared = _prepare_non_test_inputs(
        config,
        config_path,
        cuda_available or _cuda_available,
    )
    return FinalEvaluationPreflightReport(prepared)


def _configured_dataset_path(value: str, dataset_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if candidate.parts and candidate.parts[0].casefold() == dataset_root.name.casefold():
        return (dataset_root.parent / candidate).resolve(strict=False)
    return (dataset_root / candidate).resolve(strict=False)


def _coco_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load final test COCO {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("final test COCO must be a JSON object")
    return payload


def _load_test_split(
    config: FinalEvaluationConfig,
    prepared: PreparedFinalEvaluation,
    name: str,
) -> LoadedTestSplit:
    from .coco import validate_coco
    from .detector_training import _difficulty
    from .registry import load_class_registry

    if name not in {"base", "incremental"}:
        raise DataValidationError(f"unknown final test split: {name}")
    selected = config.test_splits[name]
    coco_path = _configured_dataset_path(selected.coco_path, prepared.dataset_root)
    registry = load_class_registry(prepared.registry_path)
    validate_coco(coco_path, registry, selected.expected_phase)
    payload = _coco_payload(coco_path)
    raw_images = payload.get("images")
    raw_annotations = payload.get("annotations")
    if not isinstance(raw_images, list) or not isinstance(raw_annotations, list):
        raise DataValidationError("final test COCO images/annotations must be lists")
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            raise DataValidationError("final test annotation must be an object")
        image_id = annotation["image_id"]
        annotations_by_image.setdefault(image_id, []).append(annotation)
    images = []
    for raw_image in raw_images:
        if not isinstance(raw_image, dict):
            raise DataValidationError("final test image must be an object")
        coco_image_id = raw_image["id"]
        file_name = raw_image["file_name"]
        image_path = (coco_path.parent / file_name).resolve(strict=False)
        objects = []
        for annotation in sorted(
            annotations_by_image.get(coco_image_id, []), key=lambda item: item["id"]
        ):
            record = registry.by_category_id.get(annotation["category_id"])
            if record is None or record.phase != selected.expected_phase:
                raise DataValidationError(
                    "final test annotation category does not match frozen phase"
                )
            x, y, width, height = (float(value) for value in annotation["bbox"])
            objects.append(
                TestObjectRecord(
                    sample_id=f"{name}:{file_name}:annotation-{annotation['id']}",
                    annotation_id=int(annotation["id"]),
                    bbox_xyxy=(x, y, x + width, y + height),
                    category_id=record.category_id,
                    model_index=record.model_index,
                    phase=record.phase,
                )
            )
        images.append(
            TestImageRecord(
                image_id=str(file_name),
                image_path=image_path,
                width=int(raw_image["width"]),
                height=int(raw_image["height"]),
                difficulty=_difficulty(str(file_name)),
                objects=tuple(objects),
            )
        )
    if not images:
        raise DataValidationError(f"{name} final test must contain images")
    return LoadedTestSplit(name, selected.expected_phase, coco_path, tuple(images))


def _lock_payload(
    config: FinalEvaluationConfig,
    prepared: PreparedFinalEvaluation,
) -> dict[str, Any]:
    return {
        "lock_version": 1,
        "evaluation_id": config.evaluation_id,
        "run_name": config.run_name,
        "config_path": str(prepared.config_path),
        "config_sha256": prepared.config_sha256,
        "status": "started",
        "test_access_started_at": datetime.now(timezone.utc).isoformat(),
        "test_splits": {
            name: {
                "coco_path": selected.coco_path,
                "expected_phase": selected.expected_phase,
            }
            for name, selected in config.test_splits.items()
        },
    }


def _create_start_lock(
    config: FinalEvaluationConfig, prepared: PreparedFinalEvaluation
) -> None:
    prepared.lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _lock_payload(config, prepared), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    try:
        descriptor = os.open(
            prepared.lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise DataValidationError(
            f"final evaluation one-shot lock already exists: {prepared.lock_path}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _update_start_lock(
    prepared: PreparedFinalEvaluation,
    *,
    status: str,
    error: str | None = None,
) -> None:
    if status not in {"completed", "failed"}:
        raise DataValidationError("final evaluation lock status is invalid")
    try:
        payload = json.loads(prepared.lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataValidationError("cannot update final evaluation one-shot lock") from exc
    if not isinstance(payload, dict) or payload.get("status") != "started":
        raise DataValidationError("final evaluation one-shot lock is not active")
    payload["status"] = status
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        payload["error"] = error
    temporary = prepared.lock_path.with_name(
        f".{prepared.lock_path.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(prepared.lock_path)


def _load_locked_test_splits(
    config: FinalEvaluationConfig,
    prepared: PreparedFinalEvaluation,
    *,
    loader: Callable[
        [FinalEvaluationConfig, PreparedFinalEvaluation, str], Any
    ] = _load_test_split,
) -> dict[str, Any]:
    _create_start_lock(config, prepared)
    try:
        return {
            name: loader(config, prepared, name)
            for name in ("base", "incremental")
        }
    except Exception as exc:
        _update_start_lock(prepared, status="failed", error=str(exc))
        raise


class FinalInferenceBackend:
    def __init__(
        self,
        *,
        detector_factory: Callable[[Path], Any] | None = None,
        classifier_loader: Callable[..., Any] | None = None,
        transform_factory: Callable[[int], Any] | None = None,
        device_factory: Callable[[str], Any] | None = None,
        stacker: Callable[[Sequence[Any], Any], Any] | None = None,
    ) -> None:
        self._detector_factory = detector_factory or self._default_detector_factory
        self._classifier_loader = classifier_loader or self._default_classifier_loader
        self._transform_factory = transform_factory or self._default_transform_factory
        self._device_factory = device_factory or self._default_device_factory
        self._stacker = stacker or self._default_stacker
        self._strict_device_checks = stacker is None and device_factory is None

    @staticmethod
    def _default_detector_factory(checkpoint: Path):
        from ultralytics import YOLO

        return YOLO(str(checkpoint))

    @staticmethod
    def _default_classifier_loader(*args, **kwargs):
        from .classifier_training import _load_classifier_checkpoint_model

        return _load_classifier_checkpoint_model(*args, **kwargs)

    @staticmethod
    def _default_transform_factory(image_size: int):
        from .classifier_training import _classifier_transforms

        _train, validation = _classifier_transforms(image_size)
        return validation

    @staticmethod
    def _default_device_factory(value: str):
        from .classifier_training import _torch_device

        return _torch_device(value)

    @staticmethod
    def _default_stacker(items: Sequence[Any], device):
        import torch

        return torch.stack(tuple(items)).to(device, non_blocking=True)

    @staticmethod
    def _module_on_device(module: Any, device) -> bool:
        parameters = tuple(module.parameters()) if hasattr(module, "parameters") else ()
        buffers = tuple(module.buffers()) if hasattr(module, "buffers") else ()
        return all(value.device == device for value in (*parameters, *buffers))

    def _classify(
        self,
        crops: Sequence[Any],
        *,
        model: Any,
        device: Any,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        import torch

        if not crops:
            return (), ()
        batch = self._stacker(crops, device)
        if self._strict_device_checks and batch.device != device:
            raise DataValidationError("final classifier input batch is not on CUDA")
        with torch.inference_mode():
            probabilities = model(batch).softmax(dim=1)
            confidences, indices = probabilities.max(dim=1)
        if self._strict_device_checks and probabilities.device != device:
            raise DataValidationError("final classifier output is not on CUDA")
        return (
            tuple(int(value) for value in indices.detach().cpu().tolist()),
            tuple(float(value) for value in confidences.detach().cpu().tolist()),
        )

    def predict(
        self,
        *,
        splits: Mapping[str, LoadedTestSplit],
        detector_checkpoint: Path,
        classifier_checkpoints: Mapping[str, Path],
        classifier_contexts: Mapping[str, Mapping[str, Any]],
        config: FinalEvaluationConfig,
    ) -> FinalInferenceResult:
        from PIL import Image, UnidentifiedImageError

        from .classifier_evaluation import ClassifierPrediction
        from .detector_evaluation import Detection
        from .e2e_evaluation import EndToEndPrediction

        if set(splits) != {"base", "incremental"}:
            raise DataValidationError("final inference requires both test splits")
        if set(classifier_checkpoints) != {"base", "incremental"} or set(
            classifier_contexts
        ) != {"base", "incremental"}:
            raise DataValidationError("final inference classifier inputs are invalid")
        device = self._device_factory(config.inference.device)
        if self._strict_device_checks and getattr(device, "type", None) != "cuda":
            raise DataValidationError("final evaluation backend requires CUDA")
        detector = self._detector_factory(detector_checkpoint)
        names = detector.names
        if not isinstance(names, Mapping) or tuple(
            names[index] for index in sorted(names)
        ) != ("bread",):
            raise DataValidationError("final detector must declare one bread class")

        models = {}
        transforms = {}
        for name in ("base", "incremental"):
            frozen = config.classifiers[name]
            model = self._classifier_loader(
                checkpoint=classifier_checkpoints[name],
                output_dimension=frozen.output_dimension,
                checkpoint_context=classifier_contexts[name],
                image_size=frozen.image_size,
                device=device,
            )
            if self._strict_device_checks and not self._module_on_device(model, device):
                raise DataValidationError(
                    f"{name} classifier model is not entirely on CUDA"
                )
            models[name] = model
            transforms[name] = self._transform_factory(frozen.image_size)

        split_results = {}
        for split_name in ("base", "incremental"):
            split = splits[split_name]
            relevant_models = (
                ("base", "incremental")
                if split_name == "base"
                else ("incremental",)
            )
            detector_results = detector.predict(
                source=[str(image.image_path) for image in split.images],
                conf=config.detector.confidence_floor,
                iou=config.detector.nms_iou,
                imgsz=config.detector.image_size,
                device=config.inference.device,
                verbose=False,
                stream=False,
            )
            if len(detector_results) != len(split.images):
                raise DataValidationError(
                    "final detector result count does not match test images"
                )
            detector_predictions = {}
            classifier_predictions = {name: [] for name in relevant_models}
            e2e_predictions = {
                name: {} for name in relevant_models
            }
            batch_sizes = {
                name: {"ground_truth": [], "detections": []}
                for name in relevant_models
            }
            for image, detector_result in zip(
                split.images, detector_results, strict=True
            ):
                try:
                    with Image.open(image.image_path) as source:
                        scene = source.convert("RGB")
                except (OSError, UnidentifiedImageError) as exc:
                    raise DataValidationError(
                        f"cannot load final test image {image.image_path}: {exc}"
                    ) from exc
                if scene.size != (image.width, image.height):
                    raise DataValidationError(
                        f"final test image size drifted: {image.image_id}"
                    )
                detections = []
                for xyxy, confidence, class_index in zip(
                    detector_result.boxes.xyxy.detach().cpu().tolist(),
                    detector_result.boxes.conf.detach().cpu().tolist(),
                    detector_result.boxes.cls.detach().cpu().tolist(),
                    strict=True,
                ):
                    if int(class_index) != 0:
                        raise DataValidationError(
                            "final detector emitted a non-bread class"
                        )
                    x1 = max(0.0, min(float(image.width), float(xyxy[0])))
                    y1 = max(0.0, min(float(image.height), float(xyxy[1])))
                    x2 = max(0.0, min(float(image.width), float(xyxy[2])))
                    y2 = max(0.0, min(float(image.height), float(xyxy[3])))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append(
                        Detection((x1, y1, x2, y2), float(confidence), 0)
                    )
                detector_predictions[image.image_id] = tuple(detections)

                for model_name in relevant_models:
                    transform = transforms[model_name]
                    gt_crops = [
                        transform(
                            scene.crop(
                                (
                                    math.floor(obj.bbox_xyxy[0]),
                                    math.floor(obj.bbox_xyxy[1]),
                                    math.ceil(obj.bbox_xyxy[2]),
                                    math.ceil(obj.bbox_xyxy[3]),
                                )
                            )
                        )
                        for obj in image.objects
                    ]
                    gt_indices, gt_confidences = self._classify(
                        gt_crops, model=models[model_name], device=device
                    )
                    batch_sizes[model_name]["ground_truth"].append(len(gt_crops))
                    for obj, predicted_index, confidence in zip(
                        image.objects,
                        gt_indices,
                        gt_confidences,
                        strict=True,
                    ):
                        classifier_predictions[model_name].append(
                            ClassifierPrediction(
                                obj.sample_id,
                                obj.model_index,
                                predicted_index,
                                confidence,
                            )
                        )

                    detection_crops = [
                        transform(
                            scene.crop(
                                (
                                    math.floor(item.bbox_xyxy[0]),
                                    math.floor(item.bbox_xyxy[1]),
                                    math.ceil(item.bbox_xyxy[2]),
                                    math.ceil(item.bbox_xyxy[3]),
                                )
                            )
                        )
                        for item in detections
                    ]
                    predicted_indices, confidences = self._classify(
                        detection_crops, model=models[model_name], device=device
                    )
                    batch_sizes[model_name]["detections"].append(
                        len(detection_crops)
                    )
                    image_predictions = []
                    for detection, predicted_index, confidence in zip(
                        detections,
                        predicted_indices,
                        confidences,
                        strict=True,
                    ):
                        output_dimension = config.classifiers[
                            model_name
                        ].output_dimension
                        if not 0 <= predicted_index < output_dimension:
                            raise DataValidationError(
                                f"{model_name} classifier emitted invalid model_index"
                            )
                        image_predictions.append(
                            EndToEndPrediction(
                                image_id=image.image_id,
                                bbox_xyxy=detection.bbox_xyxy,
                                model_index=predicted_index,
                                detector_confidence=detection.confidence,
                                classifier_confidence=confidence,
                                score=detection.confidence * confidence,
                            )
                        )
                    e2e_predictions[model_name][image.image_id] = tuple(
                        image_predictions
                    )

            split_results[split_name] = FinalSplitInferenceResult(
                detector_predictions=detector_predictions,
                classifier_predictions={
                    name: tuple(values)
                    for name, values in classifier_predictions.items()
                },
                e2e_predictions={
                    name: dict(values) for name, values in e2e_predictions.items()
                },
                batch_sizes={
                    name: {
                        kind: tuple(values)
                        for kind, values in groups.items()
                    }
                    for name, groups in batch_sizes.items()
                },
            )
        return FinalInferenceResult(split_results)


def _delta(new_value: Any, old_value: Any) -> float | None:
    if new_value is None or old_value is None:
        return None
    return float(new_value) - float(old_value)


def _environment_metadata() -> dict[str, Any]:
    import importlib.metadata

    dependencies = {}
    for distribution in ("torch", "torchvision", "ultralytics", "Pillow", "PyYAML"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = "unavailable"
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda = torch.version.cuda
    except (ImportError, RuntimeError):
        gpu = None
        cuda = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "cuda": cuda,
        "dependencies": dependencies,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluation_inputs(split: LoadedTestSplit):
    from .detector_evaluation import EvaluationImage, EvaluationObject
    from .e2e_evaluation import EndToEndImage, EndToEndTruth

    detector_images = tuple(
        EvaluationImage(
            image.image_id,
            image.difficulty,
            tuple(
                EvaluationObject(obj.bbox_xyxy, obj.category_id, obj.phase)
                for obj in image.objects
            ),
        )
        for image in split.images
    )
    e2e_images = tuple(
        EndToEndImage(
            image.image_id,
            tuple(
                EndToEndTruth(obj.bbox_xyxy, obj.model_index)
                for obj in image.objects
            ),
        )
        for image in split.images
    )
    return detector_images, e2e_images


def _metrics(
    config: FinalEvaluationConfig,
    splits: Mapping[str, LoadedTestSplit],
    result: FinalInferenceResult,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from .classifier_evaluation import evaluate_classifier_predictions
    from .detector_evaluation import (
        EvaluationThresholds,
        evaluate_detector_predictions,
    )
    from .e2e_evaluation import evaluate_end_to_end_predictions

    if set(result.splits) != {"base", "incremental"}:
        raise DataValidationError("final inference result splits are invalid")
    thresholds = EvaluationThresholds(
        config.detector.confidence_floor,
        config.detector.operating_confidence,
        config.detector.nms_iou,
        config.detector.matching_iou,
    )
    detector_metrics = {}
    classifier_metrics = {}
    e2e_metrics = {}
    for split_name in ("base", "incremental"):
        detector_images, e2e_images = _evaluation_inputs(splits[split_name])
        split_result = result.splits[split_name]
        detector_metrics[f"{split_name}_test"] = evaluate_detector_predictions(
            detector_images,
            split_result.detector_predictions,
            thresholds,
        )
        relevant = (
            ("base", "incremental")
            if split_name == "base"
            else ("incremental",)
        )
        if set(split_result.classifier_predictions) != set(relevant) or set(
            split_result.e2e_predictions
        ) != set(relevant):
            raise DataValidationError(
                f"{split_name} final classifier variants are invalid"
            )
        for model_name in relevant:
            key = f"{model_name}_model_on_{split_name}_test"
            output_dimension = config.classifiers[model_name].output_dimension
            expected_truth = tuple(
                (obj.sample_id, obj.model_index)
                for image in splits[split_name].images
                for obj in image.objects
            )
            actual_truth = tuple(
                (item.sample_id, item.target_index)
                for item in split_result.classifier_predictions[model_name]
            )
            if actual_truth != expected_truth:
                raise DataValidationError(
                    f"{key} classifier predictions do not match COCO ground truth"
                )
            classifier_metrics[key] = evaluate_classifier_predictions(
                split_result.classifier_predictions[model_name],
                output_dimension=output_dimension,
            )
            e2e_metrics[key] = evaluate_end_to_end_predictions(
                e2e_images,
                split_result.e2e_predictions[model_name],
                output_dimension=output_dimension,
                count_detector_confidence=config.detector.operating_confidence,
            )
    base_classifier = classifier_metrics["base_model_on_base_test"]
    incremental_base_classifier = classifier_metrics[
        "incremental_model_on_base_test"
    ]
    base_e2e = e2e_metrics["base_model_on_base_test"]
    incremental_base_e2e = e2e_metrics["incremental_model_on_base_test"]
    incremental_new_classifier = classifier_metrics[
        "incremental_model_on_incremental_test"
    ]
    incremental_new_e2e = e2e_metrics["incremental_model_on_incremental_test"]
    summary = {
        "summary_version": 1,
        "selection_status": config.selection_status,
        "selection_basis": config.selection_basis,
        "configuration_changed_after_test": False,
        "base_retention_delta": {
            "classifier_top1": _delta(
                incremental_base_classifier["top1_accuracy"],
                base_classifier["top1_accuracy"],
            ),
            "classifier_macro_f1": _delta(
                incremental_base_classifier["macro_f1"],
                base_classifier["macro_f1"],
            ),
            "e2e_map50": _delta(
                incremental_base_e2e["map50"], base_e2e["map50"]
            ),
            "e2e_map50_95": _delta(
                incremental_base_e2e["map50_95"], base_e2e["map50_95"]
            ),
            "e2e_supported_exact_count_accuracy": _delta(
                incremental_base_e2e["supported_macro_exact_count_accuracy"],
                base_e2e["supported_macro_exact_count_accuracy"],
            ),
        },
        "incremental_new_classes": {
            "classifier_top1": incremental_new_classifier["top1_accuracy"],
            "classifier_macro_f1": incremental_new_classifier["macro_f1"],
            "e2e_map50": incremental_new_e2e["map50"],
            "e2e_map50_95": incremental_new_e2e["map50_95"],
            "e2e_supported_exact_count_accuracy": incremental_new_e2e[
                "supported_macro_exact_count_accuracy"
            ],
            "detector_recall": detector_metrics["incremental_test"]["global"][
                "recall"
            ],
        },
        "base_test": {
            "base_classifier_top1": base_classifier["top1_accuracy"],
            "incremental_classifier_top1": incremental_base_classifier[
                "top1_accuracy"
            ],
            "base_e2e_map50": base_e2e["map50"],
            "incremental_e2e_map50": incremental_base_e2e["map50"],
            "detector_recall": detector_metrics["base_test"]["global"]["recall"],
        },
    }
    return detector_metrics, classifier_metrics, e2e_metrics, summary


def _prediction_payload(
    splits: Mapping[str, LoadedTestSplit], result: FinalInferenceResult
) -> dict[str, Any]:
    payload = {"prediction_version": 1, "splits": {}}
    for split_name in ("base", "incremental"):
        split_result = result.splits[split_name]
        payload["splits"][split_name] = {
            "image_ids": [image.image_id for image in splits[split_name].images],
            "detector": {
                image_id: [
                    {
                        "bbox_xyxy": list(item.bbox_xyxy),
                        "confidence": item.confidence,
                        "class_index": item.class_index,
                    }
                    for item in items
                ]
                for image_id, items in split_result.detector_predictions.items()
            },
            "classifier": {
                name: [item.to_dict() for item in items]
                for name, items in split_result.classifier_predictions.items()
            },
            "end_to_end": {
                name: {
                    image_id: [item.to_dict() for item in items]
                    for image_id, items in predictions.items()
                }
                for name, predictions in split_result.e2e_predictions.items()
            },
            "batch_sizes": {
                name: {kind: list(values) for kind, values in groups.items()}
                for name, groups in split_result.batch_sizes.items()
            },
        }
    return payload


def _report_markdown(summary: Mapping[str, Any]) -> str:
    base = summary["base_test"]
    delta = summary["base_retention_delta"]
    new = summary["incremental_new_classes"]
    return "\n".join(
        (
            "# Bakery Scanner 동결 최종 평가",
            "",
            "이 보고서는 test 접근 전에 동결된 설정을 one-shot으로 실행한 결과입니다.",
            "test 결과를 확인한 뒤 모델, threshold, checkpoint 또는 코드를 변경하지 않았습니다.",
            "",
            "## Base test",
            "",
            f"- Detector Recall@0.5: {base['detector_recall']}",
            f"- Base classifier Top-1: {base['base_classifier_top1']}",
            f"- Incremental classifier Top-1: {base['incremental_classifier_top1']}",
            f"- Base end-to-end mAP50: {base['base_e2e_map50']}",
            f"- Incremental end-to-end mAP50: {base['incremental_e2e_map50']}",
            "",
            "## Base retention delta (Incremental - Base)",
            "",
            *(f"- {key}: {value}" for key, value in delta.items()),
            "",
            "## Incremental test 신규 5개 클래스",
            "",
            *(f"- {key}: {value}" for key, value in new.items()),
            "",
        )
    )


def _validate_completed_output(output_dir: Path) -> None:
    expected = {
        "classifier_metrics.json",
        "detector_metrics.json",
        "e2e_metrics.json",
        "frozen_config.yaml",
        "metadata.json",
        "predictions.json",
        "report.md",
        "summary.json",
    }
    actual = {path.name for path in output_dir.iterdir()}
    if actual != expected:
        raise DataValidationError(
            f"final evaluation files are invalid: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for filename in expected - {"frozen_config.yaml", "report.md"}:
        _coco_payload(output_dir / filename)


def _fail_active_lock(prepared: PreparedFinalEvaluation, error: Exception) -> None:
    if not prepared.lock_path.is_file():
        return
    try:
        payload = json.loads(prepared.lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if isinstance(payload, dict) and payload.get("status") == "started":
        _update_start_lock(prepared, status="failed", error=str(error))


def run_final_evaluation(
    config: FinalEvaluationConfig,
    config_path: str | Path,
    backend: FinalInferenceBackend | None = None,
) -> FinalEvaluationReport:
    preflight = preflight_final_evaluation(config, config_path)
    prepared = preflight.prepared
    selected_backend = backend or FinalInferenceBackend()
    checkpoint_paths = {
        "detector": prepared.detector_checkpoint,
        "base_classifier": prepared.classifier_checkpoints["base"],
        "incremental_classifier": prepared.classifier_checkpoints["incremental"],
    }
    hashes_before = {name: _sha256(path) for name, path in checkpoint_paths.items()}
    staging_dir = prepared.output_dir.parent / f".{prepared.output_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        splits = _load_locked_test_splits(
            config, prepared, loader=_load_test_split
        )
        staging_dir.mkdir(parents=True)
        result = selected_backend.predict(
            splits=splits,
            detector_checkpoint=prepared.detector_checkpoint,
            classifier_checkpoints=prepared.classifier_checkpoints,
            classifier_contexts=prepared.classifier_contexts,
            config=config,
        )
        if _sha256(prepared.config_path) != prepared.config_sha256:
            raise DataValidationError(
                "frozen final evaluation configuration changed during inference"
            )
        hashes_after = {name: _sha256(path) for name, path in checkpoint_paths.items()}
        if hashes_after != hashes_before:
            raise DataValidationError(
                "one or more checkpoints changed during final evaluation"
            )
        detector_metrics, classifier_metrics, e2e_metrics, summary = _metrics(
            config, splits, result
        )
        _write_json(staging_dir / "detector_metrics.json", detector_metrics)
        _write_json(staging_dir / "classifier_metrics.json", classifier_metrics)
        _write_json(staging_dir / "e2e_metrics.json", e2e_metrics)
        _write_json(staging_dir / "summary.json", summary)
        _write_json(staging_dir / "predictions.json", _prediction_payload(splits, result))
        frozen_copy = staging_dir / "frozen_config.yaml"
        shutil.copy2(prepared.config_path, frozen_copy)
        if _sha256(frozen_copy) != prepared.config_sha256:
            raise DataValidationError(
                "published frozen final evaluation configuration SHA-256 is invalid"
            )
        checkpoints = {
            name: {
                "path": str(path),
                "sha256_before": hashes_before[name],
                "sha256_after": hashes_after[name],
                "unchanged": hashes_before[name] == hashes_after[name],
            }
            for name, path in checkpoint_paths.items()
        }
        _write_json(
            staging_dir / "metadata.json",
            {
                "metadata_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_id": config.evaluation_id,
                "selection": {
                    "status": config.selection_status,
                    "basis": config.selection_basis,
                    "frozen_at_utc": config.frozen_at_utc,
                    "configuration_changed_after_test": False,
                },
                "configuration": {
                    "path": str(prepared.config_path),
                    "sha256": prepared.config_sha256,
                },
                "test_splits": {
                    name: {
                        "phase": split.phase,
                        "coco_path": str(split.coco_path),
                        "coco_sha256": _sha256(split.coco_path),
                        "image_count": len(split.images),
                        "annotation_count": sum(
                            len(image.objects) for image in split.images
                        ),
                    }
                    for name, split in splits.items()
                },
                "thresholds": {
                    "confidence_floor": config.detector.confidence_floor,
                    "operating_confidence": config.detector.operating_confidence,
                    "nms_iou": config.detector.nms_iou,
                    "matching_iou": config.detector.matching_iou,
                    "e2e_iou_thresholds": [
                        round(0.5 + 0.05 * index, 2) for index in range(10)
                    ],
                },
                "inference": {
                    "device": config.inference.device,
                    "detector_image_size": config.detector.image_size,
                    "classifier_image_sizes": {
                        name: item.image_size
                        for name, item in config.classifiers.items()
                    },
                    "classifier_batch_strategy": config.inference.classifier_batch_strategy,
                    "combined_score": config.inference.combined_score,
                    "batch_sizes": {
                        split_name: {
                            model_name: {
                                kind: list(values)
                                for kind, values in groups.items()
                            }
                            for model_name, groups in split_result.batch_sizes.items()
                        }
                        for split_name, split_result in result.splits.items()
                    },
                },
                "checkpoints": checkpoints,
                "provenance": dict(prepared.provenance),
                "metric_versions": {
                    "detector": 1,
                    "classifier": 1,
                    "end_to_end": 1,
                    "summary": 1,
                },
                "environment": _environment_metadata(),
            },
        )
        (staging_dir / "report.md").write_text(
            _report_markdown(summary), encoding="utf-8"
        )
        _validate_completed_output(staging_dir)
        final_hashes = {
            name: _sha256(path) for name, path in checkpoint_paths.items()
        }
        if (
            _sha256(prepared.config_path) != prepared.config_sha256
            or _sha256(frozen_copy) != prepared.config_sha256
            or final_hashes != hashes_before
        ):
            raise DataValidationError(
                "frozen configuration or checkpoint changed immediately before publication"
            )
        staging_dir.rename(prepared.output_dir)
        _update_start_lock(prepared, status="completed")
        return FinalEvaluationReport(
            prepared.output_dir,
            prepared.output_dir / "summary.json",
            prepared.output_dir / "metadata.json",
        )
    except Exception as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        _fail_active_lock(prepared, exc)
        raise
