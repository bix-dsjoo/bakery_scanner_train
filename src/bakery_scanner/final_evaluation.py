from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping
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


@dataclass(frozen=True, slots=True)
class FinalEvaluationPreflightReport:
    prepared: PreparedFinalEvaluation
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_path": str(self.prepared.config_path),
            "config_sha256": self.prepared.config_sha256,
            "evaluation_id": self.prepared.output_dir.name,
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
    )


def preflight_final_evaluation(
    config: FinalEvaluationConfig,
    config_path: str | Path,
    *,
    cuda_available: Callable[[str], bool] | None = None,
) -> FinalEvaluationPreflightReport:
    if not isinstance(config, FinalEvaluationConfig):
        raise DataValidationError("config must be FinalEvaluationConfig")
    prepared = _prepare_non_test_inputs(
        config,
        Path(config_path),
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
