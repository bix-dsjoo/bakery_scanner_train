from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import re
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import yaml

from .classifier_dataset import validate_classifier_dataset
from .classifier_evaluation import (
    ClassifierPrediction,
    evaluate_classifier_predictions,
)
from .errors import DataValidationError
from .registry import load_class_registry
from .safety import assert_training_paths_safe

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_FIELD_ORDER = (
    "dataset_root",
    "source_classifier_run",
    "output_root",
    "run_name",
    "architecture",
    "pretrained_model",
    "image_size",
    "epochs",
    "batch_size",
    "seed",
    "device",
    "patience",
    "workers",
    "learning_rate",
    "weight_decay",
)
_CONFIG_FIELDS = set(_CONFIG_FIELD_ORDER)
_INCREMENTAL_CONFIG_FIELD_ORDER = (
    "phase",
    "dataset_root",
    "source_classifier_run",
    "output_root",
    "run_name",
    "architecture",
    "base_checkpoint",
    "frozen_detector_checkpoint",
    "image_size",
    "epochs",
    "batch_size",
    "seed",
    "device",
    "patience",
    "workers",
    "learning_rate",
    "weight_decay",
)
_INCREMENTAL_CONFIG_FIELDS = set(_INCREMENTAL_CONFIG_FIELD_ORDER)
_PREPROCESSING_METADATA = {
    "train": [
        "RandomResizedCrop(scale=[0.8,1.0])",
        "RandomHorizontalFlip",
        "ColorJitter(0.1,0.1,0.1,0.05)",
        "ToTensor",
        "ImageNetNormalize",
    ],
    "validation": [
        "Resize(image_size/0.875)",
        "CenterCrop(image_size)",
        "ToTensor",
        "ImageNetNormalize",
    ],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass(frozen=True, slots=True)
class ClassifierTrainingConfig:
    dataset_root: str
    source_classifier_run: str
    output_root: str
    run_name: str
    architecture: str
    pretrained_model: str
    image_size: int
    epochs: int
    batch_size: int
    seed: int
    device: str
    patience: int
    workers: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class IncrementalClassifierTrainingConfig:
    phase: str
    dataset_root: str
    source_classifier_run: str
    output_root: str
    run_name: str
    architecture: str
    base_checkpoint: str
    frozen_detector_checkpoint: str
    image_size: int
    epochs: int
    batch_size: int
    seed: int
    device: str
    patience: int
    workers: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class ClassifierSample:
    sample_id: str
    image_path: Path
    target_index: int
    split: str


@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    epochs_completed: int
    history: tuple[Mapping[str, Any], ...]


class ClassifierBackend(Protocol):
    def cuda_available(self, device: str) -> bool: ...

    def train(
        self,
        *,
        pretrained_model: Path,
        train_samples: Sequence[ClassifierSample],
        validation_samples: Sequence[ClassifierSample],
        output_dimension: int,
        checkpoint_context: Mapping[str, Any],
        output_dir: Path,
        arguments: Mapping[str, object],
    ) -> BackendTrainingResult: ...

    def predict(
        self,
        *,
        checkpoint: Path,
        samples: Sequence[ClassifierSample],
        output_dimension: int,
        checkpoint_context: Mapping[str, Any],
        arguments: Mapping[str, object],
    ) -> Sequence[ClassifierPrediction]: ...


@dataclass(frozen=True, slots=True)
class ClassifierTrainingReport:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path
    history_path: Path
    split: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        metrics_payload = _json_object(self.metrics_path, "classifier metrics")
        return {
            "status": "ok",
            "split": self.split,
            "output_dir": str(self.output_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint),
            "metadata_path": str(self.metadata_path),
            "predictions_path": str(self.predictions_path),
            "metrics_path": str(self.metrics_path),
            "history_path": str(self.history_path),
            "metrics": metrics_payload["metrics"],
        }


@dataclass(frozen=True, slots=True)
class ClassifierEvaluationReport:
    output_dir: Path
    checkpoint: Path
    predictions_path: Path
    metrics_path: Path
    split: str = "validation"

    def to_dict(self) -> dict[str, Any]:
        metrics_payload = _json_object(self.metrics_path, "classifier metrics")
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


def _run_name(value: object, label: str) -> str:
    result = _text(value, label)
    if not _RUN_NAME.fullmatch(result):
        raise DataValidationError(f"{label} is invalid: {result!r}")
    return result


def _integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return value


def _number(value: object, label: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DataValidationError(f"{label} must be a finite number")
    result = float(value)
    if result < 0 if allow_zero else result <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return result


def load_classifier_training_config(path: str | Path) -> ClassifierTrainingConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(
            f"cannot load classifier config {config_path}: {exc}"
        ) from exc
    payload = _strict_object(payload, _CONFIG_FIELDS, "classifier config")
    architecture = _text(payload["architecture"], "architecture")
    if architecture != "resnet18":
        raise DataValidationError("architecture must be resnet18 for this baseline")
    device = _text(payload["device"], "device")
    if device != "0":
        raise DataValidationError("device must be CUDA device '0' for this baseline")
    return ClassifierTrainingConfig(
        dataset_root=_text(payload["dataset_root"], "dataset_root"),
        source_classifier_run=_run_name(
            payload["source_classifier_run"], "source_classifier_run"
        ),
        output_root=_text(payload["output_root"], "output_root"),
        run_name=_run_name(payload["run_name"], "run_name"),
        architecture=architecture,
        pretrained_model=_text(payload["pretrained_model"], "pretrained_model"),
        image_size=_integer(payload["image_size"], "image_size"),
        epochs=_integer(payload["epochs"], "epochs"),
        batch_size=_integer(payload["batch_size"], "batch_size"),
        seed=_integer(payload["seed"], "seed", allow_zero=True),
        device=device,
        patience=_integer(payload["patience"], "patience", allow_zero=True),
        workers=_integer(payload["workers"], "workers", allow_zero=True),
        learning_rate=_number(payload["learning_rate"], "learning_rate"),
        weight_decay=_number(
            payload["weight_decay"], "weight_decay", allow_zero=True
        ),
    )


def load_classifier_experiment_config(
    path: str | Path,
) -> ClassifierTrainingConfig | IncrementalClassifierTrainingConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(
            f"cannot load classifier config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or "phase" not in payload:
        return load_classifier_training_config(config_path)
    payload = _strict_object(
        payload,
        _INCREMENTAL_CONFIG_FIELDS,
        "incremental classifier config",
    )
    if payload["phase"] != "incremental":
        raise DataValidationError("incremental classifier phase must be incremental")
    architecture = _text(payload["architecture"], "architecture")
    if architecture != "resnet18":
        raise DataValidationError("architecture must be resnet18 for this baseline")
    device = _text(payload["device"], "device")
    if device != "0":
        raise DataValidationError("device must be CUDA device '0' for this baseline")
    return IncrementalClassifierTrainingConfig(
        phase="incremental",
        dataset_root=_text(payload["dataset_root"], "dataset_root"),
        source_classifier_run=_run_name(
            payload["source_classifier_run"], "source_classifier_run"
        ),
        output_root=_text(payload["output_root"], "output_root"),
        run_name=_run_name(payload["run_name"], "run_name"),
        architecture=architecture,
        base_checkpoint=_text(payload["base_checkpoint"], "base_checkpoint"),
        frozen_detector_checkpoint=_text(
            payload["frozen_detector_checkpoint"],
            "frozen_detector_checkpoint",
        ),
        image_size=_integer(payload["image_size"], "image_size"),
        epochs=_integer(payload["epochs"], "epochs"),
        batch_size=_integer(payload["batch_size"], "batch_size"),
        seed=_integer(payload["seed"], "seed", allow_zero=True),
        device=device,
        patience=_integer(payload["patience"], "patience", allow_zero=True),
        workers=_integer(payload["workers"], "workers", allow_zero=True),
        learning_rate=_number(payload["learning_rate"], "learning_rate"),
        weight_decay=_number(
            payload["weight_decay"], "weight_decay", allow_zero=True
        ),
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


def _config_payload(config: ClassifierTrainingConfig) -> dict[str, Any]:
    return {field: getattr(config, field) for field in _CONFIG_FIELD_ORDER}


def _environment_metadata() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in ("torch", "torchvision", "Pillow", "PyYAML"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = "unavailable"
    hardware: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import torch

        hardware.update(
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except (ImportError, RuntimeError):
        hardware.update(
            {"cuda_available": False, "cuda_version": None, "gpu": None}
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "dependencies": dependencies,
        "hardware": hardware,
    }


def _load_samples(dataset_root: Path, manifest_path: Path) -> tuple[ClassifierSample, ...]:
    payload = _json_object(manifest_path, "classifier manifest")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise DataValidationError("classifier manifest samples must be a list")
    samples: list[ClassifierSample] = []
    for position, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise DataValidationError(f"classifier sample {position} must be an object")
        try:
            output_path = raw["output_path"]
            target_index = raw["model_index"]
            split = raw["split"]
        except KeyError as exc:
            raise DataValidationError(
                f"classifier sample {position} is missing {exc.args[0]}"
            ) from exc
        if not isinstance(output_path, str) or not output_path:
            raise DataValidationError("classifier sample output_path must be text")
        if isinstance(target_index, bool) or not isinstance(target_index, int):
            raise DataValidationError("classifier sample model_index must be an integer")
        if split not in {"train", "validation"}:
            raise DataValidationError("classifier sample split is invalid")
        image_path = (dataset_root / output_path).resolve(strict=False)
        if not image_path.is_file():
            raise DataValidationError(f"classifier sample image is missing: {image_path}")
        samples.append(
            ClassifierSample(output_path, image_path, target_index, str(split))
        )
    return tuple(samples)


def _backend_arguments(config: ClassifierTrainingConfig) -> dict[str, object]:
    return {
        "architecture": config.architecture,
        "image_size": config.image_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "device": config.device,
        "patience": config.patience,
        "workers": config.workers,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }


def _checkpoint_context(
    config: ClassifierTrainingConfig,
    dataset_root: Path,
    manifest_path: Path,
    output_dimension: int,
) -> dict[str, Any]:
    manifest = _json_object(manifest_path, "classifier manifest")
    registry_payload = manifest.get("registry")
    if not isinstance(registry_payload, dict):
        raise DataValidationError("classifier manifest registry must be an object")
    registry_sha256 = registry_payload.get("sha256")
    if not isinstance(registry_sha256, str) or len(registry_sha256) != 64:
        raise DataValidationError("classifier manifest registry SHA-256 is invalid")
    registry = load_class_registry(dataset_root / "class_registry.json")
    mapping = []
    for model_index in range(output_dimension):
        record = registry.by_model_index.get(model_index)
        if record is None:
            raise DataValidationError(
                f"class registry is missing model_index {model_index}"
            )
        mapping.append(
            {
                "model_index": record.model_index,
                "category_id": record.category_id,
                "canonical_name": record.canonical_name,
            }
        )
    return {
        "context_version": 1,
        "source_manifest_sha256": _sha256(manifest_path),
        "registry_sha256": registry_sha256,
        "model_index_mapping": mapping,
        "config": _config_payload(config),
    }


def _write_evaluation(
    *,
    backend: ClassifierBackend,
    checkpoint: Path,
    samples: Sequence[ClassifierSample],
    output_dimension: int,
    checkpoint_context: Mapping[str, Any],
    arguments: Mapping[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    predictions = tuple(
        backend.predict(
            checkpoint=checkpoint,
            samples=samples,
            output_dimension=output_dimension,
            checkpoint_context=checkpoint_context,
            arguments=arguments,
        )
    )
    if len(predictions) != len(samples):
        raise DataValidationError(
            "classifier backend prediction count does not match validation samples"
        )
    expected = [(sample.sample_id, sample.target_index) for sample in samples]
    actual = [
        (prediction.sample_id, prediction.target_index) for prediction in predictions
    ]
    if actual != expected:
        raise DataValidationError(
            "classifier backend prediction sample IDs or targets do not match validation samples"
        )
    metrics = evaluate_classifier_predictions(
        predictions, output_dimension=output_dimension
    )
    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "output_dimension": output_dimension,
                "predictions": [prediction.to_dict() for prediction in predictions],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "split": "validation",
                "metric_version": 1,
                "checkpoint_sha256": _sha256(checkpoint),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return predictions_path, metrics_path


def _validate_completed_run(output_dir: Path) -> None:
    required = {
        "checkpoints",
        "config.yaml",
        "history.json",
        "metadata.json",
        "metrics.json",
        "predictions.json",
    }
    actual = {path.name for path in output_dir.iterdir()}
    if actual != required:
        raise DataValidationError(
            f"completed classifier run files are invalid: expected={sorted(required)}, actual={sorted(actual)}"
        )
    for filename in ("history.json", "metadata.json", "metrics.json", "predictions.json"):
        _json_object(output_dir / filename, filename)
    checkpoints = output_dir / "checkpoints"
    if {path.name for path in checkpoints.iterdir()} != {"best.pt", "last.pt"}:
        raise DataValidationError("classifier checkpoints are incomplete")


def train_classifier(
    config: ClassifierTrainingConfig,
    backend: ClassifierBackend | None = None,
) -> ClassifierTrainingReport:
    if not isinstance(config, ClassifierTrainingConfig):
        raise DataValidationError("config must be ClassifierTrainingConfig")
    selected_backend = backend or TorchvisionClassifierBackend()
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    source_dir = (
        dataset_root / "derived" / "classifier" / config.source_classifier_run
    )
    registry_path = dataset_root / "class_registry.json"
    output_root = Path(config.output_root).resolve(strict=False)
    pretrained_model = Path(config.pretrained_model).resolve(strict=False)
    assert_training_paths_safe(
        [source_dir, registry_path, output_root, pretrained_model], dataset_root
    )

    dataset_report = validate_classifier_dataset(
        dataset_root, config.source_classifier_run
    )
    if dataset_report.phase != "base" or dataset_report.output_dimension != 15:
        raise DataValidationError(
            "Base classifier training requires a Base dataset with 15 outputs"
        )
    if not pretrained_model.is_file():
        raise DataValidationError(
            f"classifier pretrained model is missing: {pretrained_model}"
        )
    if not selected_backend.cuda_available(config.device):
        raise DataValidationError(f"CUDA device {config.device} is unavailable")

    samples = _load_samples(dataset_root, dataset_report.manifest_path)
    checkpoint_context = _checkpoint_context(
        config,
        dataset_root,
        dataset_report.manifest_path,
        dataset_report.output_dimension,
    )
    train_samples = tuple(sample for sample in samples if sample.split == "train")
    validation_samples = tuple(
        sample for sample in samples if sample.split == "validation"
    )
    if not train_samples or not validation_samples:
        raise DataValidationError("classifier train and validation splits must be non-empty")

    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise DataValidationError(f"classifier training run already exists: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".{config.run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        arguments = _backend_arguments(config)
        backend_result = selected_backend.train(
            pretrained_model=pretrained_model,
            train_samples=train_samples,
            validation_samples=validation_samples,
            output_dimension=dataset_report.output_dimension,
            checkpoint_context=checkpoint_context,
            output_dir=staging_dir / "backend",
            arguments=arguments,
        )
        if (
            not backend_result.best_checkpoint.is_file()
            or not backend_result.last_checkpoint.is_file()
        ):
            raise DataValidationError("classifier backend checkpoints are missing")
        checkpoints_dir = staging_dir / "checkpoints"
        checkpoints_dir.mkdir()
        best_checkpoint = checkpoints_dir / "best.pt"
        last_checkpoint = checkpoints_dir / "last.pt"
        shutil.copy2(backend_result.best_checkpoint, best_checkpoint)
        shutil.copy2(backend_result.last_checkpoint, last_checkpoint)
        shutil.rmtree(staging_dir / "backend")

        predictions_path, metrics_path = _write_evaluation(
            backend=selected_backend,
            checkpoint=best_checkpoint,
            samples=validation_samples,
            output_dimension=dataset_report.output_dimension,
            checkpoint_context=checkpoint_context,
            arguments=arguments,
            output_dir=staging_dir,
        )
        history_path = staging_dir / "history.json"
        history_path.write_text(
            json.dumps(
                {
                    "best_epoch": backend_result.best_epoch,
                    "epochs_completed": backend_result.epochs_completed,
                    "selection_metric": "validation_loss",
                    "history": list(backend_result.history),
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
                    "dataset": {
                        "manifest_path": str(dataset_report.manifest_path),
                        "manifest_sha256": _sha256(dataset_report.manifest_path),
                        "registry_sha256": checkpoint_context["registry_sha256"],
                        "model_index_mapping": checkpoint_context[
                            "model_index_mapping"
                        ],
                        "source_run": config.source_classifier_run,
                        "phase": dataset_report.phase,
                        "output_dimension": dataset_report.output_dimension,
                        "train_samples": len(train_samples),
                        "validation_samples": len(validation_samples),
                    },
                    "model": {
                        "architecture": config.architecture,
                        "pretrained_path": str(pretrained_model),
                        "pretrained_sha256": _sha256(pretrained_model),
                        "best_sha256": _sha256(best_checkpoint),
                        "last_sha256": _sha256(last_checkpoint),
                    },
                    "training": {
                        "best_epoch": backend_result.best_epoch,
                        "epochs_completed": backend_result.epochs_completed,
                        "selection_metric": "validation_loss",
                    },
                    "environment": _environment_metadata(),
                    "determinism": {
                        "python_seeded": True,
                        "torch_seeded": True,
                        "cuda_seeded": True,
                        "cudnn_benchmark": False,
                        "cudnn_deterministic": True,
                        "torch_deterministic_algorithms": False,
                    },
                    "preprocessing": _PREPROCESSING_METADATA,
                    "backend_arguments": arguments,
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
        return ClassifierTrainingReport(
            output_dir=output_dir,
            best_checkpoint=output_dir / "checkpoints" / "best.pt",
            last_checkpoint=output_dir / "checkpoints" / "last.pt",
            metadata_path=output_dir / metadata_path.name,
            predictions_path=output_dir / predictions_path.name,
            metrics_path=output_dir / metrics_path.name,
            history_path=output_dir / history_path.name,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def evaluate_classifier_checkpoint(
    config: ClassifierTrainingConfig,
    checkpoint: str | Path,
    backend: ClassifierBackend | None = None,
    output_dir: str | Path | None = None,
) -> ClassifierEvaluationReport:
    if not isinstance(config, ClassifierTrainingConfig):
        raise DataValidationError("config must be ClassifierTrainingConfig")
    selected_backend = backend or TorchvisionClassifierBackend()
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    source_dir = (
        dataset_root / "derived" / "classifier" / config.source_classifier_run
    )
    registry_path = dataset_root / "class_registry.json"
    checkpoint_path = Path(checkpoint).resolve(strict=False)
    target_dir = (
        Path(output_dir).resolve(strict=False)
        if output_dir is not None
        else checkpoint_path.parent.parent / "evaluation"
    )
    assert_training_paths_safe(
        [source_dir, registry_path, checkpoint_path, target_dir], dataset_root
    )
    dataset_report = validate_classifier_dataset(
        dataset_root, config.source_classifier_run
    )
    if dataset_report.phase != "base" or dataset_report.output_dimension != 15:
        raise DataValidationError(
            "Base classifier evaluation requires a Base dataset with 15 outputs"
        )
    if not checkpoint_path.is_file():
        raise DataValidationError(
            f"classifier checkpoint is missing: {checkpoint_path}"
        )
    if not selected_backend.cuda_available(config.device):
        raise DataValidationError(f"CUDA device {config.device} is unavailable")
    samples = _load_samples(dataset_root, dataset_report.manifest_path)
    checkpoint_context = _checkpoint_context(
        config,
        dataset_root,
        dataset_report.manifest_path,
        dataset_report.output_dimension,
    )
    validation_samples = tuple(
        sample for sample in samples if sample.split == "validation"
    )
    if not validation_samples:
        raise DataValidationError("classifier validation split must be non-empty")
    if target_dir.exists():
        raise DataValidationError(
            f"classifier evaluation output already exists: {target_dir}"
        )
    target_dir.mkdir(parents=True)
    try:
        predictions_path, metrics_path = _write_evaluation(
            backend=selected_backend,
            checkpoint=checkpoint_path,
            samples=validation_samples,
            output_dimension=dataset_report.output_dimension,
            checkpoint_context=checkpoint_context,
            arguments=_backend_arguments(config),
            output_dir=target_dir,
        )
        return ClassifierEvaluationReport(
            output_dir=target_dir,
            checkpoint=checkpoint_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
        )
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise


def _build_resnet18(pretrained_model: Path, output_dimension: int):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    if output_dimension <= 0:
        raise DataValidationError("output_dimension must be positive")
    model = resnet18(weights=None)
    try:
        state_dict = torch.load(
            pretrained_model, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict, strict=True)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise DataValidationError(
            f"cannot strict-load ResNet18 pretrained state {pretrained_model}: {exc}"
        ) from exc
    model.fc = nn.Linear(model.fc.in_features, output_dimension)
    return model


class _ManifestImageDataset:
    def __init__(self, samples: Sequence[ClassifierSample], transform) -> None:
        self.samples = tuple(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image, UnidentifiedImageError

        sample = self.samples[index]
        try:
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise DataValidationError(
                f"cannot load classifier image {sample.image_path}: {exc}"
            ) from exc
        return self.transform(image), sample.target_index, sample.sample_id


def _classifier_transforms(image_size: int):
    from torchvision import transforms

    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, validation_transform


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    try:
        import numpy

        numpy.random.seed(worker_seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(worker_seed)


def _torch_device(value: object):
    import torch

    if value == "cpu":
        return torch.device("cpu")
    if value == "0":
        return torch.device("cuda:0")
    raise DataValidationError(f"unsupported classifier device: {value!r}")


def _checkpoint_payload(
    *,
    model,
    optimizer,
    output_dimension: int,
    image_size: int,
    epoch: int,
    validation_loss: float,
    checkpoint_context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "architecture": "resnet18",
        "output_dimension": output_dimension,
        "image_size": image_size,
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "context": dict(checkpoint_context),
    }


def _load_classifier_checkpoint_model(
    checkpoint: Path,
    *,
    output_dimension: int,
    checkpoint_context: Mapping[str, Any],
    image_size: int,
    device,
):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise DataValidationError(
            f"cannot load classifier checkpoint {checkpoint}: {exc}"
        ) from exc
    required = {
        "checkpoint_version",
        "architecture",
        "output_dimension",
        "image_size",
        "epoch",
        "validation_loss",
        "model_state_dict",
        "optimizer_state_dict",
        "context",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DataValidationError("classifier checkpoint schema is invalid")
    if (
        payload["checkpoint_version"] != 1
        or payload["architecture"] != "resnet18"
        or payload["output_dimension"] != output_dimension
        or payload["image_size"] != image_size
        or payload["context"] != dict(checkpoint_context)
    ):
        raise DataValidationError(
            "classifier checkpoint context does not match evaluation configuration"
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, output_dimension)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise DataValidationError(
            f"cannot strict-load classifier checkpoint: {exc}"
        ) from exc
    return model.to(device).eval()


class TorchvisionClassifierBackend:
    def cuda_available(self, device: str) -> bool:
        import torch

        return device == "0" and torch.cuda.is_available()

    def train(
        self,
        *,
        pretrained_model: Path,
        train_samples: Sequence[ClassifierSample],
        validation_samples: Sequence[ClassifierSample],
        output_dimension: int,
        checkpoint_context: Mapping[str, Any],
        output_dir: Path,
        arguments: Mapping[str, object],
    ) -> BackendTrainingResult:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        if arguments.get("architecture") != "resnet18":
            raise DataValidationError("classifier architecture must be resnet18")
        image_size = _argument_integer(arguments, "image_size")
        epochs = _argument_integer(arguments, "epochs")
        batch_size = _argument_integer(arguments, "batch_size")
        seed = _argument_integer(arguments, "seed", allow_zero=True)
        patience = _argument_integer(arguments, "patience", allow_zero=True)
        workers = _argument_integer(arguments, "workers", allow_zero=True)
        learning_rate = _argument_number(arguments, "learning_rate")
        weight_decay = _argument_number(
            arguments, "weight_decay", allow_zero=True
        )
        device = _torch_device(arguments.get("device"))

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        train_transform, validation_transform = _classifier_transforms(image_size)
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            _ManifestImageDataset(train_samples, train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
            worker_init_fn=partial(_seed_worker, base_seed=seed),
            pin_memory=device.type == "cuda",
        )
        validation_loader = DataLoader(
            _ManifestImageDataset(validation_samples, validation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=partial(_seed_worker, base_seed=seed),
            pin_memory=device.type == "cuda",
        )
        if not len(train_loader) or not len(validation_loader):
            raise DataValidationError("classifier backend received an empty split")

        model = _build_resnet18(pretrained_model, output_dimension).to(device)
        counts = torch.zeros(output_dimension, dtype=torch.float32)
        for sample in train_samples:
            if not 0 <= sample.target_index < output_dimension:
                raise DataValidationError(
                    f"classifier target is outside output dimension: {sample.target_index}"
                )
            counts[sample.target_index] += 1
        weights = torch.zeros_like(counts)
        present = counts > 0
        weights[present] = len(train_samples) / (output_dimension * counts[present])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        output_dir.mkdir(parents=True, exist_ok=False)
        best_checkpoint = output_dir / "best.pt"
        last_checkpoint = output_dir / "last.pt"
        best_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            for images, targets, _sample_ids in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                count = targets.shape[0]
                train_loss_sum += float(loss.detach()) * count
                train_correct += int((logits.argmax(dim=1) == targets).sum())
                train_count += count

            model.eval()
            validation_loss_sum = 0.0
            validation_correct = 0
            validation_count = 0
            validation_predictions: list[ClassifierPrediction] = []
            with torch.inference_mode():
                for images, targets, sample_ids in validation_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits = model(images)
                    loss = criterion(logits, targets)
                    count = targets.shape[0]
                    validation_loss_sum += float(loss) * count
                    validation_correct += int(
                        (logits.argmax(dim=1) == targets).sum()
                    )
                    validation_count += count
                    probabilities = logits.softmax(dim=1)
                    confidences, predicted_indices = probabilities.max(dim=1)
                    for sample_id, target, predicted, confidence in zip(
                        sample_ids,
                        targets.cpu().tolist(),
                        predicted_indices.cpu().tolist(),
                        confidences.cpu().tolist(),
                        strict=True,
                    ):
                        validation_predictions.append(
                            ClassifierPrediction(
                                sample_id=sample_id,
                                target_index=target,
                                predicted_index=predicted,
                                confidence=confidence,
                            )
                        )
            validation_loss = validation_loss_sum / validation_count
            validation_metrics = evaluate_classifier_predictions(
                validation_predictions, output_dimension=output_dimension
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_sum / train_count,
                    "train_top1_accuracy": train_correct / train_count,
                    "validation_loss": validation_loss,
                    "validation_top1_accuracy": (
                        validation_correct / validation_count
                    ),
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_evaluated_class_count": validation_metrics[
                        "evaluated_class_count"
                    ],
                }
            )
            checkpoint = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                output_dimension=output_dimension,
                image_size=image_size,
                epoch=epoch,
                validation_loss=validation_loss,
                checkpoint_context=checkpoint_context,
            )
            torch.save(checkpoint, last_checkpoint)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(checkpoint, best_checkpoint)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_epoch == 0 or not best_checkpoint.is_file():
            raise DataValidationError("classifier training did not select a checkpoint")
        return BackendTrainingResult(
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            best_epoch=best_epoch,
            epochs_completed=len(history),
            history=tuple(history),
        )

    def predict(
        self,
        *,
        checkpoint: Path,
        samples: Sequence[ClassifierSample],
        output_dimension: int,
        checkpoint_context: Mapping[str, Any],
        arguments: Mapping[str, object],
    ) -> Sequence[ClassifierPrediction]:
        import torch
        from torch.utils.data import DataLoader

        device = _torch_device(arguments.get("device"))
        image_size = _argument_integer(arguments, "image_size")
        batch_size = _argument_integer(arguments, "batch_size")
        workers = _argument_integer(arguments, "workers", allow_zero=True)
        model = _load_classifier_checkpoint_model(
            checkpoint,
            output_dimension=output_dimension,
            checkpoint_context=checkpoint_context,
            image_size=image_size,
            device=device,
        )
        _train_transform, validation_transform = _classifier_transforms(image_size)
        loader = DataLoader(
            _ManifestImageDataset(samples, validation_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        predictions: list[ClassifierPrediction] = []
        with torch.inference_mode():
            for images, targets, sample_ids in loader:
                probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1)
                confidences, indices = probabilities.max(dim=1)
                for sample_id, target, predicted, confidence in zip(
                    sample_ids,
                    targets.tolist(),
                    indices.cpu().tolist(),
                    confidences.cpu().tolist(),
                    strict=True,
                ):
                    predictions.append(
                        ClassifierPrediction(
                            sample_id=sample_id,
                            target_index=target,
                            predicted_index=predicted,
                            confidence=confidence,
                        )
                    )
        return tuple(predictions)


def _argument_integer(
    arguments: Mapping[str, object], label: str, *, allow_zero: bool = False
) -> int:
    return _integer(arguments.get(label), label, allow_zero=allow_zero)


def _argument_number(
    arguments: Mapping[str, object], label: str, *, allow_zero: bool = False
) -> float:
    return _number(arguments.get(label), label, allow_zero=allow_zero)
