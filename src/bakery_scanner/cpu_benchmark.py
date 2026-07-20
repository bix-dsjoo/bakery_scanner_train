from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import yaml

from .errors import DataValidationError
from .safety import assert_training_paths_safe

BENCHMARK_STAGES = (
    "detector",
    "crop_preprocess",
    "classifier_batch",
    "postprocess",
    "end_to_end",
)

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_FIELD_ORDER = (
    "dataset_root",
    "detector_config",
    "classifier_config",
    "detector_checkpoint",
    "classifier_checkpoint",
    "output_root",
    "run_name",
    "warmup_iterations",
    "repetitions",
    "intra_op_threads",
    "inter_op_threads",
)
_CONFIG_FIELDS = set(_CONFIG_FIELD_ORDER)


@dataclass(frozen=True, slots=True)
class CpuBenchmarkConfig:
    dataset_root: str
    detector_config: str
    classifier_config: str
    detector_checkpoint: str
    classifier_checkpoint: str
    output_root: str
    run_name: str
    warmup_iterations: int
    repetitions: int
    intra_op_threads: int
    inter_op_threads: int


@dataclass(frozen=True, slots=True)
class CpuBackendResult:
    stage_samples_ms: Mapping[str, Sequence[float]]
    measured_image_ids: Sequence[str]
    classifier_batch_sizes: Sequence[int]
    warmup_invocation_count: int
    runtime: str
    execution_provider: str
    device: str
    intra_op_threads: int
    inter_op_threads: int


class CpuBenchmarkBackend(Protocol):
    def run(
        self,
        *,
        image_paths: Sequence[Path],
        image_ids: Sequence[str],
        detector_checkpoint: Path,
        classifier_checkpoint: Path,
        classifier_context: Mapping[str, Any],
        output_dimension: int,
        detector_image_size: int,
        classifier_image_size: int,
        detector_confidence: float,
        detector_nms_iou: float,
        warmup_iterations: int,
        repetitions: int,
        intra_op_threads: int,
        inter_op_threads: int,
    ) -> CpuBackendResult: ...


@dataclass(frozen=True, slots=True)
class PreparedCpuBenchmark:
    dataset_root: Path
    detector_config_path: Path
    classifier_config_path: Path
    detector_checkpoint: Path
    classifier_checkpoint: Path
    detector_metadata_path: Path
    classifier_metadata_path: Path
    detector_manifest_path: Path
    classifier_manifest_path: Path
    image_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    classifier_context: Mapping[str, Any]
    output_dimension: int
    detector_image_size: int
    classifier_image_size: int
    detector_confidence: float
    detector_nms_iou: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CpuBenchmarkReport:
    output_dir: Path
    config_path: Path
    benchmark_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = _json_object(self.benchmark_path, "CPU benchmark result")
        return {
            "status": "ok",
            "output_dir": str(self.output_dir),
            "config_path": str(self.config_path),
            "benchmark_path": str(self.benchmark_path),
            "metadata_path": str(self.metadata_path),
            "benchmark": payload,
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


def _integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return value


def load_cpu_benchmark_config(path: str | Path) -> CpuBenchmarkConfig:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(
            f"cannot load CPU benchmark config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _CONFIG_FIELDS:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise DataValidationError(
            "CPU benchmark config fields are invalid: "
            f"expected={sorted(_CONFIG_FIELDS)}, actual={actual}"
        )
    return CpuBenchmarkConfig(
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
        warmup_iterations=_integer(
            payload["warmup_iterations"], "warmup_iterations", allow_zero=True
        ),
        repetitions=_integer(payload["repetitions"], "repetitions"),
        intra_op_threads=_integer(payload["intra_op_threads"], "intra_op_threads"),
        inter_op_threads=_integer(
            payload["inter_op_threads"], "inter_op_threads"
        ),
    )


def _config_payload(config: CpuBenchmarkConfig) -> dict[str, Any]:
    return {field: getattr(config, field) for field in _CONFIG_FIELD_ORDER}


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


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise DataValidationError("timing samples must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise DataValidationError("percentile quantile must be between zero and one")
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in samples)
    if not values:
        raise DataValidationError("timing samples must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise DataValidationError("timing samples must be finite non-negative numbers")
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
    }


def _initial_paths(config: CpuBenchmarkConfig) -> tuple[Path, ...]:
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    detector_checkpoint = Path(config.detector_checkpoint).resolve(strict=False)
    classifier_checkpoint = Path(config.classifier_checkpoint).resolve(strict=False)
    return (
        dataset_root / "class_registry.json",
        Path(config.detector_config).resolve(strict=False),
        Path(config.classifier_config).resolve(strict=False),
        detector_checkpoint,
        classifier_checkpoint,
        detector_checkpoint.parent.parent / "metadata.json",
        detector_checkpoint.parent.parent / "config.yaml",
        classifier_checkpoint.parent.parent / "metadata.json",
        Path(config.output_root).resolve(strict=False),
    )


def _load_benchmark_images(
    detector_dir: Path, manifest_path: Path
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    payload = _json_object(manifest_path, "detector manifest")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise DataValidationError("detector manifest samples must be a list")
    image_ids: list[str] = []
    image_paths: list[Path] = []
    for position, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise DataValidationError(f"detector sample {position} must be an object")
        if raw.get("split") != "validation":
            continue
        image_id = raw.get("sample_id")
        output_path = raw.get("output_path")
        if not isinstance(image_id, str) or not image_id:
            raise DataValidationError("detector validation sample_id is invalid")
        if not isinstance(output_path, str) or not output_path:
            raise DataValidationError("detector validation output_path is invalid")
        image_path = (detector_dir / output_path).resolve(strict=False)
        if not image_path.is_file():
            raise DataValidationError(f"benchmark image is missing: {image_path}")
        if image_id in image_ids:
            raise DataValidationError(f"duplicate benchmark image ID: {image_id}")
        image_ids.append(image_id)
        image_paths.append(image_path)
    if not image_ids:
        raise DataValidationError("detector validation split must contain images")
    return tuple(image_ids), tuple(image_paths)


def _validate_classifier_checkpoint_provenance(
    *,
    config,
    checkpoint: Path,
    metadata: Mapping[str, Any],
    classifier_manifest_path: Path,
    classifier_context: Mapping[str, Any],
    detector_checkpoint: Path,
) -> None:
    expected_checkpoint = (
        Path(config.output_root) / config.run_name / "checkpoints" / "best.pt"
    ).resolve(strict=False)
    if checkpoint != expected_checkpoint:
        raise DataValidationError(
            "classifier checkpoint path does not match the selected classifier run"
        )
    dataset = metadata.get("dataset")
    model = metadata.get("model")
    frozen_detector = metadata.get("frozen_detector")
    if (
        not isinstance(dataset, dict)
        or dataset.get("phase") != "incremental"
        or dataset.get("output_dimension") != 20
        or dataset.get("manifest_path") != str(classifier_manifest_path)
        or dataset.get("manifest_sha256") != _sha256(classifier_manifest_path)
        or dataset.get("registry_sha256") != classifier_context.get("registry_sha256")
        or dataset.get("model_index_mapping")
        != classifier_context.get("model_index_mapping")
    ):
        raise DataValidationError(
            "classifier metadata dataset does not match the selected Incremental run"
        )
    if not isinstance(model, dict) or model.get("architecture") != "resnet18":
        raise DataValidationError("classifier metadata must describe ResNet18")
    detector_hash = _sha256(detector_checkpoint)
    if (
        not isinstance(frozen_detector, dict)
        or Path(str(frozen_detector.get("checkpoint"))).resolve(strict=False)
        != detector_checkpoint
        or frozen_detector.get("sha256_before") != detector_hash
        or frozen_detector.get("sha256_after") != detector_hash
        or frozen_detector.get("detector_unchanged") is not True
    ):
        raise DataValidationError(
            "classifier metadata frozen detector does not match the selected detector"
        )


def _prepare_cpu_benchmark(config: CpuBenchmarkConfig) -> PreparedCpuBenchmark:
    from .classifier_dataset import validate_classifier_dataset
    from .classifier_training import (
        IncrementalClassifierTrainingConfig,
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
    from .yolo_dataset import validate_yolo_dataset

    dataset_root = Path(config.dataset_root).resolve(strict=False)
    detector_config_path = Path(config.detector_config).resolve(strict=False)
    classifier_config_path = Path(config.classifier_config).resolve(strict=False)
    detector_checkpoint = Path(config.detector_checkpoint).resolve(strict=False)
    classifier_checkpoint = Path(config.classifier_checkpoint).resolve(strict=False)
    detector_run_config_path = detector_checkpoint.parent.parent / "config.yaml"

    detector_config = load_detector_training_config(detector_config_path)
    detector_run_config = load_detector_training_config(detector_run_config_path)
    classifier_config = load_classifier_experiment_config(classifier_config_path)
    if Path(detector_config.dataset_root).resolve(strict=False) != dataset_root:
        raise DataValidationError(
            "detector config dataset_root does not match CPU benchmark config"
        )
    if Path(classifier_config.dataset_root).resolve(strict=False) != dataset_root:
        raise DataValidationError(
            "classifier config dataset_root does not match CPU benchmark config"
        )
    if not isinstance(classifier_config, IncrementalClassifierTrainingConfig):
        raise DataValidationError(
            "CPU benchmark requires the 20-output Incremental classifier config"
        )

    detector_model_path = Path(detector_config.model).resolve(strict=False)
    yolo_dir = (
        dataset_root / "derived" / "yolo" / detector_config.yolo_run_name
    ).resolve(strict=False)
    assert_training_paths_safe(
        [detector_model_path, yolo_dir, classifier_config.base_checkpoint], dataset_root
    )

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
    if classifier_report.phase != "incremental" or classifier_report.output_dimension != 20:
        raise DataValidationError(
            "CPU benchmark requires a validated 20-output Incremental classifier dataset"
        )
    image_ids, image_paths = _load_benchmark_images(
        detector_report.output_dir, detector_report.manifest_path
    )
    classifier_context = _checkpoint_context(
        classifier_config,
        dataset_root,
        classifier_report.manifest_path,
        classifier_report.output_dimension,
    )
    if not detector_checkpoint.is_file() or not classifier_checkpoint.is_file():
        raise DataValidationError("CPU benchmark checkpoint is missing")
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
    classifier_metadata_path, classifier_metadata = _checkpoint_metadata(
        classifier_checkpoint, "classifier"
    )
    _validate_classifier_checkpoint_provenance(
        config=classifier_config,
        checkpoint=classifier_checkpoint,
        metadata=classifier_metadata,
        classifier_manifest_path=classifier_report.manifest_path,
        classifier_context=classifier_context,
        detector_checkpoint=detector_checkpoint,
    )
    return PreparedCpuBenchmark(
        dataset_root=dataset_root,
        detector_config_path=detector_config_path,
        classifier_config_path=classifier_config_path,
        detector_checkpoint=detector_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        detector_metadata_path=detector_metadata_path,
        classifier_metadata_path=classifier_metadata_path,
        detector_manifest_path=detector_report.manifest_path,
        classifier_manifest_path=classifier_report.manifest_path,
        image_ids=image_ids,
        image_paths=image_paths,
        classifier_context=classifier_context,
        output_dimension=classifier_report.output_dimension,
        detector_image_size=detector_config.image_size,
        classifier_image_size=classifier_config.image_size,
        detector_confidence=detector_config.thresholds.operating_confidence,
        detector_nms_iou=detector_config.thresholds.nms_iou,
        provenance={
            "detector_config_sha256": _sha256(detector_config_path),
            "classifier_config_sha256": _sha256(classifier_config_path),
            "detector_manifest_sha256": _sha256(detector_report.manifest_path),
            "classifier_manifest_sha256": _sha256(classifier_report.manifest_path),
            "registry_sha256": classifier_context["registry_sha256"],
        },
    )


def _validate_backend_result(
    result: CpuBackendResult,
    *,
    image_ids: Sequence[str],
    warmup_iterations: int,
    repetitions: int,
    intra_op_threads: int,
    inter_op_threads: int,
) -> None:
    if (
        result.runtime != "pytorch"
        or result.execution_provider != "CPU"
        or result.device != "cpu"
    ):
        raise DataValidationError(
            "CPU benchmark backend must use PyTorch with the CPU execution provider"
        )
    if (
        result.intra_op_threads != intra_op_threads
        or result.inter_op_threads != inter_op_threads
    ):
        raise DataValidationError("CPU benchmark effective thread counts do not match config")
    expected_ids = tuple(image_ids) * repetitions
    expected_count = len(expected_ids)
    if tuple(result.measured_image_ids) != expected_ids:
        raise DataValidationError("CPU benchmark measured image order or sample count is invalid")
    if result.warmup_invocation_count != len(image_ids) * warmup_iterations:
        raise DataValidationError("CPU benchmark warm-up invocation count is invalid")
    if set(result.stage_samples_ms) != set(BENCHMARK_STAGES):
        raise DataValidationError("CPU benchmark timing stages are invalid")
    for stage in BENCHMARK_STAGES:
        samples = tuple(result.stage_samples_ms[stage])
        if len(samples) != expected_count:
            raise DataValidationError(
                f"CPU benchmark {stage} sample count is invalid"
            )
        _timing_statistics(samples)
    if len(result.classifier_batch_sizes) != expected_count or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result.classifier_batch_sizes
    ):
        raise DataValidationError("CPU benchmark classifier batch sizes are invalid")


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except OSError:
            pass
    return (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.machine()
    )


def _environment_metadata() -> dict[str, Any]:
    dependencies = {}
    for distribution in ("torch", "torchvision", "ultralytics", "Pillow", "PyYAML"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependencies,
    }


def _validate_completed_run(output_dir: Path) -> None:
    expected = {"benchmark.json", "config.yaml", "metadata.json"}
    actual = {path.name for path in output_dir.iterdir()}
    if actual != expected:
        raise DataValidationError(
            f"completed CPU benchmark files are invalid: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for filename in ("benchmark.json", "metadata.json"):
        _json_object(output_dir / filename, filename)


def run_cpu_benchmark(
    config: CpuBenchmarkConfig,
    backend: CpuBenchmarkBackend | None = None,
) -> CpuBenchmarkReport:
    if not isinstance(config, CpuBenchmarkConfig):
        raise DataValidationError("config must be CpuBenchmarkConfig")
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    assert_training_paths_safe(_initial_paths(config), dataset_root)
    prepared = _prepare_cpu_benchmark(config)
    selected_backend = backend
    if selected_backend is None:
        selected_backend = TorchCpuBenchmarkBackend()

    output_root = Path(config.output_root).resolve(strict=False)
    output_dir = output_root / config.run_name
    if output_dir.exists():
        raise DataValidationError(f"CPU benchmark run already exists: {output_dir}")
    detector_hash_before = _sha256(prepared.detector_checkpoint)
    classifier_hash_before = _sha256(prepared.classifier_checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".{config.run_name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    try:
        result = selected_backend.run(
            image_paths=prepared.image_paths,
            image_ids=prepared.image_ids,
            detector_checkpoint=prepared.detector_checkpoint,
            classifier_checkpoint=prepared.classifier_checkpoint,
            classifier_context=prepared.classifier_context,
            output_dimension=prepared.output_dimension,
            detector_image_size=prepared.detector_image_size,
            classifier_image_size=prepared.classifier_image_size,
            detector_confidence=prepared.detector_confidence,
            detector_nms_iou=prepared.detector_nms_iou,
            warmup_iterations=config.warmup_iterations,
            repetitions=config.repetitions,
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )
        _validate_backend_result(
            result,
            image_ids=prepared.image_ids,
            warmup_iterations=config.warmup_iterations,
            repetitions=config.repetitions,
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )
        detector_hash_after = _sha256(prepared.detector_checkpoint)
        classifier_hash_after = _sha256(prepared.classifier_checkpoint)
        if detector_hash_after != detector_hash_before:
            raise DataValidationError("detector checkpoint changed during CPU benchmark")
        if classifier_hash_after != classifier_hash_before:
            raise DataValidationError("classifier checkpoint changed during CPU benchmark")

        raw_samples = {
            stage: [float(value) for value in result.stage_samples_ms[stage]]
            for stage in BENCHMARK_STAGES
        }
        batch_sizes = [int(value) for value in result.classifier_batch_sizes]
        benchmark_payload = {
            "benchmark_version": 1,
            "warmup_iterations": config.warmup_iterations,
            "warmup_invocation_count": result.warmup_invocation_count,
            "repetitions": config.repetitions,
            "scene_count": len(prepared.image_ids),
            "measured_invocation_count": len(result.measured_image_ids),
            "image_ids": list(prepared.image_ids),
            "measured_image_ids": list(result.measured_image_ids),
            "timings_ms": {
                stage: _timing_statistics(raw_samples[stage])
                for stage in BENCHMARK_STAGES
            },
            "raw_samples_ms": raw_samples,
            "classifier_batch_sizes": {
                "raw": batch_sizes,
                "minimum": min(batch_sizes),
                "maximum": max(batch_sizes),
                "mean": sum(batch_sizes) / len(batch_sizes),
            },
        }
        benchmark_path = staging_dir / "benchmark.json"
        benchmark_path.write_text(
            json.dumps(benchmark_payload, ensure_ascii=False, indent=2, sort_keys=True)
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
                    "split": "validation (train-side)",
                    "runtime": {
                        "runtime": result.runtime,
                        "execution_provider": result.execution_provider,
                        "device": result.device,
                        "intra_op_threads": result.intra_op_threads,
                        "inter_op_threads": result.inter_op_threads,
                    },
                    "inputs": {
                        "scene_count": len(prepared.image_ids),
                        "detector_image_size": prepared.detector_image_size,
                        "classifier_image_size": prepared.classifier_image_size,
                        "classifier_batch_strategy": "one dynamic batch per scene",
                        "detector_operating_confidence": prepared.detector_confidence,
                        "detector_nms_iou": prepared.detector_nms_iou,
                    },
                    "artifacts": {
                        "detector_checkpoint": str(prepared.detector_checkpoint),
                        "detector_sha256_before": detector_hash_before,
                        "detector_sha256_after": detector_hash_after,
                        "detector_unchanged": detector_hash_before == detector_hash_after,
                        "classifier_checkpoint": str(prepared.classifier_checkpoint),
                        "classifier_sha256_before": classifier_hash_before,
                        "classifier_sha256_after": classifier_hash_after,
                        "classifier_unchanged": (
                            classifier_hash_before == classifier_hash_after
                        ),
                        "detector_metadata": str(prepared.detector_metadata_path),
                        "classifier_metadata": str(prepared.classifier_metadata_path),
                        "detector_manifest": str(prepared.detector_manifest_path),
                        "classifier_manifest": str(prepared.classifier_manifest_path),
                        **dict(prepared.provenance),
                    },
                    "environment": _environment_metadata(),
                    "limitations": {
                        "pos_device_claim": False,
                        "statement": (
                            "Current development-PC CPU measurement only; not a specific POS-device claim."
                        ),
                        "test_data_used": False,
                    },
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
        return CpuBenchmarkReport(
            output_dir=output_dir,
            config_path=output_dir / config_path.name,
            benchmark_path=output_dir / benchmark_path.name,
            metadata_path=output_dir / metadata_path.name,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


class TorchCpuBenchmarkBackend:
    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] | None = None,
        detector_factory: Callable[[Path], Any] | None = None,
        classifier_loader: Callable[..., Any] | None = None,
        transform_factory: Callable[[int], Any] | None = None,
        thread_configurer: Callable[[int, int], tuple[int, int]] | None = None,
    ) -> None:
        self._clock_ns = clock_ns or time.perf_counter_ns
        self._detector_factory = detector_factory or self._default_detector_factory
        self._classifier_loader = classifier_loader or self._default_classifier_loader
        self._transform_factory = transform_factory or self._default_transform_factory
        self._thread_configurer = thread_configurer or self._configure_threads

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

        _train_transform, validation_transform = _classifier_transforms(image_size)
        return validation_transform

    @staticmethod
    def _configure_threads(intra_op_threads: int, inter_op_threads: int) -> tuple[int, int]:
        import torch

        torch.set_num_threads(intra_op_threads)
        try:
            torch.set_num_interop_threads(inter_op_threads)
        except RuntimeError as exc:
            if torch.get_num_interop_threads() != inter_op_threads:
                raise DataValidationError(
                    "cannot apply configured PyTorch inter-op thread count"
                ) from exc
        effective = (torch.get_num_threads(), torch.get_num_interop_threads())
        if effective != (intra_op_threads, inter_op_threads):
            raise DataValidationError(
                "effective PyTorch CPU thread counts do not match config"
            )
        return effective

    @staticmethod
    def _require_cpu_module(module: Any) -> None:
        parameters = tuple(module.parameters()) if hasattr(module, "parameters") else ()
        buffers = tuple(module.buffers()) if hasattr(module, "buffers") else ()
        if any(value.device.type != "cpu" for value in (*parameters, *buffers)):
            raise DataValidationError("classifier model must be entirely on CPU")

    def run(
        self,
        *,
        image_paths: Sequence[Path],
        image_ids: Sequence[str],
        detector_checkpoint: Path,
        classifier_checkpoint: Path,
        classifier_context: Mapping[str, Any],
        output_dimension: int,
        detector_image_size: int,
        classifier_image_size: int,
        detector_confidence: float,
        detector_nms_iou: float,
        warmup_iterations: int,
        repetitions: int,
        intra_op_threads: int,
        inter_op_threads: int,
    ) -> CpuBackendResult:
        import torch
        from PIL import Image, UnidentifiedImageError

        if len(image_paths) != len(image_ids) or not image_ids:
            raise DataValidationError("CPU benchmark images and IDs must be non-empty")
        if output_dimension != 20:
            raise DataValidationError("CPU benchmark classifier must have 20 outputs")
        effective_threads = self._thread_configurer(
            intra_op_threads, inter_op_threads
        )
        if effective_threads != (intra_op_threads, inter_op_threads):
            raise DataValidationError(
                "effective PyTorch CPU thread counts do not match config"
            )
        cpu_device = torch.device("cpu")
        detector = self._detector_factory(detector_checkpoint)
        names = detector.names
        if not isinstance(names, Mapping) or tuple(
            names[index] for index in sorted(names)
        ) != ("bread",):
            raise DataValidationError("CPU benchmark detector must have one bread class")
        classifier = self._classifier_loader(
            checkpoint=classifier_checkpoint,
            output_dimension=output_dimension,
            checkpoint_context=classifier_context,
            image_size=classifier_image_size,
            device=cpu_device,
        )
        if hasattr(classifier, "to"):
            classifier = classifier.to(cpu_device)
        if hasattr(classifier, "eval"):
            classifier = classifier.eval()
        self._require_cpu_module(classifier)
        validation_transform = self._transform_factory(classifier_image_size)

        measured_ids: list[str] = []
        batch_sizes: list[int] = []
        stage_samples: dict[str, list[float]] = {
            stage: [] for stage in BENCHMARK_STAGES
        }

        def elapsed_ms(start: int, end: int) -> float:
            value = (end - start) / 1_000_000.0
            if value < 0.0:
                raise DataValidationError("CPU benchmark clock moved backwards")
            return value

        def invoke(image_id: str, image_path: Path, *, measured: bool) -> None:
            end_to_end_start = self._clock_ns()

            detector_start = self._clock_ns()
            detector_results = detector.predict(
                source=str(image_path),
                conf=detector_confidence,
                iou=detector_nms_iou,
                imgsz=detector_image_size,
                device="cpu",
                verbose=False,
                stream=False,
            )
            detector_end = self._clock_ns()
            if len(detector_results) != 1:
                raise DataValidationError(
                    "CPU detector must return exactly one result per scene"
                )
            result = detector_results[0]

            crop_start = self._clock_ns()
            try:
                with Image.open(image_path) as source:
                    scene = source.convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                raise DataValidationError(
                    f"cannot load CPU benchmark image {image_path}: {exc}"
                ) from exc
            width, height = scene.size
            crop_tensors = []
            detections: list[tuple[tuple[float, float, float, float], float]] = []
            boxes = result.boxes
            for xyxy, confidence, class_index in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
                strict=True,
            ):
                if int(class_index) != 0:
                    raise DataValidationError("CPU detector emitted a non-bread class")
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
            batch = (
                torch.stack(crop_tensors).to(cpu_device)
                if crop_tensors
                else None
            )
            if batch is not None and batch.device.type != "cpu":
                raise DataValidationError("classifier input tensor must be on CPU")
            crop_end = self._clock_ns()

            classifier_ms = 0.0
            model_indices = None
            classifier_confidences = None
            if batch is not None:
                classifier_start = self._clock_ns()
                with torch.inference_mode():
                    probabilities = classifier(batch).softmax(dim=1)
                    classifier_confidences, model_indices = probabilities.max(dim=1)
                classifier_end = self._clock_ns()
                if probabilities.device.type != "cpu":
                    raise DataValidationError("classifier output tensor must be on CPU")
                classifier_ms = elapsed_ms(classifier_start, classifier_end)

            postprocess_start = self._clock_ns()
            final_predictions = []
            if model_indices is not None and classifier_confidences is not None:
                for (bbox, detector_score), model_index, classifier_score in zip(
                    detections,
                    model_indices.detach().cpu().tolist(),
                    classifier_confidences.detach().cpu().tolist(),
                    strict=True,
                ):
                    final_predictions.append(
                        {
                            "bbox_xyxy": bbox,
                            "model_index": int(model_index),
                            "detector_confidence": detector_score,
                            "classifier_confidence": float(classifier_score),
                            "score": detector_score * float(classifier_score),
                        }
                    )
            postprocess_end = self._clock_ns()
            if len(final_predictions) != len(detections):
                raise DataValidationError(
                    "CPU benchmark postprocess count does not match detections"
                )
            end_to_end_end = self._clock_ns()

            if measured:
                measured_ids.append(image_id)
                batch_sizes.append(len(crop_tensors))
                stage_samples["detector"].append(
                    elapsed_ms(detector_start, detector_end)
                )
                stage_samples["crop_preprocess"].append(
                    elapsed_ms(crop_start, crop_end)
                )
                stage_samples["classifier_batch"].append(classifier_ms)
                stage_samples["postprocess"].append(
                    elapsed_ms(postprocess_start, postprocess_end)
                )
                stage_samples["end_to_end"].append(
                    elapsed_ms(end_to_end_start, end_to_end_end)
                )

        for _iteration in range(warmup_iterations):
            for image_id, image_path in zip(image_ids, image_paths, strict=True):
                invoke(image_id, image_path, measured=False)
        for _iteration in range(repetitions):
            for image_id, image_path in zip(image_ids, image_paths, strict=True):
                invoke(image_id, image_path, measured=True)

        return CpuBackendResult(
            stage_samples_ms={
                stage: tuple(stage_samples[stage]) for stage in BENCHMARK_STAGES
            },
            measured_image_ids=tuple(measured_ids),
            classifier_batch_sizes=tuple(batch_sizes),
            warmup_invocation_count=warmup_iterations * len(image_ids),
            runtime="pytorch",
            execution_provider="CPU",
            device="cpu",
            intra_op_threads=effective_threads[0],
            inter_op_threads=effective_threads[1],
        )
