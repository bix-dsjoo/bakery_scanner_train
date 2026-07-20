from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from .errors import DataValidationError

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
