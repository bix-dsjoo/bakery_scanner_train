from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bakery_scanner.cpu_benchmark import (
    CpuBenchmarkConfig,
    _percentile,
    _timing_statistics,
    load_cpu_benchmark_config,
)
from bakery_scanner.errors import DataValidationError


def _payload() -> dict[str, object]:
    return {
        "dataset_root": "datasets",
        "detector_config": "configs/detector/yolo11n_base.yaml",
        "classifier_config": "configs/classifier/resnet18_incremental.yaml",
        "detector_checkpoint": (
            "runs/detector/yolo11n_base_seed42/checkpoints/best.pt"
        ),
        "classifier_checkpoint": (
            "runs/classifier/resnet18_incremental_seed42/checkpoints/best.pt"
        ),
        "output_root": "runs/benchmark",
        "run_name": "incremental_resnet18_cpu",
        "warmup_iterations": 5,
        "repetitions": 30,
        "intra_op_threads": 4,
        "inter_op_threads": 1,
    }


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_load_cpu_benchmark_config_is_strict(tmp_path: Path) -> None:
    config = load_cpu_benchmark_config(_write_config(tmp_path / "cpu.yaml", _payload()))

    assert config == CpuBenchmarkConfig(**_payload())

    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(DataValidationError, match="config fields"):
        load_cpu_benchmark_config(_write_config(tmp_path / "extra.yaml", payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup_iterations", -1, "warmup_iterations"),
        ("repetitions", 0, "repetitions"),
        ("intra_op_threads", 0, "intra_op_threads"),
        ("inter_op_threads", True, "inter_op_threads"),
        ("run_name", "../escape", "run_name"),
    ],
)
def test_load_cpu_benchmark_config_rejects_invalid_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(DataValidationError, match=message):
        load_cpu_benchmark_config(_write_config(tmp_path / "invalid.yaml", payload))


def test_percentiles_use_linear_interpolation() -> None:
    samples = (1.0, 2.0, 3.0, 4.0)

    assert _percentile(samples, 0.5) == pytest.approx(2.5)
    assert _percentile(samples, 0.95) == pytest.approx(3.85)
    assert _timing_statistics(samples) == pytest.approx(
        {"count": 4, "mean_ms": 2.5, "p50_ms": 2.5, "p95_ms": 3.85}
    )


def test_timing_statistics_rejects_empty_or_invalid_samples() -> None:
    with pytest.raises(DataValidationError, match="must not be empty"):
        _timing_statistics(())
    with pytest.raises(DataValidationError, match="finite non-negative"):
        _timing_statistics((1.0, float("nan")))
    with pytest.raises(DataValidationError, match="finite non-negative"):
        _timing_statistics((-0.1,))
