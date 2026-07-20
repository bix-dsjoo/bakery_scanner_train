from __future__ import annotations

import json
from pathlib import Path

from bakery_scanner import cpu_benchmark_cli
from bakery_scanner.cpu_benchmark import (
    CpuBenchmarkConfig,
    CpuBenchmarkReport,
)
from bakery_scanner.errors import DataValidationError


def _config(tmp_path: Path) -> CpuBenchmarkConfig:
    return CpuBenchmarkConfig(
        dataset_root=str(tmp_path / "datasets"),
        detector_config=str(tmp_path / "detector.yaml"),
        classifier_config=str(tmp_path / "classifier.yaml"),
        detector_checkpoint=str(tmp_path / "detector" / "best.pt"),
        classifier_checkpoint=str(tmp_path / "classifier" / "best.pt"),
        output_root=str(tmp_path / "runs" / "benchmark"),
        run_name="incremental-cpu",
        warmup_iterations=2,
        repetitions=3,
        intra_op_threads=4,
        inter_op_threads=1,
    )


def _report(tmp_path: Path) -> CpuBenchmarkReport:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    benchmark_path = output_dir / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "timings_ms": {
                    "detector": {"mean_ms": 10.0, "p50_ms": 9.0, "p95_ms": 12.0},
                    "crop_preprocess": {"mean_ms": 2.0, "p50_ms": 2.0, "p95_ms": 3.0},
                    "classifier_batch": {"mean_ms": 4.0, "p50_ms": 4.0, "p95_ms": 5.0},
                    "postprocess": {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.5},
                    "end_to_end": {"mean_ms": 17.0, "p50_ms": 16.0, "p95_ms": 20.0},
                }
            }
        ),
        encoding="utf-8",
    )
    return CpuBenchmarkReport(
        output_dir=output_dir,
        config_path=output_dir / "config.yaml",
        benchmark_path=benchmark_path,
        metadata_path=output_dir / "metadata.json",
    )


def test_cpu_benchmark_cli_prints_preflight_result_and_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config(tmp_path)
    report = _report(tmp_path)
    monkeypatch.setattr(
        cpu_benchmark_cli, "load_cpu_benchmark_config", lambda _path: config
    )
    monkeypatch.setattr(cpu_benchmark_cli, "run_cpu_benchmark", lambda _config: report)

    assert cpu_benchmark_cli.main(["run", "--config", "config.yaml"]) == 0
    captured = capsys.readouterr()
    assert "validation (train-side)" in captured.err
    assert "device: cpu" in captured.err
    assert "intra-op threads: 4" in captured.err
    assert str(Path(config.detector_checkpoint).resolve()) in captured.err
    assert "end_to_end: mean=17.000 ms" in captured.out
    assert "not a specific POS-device claim" in captured.out

    assert cpu_benchmark_cli.main(
        ["run", "--config", "config.yaml", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["benchmark"]["timings_ms"]["detector"]["p95_ms"] == 12.0


def test_cpu_benchmark_cli_returns_one_for_invalid_config(monkeypatch, capsys) -> None:
    def fail(_path):
        raise DataValidationError("bad CPU benchmark config")

    monkeypatch.setattr(cpu_benchmark_cli, "load_cpu_benchmark_config", fail)

    assert cpu_benchmark_cli.main(["run", "--config", "bad.yaml"]) == 1
    assert "bad CPU benchmark config" in capsys.readouterr().err
