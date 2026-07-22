from __future__ import annotations

import json
from pathlib import Path

from bakery_scanner import detector_ensemble_cli
from bakery_scanner.detector_ensemble import (
    DetectorEnsembleConfig,
    DetectorEnsembleMember,
    DetectorEnsembleReport,
)


def _config(tmp_path: Path) -> DetectorEnsembleConfig:
    return DetectorEnsembleConfig(
        dataset_root=str(tmp_path / "datasets"),
        output_root=str(tmp_path / "runs"),
        run_name="ensemble",
        members=(
            DetectorEnsembleMember("first.yaml", "a" * 64, "first.pt", "b" * 64),
            DetectorEnsembleMember("second.yaml", "c" * 64, "second.pt", "d" * 64),
        ),
        cpu_threads=8,
        cpu_warmups=1,
        cpu_repetitions=3,
    )


def _report(tmp_path: Path) -> DetectorEnsembleReport:
    output = tmp_path / "runs" / "ensemble"
    output.mkdir(parents=True)
    metadata = output / "metadata.json"
    predictions = output / "predictions.json"
    metrics = output / "metrics.json"
    metadata.write_text("{}\n", encoding="utf-8")
    predictions.write_text("{}\n", encoding="utf-8")
    metrics.write_text(
        json.dumps(
            {
                "metrics": {
                    "global": {
                        "ground_truth_count": 30,
                        "true_positive_count": 30,
                        "false_positive_count": 0,
                        "recall": 1.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return DetectorEnsembleReport(output, metadata, predictions, metrics)


def test_ensemble_evaluate_cli_emits_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config(tmp_path)
    report = _report(tmp_path)
    monkeypatch.setattr(
        detector_ensemble_cli, "load_detector_ensemble_config", lambda path: config
    )
    monkeypatch.setattr(
        detector_ensemble_cli, "evaluate_detector_ensemble", lambda value: report
    )

    exit_code = detector_ensemble_cli.main(
        ["evaluate", "--config", "ensemble.yaml", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["metrics"]["global"]["recall"] == 1.0
    assert payload["metrics"]["global"]["false_positive_count"] == 0


def test_ensemble_benchmark_cli_emits_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config(tmp_path)
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps({"timing": {"count": 18, "p95_ms": 249.0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        detector_ensemble_cli, "load_detector_ensemble_config", lambda path: config
    )
    monkeypatch.setattr(
        detector_ensemble_cli,
        "benchmark_detector_ensemble_cpu",
        lambda value: benchmark,
    )

    exit_code = detector_ensemble_cli.main(
        ["benchmark", "--config", "ensemble.yaml", "--json"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["benchmark"]["timing"]["count"] == 18
    assert payload["benchmark_path"] == str(benchmark)
