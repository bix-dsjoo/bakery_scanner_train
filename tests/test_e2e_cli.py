from __future__ import annotations

import json
from pathlib import Path

from bakery_scanner import e2e_cli
from bakery_scanner.e2e_inference import EndToEndConfig, EndToEndReport
from bakery_scanner.errors import DataValidationError


def _config(tmp_path: Path) -> EndToEndConfig:
    return EndToEndConfig(
        dataset_root=str(tmp_path / "datasets"),
        detector_config=str(tmp_path / "detector.yaml"),
        classifier_config=str(tmp_path / "classifier.yaml"),
        detector_checkpoint=str(tmp_path / "detector" / "best.pt"),
        classifier_checkpoint=str(tmp_path / "classifier" / "best.pt"),
        output_root=str(tmp_path / "runs" / "e2e"),
        run_name="baseline",
    )


def _report(tmp_path: Path) -> EndToEndReport:
    output = tmp_path / "output"
    output.mkdir()
    metrics = output / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "split": "validation",
                "metrics": {
                    "map50": 0.4,
                    "map50_95": 0.3,
                    "supported_macro_exact_count_accuracy": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    return EndToEndReport(
        output_dir=output,
        metadata_path=output / "metadata.json",
        predictions_path=output / "predictions.json",
        metrics_path=metrics,
    )


def test_e2e_cli_prints_preflight_metrics_and_json(tmp_path, monkeypatch, capsys) -> None:
    config = _config(tmp_path)
    report = _report(tmp_path)
    monkeypatch.setattr(e2e_cli, "load_end_to_end_config", lambda path: config)
    monkeypatch.setattr(e2e_cli, "evaluate_end_to_end", lambda selected: report)

    assert e2e_cli.main(["evaluate", "--config", "config.yaml"]) == 0
    captured = capsys.readouterr()
    assert "validation (train-side)" in captured.err
    assert str(Path(config.detector_checkpoint).resolve()) in captured.err
    assert str(Path(config.classifier_checkpoint).resolve()) in captured.err
    assert "mAP50: 0.400000" in captured.out
    assert "mAP50:95: 0.300000" in captured.out

    assert (
        e2e_cli.main(["evaluate", "--config", "config.yaml", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["supported_macro_exact_count_accuracy"] == 0.8


def test_e2e_cli_returns_one_for_invalid_config(monkeypatch, capsys) -> None:
    def fail(path):
        raise DataValidationError("bad e2e config")

    monkeypatch.setattr(e2e_cli, "load_end_to_end_config", fail)

    assert e2e_cli.main(["evaluate", "--config", "bad.yaml"]) == 1
    assert "bad e2e config" in capsys.readouterr().err
