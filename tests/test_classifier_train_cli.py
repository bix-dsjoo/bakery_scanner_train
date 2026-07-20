from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from bakery_scanner import classifier_train_cli
from bakery_scanner.classifier_training import (
    ClassifierEvaluationReport,
    ClassifierTrainingReport,
)
from bakery_scanner.errors import DataValidationError


def _training_report(tmp_path: Path) -> ClassifierTrainingReport:
    output = tmp_path / "run"
    output.mkdir()
    metrics = output / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "split": "validation",
                "metrics": {"top1_accuracy": 0.8, "macro_f1": 0.7},
            }
        ),
        encoding="utf-8",
    )
    return ClassifierTrainingReport(
        output_dir=output,
        best_checkpoint=output / "checkpoints" / "best.pt",
        last_checkpoint=output / "checkpoints" / "last.pt",
        metadata_path=output / "metadata.json",
        predictions_path=output / "predictions.json",
        metrics_path=metrics,
        history_path=output / "history.json",
    )


def _config(tmp_path: Path):
    return SimpleNamespace(
        dataset_root=str(tmp_path / "datasets"),
        source_classifier_run="base_seed42",
        pretrained_model=str(tmp_path / "models" / "resnet18.pth"),
        output_root=str(tmp_path / "runs" / "classifier"),
        run_name="baseline",
    )


def test_train_cli_prints_validation_selection_and_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    report = _training_report(tmp_path)
    monkeypatch.setattr(
        classifier_train_cli,
        "load_classifier_training_config",
        lambda path: _config(tmp_path),
    )
    monkeypatch.setattr(
        classifier_train_cli, "train_classifier", lambda config: report
    )

    assert classifier_train_cli.main(["train", "--config", "config.yaml"]) == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "train split: train" in captured.err
    assert "validation split: validation" in captured.err
    assert str((tmp_path / "datasets").resolve()) in captured.err
    assert "base_seed42" in captured.err
    assert str((tmp_path / "models" / "resnet18.pth").resolve()) in captured.err
    assert str((tmp_path / "runs" / "classifier" / "baseline").resolve()) in captured.err
    assert "validation" in output
    assert "Top-1: 0.800000" in output
    assert "best.pt" in output

    assert (
        classifier_train_cli.main(
            ["train", "--config", "config.yaml", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["split"] == "validation"
    assert payload["metrics"]["macro_f1"] == 0.7


def test_evaluate_cli_passes_checkpoint_and_output_dir(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    metrics = output / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "split": "validation",
                "metrics": {"top1_accuracy": 0.75, "macro_f1": 0.6},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    report = ClassifierEvaluationReport(
        output_dir=output,
        checkpoint=checkpoint,
        predictions_path=output / "predictions.json",
        metrics_path=metrics,
    )
    calls = []
    monkeypatch.setattr(
        classifier_train_cli,
        "load_classifier_training_config",
        lambda path: _config(tmp_path),
    )
    monkeypatch.setattr(
        classifier_train_cli,
        "evaluate_classifier_checkpoint",
        lambda config, selected_checkpoint, output_dir=None: (
            calls.append((selected_checkpoint, output_dir)) or report
        ),
    )

    assert (
        classifier_train_cli.main(
            [
                "evaluate",
                "--config",
                "config.yaml",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    assert calls == [(checkpoint, output)]
    assert json.loads(capsys.readouterr().out)["metrics"]["top1_accuracy"] == 0.75


def test_cli_returns_one_for_invalid_config(monkeypatch, capsys) -> None:
    def fail(path):
        raise DataValidationError("bad classifier config")

    monkeypatch.setattr(classifier_train_cli, "load_classifier_training_config", fail)

    assert classifier_train_cli.main(["train", "--config", "bad.yaml"]) == 1
    assert "bad classifier config" in capsys.readouterr().err
