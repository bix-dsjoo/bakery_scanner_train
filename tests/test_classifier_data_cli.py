from __future__ import annotations

import json
from pathlib import Path

from bakery_scanner.classifier_data_cli import main


def test_generate_and_validate_cli_report_concrete_classifier_run(
    dataset_factory,
    capsys,
) -> None:
    dataset_root = dataset_factory()
    exit_code = main(
        [
            "generate",
            "--dataset-root",
            str(dataset_root),
            "--run-name",
            "cli-base",
            "--phase",
            "base",
            "--seed",
            "9",
            "--validation-fraction",
            "0.5",
            "--expected-base-images-per-class",
            "1",
            "--json",
        ]
    )
    generated = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert generated["status"] == "ok"
    assert generated["phase"] == "base"
    assert generated["output_dimension"] == 15
    assert Path(generated["output_dir"]).name == "cli-base"

    exit_code = main(
        [
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--run-name",
            "cli-base",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Classifier dataset validation: ok" in output
    assert "output dimension: 15" in output
    assert str(dataset_root.resolve()) in output


def test_generate_cli_returns_failure_for_invalid_dataset(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "generate",
            "--dataset-root",
            str(tmp_path / "missing"),
            "--run-name",
            "invalid",
            "--phase",
            "base",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error:" in captured.err
