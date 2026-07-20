from __future__ import annotations

import json
from pathlib import Path

from bakery_scanner import final_evaluation_cli
from bakery_scanner.errors import DataValidationError
from bakery_scanner.final_evaluation import (
    FinalEvaluationPreflightReport,
    FinalEvaluationReport,
    PreparedFinalEvaluation,
    load_final_evaluation_config,
)


def _prepared(tmp_path: Path) -> PreparedFinalEvaluation:
    return PreparedFinalEvaluation(
        config_path=tmp_path / "frozen.yaml",
        config_sha256="a" * 64,
        dataset_root=tmp_path / "datasets",
        registry_path=tmp_path / "registry.json",
        detector_config_path=tmp_path / "detector.yaml",
        detector_checkpoint=tmp_path / "detector.pt",
        classifier_config_paths={"base": tmp_path / "base.yaml", "incremental": tmp_path / "inc.yaml"},
        classifier_checkpoints={"base": tmp_path / "base.pt", "incremental": tmp_path / "inc.pt"},
        classifier_contexts={"base": {}, "incremental": {}},
        output_dir=tmp_path / "runs" / "frozen_v1",
        lock_path=tmp_path / "runs" / ".frozen_v1.started.json",
        provenance={},
    )


def test_final_evaluation_cli_preflight_and_run(tmp_path, monkeypatch, capsys) -> None:
    prepared = _prepared(tmp_path)
    config = object()
    monkeypatch.setattr(final_evaluation_cli, "load_final_evaluation_config", lambda _path: config)
    monkeypatch.setattr(
        final_evaluation_cli,
        "preflight_final_evaluation",
        lambda *_args: FinalEvaluationPreflightReport(prepared),
    )
    assert final_evaluation_cli.main(["preflight", "--config", "frozen.yaml", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["test_data_accessed"] is False

    prepared.output_dir.mkdir(parents=True)
    summary_path = prepared.output_dir / "summary.json"
    metadata_path = prepared.output_dir / "metadata.json"
    summary_path.write_text(json.dumps({"base_retention_delta": {"classifier_top1": -0.1}}), encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")
    report = FinalEvaluationReport(prepared.output_dir, summary_path, metadata_path)
    monkeypatch.setattr(final_evaluation_cli, "run_final_evaluation", lambda *_args: report)
    assert final_evaluation_cli.main(["run", "--config", "frozen.yaml"]) == 0
    captured = capsys.readouterr()
    assert "irreversible one-shot" in captured.err
    assert "Final evaluation: completed" in captured.out


def test_final_evaluation_cli_returns_one_on_validation_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        final_evaluation_cli,
        "load_final_evaluation_config",
        lambda _path: (_ for _ in ()).throw(DataValidationError("frozen error")),
    )
    assert final_evaluation_cli.main(["preflight", "--config", "bad.yaml"]) == 1
    assert "frozen error" in capsys.readouterr().err
