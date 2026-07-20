import json
from pathlib import Path

from bakery_scanner import detector_train_cli
from bakery_scanner.detector_training import DetectorTrainingReport

BASELINE_YAML = """\
dataset_root: datasets
source_detector_run: detector-input
yolo_run_name: yolo-input
output_root: runs/detector
run_name: detector-baseline
model: yolo11n.pt
image_size: 640
epochs: 50
batch_size: 16
seed: 42
device: "0"
patience: 10
workers: 8
thresholds:
  confidence_floor: 0.001
  operating_confidence: 0.25
  nms_iou: 0.7
  matching_iou: 0.5
"""


def _training_report(tmp_path: Path) -> DetectorTrainingReport:
    output_dir = tmp_path / "run"
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    best = checkpoints / "best.pt"
    last = checkpoints / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    metadata = output_dir / "metadata.json"
    predictions = output_dir / "predictions.json"
    metrics = output_dir / "metrics.json"
    metadata.write_text("{}\n", encoding="utf-8")
    predictions.write_text("{}\n", encoding="utf-8")
    metrics.write_text(
        json.dumps(
            {
                "split": "validation",
                "metrics": {
                    "global": {
                        "ap50": 0.5,
                        "recall": 0.6,
                        "miss_rate": 0.4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return DetectorTrainingReport(
        output_dir,
        best,
        last,
        metadata,
        predictions,
        metrics,
    )


def test_train_cli_emits_json_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(BASELINE_YAML, encoding="utf-8")
    report = _training_report(tmp_path)
    monkeypatch.setattr(detector_train_cli, "train_detector", lambda config: report)

    exit_code = detector_train_cli.main(
        ["train", "--config", str(config_path), "--json"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["split"] == "validation"
    assert payload["metrics"]["global"]["ap50"] == 0.5
    assert captured.err == ""


def test_train_cli_returns_failure_for_invalid_config(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("dataset_root: datasets\n", encoding="utf-8")

    exit_code = detector_train_cli.main(["train", "--config", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed" in captured.err.lower()
