from pathlib import Path
from collections.abc import Mapping, Sequence
import json
from types import SimpleNamespace

import pytest

from bakery_scanner.detector_evaluation import Detection, EvaluationThresholds
from bakery_scanner.detector_training import (
    BackendTrainingResult,
    DetectorTrainingConfig,
    UltralyticsBackend,
    evaluate_detector_checkpoint,
    load_detector_training_config,
    train_detector,
)
from bakery_scanner.errors import DataValidationError
from bakery_scanner.yolo_dataset import build_yolo_dataset

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


def test_load_detector_training_config_accepts_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.yaml"
    path.write_text(BASELINE_YAML, encoding="utf-8")

    config = load_detector_training_config(path)

    assert config.source_detector_run == "detector-input"
    assert config.model == "yolo11n.pt"
    assert config.image_size == 640
    assert config.device == "0"
    assert config.thresholds == EvaluationThresholds()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("image_size: true", "image_size"),
        ("device: cpu", "device"),
        ("run_name: ../escape", "run_name"),
        ("operating_confidence: 1.0", "confidence"),
    ],
)
def test_load_detector_training_config_rejects_invalid_values(
    tmp_path: Path, replacement: str, message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    if replacement.startswith("operating_confidence"):
        payload = BASELINE_YAML.replace("operating_confidence: 0.25", replacement)
    else:
        key = replacement.split(":", 1)[0]
        original = next(line for line in BASELINE_YAML.splitlines() if line.startswith(f"{key}:"))
        payload = BASELINE_YAML.replace(original, replacement)
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(DataValidationError, match=message):
        load_detector_training_config(path)


def test_load_detector_training_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text(BASELINE_YAML + "surprise: value\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="fields|unknown"):
        load_detector_training_config(path)


class RecordingBackend:
    def __init__(self) -> None:
        self.training_arguments: dict[str, object] | None = None

    def cuda_available(self, device: str) -> bool:
        return True

    def train(
        self,
        *,
        model: str,
        data_yaml: Path,
        output_dir: Path,
        arguments: Mapping[str, object],
    ) -> BackendTrainingResult:
        self.training_arguments = dict(arguments)
        output_dir.mkdir(parents=True)
        best = output_dir / "best.pt"
        last = output_dir / "last.pt"
        best.write_bytes(b"best checkpoint")
        last.write_bytes(b"last checkpoint")
        return BackendTrainingResult(best, last, ("bread",), Path(model))

    def predict(
        self,
        *,
        checkpoint: Path,
        image_paths: Sequence[Path],
        confidence_floor: float,
        nms_iou: float,
        image_size: int,
        device: str,
    ) -> Mapping[str, Sequence[Detection]]:
        return {path.name: () for path in image_paths}


def _training_config(
    dataset_root: Path, tmp_path: Path, pretrained: Path
) -> DetectorTrainingConfig:
    return DetectorTrainingConfig(
        dataset_root=str(dataset_root),
        source_detector_run="detector-input",
        yolo_run_name="yolo-input",
        output_root=str(tmp_path / "runs" / "detector"),
        run_name="detector-baseline",
        model=str(pretrained),
        image_size=640,
        epochs=50,
        batch_size=16,
        seed=42,
        device="0",
        patience=10,
        workers=8,
        thresholds=EvaluationThresholds(),
    )


def test_train_detector_publishes_reproducible_run(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_root, _ = detector_source_run
    pretrained = tmp_path / "pretrained.pt"
    pretrained.write_bytes(b"pretrained checkpoint")
    config = _training_config(dataset_root, tmp_path, pretrained)
    backend = RecordingBackend()

    report = train_detector(config, backend)

    assert report.output_dir.is_dir()
    assert report.best_checkpoint.read_bytes() == b"best checkpoint"
    assert report.last_checkpoint.read_bytes() == b"last checkpoint"
    assert backend.training_arguments == {
        "image_size": 640,
        "epochs": 50,
        "batch_size": 16,
        "seed": 42,
        "device": "0",
        "patience": 10,
        "workers": 8,
    }
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metadata["split"] == "validation"
    assert Path(metadata["dataset"]["train_path"]).parts[-2:] == ("train", "images")
    assert Path(metadata["dataset"]["validation_path"]).parts[-2:] == (
        "validation",
        "images",
    )
    assert metadata["model"]["best_sha256"]
    assert metadata["model"]["last_sha256"]
    assert metadata["model"]["pretrained_sha256"]
    assert metadata["environment"]["python"]
    assert metadata["environment"]["dependencies"]
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert metrics["metrics"]["global"]["ground_truth_count"] > 0
    assert metrics["metrics"]["global"]["recall"] == pytest.approx(0.0)


def test_evaluate_detector_checkpoint_rejects_test_path(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_root, source_run = detector_source_run
    build_yolo_dataset(dataset_root, source_run, "yolo-input")
    unsafe_checkpoint = dataset_root / "base" / "test" / "checkpoint.pt"
    unsafe_checkpoint.write_bytes(b"checkpoint")
    config = _training_config(dataset_root, tmp_path, unsafe_checkpoint)
    evaluation_dir = tmp_path / "evaluation"

    with pytest.raises(DataValidationError, match="evaluation-only"):
        evaluate_detector_checkpoint(
            config,
            unsafe_checkpoint,
            RecordingBackend(),
            evaluation_dir,
        )

    assert not evaluation_dir.exists()


def test_evaluate_detector_checkpoint_writes_validation_artifacts(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_root, source_run = detector_source_run
    build_yolo_dataset(dataset_root, source_run, "yolo-input")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = _training_config(dataset_root, tmp_path, checkpoint)
    evaluation_dir = tmp_path / "standalone-evaluation"

    report = evaluate_detector_checkpoint(
        config,
        checkpoint,
        RecordingBackend(),
        evaluation_dir,
    )

    assert report.output_dir == evaluation_dir.resolve()
    assert report.predictions_path.is_file()
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert metrics["checkpoint_sha256"]
    assert metrics["metrics"]["global"]["ground_truth_count"] > 0


def test_train_detector_rejects_pretrained_model_in_test_path(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    dataset_root, _ = detector_source_run
    unsafe_pretrained = dataset_root / "base" / "test" / "pretrained.pt"
    unsafe_pretrained.write_bytes(b"pretrained")
    config = _training_config(dataset_root, tmp_path, unsafe_pretrained)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        train_detector(config, RecordingBackend())

    assert not (Path(config.output_root) / config.run_name).exists()


def test_ultralytics_backend_reads_checkpoints_from_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pretrained = tmp_path / "yolo11n.pt"
    pretrained.write_bytes(b"pretrained")

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            self.ckpt_path = Path(model)
            self.names = {0: "bread"}
            self.trainer = None

        def train(self, **kwargs: object) -> object:
            save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
            weights = save_dir / "weights"
            weights.mkdir(parents=True)
            best = weights / "best.pt"
            last = weights / "last.pt"
            best.write_bytes(b"best")
            last.write_bytes(b"last")
            self.trainer = SimpleNamespace(best=best, last=last)
            return object()

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    output_dir = tmp_path / "backend-output"

    result = UltralyticsBackend().train(
        model=str(pretrained),
        data_yaml=tmp_path / "data.yaml",
        output_dir=output_dir,
        arguments={
            "image_size": 640,
            "epochs": 1,
            "batch_size": 1,
            "seed": 42,
            "device": "0",
            "patience": 0,
            "workers": 0,
        },
    )

    assert result.best_checkpoint.read_bytes() == b"best"
    assert result.last_checkpoint.read_bytes() == b"last"
    assert result.pretrained_checkpoint == pretrained.resolve()
    assert result.class_names == ("bread",)
