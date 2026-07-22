from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from bakery_scanner.classifier_training import ClassifierTrainingConfig
from bakery_scanner.detector_evaluation import EvaluationThresholds
from bakery_scanner.detector_training import DetectorTrainingConfig
from bakery_scanner.e2e_evaluation import EndToEndPrediction
from bakery_scanner.e2e_inference import (
    BackendInferenceResult,
    EndToEndConfig,
    TorchEndToEndBackend,
    _validate_detector_checkpoint_provenance,
    _validate_yolo_source_binding,
    evaluate_end_to_end,
    load_end_to_end_config,
)
from bakery_scanner.errors import DataValidationError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path, dataset_root: Path) -> EndToEndConfig:
    (tmp_path / "detector.yaml").write_text("detector: fixture\n", encoding="utf-8")
    (tmp_path / "classifier.yaml").write_text("classifier: fixture\n", encoding="utf-8")
    return EndToEndConfig(
        dataset_root=str(dataset_root),
        detector_config=str(tmp_path / "detector.yaml"),
        classifier_config=str(tmp_path / "classifier.yaml"),
        detector_checkpoint=str(
            tmp_path / "runs" / "detector" / "detector" / "checkpoints" / "best.pt"
        ),
        classifier_checkpoint=str(
            tmp_path / "classifier-run" / "checkpoints" / "best.pt"
        ),
        output_root=str(tmp_path / "runs" / "e2e"),
        run_name="base-e2e",
    )


def _source_configs(dataset_root: Path, tmp_path: Path):
    pretrained_model = tmp_path / "weights.pt"
    if not pretrained_model.exists():
        pretrained_model.write_bytes(b"detector pretrained")
    detector = DetectorTrainingConfig(
        dataset_root=str(dataset_root),
        source_detector_run="det-run",
        yolo_run_name="det-yolo",
        output_root=str(tmp_path / "runs" / "detector"),
        run_name="detector",
        model=str(pretrained_model),
        image_size=640,
        epochs=1,
        batch_size=1,
        seed=42,
        device="0",
        patience=1,
        workers=0,
        thresholds=EvaluationThresholds(0.001, 0.25, 0.7, 0.5),
    )
    classifier = ClassifierTrainingConfig(
        dataset_root=str(dataset_root),
        source_classifier_run="cls-run",
        output_root=str(tmp_path / "runs" / "classifier"),
        run_name="classifier",
        architecture="resnet18",
        pretrained_model=str(tmp_path / "pretrained.pt"),
        image_size=224,
        epochs=1,
        batch_size=8,
        seed=42,
        device="0",
        patience=1,
        workers=0,
        learning_rate=0.001,
        weight_decay=0.0001,
    )
    return detector, classifier


def _fixture_files(tmp_path: Path, dataset_root: Path, registry_hash: str):
    detector_dir = dataset_root / "derived" / "detector" / "det-run"
    image_dir = detector_dir / "validation" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "real__scene_e_0001.jpg"
    Image.new("RGB", (32, 32), (20, 30, 40)).save(image_path)
    detector_manifest = detector_dir / "manifest.json"
    detector_manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "real__scene_e_0001.jpg",
                        "split": "validation",
                        "output_path": "validation/images/real__scene_e_0001.jpg",
                        "original_annotations": [
                            {"category_id": 2, "bbox": [1, 2, 10, 12]}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    classifier_dir = dataset_root / "derived" / "classifier" / "cls-run"
    classifier_dir.mkdir(parents=True)
    classifier_manifest = classifier_dir / "manifest.json"
    classifier_manifest.write_text(
        json.dumps(
            {
                "registry": {"sha256": registry_hash},
                "samples": [],
            }
        ),
        encoding="utf-8",
    )
    return detector_dir, detector_manifest, classifier_dir, classifier_manifest, image_path


def _checkpoint(path: Path, content: bytes) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    (path.parent.parent / "config.yaml").write_text(
        "fixture: true\n", encoding="utf-8"
    )
    training_manifest = path.parent.parent / "det-yolo" / "manifest.json"
    training_manifest.parent.mkdir()
    training_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    pretrained_model = path.parents[4] / "weights.pt"
    metadata = path.parent.parent / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "dataset": {
                    "manifest_path": str(training_manifest),
                    "manifest_sha256": _sha256(training_manifest),
                },
                "model": {
                    "best_sha256": _sha256(path),
                    "class_names": ["bread"],
                    "pretrained_sha256": (
                        _sha256(pretrained_model)
                        if pretrained_model.is_file()
                        else "0" * 64
                    ),
                },
                "backend_arguments": {
                    "image_size": 640,
                    "epochs": 1,
                    "batch_size": 1,
                    "seed": 42,
                    "device": "0",
                    "patience": 1,
                    "workers": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path, metadata


def test_detector_checkpoint_provenance_rejects_different_run_config(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    detector_config, _ = _source_configs(dataset_root, tmp_path)
    checkpoint, metadata_path = _checkpoint(
        tmp_path
        / "runs"
        / "detector"
        / "detector"
        / "checkpoints"
        / "best.pt",
        b"detector",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    different_config = replace(detector_config, source_detector_run="other-run")

    with pytest.raises(DataValidationError, match="detector run config"):
        _validate_detector_checkpoint_provenance(
            detector_config,
            different_config,
            checkpoint,
            metadata,
            Path(metadata["dataset"]["manifest_path"]),
            project_root=tmp_path,
        )


def test_detector_checkpoint_provenance_rejects_backend_argument_drift(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    detector_config, _ = _source_configs(dataset_root, tmp_path)
    checkpoint, metadata_path = _checkpoint(
        tmp_path
        / "runs"
        / "detector"
        / "detector"
        / "checkpoints"
        / "best.pt",
        b"detector",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["backend_arguments"]["seed"] = 99

    with pytest.raises(DataValidationError, match="backend arguments"):
        _validate_detector_checkpoint_provenance(
            detector_config,
            detector_config,
            checkpoint,
            metadata,
            Path(metadata["dataset"]["manifest_path"]),
            project_root=tmp_path,
        )


def test_detector_checkpoint_provenance_rejects_manifest_hash_drift(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    detector_config, _ = _source_configs(dataset_root, tmp_path)
    checkpoint, metadata_path = _checkpoint(
        tmp_path
        / "runs"
        / "detector"
        / "detector"
        / "checkpoints"
        / "best.pt",
        b"detector",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset"]["manifest_sha256"] = "0" * 64

    with pytest.raises(DataValidationError, match="manifest SHA-256"):
        _validate_detector_checkpoint_provenance(
            detector_config,
            detector_config,
            checkpoint,
            metadata,
            Path(metadata["dataset"]["manifest_path"]),
            project_root=tmp_path,
        )


def test_detector_checkpoint_provenance_rejects_pretrained_model_drift(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    detector_config, _ = _source_configs(dataset_root, tmp_path)
    checkpoint, metadata_path = _checkpoint(
        tmp_path
        / "runs"
        / "detector"
        / "detector"
        / "checkpoints"
        / "best.pt",
        b"detector",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    Path(detector_config.model).write_bytes(b"changed pretrained")

    with pytest.raises(DataValidationError, match="pretrained model SHA-256"):
        _validate_detector_checkpoint_provenance(
            detector_config,
            detector_config,
            checkpoint,
            metadata,
            Path(metadata["dataset"]["manifest_path"]),
            project_root=tmp_path,
        )


def test_detector_checkpoint_provenance_accepts_relocated_manifest_path(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "bakery_scanner_train"
    project_root.mkdir()
    dataset_root = project_root / "datasets"
    detector_config, _ = _source_configs(dataset_root, project_root)
    checkpoint, metadata_path = _checkpoint(
        project_root
        / "runs"
        / "detector"
        / "detector"
        / "checkpoints"
        / "best.pt",
        b"detector",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    yolo_manifest = Path(metadata["dataset"]["manifest_path"])
    metadata["dataset"]["manifest_path"] = str(
        project_root
        / ".worktrees"
        / "old"
        / yolo_manifest.relative_to(project_root)
    )

    _validate_detector_checkpoint_provenance(
        detector_config,
        detector_config,
        checkpoint,
        metadata,
        yolo_manifest,
        project_root=project_root,
    )


def test_yolo_source_binding_accepts_relocated_manifest_path(tmp_path: Path) -> None:
    project_root = tmp_path / "bakery_scanner_train"
    detector_manifest = (
        project_root / "datasets" / "derived" / "detector" / "manifest.json"
    )
    detector_manifest.parent.mkdir(parents=True)
    detector_manifest.write_text('{"samples": []}\n', encoding="utf-8")
    yolo_manifest = (
        project_root / "datasets" / "derived" / "yolo" / "manifest.json"
    )
    yolo_manifest.parent.mkdir(parents=True)
    yolo_manifest.write_text(
        json.dumps(
            {
                "source": {
                    "run_name": "selected-detector-run",
                    "manifest_path": str(
                        project_root
                        / ".worktrees"
                        / "old"
                        / detector_manifest.relative_to(project_root)
                    ),
                    "manifest_sha256": _sha256(detector_manifest),
                }
            }
        ),
        encoding="utf-8",
    )

    _validate_yolo_source_binding(
        yolo_manifest,
        "selected-detector-run",
        detector_manifest,
        project_root=project_root,
    )


def test_yolo_source_binding_rejects_relocated_manifest_hash_drift(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "bakery_scanner_train"
    detector_manifest = (
        project_root / "datasets" / "derived" / "detector" / "manifest.json"
    )
    detector_manifest.parent.mkdir(parents=True)
    detector_manifest.write_text('{"samples": []}\n', encoding="utf-8")
    yolo_manifest = (
        project_root / "datasets" / "derived" / "yolo" / "manifest.json"
    )
    yolo_manifest.parent.mkdir(parents=True)
    yolo_manifest.write_text(
        json.dumps(
            {
                "source": {
                    "run_name": "selected-detector-run",
                    "manifest_path": str(
                        project_root
                        / ".worktrees"
                        / "old"
                        / detector_manifest.relative_to(project_root)
                    ),
                    "manifest_sha256": "0" * 64,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="YOLO source"):
        _validate_yolo_source_binding(
            yolo_manifest,
            "selected-detector-run",
            detector_manifest,
            project_root=project_root,
        )


def test_yolo_source_binding_rejects_different_detector_run(tmp_path: Path) -> None:
    detector_manifest = tmp_path / "detector" / "manifest.json"
    detector_manifest.parent.mkdir()
    detector_manifest.write_text('{"samples": []}\n', encoding="utf-8")
    yolo_manifest = tmp_path / "yolo" / "manifest.json"
    yolo_manifest.parent.mkdir()
    yolo_manifest.write_text(
        json.dumps(
            {
                "source": {
                    "run_name": "other-detector-run",
                    "manifest_path": str(detector_manifest),
                    "manifest_sha256": _sha256(detector_manifest),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="YOLO source"):
        _validate_yolo_source_binding(
            yolo_manifest,
            "selected-detector-run",
            detector_manifest,
            project_root=tmp_path,
        )


class RecordingBackend:
    def __init__(self, *, empty: bool = False, fail: bool = False) -> None:
        self.empty = empty
        self.fail = fail
        self.call = None

    def cuda_available(self, device: str) -> bool:
        return device == "0"

    def predict(self, **kwargs) -> BackendInferenceResult:
        self.call = kwargs
        if self.fail:
            raise RuntimeError("inference failed")
        image_id = kwargs["images"][0].image_id
        target_index = kwargs["images"][0].objects[0].model_index
        predictions = () if self.empty else (
            EndToEndPrediction(
                image_id,
                (1.0, 2.0, 11.0, 14.0),
                target_index,
                0.9,
                0.8,
                0.72,
            ),
        )
        return BackendInferenceResult(
            predictions={image_id: predictions},
            classifier_batch_sizes={image_id: len(predictions)},
        )


def test_load_e2e_config_is_strict(tmp_path: Path) -> None:
    payload = {
        "dataset_root": "datasets",
        "detector_config": "configs/detector/yolo11n_base.yaml",
        "classifier_config": "configs/classifier/resnet18_base.yaml",
        "detector_checkpoint": "runs/detector/yolo11n_base_seed42/checkpoints/best.pt",
        "classifier_checkpoint": "runs/classifier/resnet18_base_seed42/checkpoints/best.pt",
        "output_root": "runs/e2e",
        "run_name": "base_resnet18_seed42",
    }
    path = tmp_path / "e2e.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_end_to_end_config(path)

    assert config.run_name == "base_resnet18_seed42"
    payload["extra"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="fields"):
        load_end_to_end_config(path)


def test_e2e_rejects_test_path_before_manifest_reads(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "datasets"
    config = _config(tmp_path, dataset_root)
    config = replace(
        config,
        detector_checkpoint=str(dataset_root / "base" / "test" / "best.pt"),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_detector_dataset",
        lambda *args: pytest.fail("must not read detector manifest"),
    )

    with pytest.raises(DataValidationError, match="evaluation-only"):
        evaluate_end_to_end(config, RecordingBackend())


def test_e2e_publishes_metrics_and_preserves_detector_checkpoint(
    tmp_path: Path, dataset_factory, monkeypatch
) -> None:
    dataset_root = dataset_factory()
    registry_hash = _sha256(dataset_root / "class_registry.json")
    detector_dir, detector_manifest, classifier_dir, classifier_manifest, _ = (
        _fixture_files(tmp_path, dataset_root, registry_hash)
    )
    config = _config(tmp_path, dataset_root)
    detector_config, classifier_config = _source_configs(dataset_root, tmp_path)
    detector_checkpoint, _ = _checkpoint(Path(config.detector_checkpoint), b"detector")
    classifier_checkpoint, _ = _checkpoint(
        Path(config.classifier_checkpoint), b"classifier"
    )
    detector_hash_before = _sha256(detector_checkpoint)
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.load_detector_training_config",
        lambda path: detector_config,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.load_classifier_training_config",
        lambda path: classifier_config,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_detector_dataset",
        lambda root, run: SimpleNamespace(
            output_dir=detector_dir, manifest_path=detector_manifest
        ),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_yolo_dataset",
        lambda root, run: SimpleNamespace(
            manifest_path=detector_checkpoint.parent.parent
            / "det-yolo"
            / "manifest.json"
        ),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference._validate_yolo_source_binding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_classifier_dataset",
        lambda root, run: SimpleNamespace(
            output_dir=classifier_dir,
            manifest_path=classifier_manifest,
            phase="base",
            output_dimension=15,
        ),
    )
    backend = RecordingBackend()

    report = evaluate_end_to_end(config, backend)

    assert _sha256(detector_checkpoint) == detector_hash_before
    assert {path.name for path in report.output_dir.iterdir()} == {
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "predictions.json",
    }
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert metrics["metrics"]["map50"] == 1.0
    assert metadata["model"]["detector_unchanged"] is True
    assert metadata["model"]["classifier_unchanged"] is True
    assert metadata["model"]["detector_sha256_before"] == detector_hash_before
    assert metadata["inference"]["classifier_batch_sizes"] == {
        "real__scene_e_0001.jpg": 1
    }
    assert backend.call["output_dimension"] == 15
    assert backend.call["arguments"]["detector_confidence"] == 0.001
    assert backend.call["arguments"]["detector_operating_confidence"] == 0.25
    assert metadata["configuration"]["detector_config_sha256"] == _sha256(
        Path(config.detector_config)
    )
    assert metadata["configuration"]["classifier_config_sha256"] == _sha256(
        Path(config.classifier_config)
    )


def test_e2e_accepts_empty_detections_and_cleans_failed_run(
    tmp_path: Path, dataset_factory, monkeypatch
) -> None:
    dataset_root = dataset_factory()
    registry_hash = _sha256(dataset_root / "class_registry.json")
    detector_dir, detector_manifest, classifier_dir, classifier_manifest, _ = (
        _fixture_files(tmp_path, dataset_root, registry_hash)
    )
    config = _config(tmp_path, dataset_root)
    detector_config, classifier_config = _source_configs(dataset_root, tmp_path)
    detector_checkpoint, _ = _checkpoint(
        Path(config.detector_checkpoint), b"detector"
    )
    _checkpoint(Path(config.classifier_checkpoint), b"classifier")
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.load_detector_training_config",
        lambda path: detector_config,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.load_classifier_training_config",
        lambda path: classifier_config,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_detector_dataset",
        lambda root, run: SimpleNamespace(
            output_dir=detector_dir, manifest_path=detector_manifest
        ),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_yolo_dataset",
        lambda root, run: SimpleNamespace(
            manifest_path=detector_checkpoint.parent.parent
            / "det-yolo"
            / "manifest.json"
        ),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference._validate_yolo_source_binding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_classifier_dataset",
        lambda root, run: SimpleNamespace(
            output_dir=classifier_dir,
            manifest_path=classifier_manifest,
            phase="base",
            output_dimension=15,
        ),
    )

    report = evaluate_end_to_end(config, RecordingBackend(empty=True))
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    assert metrics["metrics"]["prediction_count"] == 0
    assert metrics["metrics"]["map50"] == 0.0

    failed_config = replace(config, run_name="failed-e2e")
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluate_end_to_end(failed_config, RecordingBackend(fail=True))
    output_root = Path(config.output_root)
    assert not (output_root / "failed-e2e").exists()
    assert not list(output_root.glob(".failed-e2e.tmp-*"))


def test_e2e_rejects_invalid_classifier_batch_accounting(
    tmp_path: Path, dataset_factory, monkeypatch
) -> None:
    dataset_root = dataset_factory()
    registry_hash = _sha256(dataset_root / "class_registry.json")
    detector_dir, detector_manifest, classifier_dir, classifier_manifest, _ = (
        _fixture_files(tmp_path, dataset_root, registry_hash)
    )
    config = _config(tmp_path, dataset_root)
    detector_config, classifier_config = _source_configs(dataset_root, tmp_path)
    detector_checkpoint, _ = _checkpoint(
        Path(config.detector_checkpoint), b"detector"
    )
    _checkpoint(Path(config.classifier_checkpoint), b"classifier")
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.load_detector_training_config",
        lambda path: detector_config,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.load_classifier_training_config",
        lambda path: classifier_config,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_detector_dataset",
        lambda root, run: SimpleNamespace(
            output_dir=detector_dir, manifest_path=detector_manifest
        ),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_yolo_dataset",
        lambda root, run: SimpleNamespace(
            manifest_path=detector_checkpoint.parent.parent
            / "det-yolo"
            / "manifest.json"
        ),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference._validate_yolo_source_binding",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference.validate_classifier_dataset",
        lambda root, run: SimpleNamespace(
            output_dir=classifier_dir,
            manifest_path=classifier_manifest,
            phase="base",
            output_dimension=15,
        ),
    )

    class InvalidBatchBackend(RecordingBackend):
        def predict(self, **kwargs):
            result = super().predict(**kwargs)
            return BackendInferenceResult(result.predictions, {})

    with pytest.raises(DataValidationError, match="batch sizes"):
        evaluate_end_to_end(config, InvalidBatchBackend())


def test_torch_backend_classifies_all_crops_once_per_scene(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    image_paths = []
    images = []
    for index in range(2):
        path = tmp_path / f"scene-{index}.jpg"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(path)
        image_paths.append(path)
        images.append(SimpleNamespace(image_id=path.name))

    class FakeBoxes:
        def __init__(self, count: int) -> None:
            self.xyxy = torch.tensor(
                [[1, 1, 10, 10], [12, 12, 20, 20]][:count], dtype=torch.float32
            )
            self.conf = torch.tensor([0.9, 0.8][:count], dtype=torch.float32)
            self.cls = torch.zeros(count, dtype=torch.float32)

    class FakeDetector:
        names = {0: "bread"}

        def __init__(self, checkpoint: str) -> None:
            self.checkpoint = checkpoint

        def predict(self, **kwargs):
            return [
                SimpleNamespace(boxes=FakeBoxes(2)),
                SimpleNamespace(boxes=FakeBoxes(1)),
            ]

    calls = []

    class FakeClassifier:
        def __call__(self, batch):
            calls.append(tuple(batch.shape))
            logits = torch.zeros((batch.shape[0], 15), dtype=torch.float32)
            logits[:, 0] = 2.0
            return logits

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeDetector))
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference._load_classifier_checkpoint_model",
        lambda *args, **kwargs: FakeClassifier(),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference._classifier_transforms",
        lambda size: (None, lambda image: torch.zeros((3, 8, 8))),
    )
    monkeypatch.setattr(
        "bakery_scanner.e2e_inference._torch_device",
        lambda value: torch.device("cpu"),
    )

    result = TorchEndToEndBackend().predict(
        image_paths=image_paths,
        images=images,
        detector_checkpoint=tmp_path / "detector.pt",
        classifier_checkpoint=tmp_path / "classifier.pt",
        classifier_context={"context_version": 1},
        output_dimension=15,
        arguments={
            "device": "0",
            "detector_image_size": 640,
            "detector_confidence": 0.25,
            "detector_nms_iou": 0.7,
            "classifier_image_size": 224,
        },
    )

    assert calls == [(2, 3, 8, 8), (1, 3, 8, 8)]
    assert result.classifier_batch_sizes == {"scene-0.jpg": 2, "scene-1.jpg": 1}
    assert all(
        prediction.model_index == 0
        for predictions in result.predictions.values()
        for prediction in predictions
    )
