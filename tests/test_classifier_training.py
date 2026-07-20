from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from bakery_scanner.classifier_dataset import (
    ClassifierDatasetConfig,
    build_classifier_dataset,
)
from bakery_scanner.classifier_evaluation import ClassifierPrediction
from bakery_scanner.classifier_training import (
    BackendTrainingResult,
    ClassifierSample,
    ClassifierTrainingConfig,
    IncrementalClassifierTrainingConfig,
    TorchvisionClassifierBackend,
    _balanced_class_statistics,
    _build_incremental_resnet18,
    _build_resnet18,
    _validate_incremental_validation_support,
    evaluate_classifier_checkpoint,
    load_classifier_experiment_config,
    load_classifier_training_config,
    train_classifier,
)
from bakery_scanner.errors import DataValidationError


def _config(dataset_root: Path, pretrained: Path, *, source: str = "base-run"):
    return ClassifierTrainingConfig(
        dataset_root=str(dataset_root),
        source_classifier_run=source,
        output_root=str(dataset_root.parent / "runs" / "classifier"),
        run_name="baseline",
        architecture="resnet18",
        pretrained_model=str(pretrained),
        image_size=224,
        epochs=3,
        batch_size=8,
        seed=42,
        device="0",
        patience=1,
        workers=0,
        learning_rate=0.001,
        weight_decay=0.0001,
    )


def _build_base_run(dataset_root: Path) -> None:
    build_classifier_dataset(
        ClassifierDatasetConfig(
            dataset_root=dataset_root,
            run_name="base-run",
            phase="base",
            seed=42,
            validation_fraction=0.5,
            expected_base_images_per_class=1,
            expected_incremental_images_per_class=7,
        )
    )


def _build_incremental_run(dataset_root: Path) -> None:
    for class_dir in (dataset_root / "incremental").iterdir():
        if class_dir.is_dir():
            Image.new("RGB", (16, 16), (32, 64, 96)).save(
                class_dir / "second.jpg"
            )
    build_classifier_dataset(
        ClassifierDatasetConfig(
            dataset_root=dataset_root,
            run_name="incremental-run",
            phase="incremental",
            seed=42,
            validation_fraction=0.5,
            expected_base_images_per_class=1,
            expected_incremental_images_per_class=2,
        )
    )


def _incremental_config(
    dataset_root: Path,
    base_checkpoint: Path,
    detector_checkpoint: Path,
) -> IncrementalClassifierTrainingConfig:
    return IncrementalClassifierTrainingConfig(
        phase="incremental",
        dataset_root=str(dataset_root),
        source_classifier_run="incremental-run",
        output_root=str(dataset_root.parent / "runs" / "classifier"),
        run_name="incremental-baseline",
        architecture="resnet18",
        base_checkpoint=str(base_checkpoint),
        frozen_detector_checkpoint=str(detector_checkpoint),
        image_size=224,
        epochs=3,
        batch_size=8,
        seed=42,
        device="0",
        patience=1,
        workers=0,
        learning_rate=0.001,
        weight_decay=0.0001,
    )


def _recorded_checkpoint(
    path: Path,
    content: bytes,
    *,
    kind: str,
) -> Path:
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    if kind == "detector":
        metadata = {
            "model": {
                "best_sha256": hashlib.sha256(content).hexdigest(),
                "class_names": ["bread"],
            }
        }
    else:
        metadata = {
            "dataset": {"output_dimension": 15},
            "model": {
                "architecture": "resnet18",
                "best_sha256": hashlib.sha256(content).hexdigest(),
            },
        }
    (path.parent.parent / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return path


class RecordingBackend:
    def __init__(self, *, available: bool = True, fail: bool = False) -> None:
        self.available = available
        self.fail = fail
        self.train_call = None
        self.predict_calls = []

    def cuda_available(self, device: str) -> bool:
        return self.available and device == "0"

    def train(self, **kwargs) -> BackendTrainingResult:
        self.train_call = kwargs
        if self.fail:
            raise RuntimeError("backend failed")
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        best = output_dir / "best.pt"
        last = output_dir / "last.pt"
        best.write_bytes(b"best classifier")
        last.write_bytes(b"last classifier")
        return BackendTrainingResult(
            best_checkpoint=best,
            last_checkpoint=last,
            best_epoch=2,
            epochs_completed=3,
            history=(
                {"epoch": 1, "train_loss": 1.2, "validation_loss": 1.0},
                {"epoch": 2, "train_loss": 0.8, "validation_loss": 0.6},
                {"epoch": 3, "train_loss": 0.5, "validation_loss": 0.7},
            ),
        )

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return tuple(
            ClassifierPrediction(
                sample.sample_id,
                sample.target_index,
                sample.target_index,
                0.9,
            )
            for sample in kwargs["samples"]
        )


def test_load_classifier_config_is_strict(tmp_path: Path) -> None:
    payload = {
        "dataset_root": "datasets",
        "source_classifier_run": "base_seed42",
        "output_root": "runs/classifier",
        "run_name": "resnet18_base_seed42",
        "architecture": "resnet18",
        "pretrained_model": "models/pretrained/resnet18-f37072fd.pth",
        "image_size": 224,
        "epochs": 30,
        "batch_size": 64,
        "seed": 42,
        "device": "0",
        "patience": 5,
        "workers": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_classifier_training_config(path)

    assert config.architecture == "resnet18"
    assert config.learning_rate == 0.001

    payload["unexpected"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="fields"):
        load_classifier_training_config(path)


def test_load_incremental_classifier_config_is_strict(tmp_path: Path) -> None:
    payload = {
        "phase": "incremental",
        "dataset_root": "datasets",
        "source_classifier_run": "incremental_seed42",
        "output_root": "runs/classifier",
        "run_name": "resnet18_incremental_seed42",
        "architecture": "resnet18",
        "base_checkpoint": "runs/classifier/resnet18_base_seed42/checkpoints/best.pt",
        "frozen_detector_checkpoint": "runs/detector/yolo11n_base_seed42/checkpoints/best.pt",
        "image_size": 224,
        "epochs": 30,
        "batch_size": 64,
        "seed": 42,
        "device": "0",
        "patience": 5,
        "workers": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    path = tmp_path / "incremental.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_classifier_experiment_config(path)

    assert isinstance(config, IncrementalClassifierTrainingConfig)
    assert config.phase == "incremental"
    assert config.base_checkpoint.endswith("best.pt")

    payload["unexpected"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="fields"):
        load_classifier_experiment_config(path)


def test_experiment_loader_preserves_legacy_base_config(tmp_path: Path) -> None:
    payload = {
        "dataset_root": "datasets",
        "source_classifier_run": "base_seed42",
        "output_root": "runs/classifier",
        "run_name": "resnet18_base_seed42",
        "architecture": "resnet18",
        "pretrained_model": "models/pretrained/resnet18-f37072fd.pth",
        "image_size": 224,
        "epochs": 30,
        "batch_size": 64,
        "seed": 42,
        "device": "0",
        "patience": 5,
        "workers": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_classifier_experiment_config(path)

    assert isinstance(config, ClassifierTrainingConfig)
    assert config == load_classifier_training_config(path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("architecture", "mobilenet", "architecture"),
        ("image_size", 0, "image_size"),
        ("device", "cpu", "device"),
        ("learning_rate", 0, "learning_rate"),
        ("weight_decay", -1, "weight_decay"),
    ],
)
def test_load_classifier_config_rejects_invalid_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = {
        "dataset_root": "datasets",
        "source_classifier_run": "base_seed42",
        "output_root": "runs/classifier",
        "run_name": "baseline",
        "architecture": "resnet18",
        "pretrained_model": "weights.pt",
        "image_size": 224,
        "epochs": 30,
        "batch_size": 64,
        "seed": 42,
        "device": "0",
        "patience": 5,
        "workers": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    payload[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match=message):
        load_classifier_training_config(path)


def test_train_classifier_validates_paths_before_dataset_reads(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "datasets"
    forbidden = dataset_root / "base" / "test" / "secret.pt"
    config = _config(dataset_root, forbidden)
    monkeypatch.setattr(
        "bakery_scanner.classifier_training.validate_classifier_dataset",
        lambda *args: pytest.fail("dataset must not be read before path safety"),
    )

    with pytest.raises(DataValidationError, match="evaluation-only"):
        train_classifier(config, RecordingBackend())


def test_incremental_training_validates_artifacts_before_dataset_reads(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "datasets"
    forbidden = dataset_root / "incremental" / "test" / "base.pt"
    config = _incremental_config(
        dataset_root,
        forbidden,
        tmp_path / "detector" / "best.pt",
    )
    monkeypatch.setattr(
        "bakery_scanner.classifier_training.validate_classifier_dataset",
        lambda *args: pytest.fail("dataset must not be read before path safety"),
    )

    with pytest.raises(DataValidationError, match="evaluation-only"):
        train_classifier(config, RecordingBackend())


def test_train_classifier_requires_base_15_class_run(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    for class_dir in (dataset_root / "incremental").iterdir():
        if class_dir.is_dir():
            Image.new("RGB", (16, 16), (32, 64, 96)).save(class_dir / "second.jpg")
    build_classifier_dataset(
        ClassifierDatasetConfig(
            dataset_root=dataset_root,
            run_name="incremental-run",
            phase="incremental",
            seed=42,
            validation_fraction=0.5,
            expected_base_images_per_class=1,
            expected_incremental_images_per_class=2,
        )
    )
    pretrained = tmp_path / "pretrained.pt"
    pretrained.write_bytes(b"weights")

    with pytest.raises(DataValidationError, match="Base.*15"):
        train_classifier(
            _config(dataset_root, pretrained, source="incremental-run"),
            RecordingBackend(),
        )


def test_train_classifier_publishes_complete_reproducible_run(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    _build_base_run(dataset_root)
    pretrained = tmp_path / "pretrained.pt"
    pretrained.write_bytes(b"official weights")
    backend = RecordingBackend()

    report = train_classifier(_config(dataset_root, pretrained), backend)

    assert report.output_dir.is_dir()
    assert {path.name for path in report.output_dir.iterdir()} == {
        "checkpoints",
        "config.yaml",
        "history.json",
        "metadata.json",
        "metrics.json",
        "predictions.json",
    }
    assert report.best_checkpoint.read_bytes() == b"best classifier"
    assert report.last_checkpoint.read_bytes() == b"last classifier"
    assert backend.train_call["output_dimension"] == 15
    context = backend.train_call["checkpoint_context"]
    assert context["context_version"] == 1
    assert len(context["model_index_mapping"]) == 15
    assert len(context["source_manifest_sha256"]) == 64
    assert {sample.split for sample in backend.train_call["train_samples"]} == {"train"}
    assert {sample.split for sample in backend.train_call["validation_samples"]} == {
        "validation"
    }
    arguments = dict(backend.train_call["arguments"])
    class_counts = arguments.pop("class_counts")
    class_weights = arguments.pop("class_weights")
    assert arguments == {
        "architecture": "resnet18",
        "image_size": 224,
        "epochs": 3,
        "batch_size": 8,
        "seed": 42,
        "device": "0",
        "patience": 1,
        "workers": 0,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    assert len(class_counts) == 15
    assert len(class_weights) == 15
    assert all(count > 0 for count in class_counts)
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))
    assert metadata["dataset"]["output_dimension"] == 15
    assert context["source_manifest_sha256"] == metadata["dataset"][
        "manifest_sha256"
    ]
    assert context["registry_sha256"] == metadata["dataset"]["registry_sha256"]
    assert metadata["model"]["pretrained_sha256"] == hashlib.sha256(
        b"official weights"
    ).hexdigest()
    assert metadata["model"]["best_sha256"] == hashlib.sha256(
        b"best classifier"
    ).hexdigest()
    assert metadata["environment"]["python"]
    assert metadata["environment"]["dependencies"]["torch"]
    assert metadata["dataset"]["registry_sha256"]
    assert metadata["class_balance"] == {
        "formula": "train_samples / (output_dimension * class_count)",
        "class_counts": list(class_counts),
        "class_weights": list(class_weights),
    }
    assert metadata["determinism"] == {
        "python_seeded": True,
        "torch_seeded": True,
        "cuda_seeded": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "torch_deterministic_algorithms": False,
    }
    assert metadata["preprocessing"] == {
        "train": [
            "RandomResizedCrop(scale=[0.8,1.0])",
            "RandomHorizontalFlip",
            "ColorJitter(0.1,0.1,0.1,0.05)",
            "ToTensor",
            "ImageNetNormalize",
        ],
        "validation": [
            "Resize(image_size/0.875)",
            "CenterCrop(image_size)",
            "ToTensor",
            "ImageNetNormalize",
        ],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    assert metrics["split"] == "validation"
    assert metrics["metric_version"] == 1
    assert metrics["metrics"]["top1_accuracy"] == 1.0


def test_train_incremental_classifier_publishes_20_outputs_and_freezes_detector(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    _build_incremental_run(dataset_root)
    base_checkpoint = _recorded_checkpoint(
        tmp_path / "runs" / "classifier" / "base" / "checkpoints" / "best.pt",
        b"base classifier",
        kind="classifier",
    )
    detector_checkpoint = _recorded_checkpoint(
        tmp_path / "runs" / "detector" / "base" / "checkpoints" / "best.pt",
        b"frozen detector",
        kind="detector",
    )
    detector_hash = hashlib.sha256(detector_checkpoint.read_bytes()).hexdigest()
    backend = RecordingBackend()

    report = train_classifier(
        _incremental_config(
            dataset_root, base_checkpoint, detector_checkpoint
        ),
        backend,
    )

    assert backend.train_call["pretrained_model"] == base_checkpoint.resolve()
    assert backend.train_call["output_dimension"] == 20
    assert len(backend.train_call["checkpoint_context"]["model_index_mapping"]) == 20
    assert "detector_checkpoint" not in backend.train_call
    assert (
        "frozen_detector_checkpoint"
        not in backend.train_call["checkpoint_context"]["config"]
    )
    assert backend.predict_calls
    assert all(
        "frozen_detector_checkpoint" not in call["checkpoint_context"]["config"]
        for call in backend.predict_calls
    )
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))["metrics"]
    assert metadata["dataset"]["phase"] == "incremental"
    assert metadata["model"]["base_checkpoint_sha256"] == hashlib.sha256(
        b"base classifier"
    ).hexdigest()
    assert metadata["initialization"] == {
        "source_output_dimension": 15,
        "target_output_dimension": 20,
        "copied_base_rows": 15,
        "new_rows": 5,
    }
    assert metadata["frozen_detector"] == {
        "checkpoint": str(detector_checkpoint.resolve()),
        "sha256_before": detector_hash,
        "sha256_after": detector_hash,
        "detector_unchanged": True,
    }
    assert metrics["phase"]["base"]["sample_count"] > 0
    assert metrics["phase"]["incremental"]["sample_count"] == 5


def test_incremental_training_detects_detector_mutation_and_cleans_staging(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    _build_incremental_run(dataset_root)
    base_checkpoint = _recorded_checkpoint(
        tmp_path / "runs" / "classifier" / "base" / "checkpoints" / "best.pt",
        b"base classifier",
        kind="classifier",
    )
    detector_checkpoint = _recorded_checkpoint(
        tmp_path / "runs" / "detector" / "base" / "checkpoints" / "best.pt",
        b"frozen detector",
        kind="detector",
    )

    class MutatingBackend(RecordingBackend):
        def train(self, **kwargs):
            result = super().train(**kwargs)
            detector_checkpoint.write_bytes(b"mutated detector")
            return result

    config = _incremental_config(
        dataset_root, base_checkpoint, detector_checkpoint
    )
    with pytest.raises(DataValidationError, match="detector checkpoint changed"):
        train_classifier(config, MutatingBackend())
    assert not (Path(config.output_root) / config.run_name).exists()
    assert not list(Path(config.output_root).glob(f".{config.run_name}.tmp-*"))


def test_evaluate_incremental_checkpoint_uses_20_outputs_and_freezes_detector(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    _build_incremental_run(dataset_root)
    base_checkpoint = _recorded_checkpoint(
        tmp_path / "runs" / "classifier" / "base" / "checkpoints" / "best.pt",
        b"base classifier",
        kind="classifier",
    )
    detector_checkpoint = _recorded_checkpoint(
        tmp_path / "runs" / "detector" / "base" / "checkpoints" / "best.pt",
        b"frozen detector",
        kind="detector",
    )
    incremental_checkpoint = tmp_path / "incremental-best.pt"
    incremental_checkpoint.write_bytes(b"incremental classifier")
    backend = RecordingBackend()
    output = tmp_path / "incremental-evaluation"

    report = evaluate_classifier_checkpoint(
        _incremental_config(
            dataset_root, base_checkpoint, detector_checkpoint
        ),
        incremental_checkpoint,
        backend,
        output,
    )

    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))["metrics"]
    assert backend.train_call is None
    assert metrics["phase"]["incremental"]["sample_count"] == 5
    assert report.output_dir == output


def test_train_classifier_rejects_missing_cuda_and_cleans_failed_staging(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    _build_base_run(dataset_root)
    pretrained = tmp_path / "pretrained.pt"
    pretrained.write_bytes(b"weights")
    config = _config(dataset_root, pretrained)

    with pytest.raises(DataValidationError, match="CUDA"):
        train_classifier(config, RecordingBackend(available=False))
    assert not (tmp_path / "runs" / "classifier" / "baseline").exists()

    with pytest.raises(RuntimeError, match="backend failed"):
        train_classifier(config, RecordingBackend(fail=True))
    output_root = tmp_path / "runs" / "classifier"
    assert not (output_root / "baseline").exists()
    assert not list(output_root.glob(".baseline.tmp-*"))


def test_train_classifier_rejects_predictions_for_wrong_sample_ids(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    _build_base_run(dataset_root)
    pretrained = tmp_path / "pretrained.pt"
    pretrained.write_bytes(b"weights")

    class WrongSampleBackend(RecordingBackend):
        def predict(self, **kwargs):
            return tuple(
                ClassifierPrediction(
                    f"wrong-{index}", sample.target_index, sample.target_index, 0.9
                )
                for index, sample in enumerate(kwargs["samples"])
            )

    with pytest.raises(DataValidationError, match="sample IDs"):
        train_classifier(_config(dataset_root, pretrained), WrongSampleBackend())
    assert not (tmp_path / "runs" / "classifier" / "baseline").exists()


def test_build_resnet18_strictly_loads_imagenet_state_and_replaces_head(
    tmp_path: Path,
) -> None:
    import torch
    from torchvision.models import resnet18

    pretrained = tmp_path / "resnet18.pth"
    torch.save(resnet18(weights=None).state_dict(), pretrained)

    model = _build_resnet18(pretrained, output_dimension=15)

    assert model.fc.in_features == 512
    assert model.fc.out_features == 15
    invalid = tmp_path / "invalid.pth"
    torch.save({"conv1.weight": model.conv1.weight.detach()}, invalid)
    with pytest.raises(DataValidationError, match="strict"):
        _build_resnet18(invalid, output_dimension=15)


def _incremental_checkpoint_fixture(tmp_path: Path):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 15)
    with torch.no_grad():
        base_model.fc.weight.copy_(
            torch.arange(base_model.fc.weight.numel(), dtype=torch.float32).reshape_as(
                base_model.fc.weight
            )
            / 1000
        )
        base_model.fc.bias.copy_(torch.arange(15, dtype=torch.float32))
    mapping = [
        {
            "model_index": index,
            "category_id": 100 + index,
            "canonical_name": f"class-{index}",
        }
        for index in range(20)
    ]
    base_context = {
        "context_version": 1,
        "source_manifest_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "model_index_mapping": mapping[:15],
        "config": {"run_name": "base"},
    }
    checkpoint = tmp_path / "base-best.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "architecture": "resnet18",
            "output_dimension": 15,
            "image_size": 224,
            "epoch": 3,
            "validation_loss": 0.5,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": {},
            "context": base_context,
        },
        checkpoint,
    )
    incremental_context = {
        **base_context,
        "source_manifest_sha256": "c" * 64,
        "model_index_mapping": mapping,
        "config": {"run_name": "incremental"},
    }
    return checkpoint, base_model, incremental_context


def test_incremental_resnet18_copies_backbone_and_first_fifteen_head_rows(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint, base_model, context = _incremental_checkpoint_fixture(tmp_path)

    expanded, evidence = _build_incremental_resnet18(
        checkpoint,
        output_dimension=20,
        checkpoint_context=context,
        image_size=224,
    )

    for key, tensor in base_model.state_dict().items():
        if not key.startswith("fc."):
            assert torch.equal(expanded.state_dict()[key], tensor)
    assert torch.equal(expanded.fc.weight[:15], base_model.fc.weight)
    assert torch.equal(expanded.fc.bias[:15], base_model.fc.bias)
    assert expanded.fc.out_features == 20
    assert evidence == {
        "source_output_dimension": 15,
        "target_output_dimension": 20,
        "copied_base_rows": 15,
        "new_rows": 5,
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("dimension", "20 outputs"),
        ("registry", "registry"),
        ("mapping", "mapping"),
        ("image_size", "image size"),
    ],
)
def test_incremental_resnet18_rejects_incompatible_base_checkpoint(
    tmp_path: Path, mutation: str, message: str
) -> None:
    import torch

    checkpoint, _base_model, context = _incremental_checkpoint_fixture(tmp_path)
    output_dimension = 20
    image_size = 224
    if mutation == "dimension":
        output_dimension = 19
    elif mutation == "registry":
        context = {**context, "registry_sha256": "d" * 64}
    elif mutation == "mapping":
        mapping = list(context["model_index_mapping"])
        mapping[0] = {**mapping[0], "canonical_name": "changed"}
        context = {**context, "model_index_mapping": mapping}
    elif mutation == "image_size":
        image_size = 128

    with pytest.raises(DataValidationError, match=message):
        _build_incremental_resnet18(
            checkpoint,
            output_dimension=output_dimension,
            checkpoint_context=context,
            image_size=image_size,
        )


def test_balanced_class_statistics_require_support_and_match_formula(
    tmp_path: Path,
) -> None:
    samples = tuple(
        ClassifierSample(str(index), tmp_path / str(index), target, "train")
        for index, target in enumerate((0, 0, 0, 0, 1, 1, 2))
    )

    counts, weights = _balanced_class_statistics(samples, output_dimension=3)

    assert counts == (4, 2, 1)
    assert weights == pytest.approx((7 / 12, 7 / 6, 7 / 3))
    with pytest.raises(DataValidationError, match="every output"):
        _balanced_class_statistics(samples[:-1], output_dimension=3)


def test_incremental_validation_requires_all_five_new_classes(tmp_path: Path) -> None:
    samples = tuple(
        ClassifierSample(str(index), tmp_path / str(index), index, "validation")
        for index in range(15, 19)
    )

    with pytest.raises(DataValidationError, match="new class"):
        _validate_incremental_validation_support(samples)

    complete = samples + (
        ClassifierSample("19", tmp_path / "19", 19, "validation"),
    )
    _validate_incremental_validation_support(complete)


def _tiny_samples(tmp_path: Path, split: str, count: int) -> tuple[ClassifierSample, ...]:
    samples = []
    for index in range(count):
        path = tmp_path / f"{split}-{index}.jpg"
        Image.new(
            "RGB", (64, 64), ((index * 53) % 255, (index * 97) % 255, 128)
        ).save(path)
        samples.append(
            ClassifierSample(
                sample_id=path.name,
                image_path=path,
                target_index=index % 2,
                split=split,
            )
        )
    return tuple(samples)


def test_torchvision_backend_writes_checkpoint_schema_and_predicts_probabilities(
    tmp_path: Path,
) -> None:
    import torch
    from torchvision.models import resnet18

    pretrained = tmp_path / "resnet18.pth"
    torch.save(resnet18(weights=None).state_dict(), pretrained)
    train_samples = _tiny_samples(tmp_path, "train", 4)
    validation_samples = _tiny_samples(tmp_path, "validation", 2)
    arguments = {
        "architecture": "resnet18",
        "image_size": 32,
        "epochs": 1,
        "batch_size": 2,
        "seed": 7,
        "device": "cpu",
        "patience": 0,
        "workers": 0,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    checkpoint_context = {
        "context_version": 1,
        "source_manifest_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "model_index_mapping": [
            {"model_index": 0, "category_id": 10, "canonical_name": "zero"},
            {"model_index": 1, "category_id": 20, "canonical_name": "one"},
        ],
        "config": {"run_name": "tiny"},
    }
    backend = TorchvisionClassifierBackend()

    result = backend.train(
        pretrained_model=pretrained,
        train_samples=train_samples,
        validation_samples=validation_samples,
        output_dimension=2,
        checkpoint_context=checkpoint_context,
        output_dir=tmp_path / "backend",
        arguments=arguments,
    )

    best = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)
    assert set(best) == {
        "checkpoint_version",
        "architecture",
        "output_dimension",
        "image_size",
        "epoch",
        "validation_loss",
        "model_state_dict",
        "optimizer_state_dict",
        "context",
    }
    assert best["architecture"] == "resnet18"
    assert best["output_dimension"] == 2
    assert result.best_epoch == 1
    assert result.history[0]["validation_macro_f1"] is not None
    assert result.history[0]["validation_evaluated_class_count"] == 2
    predictions = backend.predict(
        checkpoint=result.best_checkpoint,
        samples=validation_samples,
        output_dimension=2,
        checkpoint_context=checkpoint_context,
        arguments=arguments,
    )
    assert [prediction.sample_id for prediction in predictions] == [
        sample.sample_id for sample in validation_samples
    ]
    assert all(0 <= prediction.confidence <= 1 for prediction in predictions)
    assert all(0 <= prediction.predicted_index < 2 for prediction in predictions)

    mismatched_context = dict(checkpoint_context)
    mismatched_context["source_manifest_sha256"] = "c" * 64
    with pytest.raises(DataValidationError, match="context"):
        backend.predict(
            checkpoint=result.best_checkpoint,
            samples=validation_samples,
            output_dimension=2,
            checkpoint_context=mismatched_context,
            arguments=arguments,
        )


def test_torchvision_backend_wraps_corrupt_checkpoint_as_validation_error(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    sample = _tiny_samples(tmp_path, "validation", 1)

    with pytest.raises(DataValidationError, match="cannot load classifier checkpoint"):
        TorchvisionClassifierBackend().predict(
            checkpoint=checkpoint,
            samples=sample,
            output_dimension=2,
            checkpoint_context={"context_version": 1},
            arguments={
                "image_size": 32,
                "batch_size": 1,
                "workers": 0,
                "device": "cpu",
            },
        )
