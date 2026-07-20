from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bakery_scanner.errors import DataValidationError
from bakery_scanner.final_evaluation import (
    FinalEvaluationConfig,
    FrozenClassifierConfig,
    FrozenDetectorConfig,
    FrozenInferenceConfig,
    FrozenRegistryConfig,
    FrozenTestSplitConfig,
    FinalInferenceBackend,
    FinalInferenceResult,
    FinalSplitInferenceResult,
    FinalEvaluationPreflightReport,
    LoadedTestSplit,
    PreparedFinalEvaluation,
    _create_start_lock,
    _load_locked_test_splits,
    _load_test_split,
    _update_start_lock,
    load_final_evaluation_config,
    preflight_final_evaluation,
    run_final_evaluation,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _payload() -> dict[str, object]:
    return {
        "freeze_version": 1,
        "evaluation_id": "fixture_frozen_v1",
        "frozen_at_utc": "2026-07-20T10:38:51Z",
        "selection_status": "frozen_before_test_access",
        "selection_basis": "train_side_validation_only",
        "dataset_root": "datasets",
        "registry": {"path": "datasets/class_registry.json", "sha256": "a" * 64},
        "test_splits": {
            "base": {
                "coco_path": "datasets/base/test/instances_test.json",
                "expected_phase": "base",
            },
            "incremental": {
                "coco_path": "datasets/incremental/test/instances_test.json",
                "expected_phase": "incremental",
            },
        },
        "detector": {
            "config_path": "configs/detector.yaml",
            "config_sha256": "b" * 64,
            "checkpoint": "runs/detector/checkpoints/best.pt",
            "checkpoint_sha256": "c" * 64,
            "image_size": 640,
            "confidence_floor": 0.001,
            "operating_confidence": 0.25,
            "nms_iou": 0.7,
            "matching_iou": 0.5,
        },
        "classifiers": {
            "base": {
                "config_path": "configs/base.yaml",
                "config_sha256": "d" * 64,
                "checkpoint": "runs/base/checkpoints/best.pt",
                "checkpoint_sha256": "e" * 64,
                "output_dimension": 15,
                "image_size": 224,
            },
            "incremental": {
                "config_path": "configs/incremental.yaml",
                "config_sha256": "f" * 64,
                "checkpoint": "runs/incremental/checkpoints/best.pt",
                "checkpoint_sha256": "1" * 64,
                "output_dimension": 20,
                "image_size": 224,
            },
        },
        "inference": {
            "device": "0",
            "classifier_batch_strategy": "one_batch_per_scene",
            "combined_score": "detector_confidence_times_classifier_confidence",
        },
        "output_root": "runs/final_evaluation",
        "run_name": "frozen_v1",
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_final_evaluation_config_is_strict(tmp_path: Path) -> None:
    config = load_final_evaluation_config(_write(tmp_path / "frozen.yaml", _payload()))

    assert isinstance(config, FinalEvaluationConfig)
    assert config.freeze_version == 1
    assert config.registry == FrozenRegistryConfig("datasets/class_registry.json", "a" * 64)
    assert config.test_splits["base"] == FrozenTestSplitConfig(
        "datasets/base/test/instances_test.json", "base"
    )
    assert isinstance(config.detector, FrozenDetectorConfig)
    assert config.detector.operating_confidence == 0.25
    assert config.classifiers["base"] == FrozenClassifierConfig(
        "configs/base.yaml", "d" * 64, "runs/base/checkpoints/best.pt", "e" * 64, 15, 224
    )
    assert config.inference == FrozenInferenceConfig(
        "0", "one_batch_per_scene", "detector_confidence_times_classifier_confidence"
    )

    payload = _payload()
    payload["extra"] = True
    with pytest.raises(DataValidationError, match="config fields"):
        load_final_evaluation_config(_write(tmp_path / "extra.yaml", payload))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p.update(freeze_version=2), "freeze_version"),
        (lambda p: p.update(selection_status="draft"), "selection_status"),
        (lambda p: p["registry"].update(sha256="bad"), "SHA-256"),
        (lambda p: p["detector"].update(operating_confidence=0.0), "threshold"),
        (lambda p: p["classifiers"]["base"].update(output_dimension=20), "Base classifier"),
        (lambda p: p["test_splits"]["base"].update(expected_phase="incremental"), "Base test"),
        (lambda p: p["inference"].update(device="cpu"), "device"),
        (lambda p: p.update(run_name="../escape"), "run_name"),
    ],
)
def test_load_final_evaluation_config_rejects_drift(
    tmp_path: Path, mutator, message: str
) -> None:
    payload = _payload()
    mutator(payload)
    with pytest.raises(DataValidationError, match=message):
        load_final_evaluation_config(_write(tmp_path / "invalid.yaml", payload))


def _prepared(tmp_path: Path) -> PreparedFinalEvaluation:
    return PreparedFinalEvaluation(
        config_path=tmp_path / "frozen.yaml",
        config_sha256="2" * 64,
        dataset_root=tmp_path / "datasets",
        registry_path=tmp_path / "datasets" / "class_registry.json",
        detector_config_path=tmp_path / "detector.yaml",
        detector_checkpoint=tmp_path / "detector.pt",
        classifier_config_paths={"base": tmp_path / "base.yaml", "incremental": tmp_path / "inc.yaml"},
        classifier_checkpoints={"base": tmp_path / "base.pt", "incremental": tmp_path / "inc.pt"},
        classifier_contexts={"base": {}, "incremental": {}},
        output_dir=tmp_path / "runs" / "final" / "frozen_v1",
        lock_path=tmp_path / "runs" / "final" / ".frozen_v1.started.json",
        provenance={"fixture": True},
    )


def test_preflight_does_not_access_test_paths(tmp_path: Path, monkeypatch) -> None:
    config = load_final_evaluation_config(_write(tmp_path / "frozen.yaml", _payload()))
    prepared = _prepared(tmp_path)
    accessed = []
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        normalized = path.as_posix().casefold()
        if "/base/test/" in normalized or "/incremental/test/" in normalized:
            accessed.append(path)
            raise AssertionError("preflight touched test data")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        "bakery_scanner.final_evaluation._prepare_non_test_inputs",
        lambda selected, path, cuda_available: prepared,
    )

    report = preflight_final_evaluation(
        config, tmp_path / "frozen.yaml", cuda_available=lambda device: device == "0"
    )

    assert report.prepared == prepared
    assert report.status == "ready"
    assert accessed == []


def _synthetic_config(tmp_path: Path, dataset_root: Path) -> FinalEvaluationConfig:
    config = load_final_evaluation_config(_write(tmp_path / "frozen.yaml", _payload()))
    return replace(
        config,
        dataset_root=str(dataset_root),
        registry=replace(
            config.registry, path=str(dataset_root / "class_registry.json")
        ),
        test_splits={
            "base": FrozenTestSplitConfig(
                str(dataset_root / "base" / "test" / "instances_test.json"), "base"
            ),
            "incremental": FrozenTestSplitConfig(
                str(
                    dataset_root
                    / "incremental"
                    / "test"
                    / "instances_test.json"
                ),
                "incremental",
            ),
        },
    )


def test_load_test_split_maps_coco_categories_to_model_indices(
    tmp_path: Path, dataset_factory
) -> None:
    dataset_root = dataset_factory()
    config = _synthetic_config(tmp_path, dataset_root)
    prepared = replace(
        _prepared(tmp_path),
        dataset_root=dataset_root,
        registry_path=dataset_root / "class_registry.json",
    )

    base = _load_test_split(config, prepared, "base")
    incremental = _load_test_split(config, prepared, "incremental")

    assert isinstance(base, LoadedTestSplit)
    assert base.phase == "base"
    assert [image.difficulty for image in base.images] == ["easy", "medium", "hard"]
    assert all(obj.model_index < 15 for image in base.images for obj in image.objects)
    assert all(obj.phase == "base" for image in base.images for obj in image.objects)
    assert all(image.image_path.is_file() for image in base.images)
    assert all(
        obj.model_index >= 15
        for image in incremental.images
        for obj in image.objects
    )
    assert len({obj.sample_id for image in base.images for obj in image.objects}) == 3


def test_one_shot_lock_precedes_test_load_and_refuses_second_start(
    tmp_path: Path
) -> None:
    config = load_final_evaluation_config(_write(tmp_path / "frozen.yaml", _payload()))
    prepared = _prepared(tmp_path)
    observed = []

    def loader(_config, selected, name):
        payload = json.loads(selected.lock_path.read_text(encoding="utf-8"))
        observed.append((name, payload["status"]))
        return name

    splits = _load_locked_test_splits(config, prepared, loader=loader)

    assert splits == {"base": "base", "incremental": "incremental"}
    assert observed == [("base", "started"), ("incremental", "started")]
    with pytest.raises(DataValidationError, match="one-shot lock"):
        _create_start_lock(config, prepared)
    _update_start_lock(prepared, status="completed")
    payload = json.loads(prepared.lock_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["config_sha256"] == prepared.config_sha256


def test_one_shot_lock_persists_as_failed_when_test_load_fails(tmp_path: Path) -> None:
    config = load_final_evaluation_config(_write(tmp_path / "frozen.yaml", _payload()))
    prepared = _prepared(tmp_path)

    def fail(_config, selected, _name):
        assert selected.lock_path.is_file()
        raise DataValidationError("synthetic COCO failure")

    with pytest.raises(DataValidationError, match="synthetic COCO failure"):
        _load_locked_test_splits(config, prepared, loader=fail)

    payload = json.loads(prepared.lock_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "synthetic COCO failure" in payload["error"]


class _FakeFinalDetector:
    names = {0: "bread"}

    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        results = []
        for source in kwargs["source"]:
            if "incremental" in str(source):
                xyxy = self.torch.empty((0, 4), dtype=self.torch.float32)
                conf = self.torch.empty((0,), dtype=self.torch.float32)
                cls = self.torch.empty((0,), dtype=self.torch.float32)
            else:
                xyxy = self.torch.tensor([[1.0, 2.0, 11.0, 10.0]])
                conf = self.torch.tensor([0.8])
                cls = self.torch.tensor([0.0])
            results.append(
                SimpleNamespace(
                    boxes=SimpleNamespace(xyxy=xyxy, conf=conf, cls=cls)
                )
            )
        return results


def _manual_split(tmp_path: Path, name: str, model_index: int) -> LoadedTestSplit:
    from PIL import Image
    from bakery_scanner.final_evaluation import TestImageRecord, TestObjectRecord

    directory = tmp_path / name
    directory.mkdir()
    image_path = directory / f"{name}_scene_e_1.jpg"
    Image.new("RGB", (20, 20), (20, 30, 40)).save(image_path)
    coco_path = directory / "instances_test.json"
    coco_path.write_text("{}", encoding="utf-8")
    phase = "base" if name == "base" else "incremental"
    return LoadedTestSplit(
        name=name,
        phase=phase,
        coco_path=coco_path,
        images=(
            TestImageRecord(
                image_id=image_path.name,
                image_path=image_path,
                width=20,
                height=20,
                difficulty="easy",
                objects=(
                    TestObjectRecord(
                        sample_id=f"{name}:sample",
                        annotation_id=1,
                        bbox_xyxy=(1.0, 2.0, 11.0, 10.0),
                        category_id=1,
                        model_index=model_index,
                        phase=phase,
                    ),
                ),
            ),
        ),
    )


def test_final_inference_reuses_detector_and_batches_relevant_classifiers(
    tmp_path: Path
) -> None:
    import torch

    detector = _FakeFinalDetector(torch)

    class FakeClassifier(torch.nn.Module):
        def __init__(self, output_dimension: int, predicted_index: int):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.output_dimension = output_dimension
            self.predicted_index = predicted_index
            self.calls = []

        def forward(self, batch):
            self.calls.append(batch)
            logits = torch.zeros(
                (batch.shape[0], self.output_dimension), device=batch.device
            )
            logits[:, self.predicted_index] = 2.0
            return logits

    base_model = FakeClassifier(15, 3)
    incremental_model = FakeClassifier(20, 16)
    loaded = {}

    def classifier_loader(*, checkpoint, output_dimension, **kwargs):
        loaded[checkpoint.name] = kwargs
        return base_model if output_dimension == 15 else incremental_model

    backend = FinalInferenceBackend(
        detector_factory=lambda _checkpoint: detector,
        classifier_loader=classifier_loader,
        transform_factory=lambda _size: (
            lambda _image: torch.ones((3, 8, 8), dtype=torch.float32)
        ),
        device_factory=lambda _value: torch.device("cuda"),
        stacker=lambda items, _device: torch.stack(items),
    )
    config = load_final_evaluation_config(_write(tmp_path / "frozen.yaml", _payload()))

    result = backend.predict(
        splits={
            "base": _manual_split(tmp_path, "base", 3),
            "incremental": _manual_split(tmp_path, "incremental", 16),
        },
        detector_checkpoint=tmp_path / "detector.pt",
        classifier_checkpoints={
            "base": tmp_path / "base.pt",
            "incremental": tmp_path / "incremental.pt",
        },
        classifier_contexts={"base": {}, "incremental": {}},
        config=config,
    )

    assert len(detector.calls) == 2
    assert all(call["device"] == "0" for call in detector.calls)
    assert all(call["conf"] == 0.001 for call in detector.calls)
    assert all(call["iou"] == 0.7 for call in detector.calls)
    assert set(result.splits["base"].classifier_predictions) == {
        "base",
        "incremental",
    }
    assert set(result.splits["incremental"].classifier_predictions) == {
        "incremental"
    }
    assert result.splits["base"].classifier_predictions["base"][0].predicted_index == 3
    assert result.splits["incremental"].classifier_predictions["incremental"][0].predicted_index == 16
    assert len(result.splits["base"].e2e_predictions["base"]["base_scene_e_1.jpg"]) == 1
    assert result.splits["incremental"].e2e_predictions["incremental"]["incremental_scene_e_1.jpg"] == ()
    assert result.splits["base"].batch_sizes["base"] == {
        "ground_truth": (1,),
        "detections": (1,),
    }
    assert result.splits["incremental"].batch_sizes["incremental"] == {
        "ground_truth": (1,),
        "detections": (0,),
    }
    assert len(base_model.calls) == 2
    assert len(incremental_model.calls) == 3
    assert set(loaded) == {"base.pt", "incremental.pt"}
    assert all(call["device"].type == "cuda" for call in loaded.values())


def _perfect_backend_result(splits):
    from bakery_scanner.classifier_evaluation import ClassifierPrediction
    from bakery_scanner.detector_evaluation import Detection
    from bakery_scanner.e2e_evaluation import EndToEndPrediction

    results = {}
    for split_name, split in splits.items():
        relevant = ("base", "incremental") if split_name == "base" else ("incremental",)
        detector_predictions = {}
        classifier_predictions = {name: [] for name in relevant}
        e2e_predictions = {name: {} for name in relevant}
        batch_sizes = {
            name: {"ground_truth": [], "detections": []} for name in relevant
        }
        for image in split.images:
            detection_items = tuple(
                Detection(obj.bbox_xyxy, 0.9, 0) for obj in image.objects
            )
            detector_predictions[image.image_id] = detection_items
            for model_name in relevant:
                model_predictions = []
                for obj in image.objects:
                    predicted = obj.model_index
                    if split_name == "base" and model_name == "incremental":
                        predicted = 14 if obj.model_index != 14 else 13
                    classifier_predictions[model_name].append(
                        ClassifierPrediction(
                            obj.sample_id, obj.model_index, predicted, 0.8
                        )
                    )
                    model_predictions.append(
                        EndToEndPrediction(
                            image.image_id,
                            obj.bbox_xyxy,
                            predicted,
                            0.9,
                            0.8,
                            0.72,
                        )
                    )
                e2e_predictions[model_name][image.image_id] = tuple(model_predictions)
                batch_sizes[model_name]["ground_truth"].append(len(image.objects))
                batch_sizes[model_name]["detections"].append(len(image.objects))
        results[split_name] = FinalSplitInferenceResult(
            detector_predictions,
            {name: tuple(items) for name, items in classifier_predictions.items()},
            e2e_predictions,
            {
                name: {kind: tuple(values) for kind, values in groups.items()}
                for name, groups in batch_sizes.items()
            },
        )
    return FinalInferenceResult(results)


def test_run_final_evaluation_publishes_metrics_deltas_and_completes_lock(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _write(tmp_path / "frozen.yaml", _payload())
    config = load_final_evaluation_config(config_path)
    prepared = _prepared(tmp_path)
    for checkpoint in (
        prepared.detector_checkpoint,
        *prepared.classifier_checkpoints.values(),
    ):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(checkpoint.name.encode())
    splits = {
        "base": _manual_split(tmp_path, "base", 3),
        "incremental": _manual_split(tmp_path, "incremental", 16),
    }
    monkeypatch.setattr(
        "bakery_scanner.final_evaluation.preflight_final_evaluation",
        lambda *_args, **_kwargs: FinalEvaluationPreflightReport(prepared),
    )
    monkeypatch.setattr(
        "bakery_scanner.final_evaluation._load_test_split",
        lambda _config, _prepared, name: splits[name],
    )

    class Backend:
        def predict(self, **kwargs):
            return _perfect_backend_result(kwargs["splits"])

    report = run_final_evaluation(config, config_path, Backend())

    assert set(path.name for path in report.output_dir.iterdir()) == {
        "classifier_metrics.json",
        "detector_metrics.json",
        "e2e_metrics.json",
        "frozen_config.yaml",
        "metadata.json",
        "predictions.json",
        "report.md",
        "summary.json",
    }
    classifier = json.loads(
        (report.output_dir / "classifier_metrics.json").read_text(encoding="utf-8")
    )
    assert classifier["base_model_on_base_test"]["top1_accuracy"] == 1.0
    assert classifier["incremental_model_on_base_test"]["top1_accuracy"] == 0.0
    assert classifier["incremental_model_on_incremental_test"]["top1_accuracy"] == 1.0
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["base_retention_delta"]["classifier_top1"] == -1.0
    assert summary["base_retention_delta"]["e2e_map50"] == -1.0
    assert summary["incremental_new_classes"]["classifier_top1"] == 1.0
    lock = json.loads(prepared.lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "completed"
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metadata["selection"]["configuration_changed_after_test"] is False
    assert all(
        item["unchanged"] for item in metadata["checkpoints"].values()
    )


def test_run_final_evaluation_failure_cleans_staging_and_marks_lock(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _write(tmp_path / "frozen.yaml", _payload())
    config = load_final_evaluation_config(config_path)
    prepared = _prepared(tmp_path)
    for checkpoint in (
        prepared.detector_checkpoint,
        *prepared.classifier_checkpoints.values(),
    ):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(checkpoint.name.encode())
    monkeypatch.setattr(
        "bakery_scanner.final_evaluation.preflight_final_evaluation",
        lambda *_args, **_kwargs: FinalEvaluationPreflightReport(prepared),
    )
    monkeypatch.setattr(
        "bakery_scanner.final_evaluation._load_test_split",
        lambda _config, _prepared, name: _manual_split(tmp_path, name, 3 if name == "base" else 16),
    )

    class Backend:
        def predict(self, **_kwargs):
            raise RuntimeError("synthetic inference failure")

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        run_final_evaluation(config, config_path, Backend())

    assert not prepared.output_dir.exists()
    assert not list(prepared.output_dir.parent.glob(".frozen_v1.tmp-*"))
    lock = json.loads(prepared.lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "failed"
