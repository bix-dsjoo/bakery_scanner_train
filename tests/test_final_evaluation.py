from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

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
    PreparedFinalEvaluation,
    load_final_evaluation_config,
    preflight_final_evaluation,
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
