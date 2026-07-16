from pathlib import Path
from typing import Callable

from bakery_scanner.audit import audit_dataset


def test_audit_dataset_combines_integrity_checks_and_statistics(
    dataset_factory: Callable[[], Path],
) -> None:
    dataset_root = dataset_factory()

    report = audit_dataset(dataset_root, validation_fraction=0.5, seed=42)

    assert report.registry_total_classes == 20
    assert report.registry_phase_counts == {"base": 15, "incremental": 5}
    assert report.class_image_totals == {"base": 15, "incremental": 5}
    assert report.coco_splits["base_scene_train"].image_count == 6
    assert report.coco_splits["base_test"].category_count == 15
    assert report.coco_splits["incremental_test"].category_count == 5
    assert len(report.scene_split.train_paths) == 3
    assert len(report.scene_split.validation_paths) == 3


def test_audit_report_is_json_serializable(
    dataset_factory: Callable[[], Path],
) -> None:
    report = audit_dataset(dataset_factory(), validation_fraction=0.5, seed=7)

    payload = report.to_dict()

    assert payload["status"] == "ok"
    assert payload["registry"]["phase_counts"]["base"] == 15
    assert payload["scene_split"]["seed"] == 7
    assert payload["scene_split"]["train_image_count"] == 3
