from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .coco import CocoStats, validate_coco
from .registry import IMAGE_SUFFIXES, load_class_registry, validate_class_directories
from .splits import SceneSplit, split_scene_paths


@dataclass(frozen=True, slots=True)
class AuditReport:
    dataset_root: Path
    registry_version: int
    registry_total_classes: int
    registry_phase_counts: dict[str, int]
    class_image_counts: dict[str, dict[str, int]]
    class_image_totals: dict[str, int]
    coco_splits: dict[str, CocoStats]
    scene_split: SceneSplit
    validation_fraction: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset_root": str(self.dataset_root),
            "registry": {
                "version": self.registry_version,
                "total_classes": self.registry_total_classes,
                "phase_counts": self.registry_phase_counts,
            },
            "single_object_images": {
                "by_class": self.class_image_counts,
                "phase_totals": self.class_image_totals,
            },
            "coco_splits": {
                name: {
                    "image_count": stats.image_count,
                    "annotation_count": stats.annotation_count,
                    "category_count": stats.category_count,
                }
                for name, stats in self.coco_splits.items()
            },
            "scene_split": {
                "seed": self.seed,
                "validation_fraction": self.validation_fraction,
                "train_scene_ids": list(self.scene_split.train_scene_ids),
                "validation_scene_ids": list(self.scene_split.validation_scene_ids),
                "train_image_count": len(self.scene_split.train_paths),
                "validation_image_count": len(self.scene_split.validation_paths),
                "train_paths": [str(path) for path in self.scene_split.train_paths],
                "validation_paths": [
                    str(path) for path in self.scene_split.validation_paths
                ],
            },
        }


def audit_dataset(
    dataset_root: str | Path, validation_fraction: float = 0.2, seed: int = 42
) -> AuditReport:
    root = Path(dataset_root).resolve(strict=False)
    registry = load_class_registry(root / "class_registry.json")
    class_counts = validate_class_directories(root, registry)

    coco_splits = {
        "base_scene_train": validate_coco(
            root / "base" / "val" / "instances_val.json",
            registry,
            expected_phase="base",
        ),
        "base_test": validate_coco(
            root / "base" / "test" / "instances_test.json",
            registry,
            expected_phase="base",
        ),
        "incremental_test": validate_coco(
            root / "incremental" / "test" / "instances_test.json",
            registry,
            expected_phase="incremental",
        ),
    }
    scene_train_root = root / "base" / "val"
    scene_paths = sorted(
        (
            path
            for path in scene_train_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix(),
    )
    scene_split = split_scene_paths(scene_paths, validation_fraction, seed)
    class_totals = {
        phase: sum(per_class.values()) for phase, per_class in class_counts.items()
    }

    return AuditReport(
        dataset_root=root,
        registry_version=registry.version,
        registry_total_classes=len(registry.classes),
        registry_phase_counts=registry.phase_counts,
        class_image_counts=class_counts,
        class_image_totals=class_totals,
        coco_splits=coco_splits,
        scene_split=scene_split,
        validation_fraction=validation_fraction,
        seed=seed,
    )
