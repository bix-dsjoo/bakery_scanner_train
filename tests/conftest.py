import json
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

from bakery_scanner.detector_dataset import (
    DetectorDatasetConfig,
    build_detector_dataset,
)
from bakery_scanner.registry import ClassRegistry, load_class_registry
from bakery_scanner.synthetic import SyntheticConfig, generate_synthetic_dataset


@pytest.fixture
def registry_factory(tmp_path: Path) -> Callable[[], ClassRegistry]:
    def create() -> ClassRegistry:
        classes = []
        for model_index in range(20):
            phase = "base" if model_index < 15 else "incremental"
            category_id = (model_index * 7) % 20 + 1
            classes.append(
                {
                    "category_id": category_id,
                    "model_index": model_index,
                    "canonical_name": f"Bread {category_id}",
                    "folder_name": f"bread_{category_id:02d}_item",
                    "phase": phase,
                }
            )
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "classes": classes}), encoding="utf-8"
        )
        return load_class_registry(path)

    return create


@pytest.fixture
def dataset_factory(tmp_path: Path) -> Callable[[], Path]:
    def create() -> Path:
        dataset_root = tmp_path / "datasets"
        dataset_root.mkdir()
        classes = []
        for model_index in range(20):
            phase = "base" if model_index < 15 else "incremental"
            category_id = (model_index * 7) % 20 + 1
            record = {
                "category_id": category_id,
                "model_index": model_index,
                "canonical_name": f"Bread {category_id}",
                "folder_name": f"bread_{category_id:02d}_item",
                "phase": phase,
            }
            classes.append(record)
            class_dir = dataset_root / phase / str(record["folder_name"])
            class_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(class_dir / "object.jpg")
        (dataset_root / "class_registry.json").write_text(
            json.dumps({"version": 1, "classes": classes}), encoding="utf-8"
        )

        def write_scene(
            relative_dir: str, phase: str, scene_ids: tuple[str, ...], json_name: str
        ) -> None:
            directory = dataset_root / relative_dir
            directory.mkdir()
            phase_classes = [item for item in classes if item["phase"] == phase]
            images = []
            annotations = []
            for offset, (scene_id, difficulty) in enumerate(
                (pair for scene_id in scene_ids for pair in ((scene_id, "e"), (scene_id, "m"), (scene_id, "h"))),
                start=1,
            ):
                file_name = f"scene_{difficulty}_{scene_id}.jpg"
                Image.new("RGB", (40, 30), "white").save(directory / file_name)
                images.append(
                    {"id": offset, "file_name": file_name, "width": 40, "height": 30}
                )
                annotations.append(
                    {
                        "id": offset,
                        "image_id": offset,
                        "category_id": phase_classes[(offset - 1) % len(phase_classes)]["category_id"],
                        "bbox": [1, 2, 10, 8],
                    }
                )
            categories = [
                {"id": item["category_id"], "name": item["canonical_name"]}
                for item in phase_classes
            ]
            (directory / json_name).write_text(
                json.dumps(
                    {"images": images, "annotations": annotations, "categories": categories}
                ),
                encoding="utf-8",
            )

        write_scene("base/val", "base", ("0001", "0002"), "instances_val.json")
        write_scene("base/test", "base", ("0003",), "instances_test.json")
        write_scene(
            "incremental/test", "incremental", ("0004",), "instances_test.json"
        )
        return dataset_root

    return create


@pytest.fixture
def detector_source_run(
    dataset_factory: Callable[[], Path], tmp_path: Path
) -> tuple[Path, str]:
    dataset_root = dataset_factory()
    for model_index, path in enumerate(sorted(dataset_root.glob("*/bread_*/*.jpg"))):
        source = Image.new("RGB", (12, 10), "white")
        for x in range(2, 10):
            for y in range(3, 8):
                source.putpixel((x, y), (170, 60 + model_index, 20))
        source.save(path)

    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    Image.new("RGB", (80, 60), (210, 210, 210)).save(backgrounds / "tray.png")
    generate_synthetic_dataset(
        dataset_root,
        backgrounds,
        "synthetic-input",
        SyntheticConfig(
            seed=5,
            scene_count=2,
            objects_per_scene=1,
            size_fraction_range=(0.2, 0.2),
            rotation_range=(0.0, 0.0),
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
        ),
    )
    source_run = "detector-input"
    build_detector_dataset(
        dataset_root,
        "synthetic-input",
        source_run,
        DetectorDatasetConfig(seed=11, validation_fraction=0.25),
    )
    return dataset_root, source_run
