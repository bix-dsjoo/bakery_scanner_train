import json
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

from bakery_scanner.detector_dataset import (
    DetectorDatasetConfig,
    build_detector_dataset,
)
from bakery_scanner.errors import DataValidationError
from bakery_scanner.synthetic import SyntheticConfig, generate_synthetic_dataset
from bakery_scanner.yolo_dataset import build_yolo_dataset, validate_yolo_dataset


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


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_build_yolo_dataset_converts_boxes_and_records_provenance(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run

    report = build_yolo_dataset(dataset_root, source_run, "yolo-fixture")

    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert report.image_count == report.train_image_count + report.validation_image_count
    assert manifest["manifest_version"] == 1
    assert manifest["source"]["run_name"] == source_run
    assert manifest["source"]["manifest_sha256"]

    sample = next(item for item in manifest["samples"] if item["annotation_count"])
    fields = (
        (report.output_dir / sample["label_path"])
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .split()
    )
    source_manifest = json.loads(
        (
            dataset_root
            / "derived"
            / "detector"
            / source_run
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    source_sample = next(
        item
        for item in source_manifest["samples"]
        if item["sample_id"] == sample["source_sample_id"]
    )
    annotation = source_sample["original_annotations"][0]
    x, y, width, height = annotation["bbox"]
    expected = (
        0.0,
        (x + width / 2) / source_sample["width"],
        (y + height / 2) / source_sample["height"],
        width / source_sample["width"],
        height / source_sample["height"],
    )

    assert tuple(float(value) for value in fields) == pytest.approx(expected)


def test_validate_yolo_dataset_rejects_tampered_label(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "tampered")
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    sample = next(item for item in manifest["samples"] if item["annotation_count"])
    label_path = report.output_dir / sample["label_path"]
    label_path.write_text("0 0.5 0.5 2.0 0.5\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="label|hash|bounds"):
        validate_yolo_dataset(dataset_root, "tampered")


def test_validate_yolo_dataset_rejects_extra_output_file(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "extra-file")
    (report.output_dir / "train" / "labels" / "unexpected.txt").write_text(
        "", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="inventory|extra"):
        validate_yolo_dataset(dataset_root, "extra-file")


def test_validate_yolo_dataset_rejects_changed_source_manifest(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    build_yolo_dataset(dataset_root, source_run, "changed-source")
    source_manifest = (
        dataset_root / "derived" / "detector" / source_run / "manifest.json"
    )
    source_manifest.write_text(
        source_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="source|hash"):
        validate_yolo_dataset(dataset_root, "changed-source")


def test_failed_publish_restores_existing_yolo_run(
    detector_source_run: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "atomic")
    before = _tree_bytes(report.output_dir)
    original_rename = Path.rename

    def fail_staging_publish(path: Path, target: Path) -> Path:
        if path.name.startswith(".atomic.tmp-") and target.name == "atomic":
            raise OSError("simulated publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        build_yolo_dataset(
            dataset_root, source_run, "atomic", overwrite=True
        )

    assert report.output_dir.is_dir()
    assert _tree_bytes(report.output_dir) == before
