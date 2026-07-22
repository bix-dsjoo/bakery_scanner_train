import json
from pathlib import Path

import pytest

from bakery_scanner.errors import DataValidationError
from bakery_scanner.yolo_dataset import build_yolo_dataset, validate_yolo_dataset


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


def test_validate_yolo_dataset_allows_ultralytics_label_caches(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "ultralytics-cache")
    for split in ("train", "validation"):
        (report.output_dir / split / "labels.cache").write_bytes(b"runtime cache")

    validated = validate_yolo_dataset(dataset_root, "ultralytics-cache")

    assert validated.output_dir == report.output_dir


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


def test_validate_yolo_dataset_accepts_same_hashed_relocated_source(
    detector_source_run: tuple[Path, str],
) -> None:
    dataset_root, source_run = detector_source_run
    report = build_yolo_dataset(dataset_root, source_run, "relocated")
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    source_manifest = (
        dataset_root / "derived" / "detector" / source_run / "manifest.json"
    )
    manifest["source"]["manifest_path"] = str(
        dataset_root.parent
        / ".worktrees"
        / "old"
        / source_manifest.relative_to(dataset_root.parent)
    )
    report.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validated = validate_yolo_dataset(dataset_root, "relocated")

    assert validated.output_dir == report.output_dir


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
