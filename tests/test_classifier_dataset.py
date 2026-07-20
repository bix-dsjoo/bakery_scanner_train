from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import PIL
from PIL import Image

from bakery_scanner import classifier_dataset
from bakery_scanner.classifier_dataset import (
    ClassifierDatasetConfig,
    build_classifier_dataset,
    validate_classifier_dataset,
)
from bakery_scanner.errors import DataValidationError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(
    dataset_root: Path,
    run_name: str,
    phase: str,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> ClassifierDatasetConfig:
    return ClassifierDatasetConfig(
        dataset_root=dataset_root,
        run_name=run_name,
        phase=phase,
        seed=seed,
        validation_fraction=validation_fraction,
        expected_base_images_per_class=1,
        expected_incremental_images_per_class=7,
    )


def test_build_base_dataset_maps_model_indices_and_preserves_scene_groups(
    dataset_factory,
) -> None:
    dataset_root = dataset_factory()
    scene_dir = dataset_root / "base" / "val"
    with Image.open(scene_dir / "scene_e_0001.jpg") as source:
        colored = source.convert("RGB")
    for x in range(1, 11):
        for y in range(2, 10):
            colored.putpixel((x, y), (12, 34, 56))
    colored.save(scene_dir / "scene_e_0001.jpg")

    report = build_classifier_dataset(
        _config(
            dataset_root, "base-fixture", "base", seed=42, validation_fraction=0.5
        )
    )

    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert report.output_dimension == 15
    assert manifest["config"]["dataset_root"] == "."
    assert manifest["registry"]["output_dimension"] == 15
    assert manifest["environment"]["dependencies"]["Pillow"] == PIL.__version__
    assert manifest["environment"]["python"]
    assert manifest["environment"]["platform"]
    assert {sample["model_index"] for sample in manifest["samples"]} == set(range(15))
    assert set(manifest["counts"]["by_split_class"]) == {"train", "validation"}
    assert sum(manifest["counts"]["by_split_class"]["validation"].values()) == 3

    single = next(
        sample
        for sample in manifest["samples"]
        if sample["source_kind"] == "single_object" and sample["model_index"] == 0
    )
    single_source = dataset_root / single["source_path"]
    single_output = dataset_root / single["output_path"]
    assert single["category_id"] != single["model_index"]
    assert single["split"] == "train"
    assert single_source.read_bytes() == single_output.read_bytes()
    assert single["source_sha256"] == _sha256(single_source)
    assert single["output_sha256"] == _sha256(single_output)

    scene_samples = [
        sample for sample in manifest["samples"] if sample["source_kind"] == "scene_crop"
    ]
    split_by_scene: dict[str, set[str]] = {}
    for sample in scene_samples:
        split_by_scene.setdefault(sample["scene_id"], set()).add(sample["split"])
    assert set(split_by_scene) == {"0001", "0002"}
    assert all(len(splits) == 1 for splits in split_by_scene.values())
    assert {next(iter(splits)) for splits in split_by_scene.values()} == {
        "train",
        "validation",
    }

    colored_crop = next(
        sample
        for sample in scene_samples
        if sample["source_path"].endswith("scene_e_0001.jpg")
    )
    with Image.open(dataset_root / colored_crop["output_path"]) as crop:
        assert crop.size == (10, 8)
        with Image.open(dataset_root / colored_crop["source_path"]) as source:
            expected_pixel = source.convert("RGB").getpixel((5, 6))
        assert crop.convert("RGB").getpixel((4, 4)) == expected_pixel

    validation = validate_classifier_dataset(dataset_root, "base-fixture")
    assert validation.sample_count == report.sample_count
    assert validation.output_dimension == 15


def _add_incremental_images(dataset_root: Path) -> None:
    registry = json.loads(
        (dataset_root / "class_registry.json").read_text(encoding="utf-8")
    )
    for record in registry["classes"]:
        if record["phase"] != "incremental":
            continue
        directory = dataset_root / "incremental" / record["folder_name"]
        original = next(directory.glob("*.jpg"))
        for number in range(2, 8):
            with Image.open(original) as image:
                colored = image.convert("RGB")
            colored.putpixel((0, 0), (number * 10, record["model_index"], 90))
            colored.save(directory / f"object_{number}.jpg")


def _incremental_split(manifest: dict) -> dict[str, str]:
    return {
        sample["source_path"]: sample["split"]
        for sample in manifest["samples"]
        if sample["source_kind"] == "single_object"
        and sample["class_phase"] == "incremental"
    }


def test_incremental_dataset_is_deterministic_and_holds_out_one_per_class(
    dataset_factory,
) -> None:
    dataset_root = dataset_factory()
    _add_incremental_images(dataset_root)

    first = build_classifier_dataset(
        _config(dataset_root, "incremental-a", "incremental", seed=17)
    )
    second = build_classifier_dataset(
        _config(dataset_root, "incremental-b", "incremental", seed=17)
    )
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    assert first.output_dimension == 20
    assert {sample["model_index"] for sample in first_manifest["samples"]} == set(
        range(20)
    )
    assert _incremental_split(first_manifest) == _incremental_split(second_manifest)
    for model_index in range(15, 20):
        samples = [
            sample
            for sample in first_manifest["samples"]
            if sample["source_kind"] == "single_object"
            and sample["model_index"] == model_index
        ]
        assert sum(sample["split"] == "train" for sample in samples) == 6
        assert sum(sample["split"] == "validation" for sample in samples) == 1
        assert {
            sample["validation_domain"]
            for sample in samples
            if sample["split"] == "validation"
        } == {"single_object"}

    assert validate_classifier_dataset(dataset_root, "incremental-a").output_dimension == 20


def test_validation_rejects_output_and_source_tampering(
    dataset_factory,
) -> None:
    dataset_root = dataset_factory()
    report = build_classifier_dataset(
        _config(dataset_root, "tamper-output", "base")
    )
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    output = dataset_root / manifest["samples"][0]["output_path"]
    output.write_bytes(b"changed")
    with pytest.raises(DataValidationError, match="output is missing or altered"):
        validate_classifier_dataset(dataset_root, "tamper-output")

    report = build_classifier_dataset(
        _config(dataset_root, "tamper-source", "base")
    )
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    source = dataset_root / manifest["samples"][0]["source_path"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(DataValidationError, match="source hash changed"):
        validate_classifier_dataset(dataset_root, "tamper-source")


def test_validation_rejects_manifest_and_inventory_tampering(dataset_factory) -> None:
    dataset_root = dataset_factory()
    report = build_classifier_dataset(
        _config(dataset_root, "tamper-manifest", "base")
    )
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataValidationError, match="fields do not match schema"):
        validate_classifier_dataset(dataset_root, "tamper-manifest")

    report = build_classifier_dataset(
        _config(dataset_root, "tamper-inventory", "base")
    )
    (report.output_dir / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(DataValidationError, match="output inventory differs"):
        validate_classifier_dataset(dataset_root, "tamper-inventory")


def test_validation_rejects_manifest_source_from_test_split(dataset_factory) -> None:
    dataset_root = dataset_factory()
    report = build_classifier_dataset(
        _config(dataset_root, "test-path", "base")
    )
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    sample = manifest["samples"][0]
    test_source = next((dataset_root / "base" / "test").glob("*.jpg"))
    sample["source_path"] = test_source.relative_to(dataset_root).as_posix()
    sample["source_sha256"] = _sha256(test_source)
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataValidationError, match="evaluation-only"):
        validate_classifier_dataset(dataset_root, "test-path")


def test_build_rejects_scene_link_to_test_before_opening_it(
    dataset_factory,
    monkeypatch,
) -> None:
    dataset_root = dataset_factory()
    scene_path = dataset_root / "base" / "val" / "scene_e_0001.jpg"
    test_path = dataset_root / "base" / "test" / "scene_e_0003.jpg"
    original_open = Image.open
    original_resolve = Path.resolve
    scene_absolute = scene_path.absolute()
    test_resolved = original_resolve(test_path)

    def resolve_scene_link(path, *args, **kwargs):
        if path.absolute() == scene_absolute:
            return test_resolved
        return original_resolve(path, *args, **kwargs)

    def reject_test_open(path, *args, **kwargs):
        if Path(path).absolute() == scene_absolute:
            raise AssertionError("test image was opened before path safety validation")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_scene_link)
    monkeypatch.setattr(Image, "open", reject_test_open)
    with pytest.raises(DataValidationError, match="evaluation-only"):
        build_classifier_dataset(
            _config(dataset_root, "unsafe-scene", "base")
        )
    assert not (dataset_root / "derived" / "classifier" / "unsafe-scene").exists()


def test_build_rejects_coco_link_to_test_before_reading_it(
    dataset_factory,
    monkeypatch,
) -> None:
    dataset_root = dataset_factory()
    coco_path = dataset_root / "base" / "val" / "instances_val.json"
    test_coco = dataset_root / "base" / "test" / "instances_test.json"
    original_read_text = Path.read_text
    original_resolve = Path.resolve
    coco_absolute = coco_path.absolute()
    test_resolved = original_resolve(test_coco)

    def resolve_coco_link(path, *args, **kwargs):
        if path.absolute() == coco_absolute:
            return test_resolved
        return original_resolve(path, *args, **kwargs)

    def reject_test_coco_read(path, *args, **kwargs):
        if path.absolute() == coco_absolute:
            raise AssertionError("test COCO was read before path safety validation")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_coco_link)
    monkeypatch.setattr(Path, "read_text", reject_test_coco_read)
    with pytest.raises(DataValidationError, match="evaluation-only"):
        build_classifier_dataset(
            _config(dataset_root, "unsafe-coco", "base")
        )
    assert not (dataset_root / "derived" / "classifier" / "unsafe-coco").exists()


def test_validation_hashes_scene_without_annotations(dataset_factory) -> None:
    dataset_root = dataset_factory()
    coco_path = dataset_root / "base" / "val" / "instances_val.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    empty_image = coco["images"][0]
    coco["annotations"] = [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] != empty_image["id"]
    ]
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    report = build_classifier_dataset(
        _config(dataset_root, "empty-scene", "base")
    )
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["scene_sources"]) == len(coco["images"])

    empty_path = coco_path.parent / empty_image["file_name"]
    Image.new("RGB", (empty_image["width"], empty_image["height"]), "black").save(
        empty_path
    )
    with pytest.raises(DataValidationError, match="scene source hash changed"):
        validate_classifier_dataset(dataset_root, "empty-scene")


def test_overwrite_cleanup_failure_keeps_committed_run_valid(
    dataset_factory,
    monkeypatch,
) -> None:
    dataset_root = dataset_factory()
    build_classifier_dataset(
        _config(dataset_root, "atomic", "base", seed=1)
    )
    original_rmtree = classifier_dataset.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if ".backup-" in Path(path).name:
            raise OSError("simulated backup cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(classifier_dataset.shutil, "rmtree", fail_backup_cleanup)
    updated = build_classifier_dataset(
        _config(dataset_root, "atomic", "base", seed=2),
        overwrite=True,
    )

    manifest = json.loads(updated.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] == 2
    validate_classifier_dataset(dataset_root, "atomic")


def test_build_rejects_class_image_count_drift(dataset_factory) -> None:
    dataset_root = dataset_factory()
    _add_incremental_images(dataset_root)
    incremental_dir = next((dataset_root / "incremental").glob("bread_*"))
    Image.new("RGB", (8, 8), "red").save(incremental_dir / "unexpected.jpg")

    with pytest.raises(DataValidationError, match="must contain exactly 7 images"):
        build_classifier_dataset(
            _config(dataset_root, "count-drift", "incremental")
        )


def test_validation_rejects_pillow_version_drift(dataset_factory) -> None:
    dataset_root = dataset_factory()
    report = build_classifier_dataset(_config(dataset_root, "pillow-drift", "base"))
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["dependencies"]["Pillow"] = "0.0.0"
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataValidationError, match="Pillow version does not match"):
        validate_classifier_dataset(dataset_root, "pillow-drift")


def test_build_rejects_registry_link_to_test_before_reading_it(
    dataset_factory,
    monkeypatch,
) -> None:
    dataset_root = dataset_factory()
    registry_path = dataset_root / "class_registry.json"
    test_json = dataset_root / "base" / "test" / "instances_test.json"
    original_read_text = Path.read_text
    original_resolve = Path.resolve
    registry_absolute = registry_path.absolute()
    test_resolved = original_resolve(test_json)

    def resolve_registry_link(path, *args, **kwargs):
        if path.absolute() == registry_absolute:
            return test_resolved
        return original_resolve(path, *args, **kwargs)

    def reject_test_registry_read(path, *args, **kwargs):
        if path.absolute() == registry_absolute:
            raise AssertionError("test JSON was read before registry safety validation")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_registry_link)
    monkeypatch.setattr(Path, "read_text", reject_test_registry_read)
    with pytest.raises(DataValidationError, match="evaluation-only"):
        build_classifier_dataset(
            _config(dataset_root, "unsafe-registry", "base")
        )


def test_classifier_config_defaults_match_repository_counts() -> None:
    config = ClassifierDatasetConfig(Path("datasets"), "defaults", "base")
    assert config.expected_base_images_per_class == 84
    assert config.expected_incremental_images_per_class == 7
