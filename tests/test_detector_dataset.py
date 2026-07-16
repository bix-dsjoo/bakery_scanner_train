import json
import hashlib
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

import bakery_scanner.detector_dataset as detector_module
from bakery_scanner.detector_dataset import (
    BUILDER_VERSION,
    DetectorDatasetConfig,
    build_detector_dataset,
    validate_detector_dataset,
)
from bakery_scanner.errors import DataValidationError
from bakery_scanner.synthetic import SyntheticConfig, generate_synthetic_dataset


def _prepare_sources(dataset_root: Path) -> None:
    for model_index, path in enumerate(sorted(dataset_root.glob("*/bread_*/*.jpg"))):
        source = Image.new("RGB", (12, 10), "white")
        for x in range(2, 10):
            for y in range(3, 8):
                source.putpixel((x, y), (170, 60 + model_index, 20))
        source.save(path)


def _generate_input_run(
    dataset_root: Path,
    tmp_path: Path,
    scene_count: int = 3,
    run_name: str = "input",
) -> None:
    _prepare_sources(dataset_root)
    backgrounds = tmp_path / f"backgrounds-{run_name}"
    backgrounds.mkdir()
    Image.new("RGB", (80, 60), (210, 210, 210)).save(backgrounds / "tray.png")
    generate_synthetic_dataset(
        dataset_root,
        backgrounds,
        run_name,
        SyntheticConfig(
            seed=73,
            scene_count=scene_count,
            objects_per_scene=1,
            size_fraction_range=(0.2, 0.2),
            rotation_range=(0.0, 0.0),
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
        ),
    )


@pytest.fixture
def detector_inputs(
    dataset_factory: Callable[[], Path], tmp_path: Path
) -> Path:
    dataset_root = dataset_factory()
    _generate_input_run(dataset_root, tmp_path)
    return dataset_root


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def _make_real_images_unique(dataset_root: Path) -> None:
    for index, path in enumerate(
        sorted((dataset_root / "base" / "val").glob("scene_*.jpg")), start=1
    ):
        with Image.open(path) as decoded:
            changed = decoded.convert("RGB")
        for x in range(10):
            for y in range(10):
                changed.putpixel((x, y), (index * 30, index * 20, index * 10))
        changed.save(path)


def test_build_converts_every_annotation_to_bread_and_preserves_category_provenance(
    detector_inputs: Path,
) -> None:
    report = build_detector_dataset(
        detector_inputs,
        "input",
        "unit",
        DetectorDatasetConfig(seed=19, validation_fraction=1 / 3),
    )

    assert report.output_dir == (
        detector_inputs / "derived" / "detector" / "unit"
    ).resolve()
    assert report.builder_version == BUILDER_VERSION
    assert report.image_count == 9
    assert report.annotation_count == 9
    for split in ("train", "validation"):
        coco = _load_json(report.output_dir / split / "instances.json")
        assert coco["categories"] == [{"id": 1, "name": "bread"}]
        assert all(item["category_id"] == 1 for item in coco["annotations"])

    manifest = _load_json(report.manifest_path)
    assert manifest["manifest_version"] == 1
    assert manifest["builder_version"] == BUILDER_VERSION
    assert "model_index" not in report.manifest_path.read_text(encoding="utf-8")
    samples = manifest["samples"]
    assert isinstance(samples, list)
    original_ids = {
        annotation["category_id"]
        for sample in samples
        for annotation in sample["original_annotations"]
    }
    assert original_ids != {1}
    assert all(sample["split"] in {"train", "validation"} for sample in samples)


def test_split_is_deterministic_and_keeps_real_and_synthetic_resources_together(
    detector_inputs: Path,
) -> None:
    config = DetectorDatasetConfig(seed=123, validation_fraction=1 / 3)
    first = build_detector_dataset(detector_inputs, "input", "first", config)
    second = build_detector_dataset(detector_inputs, "input", "second", config)

    first_samples = _load_json(first.manifest_path)["samples"]
    second_samples = _load_json(second.manifest_path)["samples"]
    first_assignment = {item["source_path"]: item["split"] for item in first_samples}
    second_assignment = {item["source_path"]: item["split"] for item in second_samples}
    assert first_assignment == second_assignment

    real_by_scene: dict[str, set[str]] = {}
    synthetic_splits: set[str] = set()
    for sample in first_samples:
        if sample["origin"] == "real":
            real_by_scene.setdefault(sample["provenance"]["scene_id"], set()).add(
                sample["split"]
            )
        else:
            synthetic_splits.add(sample["split"])
    assert all(len(splits) == 1 for splits in real_by_scene.values())
    assert len(synthetic_splits) == 1


def test_split_falls_back_to_global_components_when_origins_are_indivisible(
    detector_inputs: Path,
) -> None:
    report = build_detector_dataset(
        detector_inputs,
        "input",
        "fallback",
        DetectorDatasetConfig(seed=123, validation_fraction=1 / 3),
    )

    samples = _load_json(report.manifest_path)["samples"]
    origin_splits = {
        origin: {item["split"] for item in samples if item["origin"] == origin}
        for origin in ("real", "synthetic")
    }
    assert origin_splits["real"] in ({"train"}, {"validation"})
    assert origin_splits["synthetic"] in ({"train"}, {"validation"})
    assert {item["split"] for item in samples} == {"train", "validation"}
    assert validate_detector_dataset(detector_inputs, "fallback").image_count == 9


def test_split_keeps_cross_origin_image_hash_bridge_together(
    detector_inputs: Path,
) -> None:
    _make_real_images_unique(detector_inputs)
    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    synthetic_path = (
        detector_inputs / "derived" / "synthetic" / "input" / "scene_000000.png"
    )
    bridged_name = "scene_e_0001.jpg"
    bridged_image = next(item for item in coco["images"] if item["file_name"] == bridged_name)
    bridged_path = coco_path.parent / bridged_name
    bridged_path.write_bytes(synthetic_path.read_bytes())
    with Image.open(synthetic_path) as decoded:
        bridged_image["width"], bridged_image["height"] = decoded.size
    _write_json(coco_path, coco)

    report = build_detector_dataset(
        detector_inputs,
        "input",
        "cross-origin-bridge",
        DetectorDatasetConfig(seed=42, validation_fraction=0.2),
    )

    samples = _load_json(report.manifest_path)["samples"]
    bridge_split = next(
        item["split"]
        for item in samples
        if item["origin"] == "real" and item["provenance"]["scene_id"] == "0001"
    )
    assert {
        item["split"]
        for item in samples
        if item["origin"] == "real" and item["provenance"]["scene_id"] == "0001"
    } == {bridge_split}
    assert {
        item["split"] for item in samples if item["origin"] == "synthetic"
    } == {bridge_split}
    assert validate_detector_dataset(detector_inputs, "cross-origin-bridge").image_count == 9


def test_split_keeps_real_scene_groups_in_both_splits_when_synthetic_is_indivisible(
    detector_inputs: Path, tmp_path: Path
) -> None:
    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    for difficulty in "emh":
        file_name = f"scene_{difficulty}_0003.jpg"
        Image.new("RGB", (40, 30), "white").save(coco_path.parent / file_name)
        coco["images"].append(
            {
                "id": len(coco["images"]) + 1,
                "file_name": file_name,
                "width": 40,
                "height": 30,
            }
        )
    _write_json(coco_path, coco)
    _make_real_images_unique(detector_inputs)
    _generate_input_run(detector_inputs, tmp_path, scene_count=100, run_name="large-input")
    report = build_detector_dataset(
        detector_inputs,
        "large-input",
        "origin-aware",
        DetectorDatasetConfig(seed=42, validation_fraction=0.2),
    )

    samples = _load_json(report.manifest_path)["samples"]
    real_samples = [item for item in samples if item["origin"] == "real"]
    synthetic_samples = [item for item in samples if item["origin"] == "synthetic"]
    assert {item["split"] for item in real_samples} == {"train", "validation"}
    assert len({item["split"] for item in synthetic_samples}) == 1

    real_splits_by_scene: dict[str, set[str]] = {}
    for item in real_samples:
        real_splits_by_scene.setdefault(item["provenance"]["scene_id"], set()).add(
            item["split"]
        )
    assert all(len(splits) == 1 for splits in real_splits_by_scene.values())
    assert validate_detector_dataset(detector_inputs, "origin-aware").image_count == 109


def test_seed_controls_which_safe_component_is_used_for_validation(
    detector_inputs: Path,
) -> None:
    _make_real_images_unique(detector_inputs)

    first = build_detector_dataset(
        detector_inputs,
        "input",
        "seed-zero",
        DetectorDatasetConfig(seed=0, validation_fraction=1 / 3),
    )
    second = build_detector_dataset(
        detector_inputs,
        "input",
        "seed-one",
        DetectorDatasetConfig(seed=1, validation_fraction=1 / 3),
    )

    first_validation = {
        item["source_path"]
        for item in _load_json(first.manifest_path)["samples"]
        if item["split"] == "validation"
    }
    second_validation = {
        item["source_path"]
        for item in _load_json(second.manifest_path)["samples"]
        if item["split"] == "validation"
    }
    assert first_validation != second_validation


def test_split_is_independent_of_coco_array_order(detector_inputs: Path) -> None:
    _make_real_images_unique(detector_inputs)
    config = DetectorDatasetConfig(seed=0, validation_fraction=1 / 3)
    first = build_detector_dataset(detector_inputs, "input", "ordered", config)

    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    scene_one = [item for item in coco["images"] if "0001" in item["file_name"]]
    scene_two = [item for item in coco["images"] if "0002" in item["file_name"]]
    coco["images"] = [*reversed(scene_one), *scene_two]
    coco["annotations"].reverse()
    _write_json(coco_path, coco)
    second = build_detector_dataset(detector_inputs, "input", "reversed", config)

    first_assignment = {
        item["source_path"]: item["split"]
        for item in _load_json(first.manifest_path)["samples"]
    }
    second_assignment = {
        item["source_path"]: item["split"]
        for item in _load_json(second.manifest_path)["samples"]
    }
    assert first_assignment == second_assignment


def test_build_and_validate_allow_a_normal_empty_real_scene(
    detector_inputs: Path,
) -> None:
    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    for difficulty in "emh":
        file_name = f"scene_{difficulty}_0003.jpg"
        Image.new("RGB", (40, 30), "white").save(coco_path.parent / file_name)
        coco["images"].append(
            {"id": len(coco["images"]) + 1, "file_name": file_name, "width": 40, "height": 30}
        )
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    build_detector_dataset(
        detector_inputs,
        "input",
        "empty",
        DetectorDatasetConfig(seed=7, validation_fraction=0.25),
    )
    validated = validate_detector_dataset(detector_inputs, "empty")

    assert validated.image_count == 12
    assert validated.annotation_count == 9


def test_impossible_single_leakage_component_fails_without_output(
    detector_inputs: Path,
) -> None:
    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    keep_names = {f"scene_{difficulty}_0001.jpg" for difficulty in "emh"}
    coco["images"] = [item for item in coco["images"] if item["file_name"] in keep_names]
    keep_ids = {item["id"] for item in coco["images"]}
    coco["annotations"] = [
        item for item in coco["annotations"] if item["image_id"] in keep_ids
    ]
    for path in coco_path.parent.glob("scene_*_0002.jpg"):
        path.unlink()

    synthetic_image = detector_inputs / "derived" / "synthetic" / "input" / "scene_000000.png"
    with Image.open(synthetic_image) as decoded:
        size = decoded.size
    for image in coco["images"]:
        target = coco_path.parent / image["file_name"]
        target.write_bytes(synthetic_image.read_bytes())
        image["width"], image["height"] = size
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    with pytest.raises(DataValidationError, match="safe.*split|split.*safe"):
        build_detector_dataset(
            detector_inputs,
            "input",
            "impossible",
            DetectorDatasetConfig(seed=1, validation_fraction=0.2),
        )

    assert not (detector_inputs / "derived" / "detector" / "impossible").exists()


def test_build_rejects_incomplete_real_scene_group(detector_inputs: Path) -> None:
    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    removed = next(
        item for item in coco["images"] if item["file_name"] == "scene_h_0002.jpg"
    )
    coco["images"].remove(removed)
    coco["annotations"] = [
        item for item in coco["annotations"] if item["image_id"] != removed["id"]
    ]
    (coco_path.parent / removed["file_name"]).unlink()
    _write_json(coco_path, coco)

    with pytest.raises(DataValidationError, match="exactly e, m, and h"):
        build_detector_dataset(
            detector_inputs, "input", "incomplete", DetectorDatasetConfig()
        )


def test_validate_rejects_changed_source_coco(detector_inputs: Path) -> None:
    build_detector_dataset(
        detector_inputs, "input", "source-coco", DetectorDatasetConfig()
    )
    source_coco = detector_inputs / "base" / "val" / "instances_val.json"
    source_coco.write_text(source_coco.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="source.*COCO|real COCO.*sha256"):
        validate_detector_dataset(detector_inputs, "source-coco")


def test_validate_rejects_changed_synthetic_manifest(detector_inputs: Path) -> None:
    build_detector_dataset(
        detector_inputs, "input", "source-synthetic", DetectorDatasetConfig()
    )
    source_manifest = detector_inputs / "derived" / "synthetic" / "input" / "manifest.json"
    source_manifest.write_text(
        source_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="synthetic manifest.*sha256"):
        validate_detector_dataset(detector_inputs, "source-synthetic")


def test_validate_rejects_unknown_manifest_field(detector_inputs: Path) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "schema", DetectorDatasetConfig()
    )
    manifest = _load_json(report.manifest_path)
    manifest["unexpected"] = True
    _write_json(report.manifest_path, manifest)

    with pytest.raises(DataValidationError, match="schema|fields"):
        validate_detector_dataset(detector_inputs, "schema")


def test_validate_wraps_invalid_utf8_as_data_error(detector_inputs: Path) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "invalid-utf8", DetectorDatasetConfig()
    )
    report.manifest_path.write_bytes(b"\xff")

    with pytest.raises(DataValidationError, match="cannot load detector manifest"):
        validate_detector_dataset(detector_inputs, "invalid-utf8")


def test_validate_rejects_manifest_annotation_that_disagrees_with_coco(
    detector_inputs: Path,
) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "annotation", DetectorDatasetConfig()
    )
    manifest = _load_json(report.manifest_path)
    sample = next(item for item in manifest["samples"] if item["original_annotations"])
    sample["original_annotations"][0]["bbox"][0] += 1
    _write_json(report.manifest_path, manifest)

    with pytest.raises(DataValidationError, match="manifest.*COCO|annotation.*match"):
        validate_detector_dataset(detector_inputs, "annotation")


def test_validate_rejects_cross_split_image_hash_leakage(detector_inputs: Path) -> None:
    report = build_detector_dataset(
        detector_inputs,
        "input",
        "leakage",
        DetectorDatasetConfig(seed=5, validation_fraction=1 / 3),
    )
    manifest = _load_json(report.manifest_path)
    train_sample = next(item for item in manifest["samples"] if item["split"] == "train")
    validation_sample = next(
        item for item in manifest["samples"] if item["split"] == "validation"
    )
    validation_sample["source_sha256"] = train_sample["source_sha256"]
    _write_json(report.manifest_path, manifest)

    with pytest.raises(DataValidationError, match="leak|split"):
        validate_detector_dataset(detector_inputs, "leakage")


def test_validate_rejects_test_path_in_manifest(detector_inputs: Path) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "unsafe-manifest", DetectorDatasetConfig()
    )
    manifest = _load_json(report.manifest_path)
    manifest["config"]["real_coco_path"] = "../../../base/test/instances_test.json"
    manifest["inputs"]["real_coco"]["path"] = "../../../base/test/instances_test.json"
    _write_json(report.manifest_path, manifest)

    with pytest.raises(DataValidationError, match="evaluation-only"):
        validate_detector_dataset(detector_inputs, "unsafe-manifest")


def test_validate_rejects_split_that_does_not_match_recorded_config(
    detector_inputs: Path,
) -> None:
    report = build_detector_dataset(
        detector_inputs,
        "input",
        "split-config",
        DetectorDatasetConfig(seed=9, validation_fraction=0.2),
    )
    manifest = _load_json(report.manifest_path)
    manifest["config"]["validation_fraction"] = 0.6
    _write_json(report.manifest_path, manifest)

    with pytest.raises(DataValidationError, match="split.*config|config.*split"):
        validate_detector_dataset(detector_inputs, "split-config")


def test_validate_rejects_extra_output_file(detector_inputs: Path) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "extra", DetectorDatasetConfig()
    )
    (report.output_dir / "train" / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(DataValidationError, match="inventory|unexpected|files"):
        validate_detector_dataset(detector_inputs, "extra")


def test_validate_rejects_missing_and_tampered_output_images(
    detector_inputs: Path,
) -> None:
    missing = build_detector_dataset(
        detector_inputs, "input", "missing-image", DetectorDatasetConfig()
    )
    missing_manifest = _load_json(missing.manifest_path)
    missing_path = missing.output_dir / missing_manifest["samples"][0]["output_path"]
    missing_path.unlink()
    with pytest.raises(DataValidationError, match="cannot hash|missing|inventory"):
        validate_detector_dataset(detector_inputs, "missing-image")

    tampered = build_detector_dataset(
        detector_inputs, "input", "tampered-image", DetectorDatasetConfig()
    )
    tampered_manifest = _load_json(tampered.manifest_path)
    tampered_path = tampered.output_dir / tampered_manifest["samples"][0]["output_path"]
    tampered_path.write_bytes(tampered_path.read_bytes() + b"tampered")
    with pytest.raises(DataValidationError, match="sha256"):
        validate_detector_dataset(detector_inputs, "tampered-image")


def test_validate_rejects_changed_source_image(detector_inputs: Path) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "source-image", DetectorDatasetConfig()
    )
    manifest = _load_json(report.manifest_path)
    real_sample = next(item for item in manifest["samples"] if item["origin"] == "real")
    source_path = (report.output_dir / real_sample["source_path"]).resolve()
    source_path.write_bytes(source_path.read_bytes() + b"tampered")

    with pytest.raises(DataValidationError, match="source sha256|source.*match"):
        validate_detector_dataset(detector_inputs, "source-image")


def test_validate_rejects_invalid_bbox_even_if_coco_hash_is_updated(
    detector_inputs: Path,
) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "bbox", DetectorDatasetConfig()
    )
    manifest = _load_json(report.manifest_path)
    split = next(
        name
        for name in ("train", "validation")
        if manifest["splits"][name]["annotation_count"]
    )
    coco_path = report.output_dir / split / "instances.json"
    coco = _load_json(coco_path)
    coco["annotations"][0]["bbox"][2] = 0
    _write_json(coco_path, coco)
    manifest["splits"][split]["coco_sha256"] = hashlib.sha256(
        coco_path.read_bytes()
    ).hexdigest()
    _write_json(report.manifest_path, manifest)

    with pytest.raises(DataValidationError, match="positive width and height"):
        validate_detector_dataset(detector_inputs, "bbox")


def test_failed_atomic_publish_restores_existing_run(
    detector_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "stable", DetectorDatasetConfig(seed=1)
    )
    before = _tree_bytes(report.output_dir)
    original_rename = Path.rename

    def fail_staging_publish(source: Path, target: Path) -> Path:
        if source.name.startswith(".stable.tmp-") and target.name == "stable":
            raise OSError("forced publish failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_staging_publish)
    with pytest.raises(OSError, match="forced publish failure"):
        build_detector_dataset(
            detector_inputs,
            "input",
            "stable",
            DetectorDatasetConfig(seed=2),
            overwrite=True,
        )

    assert _tree_bytes(report.output_dir) == before
    detector_root = report.output_dir.parent
    assert not list(detector_root.glob(".stable.tmp-*"))
    assert not list(detector_root.glob(".stable.backup-*"))


def test_non_overwrite_rejects_run_that_appears_during_staging(
    detector_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_validate = detector_module._validate_run_dir

    def create_concurrent_run(root: Path, output_dir: Path):
        validated = original_validate(root, output_dir)
        if output_dir.name.startswith(".race.tmp-"):
            concurrent = root / "derived" / "detector" / "race"
            concurrent.mkdir()
            (concurrent / "sentinel.txt").write_text("keep", encoding="utf-8")
        return validated

    monkeypatch.setattr(detector_module, "_validate_run_dir", create_concurrent_run)
    with pytest.raises(DataValidationError, match="appeared|already exists"):
        build_detector_dataset(
            detector_inputs, "input", "race", DetectorDatasetConfig()
        )

    concurrent = detector_inputs / "derived" / "detector" / "race"
    assert (concurrent / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not list(concurrent.parent.glob(".race.tmp-*"))
    assert not list(concurrent.parent.glob(".race.backup-*"))


def test_transient_backup_cleanup_failure_still_commits_successfully(
    detector_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = build_detector_dataset(
        detector_inputs, "input", "cleanup", DetectorDatasetConfig(seed=1)
    )
    original_rmtree = detector_module.shutil.rmtree
    failed_once = False

    def fail_backup_once(path: Path, *args, **kwargs) -> None:
        nonlocal failed_once
        if Path(path).name.startswith(".cleanup.backup-") and not failed_once:
            failed_once = True
            raise PermissionError("transient backup lock")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(detector_module.shutil, "rmtree", fail_backup_once)
    replaced = build_detector_dataset(
        detector_inputs,
        "input",
        "cleanup",
        DetectorDatasetConfig(seed=2),
        overwrite=True,
    )

    assert failed_once
    assert _load_json(replaced.manifest_path)["config"]["seed"] == 2
    assert replaced.output_dir == report.output_dir
    assert not list(report.output_dir.parent.glob(".cleanup.backup-*"))


def test_generate_rejects_existing_run_without_overwrite_and_unsafe_names(
    detector_inputs: Path,
) -> None:
    build_detector_dataset(
        detector_inputs, "input", "existing", DetectorDatasetConfig()
    )
    with pytest.raises(DataValidationError, match="already exists"):
        build_detector_dataset(
            detector_inputs, "input", "existing", DetectorDatasetConfig()
        )
    with pytest.raises(DataValidationError, match="run_name"):
        build_detector_dataset(
            detector_inputs, "input", "../escape", DetectorDatasetConfig()
        )


def test_generate_rejects_invalid_source_bbox_and_test_input(
    detector_inputs: Path,
) -> None:
    coco_path = detector_inputs / "base" / "val" / "instances_val.json"
    coco = _load_json(coco_path)
    coco["annotations"][0]["bbox"][2] = 0
    _write_json(coco_path, coco)
    with pytest.raises(DataValidationError, match="positive width and height"):
        build_detector_dataset(
            detector_inputs, "input", "invalid-bbox", DetectorDatasetConfig()
        )

    with pytest.raises(DataValidationError, match="evaluation-only"):
        build_detector_dataset(
            detector_inputs,
            "input",
            "unsafe-input",
            DetectorDatasetConfig(real_coco_path="base/test/instances_test.json"),
        )
