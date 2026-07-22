import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from bakery_scanner.errors import DataValidationError
from bakery_scanner.synthetic import (
    GENERATOR_VERSION,
    SyntheticConfig,
    generate_synthetic_dataset,
    generate_synthetic_dataset_from_backgrounds,
    validate_synthetic_dataset,
)


def _write_registry(dataset_root: Path) -> None:
    classes = []
    for model_index in range(20):
        phase = "base" if model_index < 15 else "incremental"
        category_id = (model_index * 7) % 20 + 1
        folder_name = f"bread_{category_id:02d}_item"
        classes.append(
            {
                "category_id": category_id,
                "model_index": model_index,
                "canonical_name": f"Bread {category_id}",
                "folder_name": folder_name,
                "phase": phase,
            }
        )
        source_dir = dataset_root / phase / folder_name
        source_dir.mkdir(parents=True)
        source = Image.new("RGB", (12, 10), "white")
        for x in range(2, 10):
            for y in range(3, 8):
                source.putpixel((x, y), (180, 60 + model_index, 20))
        source.save(source_dir / "object.png")
    (dataset_root / "class_registry.json").write_text(
        json.dumps({"version": 1, "classes": classes}), encoding="utf-8"
    )


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "datasets"
    _write_registry(dataset_root)
    background_dir = tmp_path / "backgrounds"
    background_dir.mkdir()
    Image.new("RGB", (80, 60), (210, 210, 210)).save(
        background_dir / "tray.png"
    )
    return dataset_root, background_dir


def _fixed_config(**overrides: object) -> SyntheticConfig:
    values = {
        "seed": 123,
        "scene_count": 2,
        "objects_per_scene": 2,
        "phase": "base",
        "size_fraction_range": (0.2, 0.2),
        "rotation_range": (0.0, 0.0),
        "brightness_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "foreground_threshold": 245,
    }
    values.update(overrides)
    return SyntheticConfig(**values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_writes_versioned_manifest_with_required_provenance(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs

    report = generate_synthetic_dataset(
        dataset_root, background_dir, "unit", _fixed_config()
    )

    expected_root = dataset_root / "derived" / "synthetic" / "unit"
    assert report.output_dir == expected_root.resolve()
    assert report.manifest_path == expected_root.resolve() / "manifest.json"
    assert report.image_count == 2
    assert report.object_count == 4
    assert {path.name for path in expected_root.iterdir()} == {
        "manifest.json",
        "scene_000000.png",
        "scene_000001.png",
    }
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 1
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["seed"] == 123
    assert manifest["config"]["phase"] == "base"
    scene = manifest["scenes"][0]
    assert scene["seed"] >= 0
    assert scene["background_path"].endswith("tray.png")
    assert len(scene["sha256"]) == 64
    obj = scene["objects"][0]
    assert obj["source_path"].endswith("object.png")
    assert obj["category_id"] in {
        (model_index * 7) % 20 + 1 for model_index in range(15)
    }
    assert "model_index" not in obj
    assert set(obj["transform"]) == {
        "brightness",
        "contrast",
        "position",
        "rotation_degrees",
        "scale",
        "size",
        "target_size_fraction",
    }
    assert len(obj["bbox"]) == 4


def test_generation_is_deterministic_for_same_seed_and_inputs(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    config = _fixed_config(rotation_range=(-15.0, 15.0))

    first = generate_synthetic_dataset(dataset_root, background_dir, "first", config)
    second = generate_synthetic_dataset(dataset_root, background_dir, "second", config)

    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest == second_manifest
    assert [_sha256(path) for path in sorted(first.output_dir.glob("*.png"))] == [
        _sha256(path) for path in sorted(second.output_dir.glob("*.png"))
    ]


def test_generate_from_explicit_backgrounds_uses_only_allowlisted_files(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    second = background_dir / "tray_second.png"
    excluded = background_dir / "tray_excluded.png"
    Image.new("RGB", (80, 60), (200, 200, 200)).save(second)
    Image.new("RGB", (80, 60), (190, 190, 190)).save(excluded)

    report = generate_synthetic_dataset_from_backgrounds(
        dataset_root,
        (background_dir / "tray.png", second),
        "explicit",
        _fixed_config(scene_count=12),
    )

    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    used = {
        (report.output_dir / scene["background_path"]).resolve()
        for scene in manifest["scenes"]
    }
    assert used == {(background_dir / "tray.png").resolve(), second.resolve()}
    assert excluded.resolve() not in used
    assert validate_synthetic_dataset(dataset_root, "explicit").image_count == 12


def test_bbox_matches_composited_foreground_pixels(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    report = generate_synthetic_dataset(
        dataset_root,
        background_dir,
        "bbox",
        _fixed_config(scene_count=1, objects_per_scene=1),
    )
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    bbox = manifest["scenes"][0]["objects"][0]["bbox"]
    x, y, width, height = bbox

    with Image.open(report.output_dir / "scene_000000.png") as image:
        image.load()
        changed = {
            (px, py)
            for py in range(image.height)
            for px in range(image.width)
            if image.getpixel((px, py)) != (210, 210, 210)
        }

    assert changed
    assert min(px for px, _ in changed) == x
    assert min(py for _, py in changed) == y
    assert max(px for px, _ in changed) == x + width - 1
    assert max(py for _, py in changed) == y + height - 1


def test_generate_rejects_output_escape(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs

    with pytest.raises(DataValidationError, match="run_name"):
        generate_synthetic_dataset(
            dataset_root, background_dir, "../escape", _fixed_config()
        )


def test_generate_rejects_evaluation_only_background(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, _ = synthetic_inputs
    background_dir = dataset_root / "base" / "test"
    background_dir.mkdir()
    Image.new("RGB", (80, 60), "gray").save(background_dir / "tray.png")

    with pytest.raises(DataValidationError, match="evaluation-only"):
        generate_synthetic_dataset(
            dataset_root, background_dir, "unsafe", _fixed_config()
        )


def test_generate_requires_explicit_overwrite(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    config = _fixed_config(scene_count=1, objects_per_scene=1)
    generate_synthetic_dataset(dataset_root, background_dir, "existing", config)

    with pytest.raises(DataValidationError, match="already exists"):
        generate_synthetic_dataset(dataset_root, background_dir, "existing", config)

    regenerated = generate_synthetic_dataset(
        dataset_root, background_dir, "existing", config, overwrite=True
    )
    assert regenerated.image_count == 1


def test_generate_rejects_file_at_run_directory(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    run_path = dataset_root / "derived" / "synthetic" / "not-a-directory"
    run_path.parent.mkdir(parents=True)
    run_path.write_text("occupied", encoding="utf-8")

    with pytest.raises(DataValidationError, match="directory"):
        generate_synthetic_dataset(
            dataset_root,
            background_dir,
            "not-a-directory",
            _fixed_config(),
            overwrite=True,
        )


def test_failed_overwrite_preserves_existing_valid_run(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    config = _fixed_config(scene_count=1, objects_per_scene=1)
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "preserved", config
    )
    before = {
        path.name: path.read_bytes() for path in generated.output_dir.iterdir()
    }
    for source_path in dataset_root.glob("*/bread_*/*"):
        Image.new("RGB", (12, 10), "white").save(source_path)

    with pytest.raises(DataValidationError, match="foreground mask is empty"):
        generate_synthetic_dataset(
            dataset_root,
            background_dir,
            "preserved",
            config,
            overwrite=True,
        )

    assert {
        path.name: path.read_bytes() for path in generated.output_dir.iterdir()
    } == before


@pytest.mark.skipif(os.name != "nt", reason="Windows file-lock behavior")
def test_overwrite_waits_for_transient_windows_file_lock(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    config = _fixed_config(scene_count=1, objects_per_scene=1)
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "locked", config
    )
    ready = threading.Event()

    def hold_image_open() -> None:
        with (generated.output_dir / "scene_000000.png").open("rb"):
            ready.set()
            time.sleep(0.25)

    holder = threading.Thread(target=hold_image_open)
    holder.start()
    assert ready.wait(timeout=1)
    try:
        regenerated = generate_synthetic_dataset(
            dataset_root, background_dir, "locked", config, overwrite=True
        )
    finally:
        holder.join(timeout=1)

    assert regenerated.image_count == 1
    assert validate_synthetic_dataset(dataset_root, "locked").image_count == 1


def test_generate_rejects_run_link_that_resolves_to_another_run(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    synthetic_root = dataset_root / "derived" / "synthetic"
    target = synthetic_root / "target"
    target.mkdir(parents=True)
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    alias = synthetic_root / "alias"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(DataValidationError, match="link|junction"):
        generate_synthetic_dataset(
            dataset_root,
            background_dir,
            "alias",
            _fixed_config(scene_count=1, objects_per_scene=1),
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_validate_replays_manifest_image_and_bbox(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generate_synthetic_dataset(dataset_root, background_dir, "valid", _fixed_config())

    report = validate_synthetic_dataset(dataset_root, "valid")

    assert report.image_count == 2
    assert report.object_count == 4
    assert report.generator_version == GENERATOR_VERSION


def test_validate_rejects_unknown_manifest_field(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "unknown-field", _fixed_config()
    )
    payload = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    generated.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="schema"):
        validate_synthetic_dataset(dataset_root, "unknown-field")


def test_validate_rejects_unknown_object_field(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "unknown-object-field", _fixed_config()
    )
    payload = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
    payload["scenes"][0]["objects"][0]["unexpected"] = True
    generated.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="schema"):
        validate_synthetic_dataset(dataset_root, "unknown-object-field")


def test_validate_rejects_tampered_image(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "tampered-image", _fixed_config()
    )
    image_path = generated.output_dir / "scene_000000.png"
    with Image.open(image_path) as image:
        changed = image.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(image_path)

    with pytest.raises(DataValidationError, match="sha256|replay"):
        validate_synthetic_dataset(dataset_root, "tampered-image")


def test_validate_rejects_tampered_bbox(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "tampered-bbox", _fixed_config()
    )
    payload = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
    payload["scenes"][0]["objects"][0]["bbox"][0] += 1
    generated.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="bbox"):
        validate_synthetic_dataset(dataset_root, "tampered-bbox")


def test_validate_rejects_tampered_recorded_scale(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "tampered-scale", _fixed_config()
    )
    payload = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
    payload["scenes"][0]["objects"][0]["transform"]["scale"] += 0.1
    generated.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="scale"):
        validate_synthetic_dataset(dataset_root, "tampered-scale")


def test_validate_rejects_transform_outside_configured_range(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "tampered-range", _fixed_config()
    )
    payload = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
    payload["scenes"][0]["objects"][0]["transform"]["brightness"] = 2.0
    generated.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="brightness.*configured range"):
        validate_synthetic_dataset(dataset_root, "tampered-range")


def test_validate_rejects_scene_seed_not_derived_from_master_seed(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "tampered-seed", _fixed_config()
    )
    payload = json.loads(generated.manifest_path.read_text(encoding="utf-8"))
    payload["scenes"][0]["seed"] += 1
    generated.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="scene seed"):
        validate_synthetic_dataset(dataset_root, "tampered-seed")


def test_validate_rejects_missing_output_image(
    synthetic_inputs: tuple[Path, Path],
) -> None:
    dataset_root, background_dir = synthetic_inputs
    generated = generate_synthetic_dataset(
        dataset_root, background_dir, "missing", _fixed_config()
    )
    (generated.output_dir / "scene_000000.png").unlink()

    with pytest.raises(DataValidationError, match="output images"):
        validate_synthetic_dataset(dataset_root, "missing")
