import json
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

from bakery_scanner.detector_cli import main
from bakery_scanner.synthetic import SyntheticConfig, generate_synthetic_dataset


@pytest.fixture
def detector_cli_inputs(
    dataset_factory: Callable[[], Path], tmp_path: Path
) -> Path:
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
        "cli-input",
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
    return dataset_root


def test_generate_cli_emits_json_report(detector_cli_inputs: Path, capsys) -> None:
    exit_code = main(
        [
            "generate",
            "--dataset-root",
            str(detector_cli_inputs),
            "--synthetic-run",
            "cli-input",
            "--run-name",
            "cli-output",
            "--seed",
            "11",
            "--validation-fraction",
            "0.25",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["image_count"] == 8
    assert payload["annotation_count"] == 8
    assert payload["manifest_path"].endswith("manifest.json")
    assert captured.err == ""


def test_validate_cli_emits_json_report(detector_cli_inputs: Path, capsys) -> None:
    assert (
        main(
            [
                "generate",
                "--dataset-root",
                str(detector_cli_inputs),
                "--synthetic-run",
                "cli-input",
                "--run-name",
                "validated",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "validate",
            "--dataset-root",
            str(detector_cli_inputs),
            "--run-name",
            "validated",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["train_image_count"] > 0
    assert payload["validation_image_count"] > 0
    assert captured.err == ""


def test_generate_cli_rejects_evaluation_only_real_coco(
    detector_cli_inputs: Path, capsys
) -> None:
    exit_code = main(
        [
            "generate",
            "--dataset-root",
            str(detector_cli_inputs),
            "--synthetic-run",
            "cli-input",
            "--run-name",
            "unsafe",
            "--real-coco-path",
            "base/test/instances_test.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "evaluation-only" in captured.err
    assert not (
        detector_cli_inputs / "derived" / "detector" / "unsafe"
    ).exists()
