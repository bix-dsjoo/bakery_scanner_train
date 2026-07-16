import json
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image

from bakery_scanner.synthetic_cli import main


@pytest.fixture
def cli_synthetic_inputs(
    dataset_factory: Callable[[], Path], tmp_path: Path
) -> tuple[Path, Path]:
    dataset_root = dataset_factory()
    for path in dataset_root.glob("*/bread_*/*.jpg"):
        source = Image.new("RGB", (12, 10), "white")
        for x in range(2, 10):
            for y in range(3, 8):
                source.putpixel((x, y), (170, 80, 20))
        source.save(path)
    background_dir = tmp_path / "backgrounds"
    background_dir.mkdir()
    Image.new("RGB", (80, 60), (210, 210, 210)).save(
        background_dir / "tray.png"
    )
    return dataset_root, background_dir


def test_generate_cli_emits_json_report(
    cli_synthetic_inputs: tuple[Path, Path], capsys
) -> None:
    dataset_root, background_dir = cli_synthetic_inputs

    exit_code = main(
        [
            "generate",
            "--dataset-root",
            str(dataset_root),
            "--background-dir",
            str(background_dir),
            "--run-name",
            "cli",
            "--seed",
            "7",
            "--scene-count",
            "1",
            "--objects-per-scene",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["image_count"] == 1
    assert payload["object_count"] == 1
    assert payload["manifest_path"].endswith("manifest.json")
    assert captured.err == ""


def test_validate_cli_emits_json_report(
    cli_synthetic_inputs: tuple[Path, Path], capsys
) -> None:
    dataset_root, background_dir = cli_synthetic_inputs
    assert (
        main(
            [
                "generate",
                "--dataset-root",
                str(dataset_root),
                "--background-dir",
                str(background_dir),
                "--run-name",
                "validate-cli",
                "--scene-count",
                "1",
                "--objects-per-scene",
                "1",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "validate",
            "--dataset-root",
            str(dataset_root),
            "--run-name",
            "validate-cli",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["image_count"] == 1
    assert captured.err == ""


def test_generate_cli_returns_failure_for_missing_background(
    dataset_factory: Callable[[], Path], tmp_path: Path, capsys
) -> None:
    exit_code = main(
        [
            "generate",
            "--dataset-root",
            str(dataset_factory()),
            "--background-dir",
            str(tmp_path / "missing"),
            "--run-name",
            "invalid",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Synthetic generation failed" in captured.err
