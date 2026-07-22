from __future__ import annotations

import json
from pathlib import Path

import pytest

from bakery_scanner import base_cycle_cli
from bakery_scanner.base_cycle import BaseCycleReport, load_base_cycle_config
from bakery_scanner.errors import DataValidationError


def _report() -> BaseCycleReport:
    output = Path("datasets/derived/base_cycle/base_v2")
    return BaseCycleReport(
        output_dir=output,
        manifest_path=output / "manifest.json",
        development_scene_ids=("0503", "0509"),
        holdout_scene_id="0510",
        development_image_count=6,
        holdout_image_count=3,
        development_background_count=2,
        holdout_background_count=1,
        seeds=(42, 43, 44),
    )


def test_freeze_cli_prints_concrete_assignment(monkeypatch, capsys) -> None:
    monkeypatch.setattr(base_cycle_cli, "freeze_base_cycle", lambda *_args: _report())

    assert (
        base_cycle_cli.main(
            ["freeze", "--config", "configs/base_cycle/base_v2.yaml"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "development scene IDs: 0503, 0509" in output
    assert "cycle holdout scene ID: 0510" in output
    assert "seeds: 42, 43, 44" in output


def test_validate_cli_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(base_cycle_cli, "validate_base_cycle", lambda *_args: _report())

    assert (
        base_cycle_cli.main(
            [
                "validate",
                "--repository-root",
                ".",
                "--run-name",
                "base_v2",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["holdout_scene_id"] == "0510"


def test_cli_returns_one_for_data_error(monkeypatch, capsys) -> None:
    def fail(*_args):
        raise DataValidationError("unsafe")

    monkeypatch.setattr(base_cycle_cli, "freeze_base_cycle", fail)

    assert base_cycle_cli.main(["freeze", "--config", "bad.yaml"]) == 1
    assert "unsafe" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["freeze", "--config", "config.yaml", "--overwrite"],
        ["freeze", "--config", "config.yaml", "--output-root", "elsewhere"],
        ["validate", "--run-name", "base_v2", "--output-root", "elsewhere"],
    ],
)
def test_cli_does_not_expose_mutable_output_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        base_cycle_cli.main(argv)
    assert raised.value.code == 2


def test_checked_in_config_has_exact_approved_assignment() -> None:
    config = load_base_cycle_config("configs/base_cycle/base_v2.yaml")

    assert config.development_scene_ids == ("0503", "0509")
    assert config.holdout_scene_id == "0510"
    assert config.development_backgrounds == (
        "collected/backgrounds/tray_white_square.png",
        "collected/backgrounds/tray_wood_black_surface.png",
    )
    assert config.holdout_background == (
        "collected/backgrounds/tray_wood_white_surface.png"
    )
    assert config.seeds == (42, 43, 44)
