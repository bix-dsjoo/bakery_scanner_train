import json
from pathlib import Path
from typing import Callable

from bakery_scanner.cli import main


def test_cli_prints_human_readable_audit(
    dataset_factory: Callable[[], Path], capsys
) -> None:
    exit_code = main(["--dataset-root", str(dataset_factory()), "--validation-fraction", "0.5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dataset audit passed" in captured.out
    assert "Base scene train: 6 images" in captured.out
    assert "Evaluation-only splits were inspected but not authorized for training" in captured.out
    assert captured.err == ""


def test_cli_prints_machine_readable_json(
    dataset_factory: Callable[[], Path], capsys
) -> None:
    exit_code = main(["--dataset-root", str(dataset_factory()), "--json", "--seed", "9"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["scene_split"]["seed"] == 9
    assert captured.err == ""


def test_cli_returns_failure_for_invalid_dataset(tmp_path: Path, capsys) -> None:
    exit_code = main(["--dataset-root", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Dataset audit failed" in captured.err
