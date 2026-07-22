from __future__ import annotations

from pathlib import Path

import pytest

from bakery_scanner.detector_ensemble import (
    load_detector_ensemble_config,
    merge_member_predictions,
)
from bakery_scanner.detector_evaluation import Detection
from bakery_scanner.detector_postprocess import DetectorPostprocessConfig
from bakery_scanner.errors import DataValidationError


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _member_yaml(index: int, *, config_sha: str, checkpoint_sha: str) -> str:
    return f"""\
  - config_path: configs/detector/member_{index}.yaml
    config_sha256: {config_sha}
    checkpoint_path: runs/detector/member_{index}/checkpoints/best.pt
    checkpoint_sha256: {checkpoint_sha}
"""


def _config_yaml(*, members: str) -> str:
    return f"""\
dataset_root: datasets
output_root: runs/detector_ensemble
run_name: ensemble-fixture
members:
{members}cpu_threads: 8
cpu_warmups: 1
cpu_repetitions: 3
"""


def test_load_ensemble_config_accepts_exact_two_member_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ensemble.yaml"
    path.write_text(
        _config_yaml(
            members=_member_yaml(1, config_sha=_SHA_A, checkpoint_sha=_SHA_B)
            + _member_yaml(2, config_sha=_SHA_C, checkpoint_sha=_SHA_D)
        ),
        encoding="utf-8",
    )

    config = load_detector_ensemble_config(path)

    assert config.dataset_root == "datasets"
    assert config.run_name == "ensemble-fixture"
    assert config.cpu_threads == 8
    assert config.cpu_warmups == 1
    assert config.cpu_repetitions == 3
    assert [member.config_sha256 for member in config.members] == [_SHA_A, _SHA_C]


@pytest.mark.parametrize(
    ("members", "message"),
    [
        (_member_yaml(1, config_sha=_SHA_A, checkpoint_sha=_SHA_B), "exactly two"),
        (
            _member_yaml(1, config_sha="ABC", checkpoint_sha=_SHA_B)
            + _member_yaml(2, config_sha=_SHA_C, checkpoint_sha=_SHA_D),
            "SHA-256",
        ),
        (
            _member_yaml(1, config_sha=_SHA_A, checkpoint_sha=_SHA_B)
            + _member_yaml(1, config_sha=_SHA_C, checkpoint_sha=_SHA_D),
            "unique",
        ),
    ],
)
def test_load_ensemble_config_rejects_invalid_members(
    tmp_path: Path, members: str, message: str
) -> None:
    path = tmp_path / "ensemble.yaml"
    path.write_text(_config_yaml(members=members), encoding="utf-8")

    with pytest.raises(DataValidationError, match=message):
        load_detector_ensemble_config(path)


def test_merge_member_predictions_is_deterministic_and_preserves_empty() -> None:
    member_one = {
        "scene_a.jpg": (
            Detection((0.0, 0.0, 10.0, 10.0), 0.8),
            Detection((30.0, 0.0, 40.0, 10.0), 0.7),
        ),
        "empty.jpg": (),
    }
    member_two = {
        "scene_a.jpg": (
            Detection((1.0, 0.0, 11.0, 10.0), 0.9),
            Detection((60.0, 0.0, 70.0, 10.0), 0.6),
        ),
        "empty.jpg": (),
    }
    postprocess = DetectorPostprocessConfig(0.1, 0.5, 2.0)

    first = merge_member_predictions(
        (member_one, member_two), ("scene_a.jpg", "empty.jpg"), postprocess
    )
    second = merge_member_predictions(
        (member_one, member_two), ("scene_a.jpg", "empty.jpg"), postprocess
    )

    assert first == second
    assert first["empty.jpg"] == ()
    assert [item.confidence for item in first["scene_a.jpg"]] == [0.9, 0.7, 0.6]


@pytest.mark.parametrize("failure", ["missing", "extra", "class"])
def test_merge_member_predictions_rejects_contract_mismatch(failure: str) -> None:
    valid = {"scene.jpg": (Detection((0.0, 0.0, 10.0, 10.0), 0.8),)}
    if failure == "missing":
        invalid = {}
    elif failure == "extra":
        invalid = {**valid, "other.jpg": ()}
    else:
        invalid = {"scene.jpg": (Detection((0.0, 0.0, 10.0, 10.0), 0.8, 1),)}

    with pytest.raises(DataValidationError):
        merge_member_predictions(
            (valid, invalid),
            ("scene.jpg",),
            DetectorPostprocessConfig(0.1, 0.5, 2.0),
        )
