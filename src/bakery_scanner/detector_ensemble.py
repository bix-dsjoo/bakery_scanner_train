from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .detector_evaluation import Detection
from .detector_postprocess import DetectorPostprocessConfig, filter_detections
from .errors import DataValidationError

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_FIELDS = {
    "dataset_root",
    "output_root",
    "run_name",
    "members",
    "cpu_threads",
    "cpu_warmups",
    "cpu_repetitions",
}
_MEMBER_FIELDS = {
    "config_path",
    "config_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
}


@dataclass(frozen=True, slots=True)
class DetectorEnsembleMember:
    config_path: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class DetectorEnsembleConfig:
    dataset_root: str
    output_root: str
    run_name: str
    members: tuple[DetectorEnsembleMember, DetectorEnsembleMember]
    cpu_threads: int
    cpu_warmups: int
    cpu_repetitions: int


def _strict_object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise DataValidationError(
            f"{label} fields are invalid: expected={sorted(fields)}, actual={actual}"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    result = _text(value, label)
    if not _SHA256.fullmatch(result):
        raise DataValidationError(f"{label} must be a lowercase SHA-256")
    return result


def _integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataValidationError(f"{label} must be {qualifier}")
    return value


def load_detector_ensemble_config(path: str | Path) -> DetectorEnsembleConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot load detector ensemble config {source}: {exc}") from exc
    payload = _strict_object(payload, _CONFIG_FIELDS, "detector ensemble config")
    raw_members = payload["members"]
    if not isinstance(raw_members, list) or len(raw_members) != 2:
        raise DataValidationError("detector ensemble requires exactly two members")
    members: list[DetectorEnsembleMember] = []
    for index, value in enumerate(raw_members):
        member = _strict_object(value, _MEMBER_FIELDS, f"ensemble member {index}")
        members.append(
            DetectorEnsembleMember(
                config_path=_text(member["config_path"], "member config_path"),
                config_sha256=_sha(member["config_sha256"], "member config SHA-256"),
                checkpoint_path=_text(
                    member["checkpoint_path"], "member checkpoint_path"
                ),
                checkpoint_sha256=_sha(
                    member["checkpoint_sha256"], "member checkpoint SHA-256"
                ),
            )
        )
    if len({member.config_path for member in members}) != 2 or len(
        {member.checkpoint_path for member in members}
    ) != 2:
        raise DataValidationError("detector ensemble member paths must be unique")
    run_name = _text(payload["run_name"], "run_name")
    if not _RUN_NAME.fullmatch(run_name):
        raise DataValidationError(f"run_name is invalid: {run_name!r}")
    return DetectorEnsembleConfig(
        dataset_root=_text(payload["dataset_root"], "dataset_root"),
        output_root=_text(payload["output_root"], "output_root"),
        run_name=run_name,
        members=(members[0], members[1]),
        cpu_threads=_integer(payload["cpu_threads"], "cpu_threads"),
        cpu_warmups=_integer(payload["cpu_warmups"], "cpu_warmups", allow_zero=True),
        cpu_repetitions=_integer(payload["cpu_repetitions"], "cpu_repetitions"),
    )


def merge_member_predictions(
    member_predictions: Sequence[Mapping[str, Sequence[Detection]]],
    image_ids: Sequence[str],
    postprocess: DetectorPostprocessConfig,
) -> dict[str, tuple[Detection, ...]]:
    if len(member_predictions) != 2:
        raise DataValidationError("detector ensemble requires exactly two prediction sets")
    expected_order = tuple(image_ids)
    if len(expected_order) != len(set(expected_order)):
        raise DataValidationError("detector ensemble image IDs must be unique")
    expected = set(expected_order)
    for index, predictions in enumerate(member_predictions):
        if set(predictions) != expected:
            raise DataValidationError(
                f"ensemble member {index} image IDs do not match evaluation images"
            )
    merged: dict[str, tuple[Detection, ...]] = {}
    for image_id in expected_order:
        candidates = tuple(
            detection
            for predictions in member_predictions
            for detection in predictions[image_id]
        )
        merged[image_id] = filter_detections(candidates, postprocess)
    return merged
