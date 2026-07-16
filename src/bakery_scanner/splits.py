from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .errors import DataValidationError

SCENE_PATTERN = re.compile(
    r"^scene_(?P<difficulty>[emh])_(?P<scene_id>\d+)\.(?:jpg|jpeg|png|bmp|webp)$"
)


@dataclass(frozen=True, slots=True)
class SceneSplit:
    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    train_scene_ids: tuple[str, ...]
    validation_scene_ids: tuple[str, ...]


def _scene_parts(path: Path) -> tuple[str, str]:
    match = SCENE_PATTERN.fullmatch(path.name)
    if match is None:
        raise DataValidationError(
            f"scene filename must match scene_[e|m|h]_<digits>.<image>: {path.name}"
        )
    return match.group("scene_id"), match.group("difficulty")


def scene_id_from_path(path: str | Path) -> str:
    return _scene_parts(Path(path))[0]


def split_scene_paths(
    paths: Iterable[str | Path], validation_fraction: float, seed: int
) -> SceneSplit:
    if not 0 < validation_fraction < 1:
        raise DataValidationError("validation_fraction must be between 0 and 1")

    normalized = tuple(Path(path) for path in paths)
    if len(normalized) != len(set(normalized)):
        raise DataValidationError("duplicate scene path in split input")

    groups: dict[str, dict[str, Path]] = {}
    for path in normalized:
        scene_id, difficulty = _scene_parts(path)
        group = groups.setdefault(scene_id, {})
        if difficulty in group:
            raise DataValidationError(
                f"scene {scene_id} contains duplicate {difficulty} difficulty"
            )
        group[difficulty] = path

    if len(groups) < 2:
        raise DataValidationError("scene split requires at least two scene groups")
    for scene_id, group in groups.items():
        if set(group) != {"e", "m", "h"}:
            raise DataValidationError(
                f"scene {scene_id} must contain exactly e, m, and h images"
            )

    shuffled_ids = sorted(groups)
    random.Random(seed).shuffle(shuffled_ids)
    validation_count = max(
        1, min(len(groups) - 1, int(len(groups) * validation_fraction + 0.5))
    )
    validation_ids = tuple(sorted(shuffled_ids[:validation_count]))
    validation_id_set = set(validation_ids)
    train_ids = tuple(sorted(set(groups) - validation_id_set))

    train_paths = tuple(
        sorted(
            (
                path
                for scene_id in train_ids
                for path in groups[scene_id].values()
            ),
            key=lambda path: path.as_posix(),
        )
    )
    validation_paths = tuple(
        sorted(
            (
                path
                for scene_id in validation_ids
                for path in groups[scene_id].values()
            ),
            key=lambda path: path.as_posix(),
        )
    )
    if set(train_paths).intersection(validation_paths):
        raise DataValidationError("scene groups overlap between train and validation")
    if set(train_paths).union(validation_paths) != set(normalized):
        raise DataValidationError("scene split did not preserve every input path")

    return SceneSplit(
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_scene_ids=train_ids,
        validation_scene_ids=validation_ids,
    )
