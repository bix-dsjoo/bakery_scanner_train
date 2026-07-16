from pathlib import Path

import pytest

from bakery_scanner.errors import DataValidationError
from bakery_scanner.splits import scene_id_from_path, split_scene_paths


def _scene_paths(*scene_ids: str) -> list[Path]:
    return [
        Path(f"scene_{difficulty}_{scene_id}.jpg")
        for scene_id in scene_ids
        for difficulty in "emh"
    ]


def test_scene_id_from_path_uses_shared_numeric_suffix() -> None:
    assert scene_id_from_path(Path("nested/scene_e_0503.jpg")) == "0503"
    assert scene_id_from_path(Path("scene_h_0503.jpg")) == "0503"


def test_split_scene_paths_keeps_difficulty_variants_together() -> None:
    paths = _scene_paths("0001", "0002", "0003", "0004")

    split = split_scene_paths(paths, validation_fraction=0.25, seed=17)

    assert set(split.train_paths).isdisjoint(split.validation_paths)
    assert set(split.train_paths) | set(split.validation_paths) == set(paths)
    assert set(split.train_scene_ids).isdisjoint(split.validation_scene_ids)
    for scene_id in split.validation_scene_ids:
        assert {path.name for path in split.validation_paths if scene_id in path.name} == {
            f"scene_e_{scene_id}.jpg",
            f"scene_m_{scene_id}.jpg",
            f"scene_h_{scene_id}.jpg",
        }


def test_split_scene_paths_is_deterministic_and_order_independent() -> None:
    paths = _scene_paths("0001", "0002", "0003", "0004", "0005")

    first = split_scene_paths(paths, validation_fraction=0.4, seed=123)
    second = split_scene_paths(reversed(paths), validation_fraction=0.4, seed=123)

    assert first == second


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_split_scene_paths_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(DataValidationError, match="between 0 and 1"):
        split_scene_paths(_scene_paths("0001", "0002"), fraction, seed=1)


def test_split_scene_paths_rejects_invalid_name() -> None:
    with pytest.raises(DataValidationError, match="scene filename"):
        split_scene_paths(
            [*_scene_paths("0001", "0002"), Path("other_0003.jpg")], 0.5, seed=1
        )


def test_split_scene_paths_rejects_duplicate_path() -> None:
    paths = _scene_paths("0001", "0002")

    with pytest.raises(DataValidationError, match="duplicate scene path"):
        split_scene_paths([*paths, paths[0]], 0.5, seed=1)


def test_split_scene_paths_rejects_incomplete_difficulty_group() -> None:
    paths = _scene_paths("0001", "0002")
    paths.remove(Path("scene_h_0002.jpg"))

    with pytest.raises(DataValidationError, match="exactly e, m, and h"):
        split_scene_paths(paths, 0.5, seed=1)


def test_split_scene_paths_requires_two_scene_groups() -> None:
    with pytest.raises(DataValidationError, match="at least two"):
        split_scene_paths(_scene_paths("0001"), 0.5, seed=1)
