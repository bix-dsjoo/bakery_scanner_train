from pathlib import Path

import pytest

from bakery_scanner.errors import DataValidationError
from bakery_scanner.safety import assert_training_paths_safe


def test_training_paths_allow_train_side_data(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"

    safe = assert_training_paths_safe(
        [dataset_root / "base" / "bread_02_item", Path("base/val")],
        dataset_root,
    )

    assert safe[0] == (dataset_root / "base" / "bread_02_item").resolve()
    assert safe[1] == (dataset_root / "base" / "val").resolve()


@pytest.mark.parametrize(
    "configured_path",
    [
        Path("base/test"),
        Path("incremental/test/instances_test.json"),
        Path("base/train/../test/scene.jpg"),
        Path("datasets/base/test"),
    ],
)
def test_training_paths_reject_evaluation_data(
    tmp_path: Path, configured_path: Path
) -> None:
    dataset_root = tmp_path / "datasets"

    with pytest.raises(DataValidationError, match="evaluation-only"):
        assert_training_paths_safe([configured_path], dataset_root)


def test_training_paths_reject_absolute_case_variant_on_windows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    configured_path = Path(str(dataset_root / "base" / "test").upper())

    with pytest.raises(DataValidationError, match="evaluation-only"):
        assert_training_paths_safe([configured_path], dataset_root)


@pytest.mark.parametrize("configured_path", [Path("datasets"), Path("base")])
def test_training_paths_reject_roots_that_contain_evaluation_data(
    tmp_path: Path, configured_path: Path
) -> None:
    dataset_root = tmp_path / "datasets"

    if configured_path == Path("datasets"):
        configured_path = dataset_root

    with pytest.raises(DataValidationError, match="evaluation-only"):
        assert_training_paths_safe([configured_path], dataset_root)


def test_training_paths_reject_empty_configuration(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="at least one"):
        assert_training_paths_safe([], tmp_path / "datasets")
