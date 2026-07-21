from pathlib import Path

import pytest

from bakery_scanner.artifact_paths import recorded_artifact_path_matches


def test_exact_artifact_path_matches(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"
    actual = root / "datasets" / "derived" / "manifest.json"

    assert recorded_artifact_path_matches(actual, actual, project_root=root)


def test_linked_worktree_artifact_path_matches_root_copy(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"
    actual = root / "datasets" / "derived" / "manifest.json"
    recorded = (
        root
        / ".worktrees"
        / "classifier-foundation"
        / "datasets"
        / "derived"
        / "manifest.json"
    )

    assert recorded_artifact_path_matches(recorded, actual, project_root=root)


def test_windows_separator_worktree_path_matches_root_copy(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"
    actual = root / "runs" / "classifier" / "best.pt"
    recorded = str(
        root
        / ".worktrees"
        / "CLASSIFIER-FOUNDATION"
        / "RUNS"
        / "classifier"
        / "best.pt"
    ).replace("/", "\\")

    assert recorded_artifact_path_matches(recorded, actual, project_root=root)


@pytest.mark.parametrize(
    ("recorded", "actual"),
    [
        ("other_repo/.worktrees/w/datasets/a.json", "datasets/a.json"),
        ("bakery_scanner_train/.worktrees/datasets/a.json", "datasets/a.json"),
        (
            "bakery_scanner_train/.worktrees/w/extra/datasets/a.json",
            "datasets/a.json",
        ),
        ("bakery_scanner_train/.worktrees/w/datasets/b.json", "datasets/a.json"),
        ("bakery_scanner_train/.worktrees/w/other/a.json", "other/a.json"),
        (
            "bakery_scanner_train/.worktrees/w/datasets/../a.json",
            "datasets/a.json",
        ),
    ],
)
def test_artifact_path_rejects_non_identity(
    tmp_path: Path, recorded: str, actual: str
) -> None:
    root = tmp_path / "bakery_scanner_train"

    assert not recorded_artifact_path_matches(
        recorded, root / actual, project_root=root
    )


def test_artifact_path_rejects_actual_outside_project(tmp_path: Path) -> None:
    root = tmp_path / "bakery_scanner_train"

    assert not recorded_artifact_path_matches(
        root / "datasets" / "a.json",
        tmp_path / "outside" / "datasets" / "a.json",
        project_root=root,
    )


def test_artifact_path_rejects_absolute_traversal_even_if_it_resolves_to_actual(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bakery_scanner_train"
    actual = root / "datasets" / "manifest.json"
    recorded = root / "datasets" / "derived" / ".." / "manifest.json"

    assert not recorded_artifact_path_matches(
        recorded,
        actual,
        project_root=root,
    )
