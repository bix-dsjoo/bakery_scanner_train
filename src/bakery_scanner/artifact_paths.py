from __future__ import annotations

from pathlib import Path


_ARTIFACT_ANCHORS = frozenset({"datasets", "runs", "configs", "models"})


def _portable_parts(value: str | Path) -> tuple[str, ...] | None:
    parts = tuple(
        part for part in str(value).replace("\\", "/").split("/") if part
    )
    if not parts or any(part in {".", ".."} for part in parts):
        return None
    return parts


def _same_parts(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    return tuple(item.casefold() for item in first) == tuple(
        item.casefold() for item in second
    )


def recorded_artifact_path_matches(
    recorded_path: str | Path,
    actual_path: Path,
    *,
    project_root: Path,
) -> bool:
    if not isinstance(recorded_path, (str, Path)):
        return False
    root = Path(project_root).resolve(strict=False)
    actual = Path(actual_path).resolve(strict=False)
    try:
        relative = actual.relative_to(root)
    except ValueError:
        return False
    if (
        not relative.parts
        or relative.parts[0].casefold() not in _ARTIFACT_ANCHORS
    ):
        return False
    recorded_parts = _portable_parts(recorded_path)
    root_parts = _portable_parts(root)
    if (
        recorded_parts is None
        or root_parts is None
        or len(recorded_parts) <= len(root_parts)
        or not _same_parts(recorded_parts[: len(root_parts)], root_parts)
    ):
        return False
    if Path(recorded_path).resolve(strict=False) == actual:
        return True
    tail = recorded_parts[len(root_parts) :]
    expected = tuple(relative.parts)
    if _same_parts(tail, expected):
        return True
    return (
        len(tail) >= 3
        and tail[0].casefold() == ".worktrees"
        and tail[1] not in {"", ".", ".."}
        and _same_parts(tail[2:], expected)
    )
