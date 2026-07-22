from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .errors import DataValidationError


def _configured_path(path: str | Path, dataset_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return Path(os.path.abspath(candidate))
    if candidate.parts and candidate.parts[0].casefold() == dataset_root.name.casefold():
        return Path(os.path.abspath(dataset_root.parent / candidate))
    return Path(os.path.abspath(dataset_root / candidate))


_EVALUATION_PREFIXES = (("base", "test"), ("incremental", "test"))


def _comparison_parts(path: Path) -> tuple[str, ...]:
    return tuple(os.path.normcase(part).casefold() for part in path.parts)


def _starts_with(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and parts[: len(prefix)] == prefix


def _contains_evaluation_tree(dataset_root: Path, candidate: Path) -> bool:
    root_parts = _comparison_parts(dataset_root)
    candidate_parts = _comparison_parts(candidate)
    if _starts_with(root_parts, candidate_parts):
        return True
    if not _starts_with(candidate_parts, root_parts):
        return False
    relative = candidate_parts[len(root_parts) :]
    return any(
        _starts_with(relative, forbidden) or _starts_with(forbidden, relative)
        for forbidden in _EVALUATION_PREFIXES
    )


def assert_training_paths_safe(
    paths: Iterable[str | Path], dataset_root: str | Path
) -> tuple[Path, ...]:
    root = Path(os.path.abspath(dataset_root))
    configured = tuple(_configured_path(path, root) for path in paths)
    if not configured:
        raise DataValidationError("training configuration requires at least one data path")

    root_physical = root.resolve(strict=False)
    resolved: list[Path] = []
    for candidate in configured:
        if _contains_evaluation_tree(root, candidate):
            raise DataValidationError(
                f"training path is evaluation-only and forbidden: {candidate}"
            )
        physical = candidate.resolve(strict=False)
        if _contains_evaluation_tree(root_physical, physical):
            raise DataValidationError(
                f"training path is evaluation-only and forbidden: {candidate}"
            )
        resolved.append(physical)
    return tuple(resolved)
