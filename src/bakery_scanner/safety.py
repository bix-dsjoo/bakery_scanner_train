from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .errors import DataValidationError


def _configured_path(path: str | Path, dataset_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if candidate.parts and candidate.parts[0].casefold() == dataset_root.name.casefold():
        return (dataset_root.parent / candidate).resolve(strict=False)
    return (dataset_root / candidate).resolve(strict=False)


def _contains(root: Path, candidate: Path) -> bool:
    root_key = os.path.normcase(str(root))
    candidate_key = os.path.normcase(str(candidate))
    try:
        return os.path.commonpath([root_key, candidate_key]) == root_key
    except ValueError:
        return False


def assert_training_paths_safe(
    paths: Iterable[str | Path], dataset_root: str | Path
) -> tuple[Path, ...]:
    root = Path(dataset_root).resolve(strict=False)
    configured = tuple(_configured_path(path, root) for path in paths)
    if not configured:
        raise DataValidationError("training configuration requires at least one data path")

    forbidden = (
        (root / "base" / "test").resolve(strict=False),
        (root / "incremental" / "test").resolve(strict=False),
    )
    for candidate in configured:
        if any(
            _contains(test_root, candidate) or _contains(candidate, test_root)
            for test_root in forbidden
        ):
            raise DataValidationError(
                f"training path is evaluation-only and forbidden: {candidate}"
            )
    return configured
