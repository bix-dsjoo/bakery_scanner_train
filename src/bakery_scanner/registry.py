from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .errors import DataValidationError

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
PHASE_COUNTS = {"base": 15, "incremental": 5}


@dataclass(frozen=True, slots=True)
class ClassRecord:
    category_id: int
    model_index: int
    canonical_name: str
    folder_name: str
    phase: str


@dataclass(frozen=True, slots=True)
class ClassRegistry:
    version: int
    classes: tuple[ClassRecord, ...]
    by_category_id: Mapping[int, ClassRecord]
    by_model_index: Mapping[int, ClassRecord]

    @property
    def phase_counts(self) -> dict[str, int]:
        counts = Counter(record.phase for record in self.classes)
        return {phase: counts[phase] for phase in PHASE_COUNTS}


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"{label} must be an integer")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string")
    return value


def _reject_duplicates(records: list[ClassRecord], field: str) -> None:
    values = [getattr(record, field) for record in records]
    if len(values) != len(set(values)):
        raise DataValidationError(f"duplicate {field} in class registry")


def load_class_registry(path: str | Path) -> ClassRegistry:
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load class registry {registry_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DataValidationError("class registry root must be an object")
    version = _require_int(payload.get("version"), "registry version")
    if version != 1:
        raise DataValidationError(f"unsupported registry version: {version}")
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list) or len(raw_classes) != 20:
        raise DataValidationError("class registry must contain exactly 20 classes")

    records: list[ClassRecord] = []
    for position, raw in enumerate(raw_classes):
        if not isinstance(raw, dict):
            raise DataValidationError(f"class entry {position} must be an object")
        phase = _require_text(raw.get("phase"), f"classes[{position}].phase")
        if phase not in PHASE_COUNTS:
            raise DataValidationError(f"classes[{position}].phase is invalid: {phase}")
        records.append(
            ClassRecord(
                category_id=_require_int(
                    raw.get("category_id"), f"classes[{position}].category_id"
                ),
                model_index=_require_int(
                    raw.get("model_index"), f"classes[{position}].model_index"
                ),
                canonical_name=_require_text(
                    raw.get("canonical_name"), f"classes[{position}].canonical_name"
                ),
                folder_name=_require_text(
                    raw.get("folder_name"), f"classes[{position}].folder_name"
                ),
                phase=phase,
            )
        )

    for field in ("category_id", "model_index", "canonical_name", "folder_name"):
        _reject_duplicates(records, field)
    model_indices = sorted(record.model_index for record in records)
    if model_indices != list(range(20)):
        raise DataValidationError("model_index values must be continuous from 0 through 19")

    records.sort(key=lambda record: record.model_index)
    counts = Counter(record.phase for record in records)
    if any(counts[phase] != expected for phase, expected in PHASE_COUNTS.items()):
        raise DataValidationError("class registry must contain 15 base and 5 incremental classes")
    if any(
        record.phase != ("base" if record.model_index < 15 else "incremental")
        for record in records
    ):
        raise DataValidationError(
            "base classes must use model_index 0 through 14 and incremental classes 15 through 19"
        )

    return ClassRegistry(
        version=version,
        classes=tuple(records),
        by_category_id=MappingProxyType(
            {record.category_id: record for record in records}
        ),
        by_model_index=MappingProxyType(
            {record.model_index: record for record in records}
        ),
    )


def validate_class_directories(
    dataset_root: str | Path, registry: ClassRegistry
) -> dict[str, dict[str, int]]:
    root = Path(dataset_root)
    counts: dict[str, dict[str, int]] = {}
    problems: list[str] = []

    for phase in PHASE_COUNTS:
        phase_root = root / phase
        expected = {
            record.folder_name for record in registry.classes if record.phase == phase
        }
        actual = (
            {
                path.name
                for path in phase_root.iterdir()
                if path.is_dir() and path.name.startswith("bread_")
            }
            if phase_root.is_dir()
            else set()
        )
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            problems.append(f"{phase}: missing={missing}, extra={extra}")
        counts[phase] = {
            folder_name: sum(
                1
                for path in (phase_root / folder_name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            for folder_name in sorted(expected)
            if (phase_root / folder_name).is_dir()
        }

    if problems:
        raise DataValidationError(
            "class directories do not match registry: " + "; ".join(problems)
        )
    return counts
