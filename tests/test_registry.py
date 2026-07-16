import json
from pathlib import Path

import pytest

from bakery_scanner.errors import DataValidationError
from bakery_scanner.registry import load_class_registry, validate_class_directories


def _classes() -> list[dict[str, object]]:
    records = []
    for model_index in range(20):
        phase = "base" if model_index < 15 else "incremental"
        category_id = (model_index * 7) % 20 + 1
        records.append(
            {
                "category_id": category_id,
                "model_index": model_index,
                "canonical_name": f"Bread {category_id}",
                "folder_name": f"bread_{category_id:02d}_item",
                "phase": phase,
            }
        )
    return records


def _write_registry(path: Path, classes: list[dict[str, object]] | None = None) -> Path:
    path.write_text(
        json.dumps({"version": 1, "classes": classes or _classes()}),
        encoding="utf-8",
    )
    return path


def test_load_registry_preserves_category_to_model_mapping(tmp_path: Path) -> None:
    registry = load_class_registry(_write_registry(tmp_path / "registry.json"))

    assert len(registry.classes) == 20
    assert registry.phase_counts == {"base": 15, "incremental": 5}
    assert registry.by_category_id[1].model_index != 1
    assert [record.model_index for record in registry.classes] == list(range(20))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("category_id", 1, "duplicate category_id"),
        ("model_index", 0, "duplicate model_index"),
        ("canonical_name", "Bread 1", "duplicate canonical_name"),
        ("folder_name", "bread_01_item", "duplicate folder_name"),
    ],
)
def test_registry_rejects_duplicate_identifiers(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    classes = _classes()
    classes[1][field] = replacement

    with pytest.raises(DataValidationError, match=message):
        load_class_registry(_write_registry(tmp_path / "registry.json", classes))


def test_registry_rejects_non_continuous_model_indices(tmp_path: Path) -> None:
    classes = _classes()
    classes[-1]["model_index"] = 20

    with pytest.raises(DataValidationError, match="continuous"):
        load_class_registry(_write_registry(tmp_path / "registry.json", classes))


def test_registry_rejects_wrong_phase_counts(tmp_path: Path) -> None:
    classes = _classes()
    classes[14]["phase"] = "incremental"

    with pytest.raises(DataValidationError, match="15 base and 5 incremental"):
        load_class_registry(_write_registry(tmp_path / "registry.json", classes))


def test_registry_rejects_phase_outside_model_index_range(tmp_path: Path) -> None:
    classes = _classes()
    classes[0]["phase"] = "incremental"
    classes[15]["phase"] = "base"

    with pytest.raises(DataValidationError, match="model_index 0 through 14"):
        load_class_registry(_write_registry(tmp_path / "registry.json", classes))


def test_validate_class_directories_reports_image_counts(tmp_path: Path) -> None:
    registry = load_class_registry(_write_registry(tmp_path / "registry.json"))
    for record in registry.classes:
        class_dir = tmp_path / record.phase / record.folder_name
        class_dir.mkdir(parents=True)
        (class_dir / "sample.jpg").write_bytes(b"image bytes are not decoded here")

    counts = validate_class_directories(tmp_path, registry)

    assert counts["base"][registry.classes[0].folder_name] == 1
    assert sum(counts["incremental"].values()) == 5


def test_validate_class_directories_rejects_missing_and_unregistered_folders(
    tmp_path: Path,
) -> None:
    registry = load_class_registry(_write_registry(tmp_path / "registry.json"))
    for record in registry.classes[1:]:
        (tmp_path / record.phase / record.folder_name).mkdir(parents=True)
    (tmp_path / "base" / "bread_99_unknown").mkdir(parents=True)

    with pytest.raises(DataValidationError, match="class directories do not match registry"):
        validate_class_directories(tmp_path, registry)
