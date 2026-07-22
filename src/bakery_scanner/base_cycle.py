from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from .coco import validate_coco
from .errors import DataValidationError
from .registry import load_class_registry
from .splits import SCENE_PATTERN

CYCLE_VERSION = "1.0.0"
MANIFEST_VERSION = 1
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONFIG_FIELDS = {
    "dataset_root",
    "output_root",
    "run_name",
    "real_coco_path",
    "development_scene_ids",
    "holdout_scene_id",
    "development_backgrounds",
    "holdout_background",
    "seeds",
}


@dataclass(frozen=True, slots=True)
class BaseCycleConfig:
    dataset_root: str
    output_root: str
    run_name: str
    real_coco_path: str
    development_scene_ids: tuple[str, str]
    holdout_scene_id: str
    development_backgrounds: tuple[str, str]
    holdout_background: str
    seeds: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BaseCycleReport:
    output_dir: Path
    manifest_path: Path
    development_scene_ids: tuple[str, str]
    holdout_scene_id: str
    development_image_count: int
    holdout_image_count: int
    development_background_count: int
    holdout_background_count: int
    seeds: tuple[int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "development_scene_ids": list(self.development_scene_ids),
            "holdout_scene_id": self.holdout_scene_id,
            "development_image_count": self.development_image_count,
            "holdout_image_count": self.holdout_image_count,
            "development_background_count": self.development_background_count,
            "holdout_background_count": self.holdout_background_count,
            "seeds": list(self.seeds),
        }


def _strict_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DataValidationError(f"{label} must be an object")
    actual = set(value)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing:
        raise DataValidationError(f"{label} has missing fields: {missing}")
    if unknown:
        raise DataValidationError(f"{label} has unknown fields: {unknown}")
    return value  # type: ignore[return-value]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _text_tuple(value: object, length: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != length:
        number = "two" if length == 2 else str(length)
        raise DataValidationError(f"{label} must contain exactly {number} values")
    parsed = tuple(_text(item, label) for item in value)
    if len(set(parsed)) != len(parsed):
        raise DataValidationError(f"{label} must be unique")
    return parsed


def _seed_tuple(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise DataValidationError("seeds must contain exactly three values")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in value):
        raise DataValidationError("seeds must be non-negative integers")
    if len(set(value)) != 3:
        raise DataValidationError("seeds must be unique")
    return value[0], value[1], value[2]


def _normalized_configured_path(value: object, label: str) -> str:
    return posixpath.normpath(_text(value, label).replace("\\", "/"))


def _normalized_path_tuple(
    value: object, length: int, label: str
) -> tuple[str, ...]:
    parsed = _text_tuple(value, length, label)
    normalized = tuple(_normalized_configured_path(item, label) for item in parsed)
    if len(set(normalized)) != len(normalized):
        raise DataValidationError(f"{label} must be unique after path normalization")
    return normalized


def _normalized_contract_root(value: str, label: str) -> str:
    slash_value = value.replace("\\", "/")
    required = "derived/base_cycle" if label == "output_root" else "datasets"
    if slash_value.startswith("/") or PureWindowsPath(value).is_absolute():
        raise DataValidationError(f"{label} must be exactly {required}")
    normalized = posixpath.normpath(slash_value)
    if normalized == ".." or normalized.startswith("../"):
        raise DataValidationError(f"{label} must be exactly {required}")
    return normalized


def _lexical_parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    normalized = os.fspath(value).replace("\\", "/").casefold()
    return tuple(part for part in normalized.split("/") if part not in ("", "."))


def _reject_evaluation_path(value: str | os.PathLike[str]) -> None:
    parts = _lexical_parts(value)
    for forbidden in (("base", "test"), ("incremental", "test")):
        width = len(forbidden)
        if any(parts[index : index + width] == forbidden for index in range(len(parts) - width + 1)):
            raise DataValidationError(f"evaluation-only path is forbidden: {value}")


def _is_link_or_junction(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DataValidationError(f"cannot inspect path {path}: {exc}") from exc
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0) & 0x400
    )


def _assert_existing_components_are_physical(path: Path) -> None:
    components = list(reversed(path.parents)) + [path]
    for component in components:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise DataValidationError(f"cannot inspect path {component}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & 0x400
        ):
            raise DataValidationError(f"path must not contain a symlink or junction: {path}")


def _resolve_config_context_safely(
    config_path: str | Path,
) -> tuple[Path, Path]:
    _reject_evaluation_path(config_path)
    lexical = Path(os.path.abspath(os.fspath(config_path)))
    _assert_existing_components_are_physical(lexical)
    if not lexical.exists() or not lexical.is_file():
        raise DataValidationError(f"base cycle config is missing: {lexical}")

    repository_root: Path | None = None
    for candidate in lexical.parents:
        pyproject = candidate / "pyproject.toml"
        agents = candidate / "AGENTS.md"
        if pyproject.exists() and agents.exists():
            if _is_link_or_junction(pyproject) or _is_link_or_junction(agents):
                raise DataValidationError("repository sentinels must not be links or junctions")
            if not pyproject.is_file() or not agents.is_file():
                continue
            repository_root = candidate
            break
    if repository_root is None:
        raise DataValidationError("config repository must contain pyproject.toml and AGENTS.md")

    config_root = repository_root / "configs" / "base_cycle"
    try:
        lexical.relative_to(config_root)
    except ValueError as exc:
        raise DataValidationError("base cycle config must be below configs/base_cycle") from exc

    resolved = lexical.resolve(strict=True)
    resolved_repository = repository_root.resolve(strict=True)
    try:
        resolved.relative_to(resolved_repository)
    except ValueError as exc:
        raise DataValidationError("base cycle config escaped its repository") from exc
    return resolved, resolved_repository


def load_base_cycle_config(path: str | Path) -> BaseCycleConfig:
    source, _ = _resolve_config_context_safely(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot load base cycle config {source}: {exc}") from exc
    payload = _strict_object(raw, _CONFIG_FIELDS, "base cycle config")

    dataset_root = _normalized_contract_root(
        _text(payload["dataset_root"], "dataset_root"), "dataset_root"
    )
    if dataset_root != "datasets":
        raise DataValidationError("dataset_root must be exactly datasets")
    output_root = _normalized_contract_root(
        _text(payload["output_root"], "output_root"), "output_root"
    )
    if output_root != "derived/base_cycle":
        raise DataValidationError("output_root must be exactly derived/base_cycle")

    run_name = _text(payload["run_name"], "run_name")
    if _RUN_NAME.fullmatch(run_name) is None:
        raise DataValidationError("run_name is invalid")
    development_scene_ids = _text_tuple(
        payload["development_scene_ids"], 2, "two development scene IDs"
    )
    holdout_scene_id = _text(payload["holdout_scene_id"], "holdout scene ID")
    if holdout_scene_id in development_scene_ids:
        raise DataValidationError("development and holdout scene IDs must not overlap")
    development_backgrounds = _normalized_path_tuple(
        payload["development_backgrounds"], 2, "development backgrounds"
    )
    holdout_background = _normalized_configured_path(
        payload["holdout_background"], "holdout background"
    )
    if holdout_background in development_backgrounds:
        raise DataValidationError("development and holdout backgrounds must not overlap")

    return BaseCycleConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        run_name=run_name,
        real_coco_path=_normalized_configured_path(
            payload["real_coco_path"], "real_coco_path"
        ),
        development_scene_ids=(development_scene_ids[0], development_scene_ids[1]),
        holdout_scene_id=holdout_scene_id,
        development_backgrounds=(
            development_backgrounds[0],
            development_backgrounds[1],
        ),
        holdout_background=holdout_background,
        seeds=_seed_tuple(payload["seeds"]),
    )


def _semantic_config(config: BaseCycleConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["development_scene_ids"] = list(config.development_scene_ids)
    payload["development_backgrounds"] = list(config.development_backgrounds)
    payload["seeds"] = list(config.seeds)
    return payload


def _config_sha256(config: BaseCycleConfig) -> str:
    serialized = json.dumps(
        _semantic_config(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _assignment_lock(config: BaseCycleConfig) -> dict[str, object]:
    return {
        "lock_version": 1,
        "run_name": config.run_name,
        "config": _semantic_config(config),
        "config_sha256": _config_sha256(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "integrity_pending",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise DataValidationError(f"cannot write Base cycle artifact {path}: {exc}") from exc


def _resolve_cycle_output_root_safely(root: Path, *, create: bool) -> Path:
    candidate = root / "derived" / "base_cycle"
    _assert_existing_components_are_physical(candidate)
    if create:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DataValidationError(
                f"cannot create Base cycle output root {candidate}: {exc}"
            ) from exc
        _assert_existing_components_are_physical(candidate)
    if not candidate.exists() or not candidate.is_dir():
        raise DataValidationError(f"Base cycle output root is missing: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DataValidationError("Base cycle output root escaped dataset root") from exc
    return resolved


def _publish_assignment_lock(
    config_path: str | Path, config: BaseCycleConfig
) -> Path:
    source, repository_root = _resolve_config_context_safely(config_path)
    if load_base_cycle_config(source) != config:
        raise DataValidationError("base cycle config changed before lock publication")
    root = _resolve_dataset_root_safely(config.dataset_root, repository_root)
    output_root = _resolve_cycle_output_root_safely(root, create=True)
    output_dir = output_root / config.run_name
    if output_dir.parent != output_root:
        raise DataValidationError("run_name must select a direct cycle directory")
    if os.path.lexists(output_dir):
        raise DataValidationError(f"base cycle run already exists: {output_dir}")

    staging = output_root / f".{config.run_name}.lock-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        _write_json(staging / "assignment.lock.json", _assignment_lock(config))
        staging.rename(output_dir)
    except Exception:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output_dir / "assignment.lock.json"


def _validate_config_paths_lexically(config: BaseCycleConfig) -> None:
    _assert_cycle_paths_safe_lexically(
        (
            config.real_coco_path,
            *config.development_backgrounds,
            config.holdout_background,
        )
    )


def _assert_cycle_paths_safe_lexically(
    values: tuple[str, ...] | list[str],
) -> None:
    for value in values:
        _reject_evaluation_path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataValidationError(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def _portable(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DataValidationError(f"input is outside dataset root: {path}") from exc


def _resolve_repository_root_safely(repository_root: str | Path) -> Path:
    _reject_evaluation_path(repository_root)
    lexical = Path(os.path.abspath(os.fspath(repository_root)))
    _assert_existing_components_are_physical(lexical)
    if not lexical.exists() or not lexical.is_dir():
        raise DataValidationError(f"repository root is missing: {lexical}")
    for sentinel_name in ("pyproject.toml", "AGENTS.md"):
        sentinel = lexical / sentinel_name
        if not sentinel.exists() or not sentinel.is_file():
            raise DataValidationError(
                f"repository root must contain {sentinel_name}: {lexical}"
            )
        if _is_link_or_junction(sentinel):
            raise DataValidationError(
                f"repository sentinel must not be a link or junction: {sentinel}"
            )
    return lexical.resolve(strict=True)


def _resolve_dataset_root_safely(value: str, repository_root: Path) -> Path:
    if value != "datasets":
        raise DataValidationError("dataset_root must be exactly datasets")
    candidate = repository_root / value
    _assert_existing_components_are_physical(candidate)
    if not candidate.exists() or not candidate.is_dir():
        raise DataValidationError(f"dataset root is missing: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise DataValidationError(f"dataset root escaped repository: {candidate}") from exc
    return resolved


def _resolve_configured_input(value: str, root: Path) -> Path:
    _reject_evaluation_path(value)
    candidate_value = Path(value)
    candidate = candidate_value if candidate_value.is_absolute() else root / candidate_value
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    _assert_existing_components_are_physical(lexical)
    if not lexical.exists() or not lexical.is_file():
        raise DataValidationError(f"configured input is missing: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DataValidationError(
            f"configured input must remain inside dataset root: {value}"
        ) from exc
    return resolved


def _decode_size(path: Path, label: str) -> tuple[int, int]:
    try:
        with Image.open(path) as decoded:
            converted = decoded.convert("RGB")
            converted.load()
            width, height = converted.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise DataValidationError(f"cannot decode {label} {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise DataValidationError(f"{label} must have positive dimensions: {path}")
    return width, height


def _load_and_screen_coco_filenames(coco_path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load COCO annotation {coco_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("COCO root must be an object")
    images = payload.get("images")
    if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
        raise DataValidationError("COCO images must be a list of objects")

    filenames: list[str] = []
    for index, item in enumerate(images):
        file_name = item.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise DataValidationError(
                f"COCO images[{index}].file_name must be a non-empty string"
            )
        _reject_evaluation_path(file_name)
        if (
            file_name in {".", ".."}
            or "/" in file_name
            or "\\" in file_name
            or Path(file_name).is_absolute()
            or PureWindowsPath(file_name).is_absolute()
            or Path(file_name).name != file_name
        ):
            raise DataValidationError(
                f"COCO image file_name must be a plain basename: {file_name}"
            )
        filenames.append(file_name)
    if len(filenames) != len(set(filenames)):
        raise DataValidationError("COCO image file_name values must be unique")
    _assert_cycle_paths_safe_lexically(filenames)
    return tuple(filenames)


def _scene_inventory(
    coco_path: Path,
    development_ids: tuple[str, str],
    holdout_id: str,
    root: Path,
    *,
    filenames: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    screened = (
        _load_and_screen_coco_filenames(coco_path)
        if filenames is None
        else filenames
    )
    declared_ids = {*development_ids, holdout_id}
    grouped: dict[str, set[str]] = {}
    parsed: list[tuple[str, str, str]] = []
    for file_name in screened:
        match = SCENE_PATTERN.fullmatch(file_name)
        if match is None:
            raise DataValidationError(
                f"COCO scene filename is invalid: {file_name}"
            )
        scene_id = match.group("scene_id")
        difficulty = match.group("difficulty")
        difficulties = grouped.setdefault(scene_id, set())
        if difficulty in difficulties:
            raise DataValidationError(
                f"scene {scene_id} must contain each e/m/h difficulty exactly once"
            )
        difficulties.add(difficulty)
        parsed.append((scene_id, difficulty, file_name))
    if set(grouped) != declared_ids:
        raise DataValidationError(
            "declared scene IDs must exactly partition all COCO scene groups"
        )
    if any(difficulties != {"e", "m", "h"} for difficulties in grouped.values()):
        raise DataValidationError("each declared scene group must contain exactly e/m/h")

    records: list[dict[str, object]] = []
    relative_parent = coco_path.parent.relative_to(root)
    for scene_id, difficulty, file_name in sorted(parsed, key=lambda item: item[2]):
        image_path = _resolve_configured_input(
            (relative_parent / file_name).as_posix(), root
        )
        width, height = _decode_size(image_path, "scene image")
        records.append(
            {
                "scene_id": scene_id,
                "difficulty": difficulty,
                "split": (
                    "development"
                    if scene_id in development_ids
                    else "cycle_holdout"
                ),
                "path": _portable(image_path, root),
                "sha256": _sha256(image_path),
                "width": width,
                "height": height,
            }
        )
    return records


def _background_inventory(
    development: tuple[str, str],
    holdout: str,
    root: Path,
) -> list[dict[str, object]]:
    assignments = [
        *((value, "development") for value in development),
        (holdout, "cycle_holdout"),
    ]
    records: list[dict[str, object]] = []
    for value, split in assignments:
        path = _resolve_configured_input(value, root)
        _decode_size(path, "background")
        records.append(
            {
                "split": split,
                "path": _portable(path, root),
                "sha256": _sha256(path),
            }
        )
    return sorted(records, key=lambda record: str(record["path"]))


def _validate_utc_timestamp(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise DataValidationError(f"{label} must be a UTC ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DataValidationError(f"{label} must be a UTC ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DataValidationError(f"{label} must be a UTC ISO-8601 timestamp")


def _validate_assignment_lock(
    lock_path: str | Path,
    config: BaseCycleConfig,
    config_path: str | Path,
) -> None:
    lexical = Path(os.path.abspath(os.fspath(lock_path)))
    _reject_evaluation_path(lexical)
    raw_config = Path(os.path.abspath(os.fspath(config_path)))
    config_root = next(
        (
            parent
            for parent in raw_config.parents
            if parent.name == "base_cycle" and parent.parent.name == "configs"
        ),
        None,
    )
    if config_root is None:
        raise DataValidationError("base cycle config must be below configs/base_cycle")
    repository = config_root.parent.parent
    expected = (
        repository
        / config.dataset_root
        / config.output_root
        / config.run_name
        / "assignment.lock.json"
    )
    if os.path.normcase(os.path.normpath(lexical)) != os.path.normcase(
        os.path.normpath(expected)
    ):
        raise DataValidationError(
            "assignment lock must be the physical Base cycle direct child"
        )
    _assert_existing_components_are_physical(lexical)
    if not lexical.exists() or not lexical.is_file():
        raise DataValidationError(f"assignment lock is missing: {lexical}")
    if lexical.name != "assignment.lock.json" or lexical.parent.name != config.run_name:
        raise DataValidationError("assignment lock path does not match run_name")
    try:
        raw = json.loads(lexical.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load assignment lock: {exc}") from exc
    payload = _strict_object(
        raw,
        {
            "lock_version",
            "run_name",
            "config",
            "config_sha256",
            "created_at",
            "state",
        },
        "assignment lock",
    )
    _validate_utc_timestamp(payload["created_at"], "assignment lock created_at")
    expected = _semantic_config(config)
    if (
        type(payload["lock_version"]) is not int
        or payload["lock_version"] != 1
        or payload["run_name"] != config.run_name
        or payload["config"] != expected
        or payload["config_sha256"] != _config_sha256(config)
        or payload["state"] != "integrity_pending"
    ):
        raise DataValidationError("assignment lock does not match the Base cycle config")


def _prepare_inventory(
    config_path: str | Path,
    config: BaseCycleConfig,
    lock_path: str | Path,
) -> tuple[Path, dict[str, object]]:
    _validate_assignment_lock(lock_path, config, config_path)
    source, repository_root = _resolve_config_context_safely(config_path)
    current_config = load_base_cycle_config(source)
    if current_config != config:
        raise DataValidationError("base cycle config changed after assignment lock")
    root = _resolve_dataset_root_safely(config.dataset_root, repository_root)

    _validate_config_paths_lexically(config)
    coco_path = _resolve_configured_input(config.real_coco_path, root)
    filenames = _load_and_screen_coco_filenames(coco_path)

    registry_path = _resolve_configured_input("class_registry.json", root)
    registry = load_class_registry(registry_path)
    base_indices = tuple(
        record.model_index for record in registry.classes if record.phase == "base"
    )
    if base_indices != tuple(range(15)):
        raise DataValidationError("Base model_index values must be exactly 0 through 14")
    validate_coco(coco_path, registry, "base")

    scenes = _scene_inventory(
        coco_path,
        config.development_scene_ids,
        config.holdout_scene_id,
        root,
        filenames=filenames,
    )
    backgrounds = _background_inventory(
        config.development_backgrounds,
        config.holdout_background,
        root,
    )
    inventory: dict[str, object] = {
        "config": _semantic_config(config),
        "config_sha256": _config_sha256(config),
        "registry": {
            "path": _portable(registry_path, root),
            "sha256": _sha256(registry_path),
        },
        "real_coco": {
            "path": _portable(coco_path, root),
            "sha256": _sha256(coco_path),
        },
        "scenes": scenes,
        "backgrounds": backgrounds,
        "seeds": list(config.seeds),
    }
    return root, inventory


_LOCK_FIELDS = {
    "lock_version",
    "run_name",
    "config",
    "config_sha256",
    "created_at",
    "state",
}
_MANIFEST_FIELDS = {
    "manifest_version",
    "cycle_version",
    "created_at",
    "run_name",
    "config",
    "config_sha256",
    "registry",
    "real_coco",
    "scenes",
    "backgrounds",
    "seeds",
}
_SHA_RECORD_FIELDS = {"path", "sha256"}
_SCENE_FIELDS = {
    "scene_id",
    "difficulty",
    "split",
    "path",
    "sha256",
    "width",
    "height",
}
_BACKGROUND_FIELDS = {"split", "path", "sha256"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _config_from_semantic(value: object) -> BaseCycleConfig:
    payload = _strict_object(value, _CONFIG_FIELDS, "config")
    dataset_root = _text(payload["dataset_root"], "config dataset_root")
    output_root = _text(payload["output_root"], "config output_root")
    if dataset_root != "datasets":
        raise DataValidationError("config dataset_root must be exactly datasets")
    if output_root != "derived/base_cycle":
        raise DataValidationError(
            "config output_root must be exactly derived/base_cycle"
        )
    run_name = _text(payload["run_name"], "config run_name")
    if _RUN_NAME.fullmatch(run_name) is None:
        raise DataValidationError("config run_name is invalid")
    development_ids = _text_tuple(
        payload["development_scene_ids"], 2, "config development scene IDs"
    )
    holdout_id = _text(payload["holdout_scene_id"], "config holdout scene ID")
    if holdout_id in development_ids:
        raise DataValidationError("config scene IDs overlap")
    development_backgrounds = _normalized_path_tuple(
        payload["development_backgrounds"], 2, "config development backgrounds"
    )
    holdout_background = _normalized_configured_path(
        payload["holdout_background"], "config holdout background"
    )
    if holdout_background in development_backgrounds:
        raise DataValidationError("config backgrounds overlap")
    config = BaseCycleConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        run_name=run_name,
        real_coco_path=_normalized_configured_path(
            payload["real_coco_path"], "config real_coco_path"
        ),
        development_scene_ids=(development_ids[0], development_ids[1]),
        holdout_scene_id=holdout_id,
        development_backgrounds=(
            development_backgrounds[0],
            development_backgrounds[1],
        ),
        holdout_background=holdout_background,
        seeds=_seed_tuple(payload["seeds"]),
    )
    if _semantic_config(config) != payload:
        raise DataValidationError("config must use normalized semantic paths")
    return config


def _load_json_object(path: Path, run_dir: Path, label: str) -> dict[str, object]:
    _assert_existing_components_are_physical(path)
    if path.parent != run_dir or not path.exists() or not path.is_file():
        raise DataValidationError(f"output artifact {label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load output artifact {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"output artifact {label} must be an object")
    return payload


def _validate_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DataValidationError(f"{label} SHA-256 must be 64 lowercase hex characters")
    return value


def _validate_sha_record(
    value: object, label: str
) -> tuple[str, str]:
    record = _strict_object(value, _SHA_RECORD_FIELDS, f"{label} record")
    path = _text(record["path"], f"{label} path")
    return path, _validate_sha(record["sha256"], label)


def _validated_source(path_value: str, root: Path, label: str) -> Path:
    normalized = path_value.replace("\\", "/")
    if (
        normalized.startswith("/")
        or PureWindowsPath(path_value).is_absolute()
        or posixpath.normpath(normalized) != normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise DataValidationError(f"{label} path must be portable and normalized")
    _reject_evaluation_path(normalized)
    return _resolve_configured_input(normalized, root)


def _validate_lock_payload(
    value: object, config: BaseCycleConfig, run_name: str
) -> dict[str, object]:
    payload = _strict_object(value, _LOCK_FIELDS, "assignment lock")
    _validate_utc_timestamp(payload["created_at"], "assignment lock created_at")
    if type(payload["lock_version"]) is not int or payload["lock_version"] != 1:
        raise DataValidationError("assignment lock lock_version is invalid")
    if payload["run_name"] != run_name:
        raise DataValidationError("assignment lock run_name mismatch")
    if payload["state"] != "integrity_pending":
        raise DataValidationError("assignment lock state is invalid")
    if payload["config"] != _semantic_config(config):
        raise DataValidationError("assignment lock config mismatch")
    if payload["config_sha256"] != _config_sha256(config):
        raise DataValidationError("assignment lock config_sha256 mismatch")
    return payload


def _validate_manifest_payload(
    root: Path,
    output_dir: Path,
    value: object,
    *,
    lock_payload: dict[str, object] | None = None,
) -> BaseCycleReport:
    payload = _strict_object(value, _MANIFEST_FIELDS, "manifest")
    if type(payload["manifest_version"]) is not int or payload["manifest_version"] != 1:
        raise DataValidationError("manifest_version must be 1")
    if payload["cycle_version"] != CYCLE_VERSION:
        raise DataValidationError(f"cycle_version must be {CYCLE_VERSION}")
    _validate_utc_timestamp(payload["created_at"], "manifest created_at")
    run_name = _text(payload["run_name"], "manifest run_name")
    if run_name != output_dir.name:
        raise DataValidationError("manifest run_name mismatch")
    config = _config_from_semantic(payload["config"])
    if config.run_name != run_name:
        raise DataValidationError("config run_name mismatch")
    expected_config_hash = _config_sha256(config)
    if payload["config_sha256"] != expected_config_hash:
        raise DataValidationError("manifest config_sha256 mismatch")
    if lock_payload is not None:
        _validate_lock_payload(lock_payload, config, run_name)
        if lock_payload["config"] != payload["config"]:
            raise DataValidationError("assignment lock config mismatch")

    registry_path_value, registry_sha = _validate_sha_record(
        payload["registry"], "registry"
    )
    coco_path_value, coco_sha = _validate_sha_record(payload["real_coco"], "real_coco")
    if registry_path_value != "class_registry.json":
        raise DataValidationError("registry assignment must use class_registry.json")
    if coco_path_value != config.real_coco_path:
        raise DataValidationError("real_coco assignment mismatch")

    raw_scenes = payload["scenes"]
    raw_backgrounds = payload["backgrounds"]
    if not isinstance(raw_scenes, list):
        raise DataValidationError("manifest scenes must be a list")
    if not isinstance(raw_backgrounds, list):
        raise DataValidationError("manifest backgrounds must be a list")
    if len(raw_scenes) != 9:
        raise DataValidationError("manifest scene count must be 9")
    if len(raw_backgrounds) != 3:
        raise DataValidationError("manifest background count must be 3")

    registry_path = _validated_source(registry_path_value, root, "registry")
    coco_path = _validated_source(coco_path_value, root, "real_coco")
    if _sha256(registry_path) != registry_sha:
        raise DataValidationError(f"registry SHA-256 mismatch: {registry_path}")
    if _sha256(coco_path) != coco_sha:
        raise DataValidationError(f"real_coco SHA-256 mismatch: {coco_path}")
    screened_filenames = _load_and_screen_coco_filenames(coco_path)
    source_records: list[tuple[Path, str, str]] = []

    expected_scene_assignments: dict[str, tuple[str, str, str]] = {}
    coco_parent = posixpath.dirname(config.real_coco_path)
    declared_ids = {*config.development_scene_ids, config.holdout_scene_id}
    declared_difficulties: dict[str, set[str]] = {}
    for file_name in screened_filenames:
        match = SCENE_PATTERN.fullmatch(file_name)
        if match is None:
            raise DataValidationError(f"COCO scene filename is invalid: {file_name}")
        scene_id = match.group("scene_id")
        difficulty = match.group("difficulty")
        if scene_id not in declared_ids:
            raise DataValidationError("scene assignment mismatch")
        difficulties = declared_difficulties.setdefault(scene_id, set())
        if difficulty in difficulties:
            raise DataValidationError("scene difficulty assignment is duplicated")
        difficulties.add(difficulty)
        split = (
            "development"
            if scene_id in config.development_scene_ids
            else "cycle_holdout"
        )
        expected_scene_assignments[posixpath.join(coco_parent, file_name)] = (
            scene_id,
            difficulty,
            split,
        )
    if set(declared_difficulties) != declared_ids or any(
        values != {"e", "m", "h"} for values in declared_difficulties.values()
    ):
        raise DataValidationError("scene assignment must contain exactly e/m/h")

    seen_paths: set[str] = set()
    scene_records: list[dict[str, object]] = []
    for raw_scene in raw_scenes:
        scene = _strict_object(raw_scene, _SCENE_FIELDS, "scene")
        path_value = _text(scene["path"], "scene path")
        if path_value in seen_paths:
            raise DataValidationError("manifest contains duplicate path")
        seen_paths.add(path_value)
        if path_value not in expected_scene_assignments:
            raise DataValidationError("scene assignment mismatch")
        scene_id, difficulty, split = expected_scene_assignments[path_value]
        if (
            scene["scene_id"] != scene_id
            or scene["difficulty"] != difficulty
            or scene["split"] != split
        ):
            raise DataValidationError("scene assignment mismatch")
        width = scene["width"]
        height = scene["height"]
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise DataValidationError("scene dimensions must be positive integers")
        sha = _validate_sha(scene["sha256"], "scene")
        source = _validated_source(path_value, root, "scene")
        source_records.append((source, sha, "scene"))
        scene_records.append(scene)
    if set(seen_paths) != set(expected_scene_assignments):
        raise DataValidationError("scene assignment count mismatch")

    expected_backgrounds = {
        **{path: "development" for path in config.development_backgrounds},
        config.holdout_background: "cycle_holdout",
    }
    background_records: list[dict[str, object]] = []
    for raw_background in raw_backgrounds:
        background = _strict_object(
            raw_background, _BACKGROUND_FIELDS, "background"
        )
        path_value = _text(background["path"], "background path")
        if path_value in seen_paths:
            raise DataValidationError("manifest contains duplicate path")
        seen_paths.add(path_value)
        if (
            path_value not in expected_backgrounds
            or background["split"] != expected_backgrounds[path_value]
        ):
            raise DataValidationError("background assignment mismatch")
        sha = _validate_sha(background["sha256"], "background")
        source = _validated_source(path_value, root, "background")
        source_records.append((source, sha, "background"))
        background_records.append(background)
    if {str(record["path"]) for record in background_records} != set(expected_backgrounds):
        raise DataValidationError("background assignment count mismatch")

    seeds = payload["seeds"]
    if seeds != list(config.seeds) or seeds != [42, 43, 44]:
        raise DataValidationError("manifest seeds mismatch")

    for source, expected_sha, label in source_records:
        if _sha256(source) != expected_sha:
            raise DataValidationError(f"{label} SHA-256 mismatch: {source}")

    registry = load_class_registry(registry_path)
    _load_and_screen_coco_filenames(coco_path)
    validate_coco(coco_path, registry, "base")
    for scene in scene_records:
        source = _validated_source(str(scene["path"]), root, "scene")
        if _decode_size(source, "scene image") != (scene["width"], scene["height"]):
            raise DataValidationError("scene dimensions mismatch")
    for background in background_records:
        _decode_size(
            _validated_source(str(background["path"]), root, "background"),
            "background",
        )

    development_count = sum(
        1 for scene in scene_records if scene["split"] == "development"
    )
    holdout_count = sum(
        1 for scene in scene_records if scene["split"] == "cycle_holdout"
    )
    development_background_count = sum(
        1 for record in background_records if record["split"] == "development"
    )
    holdout_background_count = sum(
        1 for record in background_records if record["split"] == "cycle_holdout"
    )
    if (development_count, holdout_count) != (6, 3):
        raise DataValidationError("scene split count mismatch")
    if (development_background_count, holdout_background_count) != (2, 1):
        raise DataValidationError("background split count mismatch")
    return BaseCycleReport(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        development_scene_ids=config.development_scene_ids,
        holdout_scene_id=config.holdout_scene_id,
        development_image_count=development_count,
        holdout_image_count=holdout_count,
        development_background_count=development_background_count,
        holdout_background_count=holdout_background_count,
        seeds=config.seeds,
    )


def _resolve_run_dir_safely(output_root: Path, run_name: str) -> Path:
    candidate = output_root / run_name
    if candidate.parent != output_root:
        raise DataValidationError("run_name must select a direct cycle directory")
    _assert_existing_components_are_physical(candidate)
    if not candidate.exists() or not candidate.is_dir():
        raise DataValidationError(f"Base cycle run is missing: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != output_root:
        raise DataValidationError("Base cycle run escaped output root")
    return resolved


def _validate_run_dir(root: Path, output_dir: Path) -> BaseCycleReport:
    lock = _load_json_object(
        output_dir / "assignment.lock.json", output_dir, "assignment.lock.json"
    )
    manifest = _load_json_object(
        output_dir / "manifest.json", output_dir, "manifest.json"
    )
    return _validate_manifest_payload(
        root, output_dir, manifest, lock_payload=lock
    )


def freeze_base_cycle(config_path: str | Path) -> BaseCycleReport:
    source, repository_root = _resolve_config_context_safely(config_path)
    config = load_base_cycle_config(source)
    root = _resolve_dataset_root_safely(config.dataset_root, repository_root)
    lock_path = _publish_assignment_lock(source, config)
    output_dir = lock_path.parent
    manifest_tmp: Path | None = None
    try:
        _, inventory = _prepare_inventory(source, config, lock_path)
        payload: dict[str, object] = {
            "manifest_version": MANIFEST_VERSION,
            "cycle_version": CYCLE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": config.run_name,
            **inventory,
        }
        lock_payload = _load_json_object(
            lock_path, output_dir, "assignment.lock.json"
        )
        _validate_manifest_payload(
            root, output_dir, payload, lock_payload=lock_payload
        )
        manifest_tmp = output_dir / f"manifest.json.tmp-{uuid.uuid4().hex}"
        _write_json(manifest_tmp, payload)
        os.replace(manifest_tmp, output_dir / "manifest.json")
        return _validate_run_dir(root, output_dir)
    except Exception:
        if manifest_tmp is not None:
            manifest_tmp.unlink(missing_ok=True)
        raise


def validate_base_cycle(
    repository_root: str | Path, run_name: str
) -> BaseCycleReport:
    repository = _resolve_repository_root_safely(repository_root)
    if _RUN_NAME.fullmatch(run_name) is None:
        raise DataValidationError("run_name is invalid")
    root = _resolve_dataset_root_safely("datasets", repository)
    output_root = _resolve_cycle_output_root_safely(root, create=False)
    output_dir = _resolve_run_dir_safely(output_root, run_name)
    return _validate_run_dir(root, output_dir)
