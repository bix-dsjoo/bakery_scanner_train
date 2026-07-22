from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from bakery_scanner.detector_ensemble import (
    DetectorEnsembleConfig,
    DetectorEnsembleMember,
    benchmark_detector_ensemble_cpu,
    evaluate_detector_ensemble,
    load_detector_ensemble_config,
    merge_member_predictions,
)
from bakery_scanner.detector_evaluation import Detection
from bakery_scanner.detector_postprocess import DetectorPostprocessConfig
from bakery_scanner.errors import DataValidationError
from bakery_scanner.yolo_dataset import build_yolo_dataset


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _member_yaml(index: int, *, config_sha: str, checkpoint_sha: str) -> str:
    return f"""\
  - config_path: configs/detector/member_{index}.yaml
    config_sha256: {config_sha}
    checkpoint_path: runs/detector/member_{index}/checkpoints/best.pt
    checkpoint_sha256: {checkpoint_sha}
"""


def _config_yaml(*, members: str) -> str:
    return f"""\
dataset_root: datasets
output_root: runs/detector_ensemble
run_name: ensemble-fixture
members:
{members}cpu_threads: 8
cpu_warmups: 1
cpu_repetitions: 3
"""


def test_load_ensemble_config_accepts_exact_two_member_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ensemble.yaml"
    path.write_text(
        _config_yaml(
            members=_member_yaml(1, config_sha=_SHA_A, checkpoint_sha=_SHA_B)
            + _member_yaml(2, config_sha=_SHA_C, checkpoint_sha=_SHA_D)
        ),
        encoding="utf-8",
    )

    config = load_detector_ensemble_config(path)

    assert config.dataset_root == "datasets"
    assert config.run_name == "ensemble-fixture"
    assert config.cpu_threads == 8
    assert config.cpu_warmups == 1
    assert config.cpu_repetitions == 3
    assert [member.config_sha256 for member in config.members] == [_SHA_A, _SHA_C]


@pytest.mark.parametrize(
    ("members", "message"),
    [
        (_member_yaml(1, config_sha=_SHA_A, checkpoint_sha=_SHA_B), "exactly two"),
        (
            _member_yaml(1, config_sha="ABC", checkpoint_sha=_SHA_B)
            + _member_yaml(2, config_sha=_SHA_C, checkpoint_sha=_SHA_D),
            "SHA-256",
        ),
        (
            _member_yaml(1, config_sha=_SHA_A, checkpoint_sha=_SHA_B)
            + _member_yaml(1, config_sha=_SHA_C, checkpoint_sha=_SHA_D),
            "unique",
        ),
    ],
)
def test_load_ensemble_config_rejects_invalid_members(
    tmp_path: Path, members: str, message: str
) -> None:
    path = tmp_path / "ensemble.yaml"
    path.write_text(_config_yaml(members=members), encoding="utf-8")

    with pytest.raises(DataValidationError, match=message):
        load_detector_ensemble_config(path)


def test_merge_member_predictions_is_deterministic_and_preserves_empty() -> None:
    member_one = {
        "scene_a.jpg": (
            Detection((0.0, 0.0, 10.0, 10.0), 0.8),
            Detection((30.0, 0.0, 40.0, 10.0), 0.7),
        ),
        "empty.jpg": (),
    }
    member_two = {
        "scene_a.jpg": (
            Detection((1.0, 0.0, 11.0, 10.0), 0.9),
            Detection((60.0, 0.0, 70.0, 10.0), 0.6),
        ),
        "empty.jpg": (),
    }
    postprocess = DetectorPostprocessConfig(0.1, 0.5, 2.0)

    first = merge_member_predictions(
        (member_one, member_two), ("scene_a.jpg", "empty.jpg"), postprocess
    )
    second = merge_member_predictions(
        (member_one, member_two), ("scene_a.jpg", "empty.jpg"), postprocess
    )

    assert first == second
    assert first["empty.jpg"] == ()
    assert [item.confidence for item in first["scene_a.jpg"]] == [0.9, 0.7, 0.6]


@pytest.mark.parametrize("failure", ["missing", "extra", "class"])
def test_merge_member_predictions_rejects_contract_mismatch(failure: str) -> None:
    valid = {"scene.jpg": (Detection((0.0, 0.0, 10.0, 10.0), 0.8),)}
    if failure == "missing":
        invalid = {}
    elif failure == "extra":
        invalid = {**valid, "other.jpg": ()}
    else:
        invalid = {"scene.jpg": (Detection((0.0, 0.0, 10.0, 10.0), 0.8, 1),)}

    with pytest.raises(DataValidationError):
        merge_member_predictions(
            (valid, invalid),
            ("scene.jpg",),
            DetectorPostprocessConfig(0.1, 0.5, 2.0),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_selected_config(
    path: Path,
    *,
    dataset_root: Path,
    source_run: str,
    yolo_run: str,
    seed: int,
    operating_confidence: float = 0.2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""\
dataset_root: {dataset_root.as_posix()}
source_detector_run: {source_run}
yolo_run_name: {yolo_run}
output_root: {(path.parent / 'outputs').as_posix()}
run_name: selected-{seed}
model: yolo26s.pt
image_size: 640
epochs: 50
batch_size: 16
seed: {seed}
device: "0"
patience: 10
workers: 0
thresholds:
  confidence_floor: 0.001
  operating_confidence: {operating_confidence}
  nms_iou: 0.15
  matching_iou: 0.5
  max_symmetric_aspect_ratio: 2.0
""",
        encoding="utf-8",
    )


def _truth_detections(yolo_manifest: Path) -> dict[str, tuple[Detection, ...]]:
    payload = json.loads(yolo_manifest.read_text(encoding="utf-8"))
    result = {}
    for sample in payload["samples"]:
        if sample["split"] != "validation":
            continue
        detections = []
        for annotation in sample["original_annotations"]:
            x, y, width, height = annotation["bbox"]
            detections.append(
                Detection((float(x), float(y), float(x + width), float(y + height)), 0.9)
            )
        result[Path(sample["image_path"]).name] = tuple(detections)
    return result


def _evaluation_fixture(
    detector_source_run: tuple[Path, str], tmp_path: Path, *, drift: bool = False
) -> tuple[DetectorEnsembleConfig, dict[str, tuple[Detection, ...]], tuple[Path, Path]]:
    dataset_root, source_run = detector_source_run
    first_yolo = build_yolo_dataset(dataset_root, source_run, "ensemble-yolo-a")
    second_yolo = build_yolo_dataset(dataset_root, source_run, "ensemble-yolo-b")
    configs = (tmp_path / "configs" / "first.yaml", tmp_path / "configs" / "second.yaml")
    _write_selected_config(
        configs[0],
        dataset_root=dataset_root,
        source_run=source_run,
        yolo_run="ensemble-yolo-a",
        seed=42,
    )
    _write_selected_config(
        configs[1],
        dataset_root=dataset_root,
        source_run=source_run,
        yolo_run="ensemble-yolo-b",
        seed=44,
        operating_confidence=0.3 if drift else 0.2,
    )
    checkpoints = (
        tmp_path / "runs" / "first" / "checkpoints" / "best.pt",
        tmp_path / "runs" / "second" / "checkpoints" / "best.pt",
    )
    for index, checkpoint in enumerate(checkpoints):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
    members = tuple(
        DetectorEnsembleMember(
            str(config),
            _sha256(config),
            str(checkpoint),
            _sha256(checkpoint),
        )
        for config, checkpoint in zip(configs, checkpoints, strict=True)
    )
    config = DetectorEnsembleConfig(
        dataset_root=str(dataset_root),
        output_root=str(tmp_path / "ensemble-runs"),
        run_name="ensemble-evaluation",
        members=(members[0], members[1]),
        cpu_threads=8,
        cpu_warmups=1,
        cpu_repetitions=3,
    )
    return config, _truth_detections(first_yolo.manifest_path), checkpoints


class ComplementaryBackend:
    def __init__(
        self, truth: Mapping[str, Sequence[Detection]], *, empty: bool = False
    ) -> None:
        self.truth = truth
        self.empty = empty
        self.predict_calls: list[dict[str, object]] = []

    def cuda_available(self, device: str) -> bool:
        return device == "0"

    def predict(self, **kwargs) -> Mapping[str, Sequence[Detection]]:
        self.predict_calls.append(kwargs)
        member_index = len(self.predict_calls) - 1
        result = {}
        for path in kwargs["image_paths"]:
            detections = () if self.empty else tuple(
                item
                for index, item in enumerate(self.truth[path.name])
                if index % 2 == member_index
            )
            result[path.name] = detections
        return result


def test_evaluate_ensemble_recovers_complementary_candidates_and_preserves_hashes(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, checkpoints = _evaluation_fixture(detector_source_run, tmp_path)
    backend = ComplementaryBackend(truth)
    hashes_before = tuple(_sha256(path) for path in checkpoints)

    report = evaluate_detector_ensemble(config, backend)

    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))["metrics"]
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metrics["global"]["true_positive_count"] == metrics["global"][
        "ground_truth_count"
    ]
    assert metrics["global"]["false_positive_count"] == 0
    assert len(backend.predict_calls) == 2
    assert [item["checkpoint_sha256"] for item in metadata["members"]] == list(
        hashes_before
    )
    assert tuple(_sha256(path) for path in checkpoints) == hashes_before


def test_evaluate_ensemble_accepts_normal_empty_predictions(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, _checkpoints = _evaluation_fixture(detector_source_run, tmp_path)

    report = evaluate_detector_ensemble(config, ComplementaryBackend(truth, empty=True))

    metrics = json.loads(report.metrics_path.read_text(encoding="utf-8"))["metrics"]
    assert metrics["global"]["prediction_count"] == 0
    assert metrics["global"]["true_positive_count"] == 0


def test_evaluate_ensemble_rejects_config_hash_drift(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, _checkpoints = _evaluation_fixture(detector_source_run, tmp_path)
    bad_member = DetectorEnsembleMember(
        config.members[0].config_path,
        "f" * 64,
        config.members[0].checkpoint_path,
        config.members[0].checkpoint_sha256,
    )
    config = DetectorEnsembleConfig(
        config.dataset_root,
        config.output_root,
        config.run_name,
        (bad_member, config.members[1]),
        config.cpu_threads,
        config.cpu_warmups,
        config.cpu_repetitions,
    )

    with pytest.raises(DataValidationError, match="config SHA-256"):
        evaluate_detector_ensemble(config, ComplementaryBackend(truth))


def test_evaluate_ensemble_rejects_member_threshold_drift(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, _checkpoints = _evaluation_fixture(
        detector_source_run, tmp_path, drift=True
    )

    with pytest.raises(DataValidationError, match="inference arguments"):
        evaluate_detector_ensemble(config, ComplementaryBackend(truth))


def test_evaluate_ensemble_rejects_checkpoint_changed_during_inference(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, checkpoints = _evaluation_fixture(detector_source_run, tmp_path)

    class MutatingBackend(ComplementaryBackend):
        def predict(self, **kwargs) -> Mapping[str, Sequence[Detection]]:
            result = super().predict(**kwargs)
            if len(self.predict_calls) == 2:
                checkpoints[0].write_bytes(b"mutated")
            return result

    with pytest.raises(DataValidationError, match="changed during inference"):
        evaluate_detector_ensemble(config, MutatingBackend(truth))

    assert not (Path(config.output_root) / config.run_name).exists()


class RecordingCpuBackend:
    def __init__(self, *, provider: str = "torch-cpu", mutate: Path | None = None) -> None:
        self.provider = provider
        self.mutate = mutate
        self.prepare_calls: list[tuple[tuple[Path, ...], int]] = []
        self.predict_calls: list[dict[str, object]] = []

    def execution_provider(self) -> str:
        return self.provider

    def prepare(self, checkpoints: Sequence[Path], threads: int) -> None:
        self.prepare_calls.append((tuple(checkpoints), threads))

    def predict(self, **kwargs) -> Sequence[Detection]:
        self.predict_calls.append(kwargs)
        if self.mutate is not None and len(self.predict_calls) == 1:
            self.mutate.write_bytes(b"mutated-by-benchmark")
        return ()


def test_cpu_benchmark_records_complete_ensemble_samples_and_excludes_warmup(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, _checkpoints = _evaluation_fixture(detector_source_run, tmp_path)
    evaluate_detector_ensemble(config, ComplementaryBackend(truth))
    backend = RecordingCpuBackend()

    benchmark_path = benchmark_detector_ensemble_cpu(config, backend)

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    image_count = payload["context"]["image_count"]
    assert payload["context"]["device"] == "cpu"
    assert payload["context"]["execution_provider"] == "torch-cpu"
    assert payload["context"]["threads"] == 8
    assert payload["warmup_invocation_count"] == image_count
    assert payload["timing"]["count"] == image_count * 3
    assert len(payload["samples"]) == image_count * 3
    assert len(backend.predict_calls) == image_count * (1 + 3) * 2
    assert all(call["device"] == "cpu" for call in backend.predict_calls)
    assert backend.prepare_calls[0][1] == 8


def test_cpu_benchmark_rejects_non_cpu_provider(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, _checkpoints = _evaluation_fixture(detector_source_run, tmp_path)
    evaluate_detector_ensemble(config, ComplementaryBackend(truth))

    with pytest.raises(DataValidationError, match="CPU provider"):
        benchmark_detector_ensemble_cpu(
            config, RecordingCpuBackend(provider="CUDAExecutionProvider")
        )


def test_cpu_benchmark_rejects_checkpoint_mutation(
    detector_source_run: tuple[Path, str], tmp_path: Path
) -> None:
    config, truth, checkpoints = _evaluation_fixture(detector_source_run, tmp_path)
    evaluate_detector_ensemble(config, ComplementaryBackend(truth))

    with pytest.raises(DataValidationError, match="changed during CPU benchmark"):
        benchmark_detector_ensemble_cpu(
            config, RecordingCpuBackend(mutate=checkpoints[0])
        )

    assert not (Path(config.output_root) / config.run_name / "benchmark.json").exists()


def test_repository_ensemble_configs_are_frozen_and_loadable() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "0503": (
            (
                "f176a807c2e10332af41eb9aeab44229a94d66b31a5374165230ec3b29b395d9",
                "370986e06f6bd9b60bc389937e6f74b986d54f1ef2235c8728817a466dcead92",
            ),
            (
                "62a0a86548e483f92e197d92d52a1bbe395f847ace1c1d4025b704277c03ef2b",
                "3c611b1124e7de9e7bd5ac7141a3e155f4ae45e1e3d1c57ee426ff5dcef9f7f6",
            ),
        ),
        "0509": (
            (
                "9960b18cc607c6dc55dfc0b4cf6722571bbe0b28395fab972ff98d5c85255ccc",
                "b11b56600f6920e9015c31d0fe48af91f7773369cecb3d8c15d3338409b880f6",
            ),
            (
                "9d584b7a0f8011deb130240a0b2787c79e153fe8b0c6a0ad2147efaf7d5d1f75",
                "b97feb4fa0a5df6798ccff90e39f5e79a9478b263b7a2bfcdd9fb533db4acb85",
            ),
        ),
    }
    for fold, hashes in expected.items():
        config = load_detector_ensemble_config(
            root
            / "configs"
            / "detector_ensemble"
            / f"yolo26s_s42_s44_val{fold}.yaml"
        )
        assert [
            (member.config_sha256, member.checkpoint_sha256)
            for member in config.members
        ] == list(hashes)
        assert ["s42" in config.members[0].config_path, "s44" in config.members[1].config_path] == [
            True,
            True,
        ]
        assert (config.cpu_threads, config.cpu_warmups, config.cpu_repetitions) == (
            8,
            1,
            3,
        )
