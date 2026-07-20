from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bakery_scanner.cpu_benchmark import (
    BENCHMARK_STAGES,
    CpuBackendResult,
    CpuBenchmarkConfig,
    PreparedCpuBenchmark,
    TorchCpuBenchmarkBackend,
    _percentile,
    _timing_statistics,
    _validate_backend_result,
    load_cpu_benchmark_config,
    run_cpu_benchmark,
)
from bakery_scanner.errors import DataValidationError


def _payload() -> dict[str, object]:
    return {
        "dataset_root": "datasets",
        "detector_config": "configs/detector/yolo11n_base.yaml",
        "classifier_config": "configs/classifier/resnet18_incremental.yaml",
        "detector_checkpoint": (
            "runs/detector/yolo11n_base_seed42/checkpoints/best.pt"
        ),
        "classifier_checkpoint": (
            "runs/classifier/resnet18_incremental_seed42/checkpoints/best.pt"
        ),
        "output_root": "runs/benchmark",
        "run_name": "incremental_resnet18_cpu",
        "warmup_iterations": 5,
        "repetitions": 30,
        "intra_op_threads": 4,
        "inter_op_threads": 1,
    }


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_load_cpu_benchmark_config_is_strict(tmp_path: Path) -> None:
    config = load_cpu_benchmark_config(_write_config(tmp_path / "cpu.yaml", _payload()))

    assert config == CpuBenchmarkConfig(**_payload())

    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(DataValidationError, match="config fields"):
        load_cpu_benchmark_config(_write_config(tmp_path / "extra.yaml", payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup_iterations", -1, "warmup_iterations"),
        ("repetitions", 0, "repetitions"),
        ("intra_op_threads", 0, "intra_op_threads"),
        ("inter_op_threads", True, "inter_op_threads"),
        ("run_name", "../escape", "run_name"),
    ],
)
def test_load_cpu_benchmark_config_rejects_invalid_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(DataValidationError, match=message):
        load_cpu_benchmark_config(_write_config(tmp_path / "invalid.yaml", payload))


def test_percentiles_use_linear_interpolation() -> None:
    samples = (1.0, 2.0, 3.0, 4.0)

    assert _percentile(samples, 0.5) == pytest.approx(2.5)
    assert _percentile(samples, 0.95) == pytest.approx(3.85)
    assert _timing_statistics(samples) == pytest.approx(
        {"count": 4, "mean_ms": 2.5, "p50_ms": 2.5, "p95_ms": 3.85}
    )


def test_timing_statistics_rejects_empty_or_invalid_samples() -> None:
    with pytest.raises(DataValidationError, match="must not be empty"):
        _timing_statistics(())
    with pytest.raises(DataValidationError, match="finite non-negative"):
        _timing_statistics((1.0, float("nan")))
    with pytest.raises(DataValidationError, match="finite non-negative"):
        _timing_statistics((-0.1,))


def _config_object(tmp_path: Path) -> CpuBenchmarkConfig:
    payload = _payload()
    payload.update(
        {
            "dataset_root": str(tmp_path / "datasets"),
            "detector_config": str(tmp_path / "detector.yaml"),
            "classifier_config": str(tmp_path / "classifier.yaml"),
            "detector_checkpoint": str(tmp_path / "detector" / "best.pt"),
            "classifier_checkpoint": str(tmp_path / "classifier" / "best.pt"),
            "output_root": str(tmp_path / "runs" / "benchmark"),
            "warmup_iterations": 2,
            "repetitions": 3,
        }
    )
    return CpuBenchmarkConfig(**payload)


def _backend_result(*, sample_count: int = 6, provider: str = "CPU"):
    return CpuBackendResult(
        stage_samples_ms={
            stage: tuple(float(index + 1) for index in range(sample_count))
            for stage in BENCHMARK_STAGES
        },
        measured_image_ids=tuple(
            image_id
            for _ in range(sample_count // 2)
            for image_id in ("scene-a", "scene-b")
        ),
        classifier_batch_sizes=tuple(range(sample_count)),
        warmup_invocation_count=4,
        runtime="pytorch",
        execution_provider=provider,
        device="cpu",
        intra_op_threads=4,
        inter_op_threads=1,
    )


def test_validate_backend_result_requires_cpu_and_exact_measured_samples() -> None:
    _validate_backend_result(
        _backend_result(),
        image_ids=("scene-a", "scene-b"),
        warmup_iterations=2,
        repetitions=3,
        intra_op_threads=4,
        inter_op_threads=1,
    )

    with pytest.raises(DataValidationError, match="CPU execution provider"):
        _validate_backend_result(
            _backend_result(provider="CUDAExecutionProvider"),
            image_ids=("scene-a", "scene-b"),
            warmup_iterations=2,
            repetitions=3,
            intra_op_threads=4,
            inter_op_threads=1,
        )


def test_validate_backend_result_rejects_malformed_stages_batches_and_threads() -> None:
    common = {
        "image_ids": ("scene-a", "scene-b"),
        "warmup_iterations": 2,
        "repetitions": 3,
        "intra_op_threads": 4,
        "inter_op_threads": 1,
    }
    valid = _backend_result()
    missing_stage = replace(
        valid,
        stage_samples_ms={
            stage: values
            for stage, values in valid.stage_samples_ms.items()
            if stage != "postprocess"
        },
    )
    with pytest.raises(DataValidationError, match="timing stages"):
        _validate_backend_result(missing_stage, **common)

    invalid_timing = replace(
        valid,
        stage_samples_ms={
            **valid.stage_samples_ms,
            "detector": (1.0, 2.0, 3.0, 4.0, 5.0, float("inf")),
        },
    )
    with pytest.raises(DataValidationError, match="finite non-negative"):
        _validate_backend_result(invalid_timing, **common)

    with pytest.raises(DataValidationError, match="batch sizes"):
        _validate_backend_result(
            replace(valid, classifier_batch_sizes=(1, 2, 3, 4, 5, -1)), **common
        )
    with pytest.raises(DataValidationError, match="thread counts"):
        _validate_backend_result(replace(valid, intra_op_threads=8), **common)
    with pytest.raises(DataValidationError, match="sample count"):
        _validate_backend_result(
            _backend_result(sample_count=4),
            image_ids=("scene-a", "scene-b"),
            warmup_iterations=2,
            repetitions=3,
            intra_op_threads=4,
            inter_op_threads=1,
        )


class _RecordingBackend:
    def __init__(self, result=None, *, mutate: Path | None = None, fail=False):
        self.result = result or _backend_result()
        self.mutate = mutate
        self.fail = fail
        self.call = None

    def run(self, **kwargs):
        self.call = kwargs
        if self.mutate is not None:
            self.mutate.write_bytes(b"mutated")
        if self.fail:
            raise RuntimeError("benchmark failed")
        return self.result


def _prepared(tmp_path: Path) -> PreparedCpuBenchmark:
    detector_checkpoint = tmp_path / "detector" / "best.pt"
    classifier_checkpoint = tmp_path / "classifier" / "best.pt"
    detector_checkpoint.parent.mkdir(parents=True)
    classifier_checkpoint.parent.mkdir(parents=True)
    detector_checkpoint.write_bytes(b"detector")
    classifier_checkpoint.write_bytes(b"classifier")
    image_paths = (tmp_path / "scene-a.jpg", tmp_path / "scene-b.jpg")
    for image_path in image_paths:
        image_path.write_bytes(b"image")
    return PreparedCpuBenchmark(
        dataset_root=tmp_path / "datasets",
        detector_config_path=tmp_path / "detector.yaml",
        classifier_config_path=tmp_path / "classifier.yaml",
        detector_checkpoint=detector_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
        detector_metadata_path=tmp_path / "detector" / "metadata.json",
        classifier_metadata_path=tmp_path / "classifier" / "metadata.json",
        detector_manifest_path=tmp_path / "detector-manifest.json",
        classifier_manifest_path=tmp_path / "classifier-manifest.json",
        image_ids=("scene-a", "scene-b"),
        image_paths=image_paths,
        classifier_context={"fixture": True},
        output_dimension=20,
        detector_image_size=640,
        classifier_image_size=224,
        detector_confidence=0.25,
        detector_nms_iou=0.7,
        provenance={"fixture": True},
    )


def test_run_cpu_benchmark_publishes_atomic_cpu_report(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config_object(tmp_path)
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(
        "bakery_scanner.cpu_benchmark._prepare_cpu_benchmark",
        lambda _config: prepared,
    )
    backend = _RecordingBackend()

    report = run_cpu_benchmark(config, backend)

    assert set(path.name for path in report.output_dir.iterdir()) == {
        "benchmark.json",
        "config.yaml",
        "metadata.json",
    }
    payload = json.loads(report.benchmark_path.read_text(encoding="utf-8"))
    assert payload["warmup_iterations"] == 2
    assert payload["repetitions"] == 3
    assert payload["scene_count"] == 2
    assert payload["timings_ms"]["detector"]["count"] == 6
    assert payload["raw_samples_ms"]["end_to_end"] == [1, 2, 3, 4, 5, 6]
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metadata["runtime"]["device"] == "cpu"
    assert metadata["runtime"]["execution_provider"] == "CPU"
    assert metadata["limitations"]["pos_device_claim"] is False
    assert backend.call["image_paths"] == prepared.image_paths


@pytest.mark.parametrize("mode", ["mutate", "fail"])
def test_run_cpu_benchmark_cleans_staging_on_failure(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    config = _config_object(tmp_path)
    prepared = _prepared(tmp_path)
    monkeypatch.setattr(
        "bakery_scanner.cpu_benchmark._prepare_cpu_benchmark",
        lambda _config: prepared,
    )
    backend = _RecordingBackend(
        mutate=prepared.detector_checkpoint if mode == "mutate" else None,
        fail=mode == "fail",
    )

    expected = "checkpoint changed" if mode == "mutate" else "benchmark failed"
    with pytest.raises((DataValidationError, RuntimeError), match=expected):
        run_cpu_benchmark(config, backend)

    output_root = Path(config.output_root)
    assert not (output_root / config.run_name).exists()
    assert not list(output_root.glob(f".{config.run_name}.tmp-*"))


def test_run_cpu_benchmark_rejects_test_path_before_prepare(tmp_path: Path) -> None:
    config = _config_object(tmp_path)
    config = replace(
        config,
        detector_config=str(tmp_path / "datasets" / "base" / "test" / "x.yaml"),
    )

    with pytest.raises(DataValidationError, match="evaluation-only"):
        run_cpu_benchmark(config, _RecordingBackend())


class _StepClock:
    def __init__(self) -> None:
        self.value = -1_000_000

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


class _FakeDetector:
    names = {0: "bread"}

    def __init__(self, torch_module) -> None:
        self.torch = torch_module
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if Path(kwargs["source"]).stem == "scene-b":
            xyxy = self.torch.empty((0, 4), dtype=self.torch.float32)
            conf = self.torch.empty((0,), dtype=self.torch.float32)
            cls = self.torch.empty((0,), dtype=self.torch.float32)
        else:
            xyxy = self.torch.tensor([[1.0, 2.0, 12.0, 14.0]])
            conf = self.torch.tensor([0.8])
            cls = self.torch.tensor([0.0])
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy, conf=conf, cls=cls))]


def test_torch_cpu_backend_uses_cpu_and_excludes_warmup(tmp_path: Path) -> None:
    import torch
    from PIL import Image

    image_paths = (tmp_path / "scene-a.jpg", tmp_path / "scene-b.jpg")
    for image_path in image_paths:
        Image.new("RGB", (20, 20), (20, 30, 40)).save(image_path)
    detector = _FakeDetector(torch)

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.calls = []

        def forward(self, batch):
            self.calls.append(batch)
            assert batch.device.type == "cpu"
            logits = torch.zeros((batch.shape[0], 20), device=batch.device)
            logits[:, 3] = 1.0
            return logits

    classifier = FakeClassifier().cpu().eval()
    loader_calls = []

    def classifier_loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        assert kwargs["device"].type == "cpu"
        return classifier

    backend = TorchCpuBenchmarkBackend(
        clock_ns=_StepClock(),
        detector_factory=lambda _checkpoint: detector,
        classifier_loader=classifier_loader,
        transform_factory=lambda _size: (
            lambda _image: torch.ones((3, 8, 8), dtype=torch.float32)
        ),
        thread_configurer=lambda intra, inter: (intra, inter),
    )

    result = backend.run(
        image_paths=image_paths,
        image_ids=("scene-a", "scene-b"),
        detector_checkpoint=tmp_path / "detector.pt",
        classifier_checkpoint=tmp_path / "classifier.pt",
        classifier_context={"fixture": True},
        output_dimension=20,
        detector_image_size=640,
        classifier_image_size=224,
        detector_confidence=0.25,
        detector_nms_iou=0.7,
        warmup_iterations=1,
        repetitions=2,
        intra_op_threads=4,
        inter_op_threads=1,
    )

    assert result.runtime == "pytorch"
    assert result.execution_provider == "CPU"
    assert result.device == "cpu"
    assert result.warmup_invocation_count == 2
    assert result.measured_image_ids == (
        "scene-a",
        "scene-b",
        "scene-a",
        "scene-b",
    )
    assert result.classifier_batch_sizes == (1, 0, 1, 0)
    assert all(len(result.stage_samples_ms[stage]) == 4 for stage in BENCHMARK_STAGES)
    assert result.stage_samples_ms["classifier_batch"] == (1.0, 0.0, 1.0, 0.0)
    assert len(detector.calls) == 6
    assert all(call["device"] == "cpu" for call in detector.calls)
    assert all(call["conf"] == 0.25 for call in detector.calls)
    assert all(call["imgsz"] == 640 for call in detector.calls)
    assert len(classifier.calls) == 3
    assert loader_calls


def test_torch_cpu_backend_rejects_non_bread_detector(tmp_path: Path) -> None:
    import torch

    detector = _FakeDetector(torch)
    detector.names = {0: "other"}
    backend = TorchCpuBenchmarkBackend(
        detector_factory=lambda _checkpoint: detector,
        classifier_loader=lambda *args, **kwargs: torch.nn.Linear(1, 1),
        transform_factory=lambda _size: lambda _image: torch.ones((1,)),
        thread_configurer=lambda intra, inter: (intra, inter),
    )

    with pytest.raises(DataValidationError, match="one bread class"):
        backend.run(
            image_paths=(tmp_path / "scene.jpg",),
            image_ids=("scene",),
            detector_checkpoint=tmp_path / "detector.pt",
            classifier_checkpoint=tmp_path / "classifier.pt",
            classifier_context={},
            output_dimension=20,
            detector_image_size=640,
            classifier_image_size=224,
            detector_confidence=0.25,
            detector_nms_iou=0.7,
            warmup_iterations=1,
            repetitions=1,
            intra_op_threads=4,
            inter_op_threads=1,
        )
