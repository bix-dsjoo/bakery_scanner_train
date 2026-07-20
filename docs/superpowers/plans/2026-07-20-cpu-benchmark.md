# CPU-only End-to-End Benchmark Implementation Plan

> Execute task-by-task with test-driven development. Do not read either test
> split. Use the train-side detector validation manifest only.

**Goal:** Add and run an auditable CPU-only latency benchmark for the frozen
YOLO11n detector and approved 20-output Incremental ResNet18 classifier.

**Architecture:** A new orchestration module validates configs, data manifests,
checkpoint provenance, CPU runtime declarations, and atomic output publication.
A native PyTorch backend owns the five timed inference stages. The CLI only
loads the strict config, prints the selected train-side inputs, runs the
benchmark, and renders its report.

**Tech stack:** Python 3.11, PyTorch, torchvision, Ultralytics YOLO, Pillow,
PyYAML, pytest.

**Design:** `docs/superpowers/specs/2026-07-20-cpu-benchmark-design.md`

---

## Constraints

- Never read `datasets/base/test` or `datasets/incremental/test`.
- Explicitly use CPU for detector, classifier, and tensors.
- Do not call the result a POS benchmark.
- Use the detector operating confidence, not the AP confidence floor.
- Exclude all warm-up invocations from recorded timing samples.
- Record detector, crop/preprocess, classifier batch, postprocess, and
  end-to-end timing separately.
- Record CPU model, dependency versions, thread counts, input sizes, dynamic
  classifier batch sizes, repetitions, and warm-up count.
- Preserve and verify detector/classifier checkpoint hashes.
- Publish output atomically and fail on an existing run.

## Task 1: Strict config and deterministic statistics

**Files:**

- Create: `src/bakery_scanner/cpu_benchmark.py`
- Create: `tests/test_cpu_benchmark.py`

- [ ] Write failing tests for exact config fields, positive warm-up/repetition
  and thread values, valid run names, and rejection of unknown fields.
- [ ] Implement immutable `CpuBenchmarkConfig` and
  `load_cpu_benchmark_config()`.
- [ ] Write failing tests for empty timing samples and deterministic linear
  interpolation at P50/P95.
- [ ] Implement `_percentile()` and `_timing_statistics()` returning
  count/mean/P50/P95 in milliseconds.
- [ ] Run `python -m pytest tests/test_cpu_benchmark.py -q`.
- [ ] Commit: `feat(benchmark): CPU 설정과 통계를 추가한다`.

## Task 2: Safe orchestration and atomic reports

**Files:**

- Modify: `src/bakery_scanner/cpu_benchmark.py`
- Modify: `tests/test_cpu_benchmark.py`

- [ ] Add a fake backend and failing tests proving that unsafe/test paths are
  rejected before reads and that only train-side validation image paths are
  passed to the backend.
- [ ] Reuse existing detector provenance validation and generic classifier
  experiment loading/context validation. Require Incremental phase and 20
  outputs.
- [ ] Define backend/result protocols with raw per-stage timing samples,
  per-scene batch sizes, runtime, execution provider, and device declarations.
- [ ] Add failing tests for any runtime other than PyTorch CPU, missing/extra
  stages, incorrect sample counts, non-finite/negative samples, invalid batch
  counts, and mutated checkpoints.
- [ ] Implement `run_cpu_benchmark()` with checkpoint before/after hashes,
  strict result validation, statistics calculation, metadata, and atomic
  `config.yaml`/`benchmark.json`/`metadata.json` publication.
- [ ] Add failing tests for backend exceptions and malformed results leaving no
  final or staging directory.
- [ ] Run focused tests and commit:
  `feat(benchmark): CPU 실행 경계와 원자적 결과를 검증한다`.

## Task 3: Native PyTorch CPU backend

**Files:**

- Modify: `src/bakery_scanner/cpu_benchmark.py`
- Modify: `tests/test_cpu_benchmark.py`

- [ ] Write failing unit tests around an injected clock and fake
  detector/classifier adapters to prove stage boundaries, warm-up exclusion,
  stable scene order, one classifier batch per non-empty scene, and empty
  detection handling.
- [ ] Implement model loading with YOLO detector `device="cpu"`, strict
  classifier checkpoint context loading on `torch.device("cpu")`, and explicit
  CPU tensors.
- [ ] Apply configured intra-op/inter-op threads before inference and record the
  effective values.
- [ ] Time the five stages with `perf_counter_ns`; return raw measured samples
  only and all observed classifier batch sizes.
- [ ] Add adapter-level tests verifying detector arguments and classifier/tensor
  devices are CPU.
- [ ] Run focused tests and commit:
  `feat(benchmark): PyTorch CPU 추론 시간을 측정한다`.

## Task 4: CLI, default config, and documentation

**Files:**

- Create: `src/bakery_scanner/cpu_benchmark_cli.py`
- Create: `tests/test_cpu_benchmark_cli.py`
- Create: `configs/benchmark/incremental_resnet18_cpu.yaml`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] Write failing CLI tests for `run --config`, JSON output, error exit, and
  printed train-side/CPU selections.
- [ ] Implement `bakery-benchmark` entry point without inventing any other
  command.
- [ ] Add the strict default config with warm-up 5, repetitions 30, intra-op 4,
  and inter-op 1.
- [ ] Document the exact implemented command, timing definitions, CPU-only
  enforcement, output schema, and non-POS limitation. State explicitly that the
  run does not access test data.
- [ ] Run CLI/focused/full tests and commit:
  `docs(benchmark): CPU 기준선 명령을 문서화한다`.

## Task 5: Execute and verify the real benchmark

**Artifacts (ignored):**

- Create: `runs/benchmark/incremental_resnet18_cpu/`

- [ ] Replay detector and Incremental classifier dataset validation without
  touching test paths.
- [ ] Run:

  ```powershell
  bakery-benchmark run --config configs/benchmark/incremental_resnet18_cpu.yaml
  ```

- [ ] Independently recalculate every mean/P50/P95 value from raw samples and
  compare it with `benchmark.json`.
- [ ] Verify exactly `30 * scene_count` samples per stage and that the five
  warm-up iterations are absent.
- [ ] Verify metadata records PyTorch CPU execution, CPU model, thread counts,
  detector/classifier input sizes, observed batch sizes, versions, paths, and
  hashes.
- [ ] Verify detector and classifier checkpoints are unchanged and the report
  contains no test path or POS-device claim.
- [ ] Record the actual current-PC result in README without generalizing it to a
  POS device.
- [ ] Run fresh completion verification:

  ```powershell
  python -m pytest -q
  python -m compileall -q src tests
  git diff --check origin/main...HEAD
  git status --short
  ```

- [ ] Commit final result documentation, push the branch, open a Korean Ready
  PR, obtain a fresh independent review, repeat review after any diff change,
  and use a separate merge agent for Korean squash merge and remote branch
  deletion.
