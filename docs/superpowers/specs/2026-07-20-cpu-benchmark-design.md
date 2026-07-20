# CPU-only End-to-End Benchmark Design

**Date:** 2026-07-20
**Status:** Approved for implementation under the active complete-all-stages objective

## Goal

Measure reproducible CPU latency for the approved frozen bread detector and the
20-output Incremental ResNet18 classifier without reading either test split.
Publish detector, crop/preprocess, classifier-batch, postprocess, and end-to-end
latency distributions with enough environment and artifact provenance to replay
the run.

This result describes the current development PC only. It is not a claim about
any specific POS device.

## Scope and data boundary

The benchmark uses all images in the existing train-side detector validation
manifest. It does not load ground-truth annotations into the timed pipeline and
does not calculate or select accuracy metrics. `datasets/base/test` and
`datasets/incremental/test` remain untouched.

The selected artifacts are:

- the approved frozen Base class-agnostic YOLO11n detector checkpoint;
- the approved 20-output Incremental ResNet18 classifier checkpoint;
- the existing registry and train-side detector/classifier manifests needed to
  validate checkpoint context and provenance.

## Considered approaches

### 1. Add timers to the existing end-to-end evaluator

This would reuse the most code, but the evaluator currently requires CUDA and
also calculates validation accuracy. Mixing timing and metric responsibilities
would make CPU-only enforcement and warm-up exclusion harder to audit.

### 2. Standalone native PyTorch CPU benchmark

This is the selected approach. It consumes the approved `.pt` artifacts
directly, keeps evaluation behavior unchanged, and makes every timed boundary
explicit. The detector is invoked with `device="cpu"`; the classifier, input
tensors, and intermediate tensors are explicitly placed on `torch.device("cpu")`.

### 3. Export and benchmark ONNX models

This is deferred. No approved ONNX artifacts or ONNX Runtime dependency exist
yet, and introducing export would add a separate numerical-equivalence and
artifact-provenance problem. A future ONNX ablation must create sessions with
only `CPUExecutionProvider` and fail if any GPU provider is active.

## Configuration and command

A strict YAML config records dataset, training-config, checkpoint, output, and
runtime choices:

- `dataset_root`
- `detector_config`
- `classifier_config`
- `detector_checkpoint`
- `classifier_checkpoint`
- `output_root`
- `run_name`
- `warmup_iterations`
- `repetitions`
- `intra_op_threads`
- `inter_op_threads`

The implemented command will be:

```powershell
bakery-benchmark run --config configs/benchmark/incremental_resnet18_cpu.yaml
```

All fields are validated strictly. The selected classifier config must describe
the 20-output Incremental run. The detector and both checkpoint metadata files
must pass the same hash and provenance binding used by the existing evaluation
pipeline.

## Timed pipeline

Each warm-up or measured iteration processes every selected scene independently
in stable manifest order. The detector uses its approved operating confidence
and NMS IoU, because the benchmark represents the normal inference path rather
than the low confidence floor used to calculate AP.

For every scene, the backend records non-overlapping wall-clock stages with
`perf_counter_ns`:

1. `detector`: YOLO image preprocessing, CPU forward pass, and NMS.
2. `crop_preprocess`: image load, bbox clamping, crop, ResNet transform, and
   tensor stacking.
3. `classifier_batch`: one CPU ResNet batch forward pass and softmax for all
   valid detections in the scene; zero when there is no detection.
4. `postprocess`: CPU conversion and assembly of final bbox/class/score records.
5. `end_to_end`: wall time around all four stages.

Warm-up iterations execute the complete path but are never appended to measured
samples. A measured sample is one scene invocation. `repetitions` therefore
produces `repetitions * scene_count` samples per stage.

The classifier uses one dynamic batch per scene whenever detections exist. The
result records every observed batch size plus minimum, maximum, and mean batch
size. Empty detections are valid and produce an empty result.

## Statistics and output

Each stage reports milliseconds with:

- sample count;
- arithmetic mean;
- P50;
- P95.

Percentiles use deterministic linear interpolation over sorted samples. The run
is published atomically as:

- `config.yaml` — replayable selected configuration;
- `benchmark.json` — timing distributions, batch sizes, iteration counts, and
  input count;
- `metadata.json` — checkpoint/config/manifest paths and SHA-256 values,
  PyTorch CPU execution declaration, CPU model, OS, Python and dependency
  versions, thread counts, detector/classifier input sizes, and checkpoint
  before/after hashes.

Raw per-scene timing samples are retained in `benchmark.json` so aggregate
statistics can be independently recalculated.

## CPU-only enforcement

The benchmark fails closed unless:

- the runtime device is exactly CPU;
- detector invocation is configured with `device="cpu"`;
- classifier model and every classifier input tensor are on CPU;
- the backend reports `runtime="pytorch"` and `execution_provider="CPU"`;
- checkpoint hashes remain unchanged before and after all warm-up and measured
  iterations.

ONNX Runtime is not used in this baseline. Any future ONNX backend must report
the session providers and fail unless the active list is exactly
`["CPUExecutionProvider"]`.

## Validation and failure handling

Unit tests cover strict config parsing, unsafe/test paths, percentile math,
warm-up exclusion, timing sample counts, CPU runtime/provider validation,
classifier batch recording, empty detections, checkpoint mutation, and atomic
cleanup. The real run must pass the full repository suite, compileall, diff
check, independent Ready-PR review, and separate merge verification.

Acceptance requires:

- no test path is read;
- all five timing stages contain exactly `repetitions * scene_count` measured
  samples;
- warm-up samples are absent from statistics;
- mean/P50/P95 and raw samples agree;
- CPU model, threads, input sizes, and observed classifier batches are recorded;
- detector and classifier checkpoint SHA-256 values are unchanged;
- the report explicitly says it is not a POS-device claim.
