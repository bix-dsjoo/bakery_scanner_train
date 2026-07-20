from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .cpu_benchmark import (
    BENCHMARK_STAGES,
    CpuBenchmarkConfig,
    CpuBenchmarkReport,
    load_cpu_benchmark_config,
    run_cpu_benchmark,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-benchmark",
        description="Run the train-side detector-plus-classifier CPU benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="benchmark the configured pipeline on CPU"
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_preflight(config: CpuBenchmarkConfig) -> None:
    print("CPU benchmark split: validation (train-side)", file=sys.stderr)
    print("CPU benchmark device: cpu", file=sys.stderr)
    print(
        f"CPU benchmark detector: {Path(config.detector_checkpoint).resolve(strict=False)}",
        file=sys.stderr,
    )
    print(
        f"CPU benchmark classifier: {Path(config.classifier_checkpoint).resolve(strict=False)}",
        file=sys.stderr,
    )
    print(
        f"CPU benchmark warm-up/repetitions: {config.warmup_iterations}/{config.repetitions}",
        file=sys.stderr,
    )
    print(
        f"CPU benchmark intra-op threads: {config.intra_op_threads}",
        file=sys.stderr,
    )
    print(
        f"CPU benchmark inter-op threads: {config.inter_op_threads}",
        file=sys.stderr,
    )
    output = Path(config.output_root).resolve(strict=False) / config.run_name
    print(f"CPU benchmark output: {output}", file=sys.stderr)


def _print_human(report: CpuBenchmarkReport) -> None:
    payload = report.to_dict()
    timings = payload["benchmark"]["timings_ms"]
    print("CPU benchmark: ok")
    print(f"  output: {payload['output_dir']}")
    for stage in BENCHMARK_STAGES:
        stats = timings[stage]
        print(
            f"  {stage}: mean={stats['mean_ms']:.3f} ms, "
            f"P50={stats['p50_ms']:.3f} ms, P95={stats['p95_ms']:.3f} ms"
        )
    print("  Current development-PC result; not a specific POS-device claim.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_cpu_benchmark_config(args.config)
        _print_preflight(config)
        report = run_cpu_benchmark(config)
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
