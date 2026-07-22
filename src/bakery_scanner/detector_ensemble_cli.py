from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .detector_ensemble import (
    benchmark_detector_ensemble_cpu,
    evaluate_detector_ensemble,
    load_detector_ensemble_config,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-detector-ensemble",
        description="Evaluate and CPU-benchmark a frozen detector candidate ensemble.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("evaluate", "Evaluate the ensemble on train-side validation"),
        ("benchmark", "Benchmark the ensemble with CPU-only inference"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        command.add_argument("--json", action="store_true")
    return parser


def _selection(config) -> dict[str, object]:
    return {
        "dataset_root": str(Path(config.dataset_root).resolve(strict=False)),
        "output": str(
            Path(config.output_root).resolve(strict=False) / config.run_name
        ),
        "members": [
            {
                "config": str(Path(member.config_path).resolve(strict=False)),
                "checkpoint": str(
                    Path(member.checkpoint_path).resolve(strict=False)
                ),
            }
            for member in config.members
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_detector_ensemble_config(args.config)
        if not args.json:
            selection = _selection(config)
            print(f"Dataset root: {selection['dataset_root']}")
            for index, member in enumerate(selection["members"]):
                print(f"Member {index} config: {member['config']}")
                print(f"Member {index} checkpoint: {member['checkpoint']}")
            print(f"Output: {selection['output']}")
        if args.command == "evaluate":
            payload = evaluate_detector_ensemble(config).to_dict()
        else:
            benchmark_path = benchmark_detector_ensemble_cpu(config)
            payload = {
                "status": "ok",
                "benchmark_path": str(benchmark_path),
                "benchmark": json.loads(benchmark_path.read_text(encoding="utf-8")),
            }
    except DataValidationError as exc:
        print(f"Detector ensemble {args.command} failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Detector ensemble {args.command} completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
