from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .e2e_inference import (
    EndToEndConfig,
    EndToEndReport,
    evaluate_end_to_end,
    load_end_to_end_config,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-e2e",
        description="Run frozen detector-plus-classifier train-side evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate the configured Base end-to-end pipeline"
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_preflight(config: EndToEndConfig) -> None:
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    output = Path(config.output_root).resolve(strict=False) / config.run_name
    print("End-to-end split: validation (train-side)", file=sys.stderr)
    print(f"End-to-end dataset root: {dataset_root}", file=sys.stderr)
    print(
        f"End-to-end detector config: {Path(config.detector_config).resolve(strict=False)}",
        file=sys.stderr,
    )
    print(
        f"End-to-end classifier config: {Path(config.classifier_config).resolve(strict=False)}",
        file=sys.stderr,
    )
    print(
        f"End-to-end detector checkpoint: {Path(config.detector_checkpoint).resolve(strict=False)}",
        file=sys.stderr,
    )
    print(
        f"End-to-end classifier checkpoint: {Path(config.classifier_checkpoint).resolve(strict=False)}",
        file=sys.stderr,
    )
    print(f"End-to-end output: {output}", file=sys.stderr)


def _print_human(report: EndToEndReport) -> None:
    payload = report.to_dict()
    metrics = payload["metrics"]
    print("End-to-end evaluation: ok")
    print(f"  split: {payload['split']} (train-side)")
    print(f"  output: {payload['output_dir']}")
    print(f"  mAP50: {metrics['map50']:.6f}")
    print(f"  mAP50:95: {metrics['map50_95']:.6f}")
    print(
        "  supported exact-count accuracy: "
        f"{metrics['supported_macro_exact_count_accuracy']:.6f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_end_to_end_config(args.config)
        _print_preflight(config)
        report = evaluate_end_to_end(config)
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
