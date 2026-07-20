from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .detector_training import (
    DetectorEvaluationReport,
    DetectorTrainingConfig,
    DetectorTrainingReport,
    evaluate_detector_checkpoint,
    load_detector_training_config,
    train_detector,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-detector",
        description="Train and evaluate the class-agnostic bread detector baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train the configured detector")
    train.add_argument("--config", required=True)
    train.add_argument("--json", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint on train-side validation"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--json", action="store_true")
    return parser


def _selection(config: DetectorTrainingConfig) -> dict[str, str]:
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    yolo_root = dataset_root / "derived" / "yolo" / config.yolo_run_name
    return {
        "dataset_root": str(dataset_root),
        "source_detector_run": config.source_detector_run,
        "train_split": str(yolo_root / "train" / "images"),
        "validation_split": str(yolo_root / "validation" / "images"),
        "model": config.model,
        "output": str(Path(config.output_root).resolve(strict=False) / config.run_name),
    }


def _print_selection(config: DetectorTrainingConfig) -> None:
    selected = _selection(config)
    print(f"Dataset root: {selected['dataset_root']}")
    print(f"Source detector run: {selected['source_detector_run']}")
    print(f"Train split: {selected['train_split']}")
    print(f"Validation split: {selected['validation_split']}")
    print(f"Model: {selected['model']}")
    print(f"Output: {selected['output']}")


def _print_report(
    report: DetectorTrainingReport | DetectorEvaluationReport,
    payload: dict[str, Any],
) -> None:
    global_metrics = payload["metrics"]["global"]
    print(f"Detector {report.split} completed: {report.output_dir}")
    print(
        "Metrics: "
        f"AP50={global_metrics['ap50']}, "
        f"Recall={global_metrics['recall']}, "
        f"miss_rate={global_metrics['miss_rate']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_detector_training_config(args.config)
        if not args.json:
            _print_selection(config)
        if args.command == "train":
            report = train_detector(config)
        else:
            report = evaluate_detector_checkpoint(
                config,
                args.checkpoint,
                output_dir=args.output_dir,
            )
    except DataValidationError as exc:
        print(f"Detector {args.command} failed: {exc}", file=sys.stderr)
        return 1

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_report(report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
