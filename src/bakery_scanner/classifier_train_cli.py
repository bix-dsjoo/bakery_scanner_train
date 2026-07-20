from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .classifier_training import (
    ClassifierEvaluationReport,
    ClassifierTrainingReport,
    evaluate_classifier_checkpoint,
    load_classifier_experiment_config,
    train_classifier,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-classifier",
        description="Train and evaluate train-side Base or Incremental classifiers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train the configured classifier")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--json", action="store_true", dest="as_json")

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a checkpoint on train-side validation"
    )
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_human(
    report: ClassifierTrainingReport | ClassifierEvaluationReport,
    action: str,
) -> None:
    payload = report.to_dict()
    metrics = payload["metrics"]
    print(f"Classifier {action}: ok")
    print(f"  split: {payload['split']} (train-side)")
    print(f"  output: {payload['output_dir']}")
    selected = (
        payload["best_checkpoint"]
        if action == "training"
        else payload["checkpoint"]
    )
    print(f"  checkpoint: {selected}")
    print(f"  Top-1: {metrics['top1_accuracy']:.6f}")
    print(f"  Macro F1: {metrics['macro_f1']:.6f}")


def _print_preflight(args, config) -> None:
    dataset_root = Path(config.dataset_root).resolve(strict=False)
    source_run = (
        dataset_root / "derived" / "classifier" / config.source_classifier_run
    )
    phase = getattr(config, "phase", "base")
    if args.command == "train":
        output = (
            Path(config.output_root).resolve(strict=False) / config.run_name
        )
    else:
        checkpoint = args.checkpoint.resolve(strict=False)
        output = (
            args.output_dir.resolve(strict=False)
            if args.output_dir is not None
            else checkpoint.parent.parent / "evaluation"
        )
        print(f"Classifier checkpoint: {checkpoint}", file=sys.stderr)
    print(f"Classifier dataset root: {dataset_root}", file=sys.stderr)
    print(f"Classifier phase: {phase}", file=sys.stderr)
    print(f"Classifier source run: {source_run}", file=sys.stderr)
    print("Classifier train split: train", file=sys.stderr)
    print("Classifier validation split: validation (train-side)", file=sys.stderr)
    if phase == "incremental":
        print(
            "Classifier Base checkpoint: "
            f"{Path(config.base_checkpoint).resolve(strict=False)}",
            file=sys.stderr,
        )
        print(
            "Classifier frozen detector checkpoint: "
            f"{Path(config.frozen_detector_checkpoint).resolve(strict=False)}",
            file=sys.stderr,
        )
    else:
        print(
            "Classifier pretrained model: "
            f"{Path(config.pretrained_model).resolve(strict=False)}",
            file=sys.stderr,
        )
    print(f"Classifier output: {output}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_classifier_experiment_config(args.config)
        _print_preflight(args, config)
        if args.command == "train":
            report = train_classifier(config)
            action = "training"
        else:
            report = evaluate_classifier_checkpoint(
                config,
                args.checkpoint,
                output_dir=args.output_dir,
            )
            action = "evaluation"
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(report, action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
