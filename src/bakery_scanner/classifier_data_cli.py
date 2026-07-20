from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .classifier_dataset import (
    ClassifierDatasetConfig,
    ClassifierDatasetReport,
    build_classifier_dataset,
    validate_classifier_dataset,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-classifier-data",
        description="Build and independently validate train-side classifier datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="build a classifier dataset")
    generate.add_argument("--dataset-root", type=Path, required=True)
    generate.add_argument("--run-name", required=True)
    generate.add_argument("--phase", choices=("base", "incremental"), required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--validation-fraction", type=float, default=0.2)
    generate.add_argument("--expected-base-images-per-class", type=int, default=84)
    generate.add_argument("--expected-incremental-images-per-class", type=int, default=7)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--json", action="store_true", dest="as_json")

    validate = subparsers.add_parser("validate", help="validate a classifier dataset")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_human(report: ClassifierDatasetReport, action: str) -> None:
    label = "generation" if action == "generate" else "validation"
    print(f"Classifier dataset {label}: ok")
    print(f"  dataset root: {report.output_dir.parents[2]}")
    print(f"  output: {report.output_dir}")
    print(f"  manifest: {report.manifest_path}")
    print(f"  phase: {report.phase}")
    print(f"  output dimension: {report.output_dimension}")
    print(f"  samples: {report.sample_count}")
    print(f"  train samples: {report.train_sample_count}")
    print(f"  validation samples: {report.validation_sample_count}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            report = build_classifier_dataset(
                ClassifierDatasetConfig(
                    dataset_root=args.dataset_root,
                    run_name=args.run_name,
                    phase=args.phase,
                    seed=args.seed,
                    validation_fraction=args.validation_fraction,
                    expected_base_images_per_class=args.expected_base_images_per_class,
                    expected_incremental_images_per_class=(
                        args.expected_incremental_images_per_class
                    ),
                ),
                overwrite=args.overwrite,
            )
        else:
            report = validate_classifier_dataset(args.dataset_root, args.run_name)
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(report, args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
