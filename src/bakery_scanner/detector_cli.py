from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .detector_dataset import (
    DetectorDatasetConfig,
    build_detector_dataset,
    validate_detector_dataset,
)
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-detector-data",
        description="Build and validate class-agnostic bread detector COCO datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Assemble a leakage-safe detector dataset"
    )
    generate.add_argument("--dataset-root", default="datasets")
    generate.add_argument("--synthetic-run", required=True)
    generate.add_argument("--run-name", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--validation-fraction", type=float, default=0.2)
    generate.add_argument(
        "--real-coco-path", default="base/val/instances_val.json"
    )
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--json", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Independently validate a detector dataset"
    )
    validate.add_argument("--dataset-root", default="datasets")
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--json", action="store_true")
    return parser


def _print_human(payload: dict[str, Any], action: str) -> None:
    print(f"Detector dataset {action} passed: {payload['output_dir']}")
    print(
        f"Images: {payload['image_count']} "
        f"(train={payload['train_image_count']}, "
        f"validation={payload['validation_image_count']})"
    )
    print(f"Annotations: {payload['annotation_count']}")
    print(f"Manifest: {payload['manifest_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            report = build_detector_dataset(
                args.dataset_root,
                args.synthetic_run,
                args.run_name,
                DetectorDatasetConfig(
                    seed=args.seed,
                    validation_fraction=args.validation_fraction,
                    real_coco_path=args.real_coco_path,
                ),
                overwrite=args.overwrite,
            )
            action = "generation"
        else:
            report = validate_detector_dataset(args.dataset_root, args.run_name)
            action = "validation"
    except DataValidationError as exc:
        action = "generation" if args.command == "generate" else "validation"
        print(f"Detector dataset {action} failed: {exc}", file=sys.stderr)
        return 1

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(payload, action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
