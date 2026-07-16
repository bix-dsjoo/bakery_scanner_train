from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .audit import AuditReport, audit_dataset
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-audit",
        description="Validate Bakery Scanner datasets and report statistics.",
    )
    parser.add_argument(
        "--dataset-root", default="datasets", help="Dataset root (default: datasets)"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Proposed train-side scene validation fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Proposed scene split seed (default: 42)"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return parser


def _print_human(report: AuditReport) -> None:
    print(f"Dataset audit passed: {report.dataset_root}")
    print(
        "Registry: "
        f"{report.registry_total_classes} classes "
        f"({report.registry_phase_counts['base']} base, "
        f"{report.registry_phase_counts['incremental']} incremental)"
    )
    print(
        "Single-object images: "
        f"{report.class_image_totals['base']} base, "
        f"{report.class_image_totals['incremental']} incremental"
    )
    labels = {
        "base_scene_train": "Base scene train",
        "base_test": "Base test",
        "incremental_test": "Incremental test",
    }
    for name, stats in report.coco_splits.items():
        print(
            f"{labels[name]}: {stats.image_count} images, "
            f"{stats.annotation_count} annotations, {stats.category_count} categories"
        )
    print(
        "Proposed scene split: "
        f"train IDs={','.join(report.scene_split.train_scene_ids)}; "
        f"validation IDs={','.join(report.scene_split.validation_scene_ids)}; "
        f"seed={report.seed}; fraction={report.validation_fraction}"
    )
    print("Evaluation-only splits were inspected but not authorized for training.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_dataset(
            args.dataset_root,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    except DataValidationError as exc:
        print(f"Dataset audit failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0
