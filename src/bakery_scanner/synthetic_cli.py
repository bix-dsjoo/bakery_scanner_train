from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .errors import DataValidationError
from .synthetic import (
    SyntheticConfig,
    SyntheticGenerationReport,
    SyntheticValidationReport,
    generate_synthetic_dataset,
    validate_synthetic_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-synthetic",
        description="Generate and replay-validate deterministic synthetic bread scenes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a synthetic run")
    generate.add_argument("--dataset-root", default="datasets")
    generate.add_argument("--background-dir", required=True)
    generate.add_argument("--run-name", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--scene-count", type=int, default=10)
    generate.add_argument("--objects-per-scene", type=int, default=5)
    generate.add_argument(
        "--phase", choices=("base", "incremental", "all"), default="base"
    )
    generate.add_argument("--size-fraction-min", type=float, default=0.12)
    generate.add_argument("--size-fraction-max", type=float, default=0.28)
    generate.add_argument("--rotation-min", type=float, default=-25.0)
    generate.add_argument("--rotation-max", type=float, default=25.0)
    generate.add_argument("--brightness-min", type=float, default=0.85)
    generate.add_argument("--brightness-max", type=float, default=1.15)
    generate.add_argument("--contrast-min", type=float, default=0.9)
    generate.add_argument("--contrast-max", type=float, default=1.1)
    generate.add_argument("--foreground-threshold", type=int, default=245)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate", help="Replay and validate a run")
    validate.add_argument("--dataset-root", default="datasets")
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--json", action="store_true")
    return parser


def _print_human(
    report: SyntheticGenerationReport | SyntheticValidationReport, action: str
) -> None:
    print(f"Synthetic {action} passed: {report.output_dir}")
    print(
        f"Images: {report.image_count}; objects: {report.object_count}; "
        f"generator: {report.generator_version}"
    )
    print(f"Manifest: {report.manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            config = SyntheticConfig(
                seed=args.seed,
                scene_count=args.scene_count,
                objects_per_scene=args.objects_per_scene,
                phase=args.phase,
                size_fraction_range=(
                    args.size_fraction_min,
                    args.size_fraction_max,
                ),
                rotation_range=(args.rotation_min, args.rotation_max),
                brightness_range=(args.brightness_min, args.brightness_max),
                contrast_range=(args.contrast_min, args.contrast_max),
                foreground_threshold=args.foreground_threshold,
            )
            report = generate_synthetic_dataset(
                args.dataset_root,
                args.background_dir,
                args.run_name,
                config,
                overwrite=args.overwrite,
            )
            action = "generation"
        else:
            report = validate_synthetic_dataset(args.dataset_root, args.run_name)
            action = "validation"
    except DataValidationError as exc:
        action = "generation" if args.command == "generate" else "validation"
        print(f"Synthetic {action} failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(report, action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
