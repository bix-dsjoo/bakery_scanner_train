from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .base_cycle import freeze_base_cycle, validate_base_cycle
from .errors import DataValidationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-base-cycle",
        description="Freeze and validate the Base redesign cycle split.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--repository-root", default=".")
    validate.add_argument("--run-name", required=True)
    validate.add_argument("--json", action="store_true")
    return parser


def _print_human(payload: dict[str, object]) -> None:
    print(f"Base cycle validation passed: {payload['output_dir']}")
    print("development scene IDs: " + ", ".join(payload["development_scene_ids"]))
    print(f"cycle holdout scene ID: {payload['holdout_scene_id']}")
    print("seeds: " + ", ".join(str(seed) for seed in payload["seeds"]))
    print(f"manifest: {payload['manifest_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            report = freeze_base_cycle(args.config)
        else:
            report = validate_base_cycle(args.repository_root, args.run_name)
    except DataValidationError as exc:
        print(f"Base cycle command failed: {exc}", file=sys.stderr)
        return 1
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
