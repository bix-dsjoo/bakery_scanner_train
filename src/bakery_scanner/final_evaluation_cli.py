from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import DataValidationError
from .final_evaluation import (
    load_final_evaluation_config,
    preflight_final_evaluation,
    run_final_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bakery-final-eval",
        description="Preflight or execute the frozen one-shot final test evaluation.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_final_evaluation_config(args.config)
        if args.command == "preflight":
            report = preflight_final_evaluation(config, args.config)
            payload = report.to_dict()
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("Final evaluation preflight: ready")
                print("  test data accessed: false")
                print(f"  config SHA-256: {payload['config_sha256']}")
                print(f"  output: {payload['output_dir']}")
            return 0
        print(
            "WARNING: irreversible one-shot final test evaluation; the start lock will prevent reruns.",
            file=sys.stderr,
        )
        report = run_final_evaluation(config, args.config)
    except DataValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = report.to_dict()
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Final evaluation: completed")
        print(f"  output: {payload['output_dir']}")
        print(f"  summary: {payload['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
