"""A minimal, training-free command-line interface for CANOPY."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from canopy import __version__


def parse_widths(raw: str) -> tuple[int, ...]:
    """Parse positive layer widths in input,hidden...,output order."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 2 or any(not part.isdecimal() for part in parts):
        raise argparse.ArgumentTypeError(
            "Provide at least two positive integers: input,hidden...,output "
            "(for example 2,64,64,1)."
        )
    widths = tuple(int(part) for part in parts)
    if any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("Layer widths must all be positive integers.")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canopy", description="Training-free neural expressivity scoring"
    )
    parser.add_argument("--version", action="version", version=f"CANOPY {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser(
        "score", help="Score feedforward ReLU layer widths without training"
    )
    score.add_argument(
        "--widths", type=parse_widths, required=True,
        help="Input, optional hidden widths, then output (e.g. 2,64,64,1)",
    )
    score.add_argument(
        "--mode", choices=("generic", "tight"), default="generic",
        help="Generic or bottleneck-aware architecture-level region-count bound",
    )
    score.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output machine-readable JSON (score is a decimal string)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        from canopy.nas.architecture import ArchitectureSpec
        from canopy.nas.tropical_score import tropical_expressivity_score

        arch = ArchitectureSpec(layer_widths=args.widths)
        score = tropical_expressivity_score(arch, mode=args.mode)

        if args.as_json:
            payload = {
                "canopy_version": __version__,
                "layer_widths": list(args.widths),
                "mode": args.mode,
                "score": str(score),
                "score_units": "count_bound",
                "score_interpretation": (
                    "Architecture-level theoretical upper bound; "
                    "not an exact realized linear-region count."
                ),
            }
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"{score} (architecture-level region-count upper bound, {args.mode})")
        return 0

    # All commands are registered above; argparse rejects unknown commands.
    raise AssertionError("Unreachable CLI command")


if __name__ == "__main__":
    raise SystemExit(main())
