"""A minimal, training-free command-line interface for CANOPY."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
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



def bounded_limit(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer from 2 to 1000") from exc
    if not 2 <= n <= 1000:
        raise argparse.ArgumentTypeError("limit must be from 2 to 1000")
    return n


def sha256_digest(raw: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", raw):
        raise argparse.ArgumentTypeError("sha256 must be 64 hexadecimal characters")
    return raw.lower()


def check_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    if digest.hexdigest() != expected:
        raise ValueError("benchmark SHA-256 mismatch; refusing to load data")


def _finite_or_none(value: float):
    x = float(value)
    return x if math.isfinite(x) else None


def run_real_evaluation(args: argparse.Namespace) -> dict:
    """Compare only real explicitly provided NATS-Bench TSS data."""
    import numpy as np
    from canopy.nas.benchmark import NATSBenchAPI, compare_metrics

    path = args.benchmark_data.expanduser()
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise FileNotFoundError(f"Benchmark data not found: {path}")
    if args.sha256:
        if not path.is_file():
            raise ValueError("--sha256 is only supported for a benchmark file")
        check_sha256(path, args.sha256)
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")

    # NATSBenchAPI can load serialized data. Only load trusted artifacts.
    benchmark = NATSBenchAPI(str(path), dataset=args.dataset, hp=args.hp)
    if args.limit > benchmark.total_architectures:
        raise ValueError("Sample larger than benchmark")
    rng = np.random.default_rng(args.seed)
    sampled = [int(i) for i in rng.choice(
        benchmark.total_architectures, size=args.limit, replace=False
    )]

    records = []
    for idx in sampled:
        # Never hide failed lookups: omitted records bias correlations.
        record = benchmark.query_by_index(idx)
        if record is None or record.index != idx:
            raise ValueError(f"Incomplete benchmark record at index {idx}")
        if record.test_accuracy is None or not math.isfinite(float(record.test_accuracy)):
            raise ValueError(f"Invalid benchmark accuracy at index {idx}")
        records.append(record)

    class CompleteSample:
        def get_all(self, **kwargs):
            return records

    comparisons = compare_metrics(CompleteSample(), seed=args.seed)
    return {
        "schema_version": 1,
        "canopy_version": __version__,
        "benchmark": {
            "kind": "real_nats_bench",
            "search_space": "NATS-Bench TSS / NAS-Bench-201",
            "dataset": args.dataset,
            "hp": args.hp,
            "identifier": args.benchmark_id or path.name,
            "sha256": args.sha256,
            "checksum_status": (
                "user_supplied_digest_matched_file" if args.sha256 else "not_verified"
            ),
            "artifact_trust": "caller_responsibility",
        },
        "sample": {
            "method": "uniform_without_replacement",
            "seed": args.seed,
            "total_architectures": benchmark.total_architectures,
            "indices": sampled,
            "missing_lookups": 0,
        },
        "scoring_notes": (
            "Generic and tight are combinatorial architecture-level bounds, "
            "not exact realized linear-region counts."
        ),
        "evaluation_notes": (
            "Retrospective test-accuracy correlations; not independent validation "
            "of NAS-Bench-301 or an actual architecture training run."
        ),
        "metrics": [
            {
                "metric": item.metric_name,
                "spearman_rho": _finite_or_none(item.spearman_rho),
                "kendall_tau": _finite_or_none(item.kendall_tau),
                "top_10_accuracy": _finite_or_none(item.top_10_accuracy),
                "top_50_accuracy": _finite_or_none(item.top_50_accuracy),
            }
            for item in comparisons
        ],
    }


def write_evaluation_report(output: Path, report: dict) -> Path:
    """Refuse overwrite and write results only after complete evaluation."""
    output.mkdir(parents=True, exist_ok=False)
    result = output / "benchmark_comparison.json"
    try:
        result.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    except Exception:
        result.unlink(missing_ok=True)
        output.rmdir()
        raise
    return result


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
    evaluate = commands.add_parser(
        "evaluate", help="Compare scoring proxies with an explicit real benchmark"
    )
    evaluate.add_argument("--benchmark-data", required=True, type=Path,
                          help="Trusted real NATS-Bench TSS file/directory")
    evaluate.add_argument("--dataset", required=True,
                          choices=("cifar10", "cifar100", "ImageNet16-120"))
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--hp", choices=("12", "200"), default="200")
    evaluate.add_argument("--limit", type=bounded_limit, default=200)
    evaluate.add_argument("--output", required=True, type=Path,
                          help="New report directory")
    evaluate.add_argument("--benchmark-id", help="Optional provenance label")
    evaluate.add_argument("--sha256", type=sha256_digest,
                          help="Expected SHA-256 for one benchmark file")
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

    if args.command == "evaluate":
        try:
            report = run_real_evaluation(args)
            path = write_evaluation_report(args.output, report)
        except (FileNotFoundError, FileExistsError, ValueError, ImportError, OSError) as error:
            build_parser().error(
                f"Evaluation failed: {error}. Install with pip install -e '.[bench]' "
                "and use only trusted NATS-Bench assets."
            )
        print(f"Wrote benchmark comparison to {path}")
        return 0

    # All commands are registered above; argparse rejects unknown commands.
    raise AssertionError("Unreachable CLI command")


if __name__ == "__main__":
    raise SystemExit(main())
