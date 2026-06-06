"""CLI entry point for Component 3 evaluation.

Usage
-----
Default — evaluate every variant for every system, write outputs to
``reports/<system>/<variant>/``::

    python -m evaluation

Single system::

    python -m evaluation --systems pv

Just the FP32 ONNX variant::

    python -m evaluation --variants onnx_fp32

Sanity-check on the val split::

    python -m evaluation --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from api.schemas import SplitName, SystemType
from evaluation.compare_variants import compare_variants
from evaluation.runner import REPORTS_DIR, evaluate_onnx, evaluate_pytorch
from utils.logging_config import get_logger
from utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, SPLITS_DIR

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Variant catalogue: name → (kind, default-path-template)
# ---------------------------------------------------------------------------

ALL_VARIANTS: tuple[str, ...] = ("pytorch_fp32", "onnx_fp32", "onnx_int8")


def _default_artifact_path(variant: str, system: str) -> Path:
    if variant == "pytorch_fp32":
        return ARTIFACTS_DIR / f"cnn1d_{system}_best.pt"
    if variant == "onnx_fp32":
        return ARTIFACTS_DIR / f"cnn1d_{system}.onnx"
    if variant == "onnx_int8":
        return ARTIFACTS_DIR / f"cnn1d_{system}_int8.onnx"
    raise ValueError(f"Unknown variant: {variant}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentpv-evaluate")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["pv", "bess"],
        default=["pv", "bess"],
        help="Which classifier(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(ALL_VARIANTS),
        default=list(ALL_VARIANTS),
        help="Which model variants to evaluate (default: all three).",
    )
    parser.add_argument(
        "--split",
        choices=[s.value for s in SplitName],
        default=SplitName.TEST.value,
        help="Dataset split to evaluate against.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPORTS_DIR,
        help="Root reports directory (default: reports/).",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument(
        "--n-latency-runs",
        type=int,
        default=1000,
        help="Timed runs in the CPU latency benchmark (assignment requires ≥1000).",
    )
    parser.add_argument("--n-latency-warmup", type=int, default=50)
    parser.add_argument("--latency-seed", type=int, default=42)
    parser.add_argument(
        "--size-budget-mib",
        type=float,
        default=50.0,
        help="Edge model size budget per artefact (assignment §4.2).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After per-variant evaluation, run compare_variants to "
        "produce per-system comparison.{md,json,png} artefacts.",
    )
    parser.add_argument(
        "--baseline-variant",
        choices=list(ALL_VARIANTS),
        default=None,
        help="Variant used as the comparison baseline. Default: pytorch_fp32.",
    )
    return parser.parse_args(argv)


def _evaluate_one(
    *,
    variant: str,
    system_type: SystemType,
    out_dir: Path,
    split: SplitName,
    processed_dir: Path,
    splits_dir: Path,
    n_latency_runs: int,
    n_latency_warmup: int,
    latency_seed: int,
    size_budget_mib: float,
):
    sys_str = system_type.value.lower()
    artefact_path = _default_artifact_path(variant, sys_str)

    log.info(
        "evaluation_start",
        extra={
            "variant": variant,
            "system": sys_str,
            "artefact": str(artefact_path),
            "split": split.value,
            "out_dir": str(out_dir),
        },
    )

    if variant == "pytorch_fp32":
        return evaluate_pytorch(
            checkpoint_path=artefact_path,
            system_type=system_type,
            variant_name=variant,
            out_dir=out_dir,
            split=split,
            processed_dir=processed_dir,
            splits_dir=splits_dir,
            n_latency_runs=n_latency_runs,
            n_latency_warmup=n_latency_warmup,
            latency_seed=latency_seed,
            size_budget_mib=size_budget_mib,
        )
    return evaluate_onnx(
        onnx_path=artefact_path,
        system_type=system_type,
        variant_name=variant,
        out_dir=out_dir,
        split=split,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        n_latency_runs=n_latency_runs,
        n_latency_warmup=n_latency_warmup,
        latency_seed=latency_seed,
        size_budget_mib=size_budget_mib,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    split = SplitName(args.split)

    summary: dict[str, dict[str, object]] = {}
    for sys_str in args.systems:
        system_type = SystemType.PV if sys_str == "pv" else SystemType.BESS
        per_variant: dict[str, dict[str, object]] = {}
        for variant in args.variants:
            out_dir = args.out_root / sys_str / variant
            artefacts = _evaluate_one(
                variant=variant,
                system_type=system_type,
                out_dir=out_dir,
                split=split,
                processed_dir=args.processed_dir,
                splits_dir=args.splits_dir,
                n_latency_runs=args.n_latency_runs,
                n_latency_warmup=args.n_latency_warmup,
                latency_seed=args.latency_seed,
                size_budget_mib=args.size_budget_mib,
            )
            per_variant[variant] = artefacts.to_json()
        system_block: dict[str, object] = {"per_variant": per_variant}

        if args.compare and len(args.variants) >= 2:
            summary_paths = [
                args.out_root / sys_str / variant / "summary.json"
                for variant in args.variants
            ]
            md_path, json_path, png_path = compare_variants(
                summary_paths=summary_paths,
                out_dir=args.out_root / sys_str,
                baseline_variant=args.baseline_variant,
            )
            system_block["comparison"] = {
                "comparison_md": str(md_path),
                "comparison_json": str(json_path),
                "tradeoff_png": str(png_path),
            }

        summary[sys_str] = system_block

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
