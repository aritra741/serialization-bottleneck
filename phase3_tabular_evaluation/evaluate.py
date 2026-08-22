"""
Phase 3 -- evaluation for the tabular (Domain 3) benchmark.

Reads Phase 2's deduplicated JSON result exports (one file per model,
`phase2_tabular_model_results/<NN_model>/<model>_results.json`), computes every
metric in PDF Section 6, and writes a single `evaluation_report.json` plus
a printed summary. This is domain-agnostic in structure -- it only reads
the record schema Phase 2 writes -- but the interpretation (§6.2's property
categories, §9's decision rules) is specific to this tabular property set.

Usage:
    python evaluate.py                          # auto-discover all results
    python evaluate.py --results v4flash=path/to/v4flash_results.json ...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats as scipy_stats

HERE = Path(__file__).resolve().parent
PHASE2_DIR = HERE.parent / "phase2_tabular_model_results"
REPORT_PATH = HERE / "evaluation_report.json"

NUMERIC_PROPERTIES = {"correlation", "skewness"}
INTEGER_PROPERTIES = {"row_count", "null_count"}
BOOLEAN_PROPERTIES = {"is_monotonic", "has_outlier", "func_dependency"}
CATEGORICAL_PROPERTIES = {"column_dtype"}
CATEGORICAL_CLASSES = ["numeric", "categorical", "datetime"]

TIERS = ["simple", "medium", "hard"]
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


# ----------------------------------------------------------------------
# Discovery / loading
# ----------------------------------------------------------------------


def discover_results(base_dir: Path) -> dict[str, Path]:
    found = {}
    if not base_dir.exists():
        return found
    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.glob("*_results.json"):
            model_name = f.stem.replace("_results", "")
            found[model_name] = f
    return found


def load_results(paths: dict[str, Path]) -> dict[str, list[dict]]:
    out = {}
    for model, path in paths.items():
        records = json.loads(path.read_text())
        out[model] = records
    return out


# ----------------------------------------------------------------------
# Bootstrap statistics -- PDF 6.4
# ----------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Optional[dict]:
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    resampled_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_lo_95": float(lo),
        "ci_hi_95": float(hi),
        "n": int(len(values)),
    }


def bootstrap_test(a: np.ndarray, b: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> Optional[dict]:
    """Two-sided bootstrap test for a difference in means/proportions
    between two independent samples. Returns effect size in percentage
    points (a - b) and a two-sided p-value."""
    if len(a) == 0 or len(b) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(n_resamples, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_resamples, len(b)))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    observed = float(a.mean() - b.mean())
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "effect_size_pp": observed * 100,
        "p_value": float(min(p, 1.0)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


# ----------------------------------------------------------------------
# Per-property, per-tier, per-locality metrics -- PDF 6.2 / 6.3
# ----------------------------------------------------------------------


def _bool_array(records: list[dict], key: str) -> np.ndarray:
    return np.array([bool(r.get(key)) for r in records], dtype=float)


def property_metrics(records: list[dict], prop: str) -> dict:
    result: dict[str, Any] = {"n": len(records)}
    correct = _bool_array(records, "correct")
    result["accuracy"] = bootstrap_ci(correct)

    parsed_flags = _bool_array(records, "parse_success")
    result["parse_success_rate"] = float(parsed_flags.mean()) if len(records) else None
    trunc = np.array([r.get("failure_type") == "reasoning_truncated" for r in records], dtype=float)
    result["reasoning_truncation_rate"] = float(trunc.mean()) if len(records) else None

    if prop in NUMERIC_PROPERTIES:
        result["tolerance_accuracy"] = {
            "strict_1pct": bootstrap_ci(_bool_array(records, "correct_1pct")),
            "moderate_5pct": bootstrap_ci(_bool_array(records, "correct_5pct")),
            "lenient_10pct": bootstrap_ci(_bool_array(records, "correct_10pct")),
        }
        rel_errors = np.array(
            [r["relative_error"] for r in records if r.get("parse_success") and r.get("relative_error") is not None]
        )
        if len(rel_errors):
            result["relative_error"] = {
                "median": float(np.median(rel_errors)),
                "p90": float(np.percentile(rel_errors, 90)),
                "n": int(len(rel_errors)),
            }
        else:
            result["relative_error"] = None

    elif prop in INTEGER_PROPERTIES:
        result["tolerance_accuracy"] = {"exact": bootstrap_ci(correct)}

    elif prop in BOOLEAN_PROPERTIES:
        parsed = [r for r in records if r.get("parse_success")]
        tp = sum(1 for r in parsed if r["ground_truth"] is True and r["parsed_answer"] is True)
        fp = sum(1 for r in parsed if r["ground_truth"] is False and r["parsed_answer"] is True)
        fn = sum(1 for r in parsed if r["ground_truth"] is True and r["parsed_answer"] is False)
        tn = sum(1 for r in parsed if r["ground_truth"] is False and r["parsed_answer"] is False)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        result["confusion_matrix"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        result["precision"] = precision
        result["recall"] = recall
        base_rate_vals = [bool(r["ground_truth"]) for r in records]
        result["base_rate_true"] = float(np.mean(base_rate_vals)) if base_rate_vals else None

    elif prop in CATEGORICAL_PROPERTIES:
        parsed = [r for r in records if r.get("parse_success")]
        matrix = {gt: {pred: 0 for pred in CATEGORICAL_CLASSES} for gt in CATEGORICAL_CLASSES}
        for r in parsed:
            gt, pred = r["ground_truth"], r["parsed_answer"]
            if gt in matrix and pred in matrix[gt]:
                matrix[gt][pred] += 1
        result["confusion_matrix"] = matrix

    result["by_tier"] = {
        tier: bootstrap_ci(_bool_array([r for r in records if r["tier"] == tier], "correct"))
        for tier in TIERS
    }
    return result


def locality_metrics(records: list[dict]) -> dict:
    out = {}
    for locality in ("local", "global"):
        subset = [r for r in records if r["property_locality"] == locality]
        out[locality] = bootstrap_ci(_bool_array(subset, "correct"))
    return out


# ----------------------------------------------------------------------
# Aggregate metrics -- PDF 6.3 "Aggregate metrics"
# ----------------------------------------------------------------------


def local_global_gap(records: list[dict]) -> dict:
    local = _bool_array([r for r in records if r["property_locality"] == "local"], "correct")
    glob = _bool_array([r for r in records if r["property_locality"] == "global"], "correct")
    gap = (float(local.mean()) - float(glob.mean())) * 100 if len(local) and len(glob) else None
    return {
        "local_mean_accuracy": float(local.mean()) if len(local) else None,
        "global_mean_accuracy": float(glob.mean()) if len(glob) else None,
        "gap_pp": gap,
        "bootstrap_test": bootstrap_test(local, glob) if len(local) and len(glob) else None,
    }


def complexity_gradient(records: list[dict]) -> dict:
    by_prop: dict[str, dict[str, float]] = defaultdict(dict)
    for prop in sorted({r["property"] for r in records}):
        prop_records = [r for r in records if r["property"] == prop]
        simple = _bool_array([r for r in prop_records if r["tier"] == "simple"], "correct")
        hard = _bool_array([r for r in prop_records if r["tier"] == "hard"], "correct")
        if len(simple) and len(hard):
            by_prop[prop] = {
                "simple_accuracy": float(simple.mean()),
                "hard_accuracy": float(hard.mean()),
                "drop_pp": (float(simple.mean()) - float(hard.mean())) * 100,
            }
    return dict(by_prop)


def cross_model_consistency(all_model_records: dict[str, list[dict]]) -> dict:
    """Spearman rank correlation of per-property mean accuracy across
    models -- PDF 6.3: 'high rank correlation indicates the difficulty
    ordering of properties is model-independent.'"""
    models = sorted(all_model_records.keys())
    if len(models) < 2:
        return {"note": "requires >= 2 models with results", "models_available": models}

    properties = sorted({r["property"] for recs in all_model_records.values() for r in recs})
    acc_matrix = {}
    for model in models:
        recs = all_model_records[model]
        acc_matrix[model] = [
            float(np.mean([r["correct"] for r in recs if r["property"] == p])) if any(r["property"] == p for r in recs) else np.nan
            for p in properties
        ]

    pairwise = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1 :]:
            v1, v2 = np.array(acc_matrix[m1]), np.array(acc_matrix[m2])
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() < 3:
                continue
            rho, p = scipy_stats.spearmanr(v1[mask], v2[mask])
            pairwise[f"{m1}__vs__{m2}"] = {"spearman_rho": float(rho), "p_value": float(p), "n_properties": int(mask.sum())}

    return {"properties": properties, "per_model_property_accuracy": acc_matrix, "pairwise_spearman": pairwise}


# ----------------------------------------------------------------------
# Decision rules -- PDF Section 9 (single-domain reading; the PDF's rule
# is phrased across 2-of-3 domains, which doesn't apply with one domain).
# ----------------------------------------------------------------------


def decision_rule_flags(model_report: dict) -> dict:
    global_props_below_70_on_hard = [
        prop for prop, m in model_report["by_property"].items()
        if (m["by_tier"].get("hard") or {}).get("mean") is not None
        and m["by_tier"]["hard"]["mean"] < 0.70
    ]
    local_above_85 = model_report["by_locality"]["local"] and model_report["by_locality"]["local"]["mean"] > 0.85
    all_above_85_hard = all(
        (m["by_tier"].get("hard") or {}).get("mean", 0) > 0.85 for m in model_report["by_property"].values()
    )
    max_parse_failure = max(
        (1 - (m["parse_success_rate"] or 1.0)) for m in model_report["by_property"].values()
    )
    return {
        "global_properties_below_70pct_on_hard": global_props_below_70_on_hard,
        "local_accuracy_above_85pct": local_above_85,
        "reads_as_implicit_structure_bottleneck": bool(global_props_below_70_on_hard) and local_above_85,
        "all_properties_above_85pct_on_hard": all_above_85_hard,
        "max_parse_failure_rate": max_parse_failure,
        "fix_harness_flag": max_parse_failure > 0.15,
        "note": "PDF Section 9's rule requires 2-of-3 domains; only the tabular domain exists here, so this is a single-domain reading, not the full decision.",
    }


# ----------------------------------------------------------------------
# Report assembly
# ----------------------------------------------------------------------


def build_model_report(model: str, records: list[dict]) -> dict:
    properties = sorted({r["property"] for r in records})
    by_property = {prop: property_metrics([r for r in records if r["property"] == prop], prop) for prop in properties}
    report = {
        "model": model,
        "n_records": len(records),
        "by_property": by_property,
        "by_locality": locality_metrics(records),
        "local_global_gap": local_global_gap(records),
        "complexity_gradient": complexity_gradient(records),
        "overall_accuracy": bootstrap_ci(_bool_array(records, "correct")),
        "overall_parse_success_rate": float(_bool_array(records, "parse_success").mean()) if records else None,
    }
    report["decision_rule_flags"] = decision_rule_flags(report)
    return report


def build_full_report(all_model_records: dict[str, list[dict]]) -> dict:
    return {
        "domain": "tabular",
        "models": sorted(all_model_records.keys()),
        "per_model": {model: build_model_report(model, recs) for model, recs in all_model_records.items()},
        "cross_model_consistency": cross_model_consistency(all_model_records),
    }


def print_summary(report: dict) -> None:
    for model, mr in report["per_model"].items():
        print(f"\n=== {model} ===")
        oa = mr["overall_accuracy"]
        print(f"  overall accuracy: {oa['mean']:.3f} [{oa['ci_lo_95']:.3f}, {oa['ci_hi_95']:.3f}]  n={oa['n']}")
        print(f"  parse success rate: {mr['overall_parse_success_rate']:.3f}")
        gap = mr["local_global_gap"]
        if gap["gap_pp"] is not None:
            bt = gap["bootstrap_test"]
            sig = f"p={bt['p_value']:.4f}" if bt else "n/a"
            print(f"  local-global gap: {gap['gap_pp']:.1f}pp (local={gap['local_mean_accuracy']:.3f}, global={gap['global_mean_accuracy']:.3f}, {sig})")
        print("  by property:")
        for prop, m in sorted(mr["by_property"].items()):
            acc = m["accuracy"]
            print(f"    {prop:18s} acc={acc['mean']:.3f}  n={acc['n']:4d}  parse_ok={m['parse_success_rate']:.3f}")
        flags = mr["decision_rule_flags"]
        if flags["fix_harness_flag"]:
            print(f"  [!] parse failure rate exceeds 15% for at least one property -- fix harness before trusting these results")
        if flags["reads_as_implicit_structure_bottleneck"]:
            print(f"  [note] reads as an implicit-structure bottleneck: {flags['global_properties_below_70pct_on_hard']} below 70% on hard while local stays >85%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="model=path pairs, e.g. v4flash=../phase2_tabular_model_results/01_v4flash/v4flash_results.json. "
        "If omitted, auto-discovers *_results.json under phase2_tabular_model_results/.",
    )
    p.add_argument("--output", type=Path, default=REPORT_PATH)
    p.add_argument("--figures", action="store_true", help="also render the PDF §8 figure set to figures/")
    args = p.parse_args()

    if args.results:
        paths = {}
        for spec in args.results:
            model, path = spec.split("=", 1)
            paths[model] = Path(path)
    else:
        paths = discover_results(PHASE2_DIR)

    if not paths:
        print(f"No result files found under {PHASE2_DIR}. Run Phase 2 first, or pass --results model=path ...")
        return

    print(f"Loading results for: {sorted(paths.keys())}")
    all_model_records = load_results(paths)
    report = build_full_report(all_model_records)

    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    print_summary(report)

    if args.figures:
        from figures import make_all_figures

        paths = make_all_figures(report, all_model_records, HERE / "figures")
        print(f"\nWrote {len(paths)} figures to {HERE / 'figures'}")


if __name__ == "__main__":
    main()
