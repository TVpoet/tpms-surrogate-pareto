# -*- coding: utf-8 -*-
"""Analyse the locked 22-point extended Pareto/FEM validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pareto_validation_extended"
DEFAULT_MANIFEST = OUTPUT_ROOT / "selection_manifest.csv"
VALIDATION_RESULTS = OUTPUT_ROOT / "validation_results.csv"
METRICS_OVERALL = OUTPUT_ROOT / "metrics_overall.json"
METRICS_BY_LOAD = OUTPUT_ROOT / "metrics_by_load.csv"
DOMINANCE_CHECKS = OUTPUT_ROOT / "dominance_checks.csv"
FRONT_STABILITY = OUTPUT_ROOT / "front_stability.csv"
RUN_METADATA = OUTPUT_ROOT / "run_metadata.json"

OBJECTIVES = {
    "sigma_hp": ("pred_sigma_hp", "fem_sigma_hp"),
    "specific_modulus": ("pred_specific_modulus", "fem_specific_modulus"),
    "rho_rel": ("pred_rho_rel", "fem_rho_rel"),
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_summary(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("sigma_d", "E_eff", "rho_rel"):
        if key not in data:
            raise ValueError(f"{path} is missing {key}")
    sigma = float(data["sigma_d"])
    e_eff = abs(float(data["E_eff"]))
    rho = float(data["rho_rel"])
    if sigma <= 0 or e_eff <= 0 or rho <= 0:
        raise ValueError(f"Non-positive FEM objective in {path}")
    return {
        "fem_sigma_hp": sigma,
        "fem_E_eff": e_eff,
        "fem_rho_rel": rho,
        "fem_specific_modulus": e_eff / rho,
    }


def _relative_error_pct(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (predicted - reference) / reference * 100.0


def _metric_summary(relative_error_pct: Iterable[float]) -> Dict[str, float]:
    error = np.asarray(list(relative_error_pct), dtype=float)
    absolute = np.abs(error)
    return {
        "n": int(len(error)),
        "MARE_pct": float(np.mean(absolute)),
        "relative_RMSE_pct": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_error_pct": float(np.median(absolute)),
        "P90_absolute_error_pct": float(np.percentile(absolute, 90)),
        "maximum_absolute_error_pct": float(np.max(absolute)),
        "mean_signed_error_pct": float(np.mean(error)),
    }


def _nondominated_mask(sigma: np.ndarray, specific_modulus: np.ndarray) -> np.ndarray:
    n = len(sigma)
    nondominated = np.ones(n, dtype=bool)
    for i in range(n):
        no_worse = (sigma <= sigma[i]) & (specific_modulus >= specific_modulus[i])
        strictly_better = (sigma < sigma[i]) | (specific_modulus > specific_modulus[i])
        if np.any(no_worse & strictly_better):
            nondominated[i] = False
    return nondominated


def _safe_correlation(function, predicted: np.ndarray, fem: np.ndarray) -> float:
    result = function(predicted, fem)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value) if np.isfinite(value) else float("nan")


def _build_results(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, source_row in manifest.iterrows():
        row = source_row.to_dict()
        summary = _read_summary(Path(str(source_row["fem_result_path"])))
        row.update(summary)
        for objective, (predicted_column, fem_column) in OBJECTIVES.items():
            error = float(
                _relative_error_pct(
                    np.asarray([float(row[predicted_column])]),
                    np.asarray([float(row[fem_column])]),
                )[0]
            )
            row[f"{objective}_error_pct"] = error
            row[f"{objective}_absolute_error_pct"] = abs(error)
        rows.append(row)
    return pd.DataFrame(rows)


def _analyse_fronts(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_rows: List[Dict[str, object]] = []
    dominance_rows: List[Dict[str, object]] = []

    for case_name, case_df in results.groupby("load_case", sort=False):
        case_df = case_df.copy()
        pareto_df = case_df[case_df["is_pareto"]].copy()
        all_front_mask = _nondominated_mask(
            case_df["fem_sigma_hp"].to_numpy(dtype=float),
            case_df["fem_specific_modulus"].to_numpy(dtype=float),
        )
        case_df["fem_is_nondominated_within_case"] = all_front_mask
        pareto_retained = case_df.loc[case_df["is_pareto"], "fem_is_nondominated_within_case"]

        d1 = case_df[case_df["point_id"] == "D1"]
        if len(d1) != 1:
            raise ValueError(f"{case_name}: expected exactly one D1")
        d1_row = d1.iloc[0]
        reference_id = str(d1_row["dominance_reference"])
        reference = case_df[case_df["point_id"] == reference_id]
        if len(reference) != 1:
            raise ValueError(f"{case_name}: missing dominance reference {reference_id}")
        ref_row = reference.iloc[0]

        surrogate_preserved = bool(
            float(ref_row["pred_sigma_hp"]) < float(d1_row["pred_sigma_hp"])
            and float(ref_row["pred_specific_modulus"]) > float(d1_row["pred_specific_modulus"])
        )
        fem_preserved = bool(
            float(ref_row["fem_sigma_hp"]) < float(d1_row["fem_sigma_hp"])
            and float(ref_row["fem_specific_modulus"]) > float(d1_row["fem_specific_modulus"])
        )
        fem_reversed = bool(
            float(d1_row["fem_sigma_hp"]) < float(ref_row["fem_sigma_hp"])
            and float(d1_row["fem_specific_modulus"]) > float(ref_row["fem_specific_modulus"])
        )
        if fem_preserved:
            fem_relation = "preserved"
        elif fem_reversed:
            fem_relation = "reversed"
        else:
            fem_relation = "became_mutually_nondominated"
        dominance_rows.append(
            {
                "load_case": case_name,
                "dominating_point": reference_id,
                "dominated_point": "D1",
                "surrogate_dominance_expected": surrogate_preserved,
                "fem_dominance_preserved": fem_preserved,
                "fem_dominance_reversed": fem_reversed,
                "fem_relation": fem_relation,
                "pred_sigma_margin_MPa": float(d1_row["pred_sigma_hp"] - ref_row["pred_sigma_hp"]),
                "fem_sigma_margin_MPa": float(d1_row["fem_sigma_hp"] - ref_row["fem_sigma_hp"]),
                "pred_specific_modulus_margin_MPa": float(
                    ref_row["pred_specific_modulus"] - d1_row["pred_specific_modulus"]
                ),
                "fem_specific_modulus_margin_MPa": float(
                    ref_row["fem_specific_modulus"] - d1_row["fem_specific_modulus"]
                ),
            }
        )

        retained_count = int(pareto_retained.sum())
        retention_rate = float(pareto_retained.mean())
        lost_anchor_ids = case_df.loc[
            case_df["is_pareto"] & ~case_df["fem_is_nondominated_within_case"], "point_id"
        ].astype(str).tolist()
        stability_rows.append(
            {
                "load_case": case_name,
                "n_re_evaluated_points": int(len(case_df)),
                "n_pareto_anchors": int(len(pareto_df)),
                "fem_nondominated_pareto_anchors": retained_count,
                "pareto_anchor_retention_rate": retention_rate,
                "lost_pareto_anchor_ids": ";".join(lost_anchor_ids),
                "spearman_sigma_hp": _safe_correlation(
                    spearmanr,
                    pareto_df["pred_sigma_hp"].to_numpy(dtype=float),
                    pareto_df["fem_sigma_hp"].to_numpy(dtype=float),
                ),
                "kendall_sigma_hp": _safe_correlation(
                    kendalltau,
                    pareto_df["pred_sigma_hp"].to_numpy(dtype=float),
                    pareto_df["fem_sigma_hp"].to_numpy(dtype=float),
                ),
                "spearman_specific_modulus": _safe_correlation(
                    spearmanr,
                    pareto_df["pred_specific_modulus"].to_numpy(dtype=float),
                    pareto_df["fem_specific_modulus"].to_numpy(dtype=float),
                ),
                "kendall_specific_modulus": _safe_correlation(
                    kendalltau,
                    pareto_df["pred_specific_modulus"].to_numpy(dtype=float),
                    pareto_df["fem_specific_modulus"].to_numpy(dtype=float),
                ),
                "P2_D1_dominance_preserved": fem_preserved,
                "stability_status": "stable" if retention_rate == 1.0 and fem_preserved else "needs_review",
            }
        )

        results.loc[case_df.index, "fem_is_nondominated_within_case"] = case_df[
            "fem_is_nondominated_within_case"
        ]

    return pd.DataFrame(stability_rows), pd.DataFrame(dominance_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    if not RUN_METADATA.exists():
        raise FileNotFoundError(RUN_METADATA)
    with RUN_METADATA.open("r", encoding="utf-8") as handle:
        run_metadata = json.load(handle)
    if run_metadata.get("manifest_sha256") != _sha256(args.manifest):
        raise RuntimeError("Manifest hash does not match the FEM run metadata")
    if run_metadata.get("campaign_status") != "completed":
        raise RuntimeError(
            f"FEM campaign is not complete: {run_metadata.get('campaign_status', 'unknown')}"
        )
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 22:
        raise ValueError(f"Expected 22 selected points, found {len(manifest)}")
    manifest["is_pareto"] = _as_bool(manifest["is_pareto"])
    results = _build_results(manifest)

    by_load_rows = []
    overall = {}
    for objective, (_, fem_column) in OBJECTIVES.items():
        error_column = f"{objective}_error_pct"
        overall[objective] = _metric_summary(results[error_column])
        for case_name, case_df in results.groupby("load_case", sort=False):
            by_load_rows.append(
                {
                    "load_case": case_name,
                    "objective": objective,
                    **_metric_summary(case_df[error_column]),
                }
            )

    front_stability, dominance_checks = _analyse_fronts(results)
    campaign_stable = bool(
        (front_stability["stability_status"] == "stable").all()
        and dominance_checks["fem_dominance_preserved"].all()
    )
    overall_payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "n_total": int(len(results)),
        "n_pareto": int(results["is_pareto"].sum()),
        "n_near_front_dominated": int((results["point_id"] == "D1").sum()),
        "objectives": overall,
        "pareto_anchor_retention": {
            "retained": int(front_stability["fem_nondominated_pareto_anchors"].sum()),
            "total": int(front_stability["n_pareto_anchors"].sum()),
            "rate": float(
                front_stability["fem_nondominated_pareto_anchors"].sum()
                / front_stability["n_pareto_anchors"].sum()
            ),
        },
        "dominance_pairs": {
            "preserved": int(dominance_checks["fem_dominance_preserved"].sum()),
            "became_mutually_nondominated": int(
                (dominance_checks["fem_relation"] == "became_mutually_nondominated").sum()
            ),
            "reversed": int(dominance_checks["fem_dominance_reversed"].sum()),
            "total": int(len(dominance_checks)),
        },
        "campaign_status": "stable" if campaign_stable else "needs_review",
        "stability_definition": (
            "All preselected Pareto anchors remain nondominated within each FEM-re-evaluated "
            "subset and all four predeclared P2-D1 dominance pairs are preserved."
        ),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results.to_csv(VALIDATION_RESULTS, index=False, encoding="utf-8-sig")
    pd.DataFrame(by_load_rows).to_csv(METRICS_BY_LOAD, index=False, encoding="utf-8-sig")
    dominance_checks.to_csv(DOMINANCE_CHECKS, index=False, encoding="utf-8-sig")
    front_stability.to_csv(FRONT_STABILITY, index=False, encoding="utf-8-sig")
    with METRICS_OVERALL.open("w", encoding="utf-8") as handle:
        json.dump(overall_payload, handle, indent=2, ensure_ascii=False)

    print("=" * 92)
    print("Extended Pareto/FEM validation analysis")
    print(f"Points: {len(results)} (Pareto={int(results.is_pareto.sum())}, D1={(results.point_id == 'D1').sum()})")
    for objective, metrics in overall.items():
        print(
            f"{objective:<18} MARE={metrics['MARE_pct']:.2f}%  "
            f"rRMSE={metrics['relative_RMSE_pct']:.2f}%  "
            f"P90={metrics['P90_absolute_error_pct']:.2f}%  "
            f"max={metrics['maximum_absolute_error_pct']:.2f}%"
        )
    retained = overall_payload["pareto_anchor_retention"]
    dominance = overall_payload["dominance_pairs"]
    print(f"Pareto anchors retained: {retained['retained']}/{retained['total']}")
    print(
        f"Dominance pairs: preserved={dominance['preserved']}/{dominance['total']}, "
        f"mutually nondominated={dominance['became_mutually_nondominated']}, "
        f"reversed={dominance['reversed']}"
    )
    print(f"Campaign status: {overall_payload['campaign_status']}")
    print(f"Results: {VALIDATION_RESULTS}")
    print("=" * 92)


if __name__ == "__main__":
    main()
