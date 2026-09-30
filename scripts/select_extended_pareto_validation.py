# -*- coding: utf-8 -*-
"""Select a fixed extended Pareto/FEM validation set.

This script performs surrogate-only searches.  It never launches Abaqus.

The locked validation design contains 22 rows:

* Three published load cases: existing P1/P2/P3 plus new Q1/Q2/D1.
* One new mixed-sign load case: new P1/P2/P3/D1.

Q1 and Q2 are stratified-random Pareto samples on the two arc-length
segments separated by P2.  D1 is a near-front candidate that is strictly
dominated by P2 in the surrogate objectives.  All selection decisions are
made before FEM evaluation and are persisted in a manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# The project links PyTorch and scientific Python packages that may load two
# Intel OpenMP runtimes on Windows.  Existing model scripts use the same guard.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pick_three_dirichlet import (  # noqa: E402
    DENSITY_PATH,
    SURROGATE_PATH,
    compute_pareto_dirichlet,
    pick_three,
)
from surrogate_model.density_model import DensityModel  # noqa: E402
from surrogate_model.model import SurrogateModel  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pareto_validation_extended"
MANIFEST_CSV = OUTPUT_ROOT / "selection_manifest.csv"
MANIFEST_JSON = OUTPUT_ROOT / "selection_manifest.json"
CONFIG_JSON = OUTPUT_ROOT / "selection_config.json"
FRONTS_NPZ = OUTPUT_ROOT / "surrogate_fronts.npz"
PREVIEW_PNG = OUTPUT_ROOT / "selection_preview.png"
RUN_METADATA = OUTPUT_ROOT / "run_metadata.json"

EXISTING_CANDIDATES = PROJECT_ROOT / "outputs" / "validation_candidates_dirichlet.json"
TRAINING_DATA = PROJECT_ROOT / "fem_data" / "parameterized" / "training_data.csv"
EXISTING_FEM_ROOT = PROJECT_ROOT / "outputs" / "pareto_validation_dirichlet"
NEW_CASE_ROOT = OUTPUT_ROOT / "cases"

PARETO_SEED = 43
SELECTION_SEED = 47
DEFAULT_N_SAMPLES = 5_000_000

LOAD_CASES = OrderedDict(
    [
        ("Biaxial-XY", np.asarray([2.0e-6, 2.0e-6, 0.0], dtype=float)),
        ("Triaxial-1-1-2", np.asarray([1.0e-6, 1.0e-6, 2.0e-6], dtype=float)),
        ("Triaxial-1-2-2", np.asarray([1.0e-6, 2.0e-6, 2.0e-6], dtype=float)),
        ("Triaxial-m1-2-3", np.asarray([-1.0e-6, 2.0e-6, 3.0e-6], dtype=float)),
    ]
)
PUBLISHED_CASES = tuple(list(LOAD_CASES)[:3])
NEW_LOAD_CASE = "Triaxial-m1-2-3"

EXISTING_POINT_KEYS = OrderedDict(
    [
        ("P1", "A_min_sigma"),
        ("P2", "C_p3"),
        ("P3", "B_max_stiff"),
    ]
)

MANIFEST_COLUMNS = [
    "load_case",
    "point_id",
    "category",
    "is_pareto",
    "fem_action",
    "dominance_reference",
    "alpha1",
    "alpha2",
    "alpha3",
    "alpha4",
    "delta_x",
    "delta_y",
    "delta_z",
    "pred_sigma_hp",
    "pred_E_eff",
    "pred_rho_rel",
    "pred_specific_modulus",
    "front_arc_fraction",
    "selection_seed",
    "selection_rule",
    "training_duplicate",
    "fem_result_path",
]


def _configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
    )


def _safe_key(case_name: str) -> str:
    return case_name.lower().replace("-", "_")


def _normalised_front_coordinates(
    sigma: np.ndarray, specific_modulus: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return front order, arc coordinate, and normalised objectives."""
    order = np.argsort(sigma, kind="stable")
    sigma_range = float(np.ptp(sigma))
    modulus_range = float(np.ptp(specific_modulus))
    sigma_norm = (sigma - float(np.min(sigma))) / sigma_range if sigma_range > 0 else np.zeros_like(sigma)
    modulus_norm = (
        (specific_modulus - float(np.min(specific_modulus))) / modulus_range
        if modulus_range > 0
        else np.zeros_like(specific_modulus)
    )
    segment = np.hypot(np.diff(sigma_norm[order]), np.diff(modulus_norm[order]))
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] > 0:
        cumulative /= cumulative[-1]
    arc = np.empty_like(cumulative)
    arc[order] = cumulative
    return order, arc, sigma_norm, modulus_norm


def _nearest_alpha_index(front_alpha: np.ndarray, alpha: Sequence[float]) -> int:
    target = np.asarray(alpha, dtype=float)
    return int(np.argmin(np.linalg.norm(front_alpha - target[None, :], axis=1)))


def _is_training_duplicate(
    alpha: Sequence[float], delta: Sequence[float], training: pd.DataFrame
) -> bool:
    delta_values = training[["delta_x", "delta_y", "delta_z"]].to_numpy(dtype=float)
    alpha_values = training[["alpha1", "alpha2", "alpha3", "alpha4"]].to_numpy(dtype=float)
    load_match = np.max(np.abs(delta_values - np.asarray(delta)[None, :]), axis=1) <= 1.0e-12
    if not np.any(load_match):
        return False
    alpha_match = (
        np.max(np.abs(alpha_values[load_match] - np.asarray(alpha)[None, :]), axis=1) <= 1.0e-8
    )
    return bool(np.any(alpha_match))


def _sufficiently_separated(alpha: np.ndarray, selected: Iterable[np.ndarray]) -> bool:
    return all(float(np.sum(np.abs(alpha - other))) >= 1.0e-3 for other in selected)


def _choose_stratified_random(
    front_alpha: np.ndarray,
    front_sigma: np.ndarray,
    front_modulus: np.ndarray,
    arc: np.ndarray,
    p2_index: int,
    already_selected: List[np.ndarray],
    training: pd.DataFrame,
    delta: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Choose Q1 and Q2 from the central 60% of P1-P2 and P2-P3 arcs."""
    p2_arc = float(arc[p2_index])
    intervals = [
        (0.20 * p2_arc, 0.80 * p2_arc),
        (p2_arc + 0.20 * (1.0 - p2_arc), p2_arc + 0.80 * (1.0 - p2_arc)),
    ]
    sigma_range = max(float(np.ptp(front_sigma)), np.finfo(float).eps)
    modulus_range = max(float(np.ptp(front_modulus)), np.finfo(float).eps)
    chosen: List[int] = []

    for low, high in intervals:
        candidates = np.flatnonzero((arc >= low) & (arc <= high))
        if not len(candidates):
            candidates = np.asarray([int(np.argmin(np.abs(arc - 0.5 * (low + high))))])
        candidates = rng.permutation(candidates)
        selected_index = None
        for idx in candidates:
            alpha = front_alpha[idx]
            if not _sufficiently_separated(alpha, already_selected):
                continue
            if _is_training_duplicate(alpha, delta, training):
                continue
            objective_distances = [
                np.hypot(
                    (front_sigma[idx] - front_sigma[j]) / sigma_range,
                    (front_modulus[idx] - front_modulus[j]) / modulus_range,
                )
                for j in chosen
            ]
            if objective_distances and min(objective_distances) < 0.03:
                continue
            selected_index = int(idx)
            break
        if selected_index is None:
            raise RuntimeError(f"Unable to choose a non-duplicate stratified Pareto point in [{low:.3f}, {high:.3f}]")
        chosen.append(selected_index)
        already_selected.append(front_alpha[selected_index].copy())

    return chosen[0], chosen[1]


def _choose_near_dominated(
    results: Dict[str, np.ndarray],
    pareto_mask: np.ndarray,
    p2_alpha: np.ndarray,
    p2_sigma: float,
    p2_modulus: float,
    selected_alpha: List[np.ndarray],
    training: pd.DataFrame,
    delta: np.ndarray,
) -> Tuple[int, float]:
    """Choose D1, strictly dominated by P2 and 2--5% away in objective space."""
    sigma = np.asarray(results["sigma_d"])
    modulus = np.asarray(results["specific_stiffness"])
    alpha = np.asarray(results["alpha"])
    sigma_scale = max(float(np.ptp(sigma[pareto_mask])), np.finfo(float).eps)
    modulus_scale = max(float(np.ptp(modulus[pareto_mask])), np.finfo(float).eps)

    delta_sigma = (sigma - p2_sigma) / sigma_scale
    delta_modulus = (p2_modulus - modulus) / modulus_scale
    gap = np.hypot(delta_sigma, delta_modulus)
    strict = (
        (~pareto_mask)
        & (delta_sigma > 0.002)
        & (delta_modulus > 0.002)
    )

    preferred = strict & (gap >= 0.02) & (gap <= 0.05)
    fallback = strict & (gap >= 0.01) & (gap <= 0.08)
    candidate_indices = np.flatnonzero(preferred)
    if not len(candidate_indices):
        candidate_indices = np.flatnonzero(fallback)
    if not len(candidate_indices):
        candidate_indices = np.flatnonzero(strict)
    if not len(candidate_indices):
        raise RuntimeError("No strictly P2-dominated candidate was found")

    order = candidate_indices[np.argsort(np.abs(gap[candidate_indices] - 0.035), kind="stable")]
    for idx in order:
        candidate_alpha = alpha[idx]
        if not _sufficiently_separated(candidate_alpha, selected_alpha + [p2_alpha]):
            continue
        if _is_training_duplicate(candidate_alpha, delta, training):
            continue
        return int(idx), float(gap[idx])
    raise RuntimeError("All near-front dominated candidates failed duplicate/separation checks")


def _record(
    case_name: str,
    point_id: str,
    category: str,
    is_pareto: bool,
    fem_action: str,
    dominance_reference: str,
    alpha: Sequence[float],
    delta: Sequence[float],
    sigma: float,
    e_eff: float,
    rho: float,
    specific_modulus: float,
    arc_fraction: float | None,
    selection_rule: str,
    training_duplicate: bool,
    fem_result_path: Path,
) -> Dict[str, object]:
    alpha = [float(v) for v in alpha]
    delta = [float(v) for v in delta]
    return OrderedDict(
        [
            ("load_case", case_name),
            ("point_id", point_id),
            ("category", category),
            ("is_pareto", bool(is_pareto)),
            ("fem_action", fem_action),
            ("dominance_reference", dominance_reference),
            ("alpha1", alpha[0]),
            ("alpha2", alpha[1]),
            ("alpha3", alpha[2]),
            ("alpha4", alpha[3]),
            ("delta_x", delta[0]),
            ("delta_y", delta[1]),
            ("delta_z", delta[2]),
            ("pred_sigma_hp", float(sigma)),
            ("pred_E_eff", float(e_eff)),
            ("pred_rho_rel", float(rho)),
            ("pred_specific_modulus", float(specific_modulus)),
            ("front_arc_fraction", None if arc_fraction is None else float(arc_fraction)),
            ("selection_seed", SELECTION_SEED),
            ("selection_rule", selection_rule),
            ("training_duplicate", bool(training_duplicate)),
            ("fem_result_path", str(fem_result_path.resolve())),
        ]
    )


def _validate_manifest(rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    expected_by_case = {
        "Biaxial-XY": {"P1", "P2", "P3", "Q1", "Q2", "D1"},
        "Triaxial-1-1-2": {"P1", "P2", "P3", "Q1", "Q2", "D1"},
        "Triaxial-1-2-2": {"P1", "P2", "P3", "Q1", "Q2", "D1"},
        "Triaxial-m1-2-3": {"P1", "P2", "P3", "D1"},
    }
    errors = []
    if len(df) != 22:
        errors.append(f"expected 22 rows, found {len(df)}")
    if int((df["fem_action"] == "reuse").sum()) != 9:
        errors.append("expected 9 reused FEM rows")
    if int((df["fem_action"] == "run").sum()) != 13:
        errors.append("expected 13 new FEM rows")
    if int(df["is_pareto"].sum()) != 18:
        errors.append("expected 18 Pareto rows")
    if int((df["point_id"] == "D1").sum()) != 4:
        errors.append("expected 4 D1 rows")
    if bool(df["training_duplicate"].any()):
        errors.append("training duplicate detected")
    for case_name, expected_ids in expected_by_case.items():
        found = set(df.loc[df["load_case"] == case_name, "point_id"])
        if found != expected_ids:
            errors.append(f"{case_name}: expected {sorted(expected_ids)}, found {sorted(found)}")
    alpha_sum = df[["alpha1", "alpha2", "alpha3", "alpha4"]].sum(axis=1)
    if not np.allclose(alpha_sum, 1.0, atol=1.0e-6):
        errors.append("one or more alpha vectors do not sum to one")
    if (df[["alpha1", "alpha2", "alpha3", "alpha4"]] < 0).any().any():
        errors.append("negative alpha coefficient detected")
    if errors:
        raise RuntimeError("Manifest validation failed: " + "; ".join(errors))


def _draw_preview(
    rows: List[Dict[str, object]], front_store: Dict[str, np.ndarray], output_path: Path
) -> None:
    _configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    df = pd.DataFrame(rows)

    for ax, panel, case_name in zip(axes.ravel(), panel_labels, LOAD_CASES):
        key = _safe_key(case_name)
        sigma = front_store[f"{key}_sigma_hp"]
        modulus = front_store[f"{key}_specific_modulus"]
        order = np.argsort(sigma)
        ax.plot(sigma[order], modulus[order], color="0.45", lw=1.1, label="Surrogate Pareto front")
        case_df = df[df["load_case"] == case_name]
        pareto_df = case_df[case_df["is_pareto"]]
        dominated_df = case_df[~case_df["is_pareto"]]
        ax.scatter(
            pareto_df["pred_sigma_hp"],
            pareto_df["pred_specific_modulus"],
            s=34,
            facecolors="white",
            edgecolors="#1f77b4",
            linewidths=1.1,
            zorder=4,
            label="Selected Pareto point",
        )
        ax.scatter(
            dominated_df["pred_sigma_hp"],
            dominated_df["pred_specific_modulus"],
            s=38,
            marker="s",
            facecolors="white",
            edgecolors="#d95f02",
            linewidths=1.1,
            zorder=5,
            label="Near-front dominated point",
        )
        for _, row in case_df.iterrows():
            ax.annotate(
                row["point_id"],
                (row["pred_sigma_hp"], row["pred_specific_modulus"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        delta_um = LOAD_CASES[case_name] * 1.0e6
        ax.text(
            0.03,
            0.96,
            rf"$\boldsymbol{{\delta}}=({delta_um[0]:g},{delta_um[1]:g},{delta_um[2]:g})\ \mathrm{{\mu m}}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(-0.12, 1.02, panel, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
        ax.set_xlabel(r"$\sigma_{\mathrm{hp}}$ (MPa)")
        ax.set_ylabel(r"$E_{\mathrm{eff}}/\rho_{\mathrm{rel}}$ (MPa)")
        _style_axes(ax)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.12, wspace=0.25, hspace=0.28)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing locked manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if MANIFEST_CSV.exists() and not args.force:
        raise FileExistsError(
            f"Locked manifest already exists: {MANIFEST_CSV}. Use --force only before FEM selection is accepted."
        )
    if MANIFEST_CSV.exists() and args.force and RUN_METADATA.exists():
        with RUN_METADATA.open("r", encoding="utf-8") as handle:
            run_metadata = json.load(handle)
        completed_statuses = {"completed", "completed_cached"}
        completed_new_cases = [
            key
            for key, value in run_metadata.get("cases", {}).items()
            if value.get("status") in completed_statuses
        ]
        if completed_new_cases:
            raise RuntimeError(
                "FEM results already exist for the locked selection; refusing to overwrite the manifest."
            )
    if not EXISTING_CANDIDATES.exists():
        raise FileNotFoundError(EXISTING_CANDIDATES)
    if not TRAINING_DATA.exists():
        raise FileNotFoundError(TRAINING_DATA)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    NEW_CASE_ROOT.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(TRAINING_DATA)
    with EXISTING_CANDIDATES.open("r", encoding="utf-8") as handle:
        existing_candidates = json.load(handle)

    surrogate = SurrogateModel.load(SURROGATE_PATH)
    density = DensityModel.load(DENSITY_PATH)

    rows: List[Dict[str, object]] = []
    front_store: Dict[str, np.ndarray] = {}
    per_case_summary: Dict[str, Dict[str, object]] = OrderedDict()

    for case_index, (case_name, delta) in enumerate(LOAD_CASES.items()):
        print(f"\n[{case_index + 1}/{len(LOAD_CASES)}] {case_name}: surrogate Pareto search")
        results, pareto_mask = compute_pareto_dirichlet(
            surrogate,
            density,
            delta,
            n_samples=args.n_samples,
            seed=PARETO_SEED,
        )
        front_alpha = np.asarray(results["alpha"])[pareto_mask]
        front_sigma = np.asarray(results["sigma_d"])[pareto_mask]
        front_e = np.asarray(results["E_eff"])[pareto_mask]
        front_rho = np.asarray(results["rho_rel"])[pareto_mask]
        front_modulus = np.asarray(results["specific_stiffness"])[pareto_mask]
        _, front_arc, _, _ = _normalised_front_coordinates(front_sigma, front_modulus)

        key = _safe_key(case_name)
        front_store[f"{key}_alpha"] = front_alpha.astype(np.float32)
        front_store[f"{key}_sigma_hp"] = front_sigma.astype(np.float32)
        front_store[f"{key}_E_eff"] = front_e.astype(np.float32)
        front_store[f"{key}_rho_rel"] = front_rho.astype(np.float32)
        front_store[f"{key}_specific_modulus"] = front_modulus.astype(np.float32)
        front_store[f"{key}_arc_fraction"] = front_arc.astype(np.float32)

        selected_alpha: List[np.ndarray] = []
        selected_indices: Dict[str, int] = {}

        if case_name in PUBLISHED_CASES:
            case_existing = existing_candidates[case_name]
            for point_id, source_key in EXISTING_POINT_KEYS.items():
                candidate = case_existing[source_key]
                idx = _nearest_alpha_index(front_alpha, candidate["alpha"])
                selected_indices[point_id] = idx
                selected_alpha.append(front_alpha[idx].copy())
                result_path = EXISTING_FEM_ROOT / f"{case_name}__{point_id}" / "case_summary.json"
                rows.append(
                    _record(
                        case_name,
                        point_id,
                        "existing_pareto",
                        True,
                        "reuse",
                        "",
                        candidate["alpha"],
                        delta,
                        candidate["pred_sigma_d"],
                        candidate["pred_E_eff"],
                        candidate["pred_rho_rel"],
                        candidate["pred_specific_stiffness"],
                        front_arc[idx],
                        "Existing locked P1/P2/P3 candidate",
                        _is_training_duplicate(candidate["alpha"], delta, training),
                        result_path,
                    )
                )
            p2_index = selected_indices["P2"]
            rng = np.random.default_rng(SELECTION_SEED + case_index)
            q1_index, q2_index = _choose_stratified_random(
                front_alpha,
                front_sigma,
                front_modulus,
                front_arc,
                p2_index,
                selected_alpha,
                training,
                delta,
                rng,
            )
            for point_id, idx, segment_name in [
                ("Q1", q1_index, "P1-P2"),
                ("Q2", q2_index, "P2-P3"),
            ]:
                selected_indices[point_id] = idx
                result_path = NEW_CASE_ROOT / f"{case_name}__{point_id}" / "case_summary.json"
                rows.append(
                    _record(
                        case_name,
                        point_id,
                        "stratified_random_pareto",
                        True,
                        "run",
                        "",
                        front_alpha[idx],
                        delta,
                        front_sigma[idx],
                        front_e[idx],
                        front_rho[idx],
                        front_modulus[idx],
                        front_arc[idx],
                        f"Seeded random Pareto sample in the central 60% of the {segment_name} arc",
                        False,
                        result_path,
                    )
                )
        else:
            p1_index, p3_index, p2_index = pick_three(front_sigma, front_modulus, method="arc")
            selected_indices.update({"P1": p1_index, "P2": p2_index, "P3": p3_index})
            for point_id in ("P1", "P2", "P3"):
                idx = selected_indices[point_id]
                selected_alpha.append(front_alpha[idx].copy())
                result_path = NEW_CASE_ROOT / f"{case_name}__{point_id}" / "case_summary.json"
                rule = {
                    "P1": "Minimum predicted sigma_hp",
                    "P2": "Normalised Pareto arc-length midpoint",
                    "P3": "Maximum predicted specific modulus",
                }[point_id]
                rows.append(
                    _record(
                        case_name,
                        point_id,
                        "new_load_pareto",
                        True,
                        "run",
                        "",
                        front_alpha[idx],
                        delta,
                        front_sigma[idx],
                        front_e[idx],
                        front_rho[idx],
                        front_modulus[idx],
                        front_arc[idx],
                        rule,
                        _is_training_duplicate(front_alpha[idx], delta, training),
                        result_path,
                    )
                )

        p2_index = selected_indices["P2"]
        d1_global_index, d1_gap = _choose_near_dominated(
            results,
            pareto_mask,
            front_alpha[p2_index],
            float(front_sigma[p2_index]),
            float(front_modulus[p2_index]),
            selected_alpha,
            training,
            delta,
        )
        d1_alpha = np.asarray(results["alpha"])[d1_global_index]
        d1_path = NEW_CASE_ROOT / f"{case_name}__D1" / "case_summary.json"
        rows.append(
            _record(
                case_name,
                "D1",
                "near_front_dominated",
                False,
                "run",
                "P2",
                d1_alpha,
                delta,
                np.asarray(results["sigma_d"])[d1_global_index],
                np.asarray(results["E_eff"])[d1_global_index],
                np.asarray(results["rho_rel"])[d1_global_index],
                np.asarray(results["specific_stiffness"])[d1_global_index],
                None,
                f"Strictly dominated by P2; normalised objective gap={d1_gap:.6f}",
                False,
                d1_path,
            )
        )

        per_case_summary[case_name] = {
            "delta_um": [float(v * 1.0e6) for v in delta],
            "pareto_size": int(np.sum(pareto_mask)),
            "point_ids": [row["point_id"] for row in rows if row["load_case"] == case_name],
            "d1_normalised_gap": d1_gap,
        }
        print(
            f"  selected: {per_case_summary[case_name]['point_ids']} | "
            f"front={int(np.sum(pareto_mask))} | D1 gap={d1_gap:.4f}"
        )

    _validate_manifest(rows)
    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
    with MANIFEST_JSON.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    np.savez_compressed(FRONTS_NPZ, **front_store)

    config = OrderedDict(
        [
            ("created_at", datetime.now().astimezone().isoformat()),
            ("surrogate_path", str(Path(SURROGATE_PATH).resolve())),
            ("density_path", str(Path(DENSITY_PATH).resolve())),
            ("training_data", str(TRAINING_DATA.resolve())),
            ("n_samples", int(args.n_samples)),
            ("pareto_seed", PARETO_SEED),
            ("selection_seed", SELECTION_SEED),
            ("new_load_case", NEW_LOAD_CASE),
            ("new_load_delta_um", [-1.0, 2.0, 3.0]),
            ("new_load_p2_rule", "normalised Pareto arc-length midpoint"),
            ("d1_preferred_normalised_gap", [0.02, 0.05]),
            ("cases", per_case_summary),
        ]
    )
    with CONFIG_JSON.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    _draw_preview(rows, front_store, PREVIEW_PNG)

    print("\nSelection locked successfully")
    print(f"  manifest: {MANIFEST_CSV}")
    print(f"  fronts:   {FRONTS_NPZ}")
    print(f"  preview:  {PREVIEW_PNG}")
    print(
        f"  rows={len(manifest_df)}, reuse={(manifest_df.fem_action == 'reuse').sum()}, "
        f"new={(manifest_df.fem_action == 'run').sum()}, "
        f"Pareto={manifest_df.is_pareto.sum()}, D1={(manifest_df.point_id == 'D1').sum()}"
    )


if __name__ == "__main__":
    main()
