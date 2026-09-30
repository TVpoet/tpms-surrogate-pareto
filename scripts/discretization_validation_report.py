# -*- coding: utf-8 -*-
"""Aggregate and plot the C3D8R/hourglass/body-fitted validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import L, get_thickness_parameter
from scripts.run_discretization_validation import (
    CASES,
    ENERGY_FLAG_PCT,
    OUTPUT_ROOT,
    SHELL_E_CONV_TOL_PCT,
    SHELL_SIGMA_CONV_TOL_PCT,
    sample_phi,
)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing validation result: {path}")
    return read_json(path)


def selected_shell_resolution() -> int:
    selection = OUTPUT_ROOT / "selected_shell_resolution.json"
    if selection.exists():
        return int(read_json(selection)["resolution"])
    return 160


def collect_hourglass_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case_id in ("A", "B"):
        for resolution in CASES[case_id]["voxel_resolutions"]:
            case_dir = OUTPUT_ROOT / "voxel" / f"{case_id}_n{resolution}"
            summary = require_summary(case_dir / "validation_summary.json")
            energy = summary["energy"]
            ratio_se = energy["ALLAE_over_ALLSE"]
            ratio_ie = energy["ALLAE_over_ALLIE"]
            rows.append(
                {
                    "case_id": case_id,
                    "description": CASES[case_id]["description"],
                    "n": int(resolution),
                    "element_type": summary["element_type"],
                    "element_count": summary["element_count"],
                    "node_count": summary["node_count"],
                    "rho_rel": summary["rho_rel"],
                    "E_eff_MPa": summary["E_eff_MPa"],
                    "sigma_hp_MPa": summary["sigma_hp"],
                    "ALLAE_final": energy["ALLAE"][-1][1] if energy["ALLAE"] else None,
                    "ALLSE_final": energy["ALLSE"][-1][1] if energy["ALLSE"] else None,
                    "ALLIE_final": energy["ALLIE"][-1][1] if energy["ALLIE"] else None,
                    "ALLAE_over_ALLSE_final": ratio_se["final"],
                    "ALLAE_over_ALLSE_stable_max": ratio_se["stable_max"],
                    "ALLAE_over_ALLIE_final": ratio_ie["final"],
                    "ALLAE_over_ALLIE_stable_max": ratio_ie["stable_max"],
                }
            )
    return rows


def collect_body_comparison_rows() -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    shell_resolution = selected_shell_resolution()
    selected: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    for case_id in ("A", "B"):
        voxel_dir = OUTPUT_ROOT / "voxel" / f"{case_id}_n80"
        shell_dir = OUTPUT_ROOT / "shell" / f"{case_id}_s{shell_resolution}"
        voxel = require_summary(voxel_dir / "validation_summary.json")
        shell = require_summary(shell_dir / "validation_summary.json")
        mesh = require_summary(shell_dir / "mesh_metrics.json")
        e_diff = (float(shell["E_eff_MPa"]) - float(voxel["E_eff_MPa"])) / float(voxel["E_eff_MPa"]) * 100.0
        sigma_diff = (float(shell["sigma_hp"]) - float(voxel["sigma_hp"])) / float(voxel["sigma_hp"]) * 100.0
        rho_diff = (float(shell["rho_rel"]) - float(voxel["rho_rel"])) / float(voxel["rho_rel"]) * 100.0
        selected[case_id] = {
            "voxel": voxel,
            "shell": shell,
            "mesh": mesh,
            "E_eff_difference_pct": e_diff,
            "sigma_hp_difference_pct": sigma_diff,
            "rho_difference_pct": rho_diff,
            "shell_resolution": shell_resolution,
        }
        for model, result in (("C3D8R voxel", voxel), ("S3 body-fitted shell", shell)):
            rows.append(
                {
                    "case_id": case_id,
                    "description": CASES[case_id]["description"],
                    "model": model,
                    "resolution": result["resolution"],
                    "element_type": result["element_type"],
                    "element_count": result["element_count"],
                    "node_count": result["node_count"],
                    "rho_rel": result["rho_rel"],
                    "E_eff_MPa": result["E_eff_MPa"],
                    "sigma_hp_MPa": result["sigma_hp"],
                    "sigma_max_MPa": result["sigma_max"],
                    "E_eff_difference_shell_vs_voxel_pct": e_diff if model.startswith("S3") else 0.0,
                    "sigma_hp_difference_shell_vs_voxel_pct": sigma_diff if model.startswith("S3") else 0.0,
                    "rho_difference_shell_vs_voxel_pct": rho_diff if model.startswith("S3") else 0.0,
                }
            )
    return rows, selected


def collect_mesh_audit_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case_id in ("A", "B"):
        for case_dir in sorted((OUTPUT_ROOT / "shell").glob(f"{case_id}_s*")):
            metrics_path = case_dir / "mesh_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = read_json(metrics_path)
            shift_counts = metrics.get("periodic_shift_counts", {})
            dat_path = case_dir / "tpms_mesh.dat"
            dat_text = dat_path.read_text(encoding="utf-8", errors="replace") if dat_path.exists() else ""
            distorted_match = re.search(r"\*\*\*WARNING:\s+(\d+) elements are distorted", dat_text)
            distorted_count = int(distorted_match.group(1)) if distorted_match else 0
            rows.append(
                {
                    "case_id": case_id,
                    "mesh_generator_version": metrics.get("mesh_generator_version", ""),
                    "surface_resolution": metrics["surface_resolution"],
                    "node_count": metrics["node_count"],
                    "element_count": metrics["element_count"],
                    "target_voxel_density_n80": metrics["target_voxel_density_n80"],
                    "shell_density": metrics["shell_density"],
                    "density_match_error_pct": metrics["density_match_error_pct"],
                    "surface_area_mm2": metrics["surface_area_mm2"],
                    "thickness_min_mm": metrics["thickness_min_mm"],
                    "thickness_p01_mm": metrics["thickness_p01_mm"],
                    "thickness_mean_mm": metrics["thickness_mean_mm"],
                    "thickness_max_mm": metrics["thickness_max_mm"],
                    "thickness_p01_in_n80_voxels": metrics["thickness_p01_in_n80_voxels"],
                    "thickness_median_in_n80_voxels": metrics["thickness_median_in_n80_voxels"],
                    "surface_fraction_below_2_voxels": metrics["surface_fraction_below_2_voxels"],
                    "surface_fraction_below_3_voxels": metrics["surface_fraction_below_3_voxels"],
                    "triangle_min_angle_min_deg": metrics["triangle_min_angle_min_deg"],
                    "triangle_min_angle_p01_deg": metrics["triangle_min_angle_p01_deg"],
                    "triangle_aspect_p99": metrics["triangle_aspect_p99"],
                    "triangle_aspect_max": metrics["triangle_aspect_max"],
                    "boundary_node_count": metrics["boundary_node_count"],
                    "periodic_class_count": metrics["periodic_class_count"],
                    "periodic_relation_count": metrics["periodic_relation_count"],
                    "max_periodic_coordinate_mismatch_mm": metrics["max_periodic_coordinate_mismatch_mm"],
                    "periodic_x_relations": shift_counts.get("x", 0),
                    "periodic_y_relations": shift_counts.get("y", 0),
                    "periodic_z_relations": shift_counts.get("z", 0),
                    "periodic_edge_relations": sum(shift_counts.get(key, 0) for key in ("xy", "xz", "yz")),
                    "periodic_corner_relations": shift_counts.get("xyz", 0),
                    "boundary_edge_count": metrics["boundary_edge_count"],
                    "nonmanifold_edge_count": metrics["nonmanifold_edge_count"],
                    "abaqus_error_count": dat_text.count("***ERROR"),
                    "abaqus_warning_count": dat_text.count("***WARNING"),
                    "abaqus_distorted_element_count": distorted_count,
                    "abaqus_distorted_element_fraction_pct": (
                        100.0 * distorted_count / float(metrics["element_count"])
                    ),
                }
            )
    return rows


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )


def style_axis(axis) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    axis.tick_params(axis="both", which="both", direction="in", top=False, right=False)


def panel_label(axis, text: str) -> None:
    axis.text(-0.12, 1.04, text, transform=axis.transAxes, ha="left", va="bottom", fontsize=11)


def plot_main_figure(hourglass_rows: List[Dict[str, object]], output_dir: Path) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(4.8, 3.55), constrained_layout=False)
    plt.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.97)

    a_rows = sorted((row for row in hourglass_rows if row["case_id"] == "A"), key=lambda row: row["n"])
    b_row = next(row for row in hourglass_rows if row["case_id"] == "B" and row["n"] == 80)
    axis.plot(
        [row["n"] for row in a_rows],
        [100.0 * float(row["ALLAE_over_ALLSE_final"]) for row in a_rows],
        marker="o",
        color="#1f77b4",
        linewidth=1.4,
        markersize=5,
        label="Configuration A",
    )
    axis.scatter(
        [80],
        [100.0 * float(b_row["ALLAE_over_ALLSE_final"])],
        marker="s",
        s=32,
        color="#d95f02",
        zorder=3,
        label=r"Configuration B ($n=80$)",
    )
    axis.set_xlabel(r"Voxel resolution, $n$")
    axis.set_ylabel(r"$\mathrm{ALLAE}/\mathrm{ALLSE}$ (%)")
    axis.set_xticks([60, 80, 100, 120])
    axis.set_xlim(56, 124)
    axis.set_ylim(0.0, 0.8)
    axis.legend(frameon=False, loc="best")
    style_axis(axis)

    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"mesh_robustness_validation.{suffix}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def central_slice(alpha: List[float], resolution: int = 401) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 2.0 * np.pi, resolution)
    x, y = np.meshgrid(phase, phase, indexing="ij")
    z = np.pi
    sx, sy, sz = np.sin(x), np.sin(y), math.sin(z)
    cx, cy, cz = np.cos(x), np.cos(y), math.cos(z)
    phi = (
        alpha[0] * (sx + sy + sz)
        + alpha[1] * (cx * sx + cy * sy + cz * sz)
        + alpha[2] * (cx * sy + cy * sz + cz * sx)
        + alpha[3] * (cx * sz + cy * sx + cz * sy)
    )
    coordinate_mm = phase / (2.0 * np.pi) * (L * 1000.0)
    return coordinate_mm, coordinate_mm, phi


def plot_slice_comparison(output_dir: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), constrained_layout=False)
    plt.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.965, wspace=0.18, hspace=0.22)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for row, case_id in enumerate(("A", "B")):
        alpha = [float(value) for value in CASES[case_id]["alpha"]]
        x, y, phi = central_slice(alpha)
        threshold = get_thickness_parameter() / 2.0

        node_phase = np.linspace(0.0, 2.0 * np.pi, 80)
        centre_phase = 0.5 * (node_phase[:-1] + node_phase[1:])
        xx, yy = np.meshgrid(centre_phase, centre_phase, indexing="ij")
        zz = np.pi
        sx, sy, sz = np.sin(xx), np.sin(yy), math.sin(zz)
        cx, cy, cz = np.cos(xx), np.cos(yy), math.cos(zz)
        voxel_phi = (
            alpha[0] * (sx + sy + sz)
            + alpha[1] * (cx * sx + cy * sy + cz * sz)
            + alpha[2] * (cx * sy + cy * sz + cz * sx)
            + alpha[3] * (cx * sz + cy * sx + cz * sy)
        )
        occupancy = np.abs(voxel_phi) <= threshold
        extent = (0.0, L * 1000.0, 0.0, L * 1000.0)

        axis = axes[row, 0]
        axis.imshow(occupancy.T, origin="lower", extent=extent, interpolation="nearest", cmap="Greys")
        axis.set_xlabel(r"$x$ (mm)")
        axis.set_ylabel(r"$y$ (mm)")
        axis.set_aspect("equal")
        style_axis(axis)
        panel_label(axis, labels[2 * row])

        axis = axes[row, 1]
        axis.contourf(x, y, (np.abs(phi) <= threshold).T, levels=[0.5, 1.5], colors=["0.78"])
        axis.contour(x, y, phi.T, levels=[-threshold, threshold], colors=["black"], linewidths=0.8)
        axis.contour(x, y, phi.T, levels=[0.0], colors=["#d95f02"], linewidths=1.0)
        axis.set_xlabel(r"$x$ (mm)")
        axis.set_ylabel(r"$y$ (mm)")
        axis.set_aspect("equal")
        style_axis(axis)
        panel_label(axis, labels[2 * row + 1])

    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"voxel_vs_body_fitted_slice.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(figure)


def voxel_mask(case_id: str, n_grid: int = 80) -> np.ndarray:
    alpha = [float(value) for value in CASES[case_id]["alpha"]]
    node_phase = np.linspace(0.0, 2.0 * np.pi, n_grid)
    phase = 0.5 * (node_phase[:-1] + node_phase[1:])
    s = np.sin(phase)
    c = np.cos(phase)
    sx, sy, sz = s[:, None, None], s[None, :, None], s[None, None, :]
    cx, cy, cz = c[:, None, None], c[None, :, None], c[None, None, :]
    phi = (
        alpha[0] * (sx + sy + sz)
        + alpha[1] * (cx * sx + cy * sy + cz * sz)
        + alpha[2] * (cx * sy + cy * sz + cz * sx)
        + alpha[3] * (cx * sz + cy * sx + cz * sy)
    )
    return np.abs(phi) <= (get_thickness_parameter() / 2.0)


def plot_mesh_appearance(output_dir: Path) -> None:
    """Render actual voxel boundaries and S3 mid-surface meshes for review."""
    import pyvista as pv

    pv.global_theme.font.family = "times"
    shell_resolution = selected_shell_resolution()
    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(2200, 1900), border=False)
    plotter.set_background("white")
    camera_position = [(17.5, 17.5, 14.0), (5.0, 5.0, 5.0), (0.0, 0.0, 1.0)]
    panel_names = ["(a)", "(b)", "(c)", "(d)"]

    for row, case_id in enumerate(("A", "B")):
        mask = voxel_mask(case_id, 80)
        spacing_mm = L * 1000.0 / 79.0
        grid = pv.ImageData(
            dimensions=(80, 80, 80),
            spacing=(spacing_mm, spacing_mm, spacing_mm),
            origin=(0.0, 0.0, 0.0),
        )
        grid.cell_data["solid"] = mask.astype(np.uint8).ravel(order="F")
        voxel_surface = grid.threshold(0.5, scalars="solid", preference="cell").extract_surface()

        shell_path = OUTPUT_ROOT / "shell" / f"{case_id}_s{shell_resolution}" / "shell_mesh_preview.npz"
        with np.load(shell_path) as shell_data:
            shell_vertices = np.asarray(shell_data["vertices"], dtype=np.float64)
            shell_faces = np.asarray(shell_data["faces"], dtype=np.int64)
        face_array = np.column_stack(
            (np.full(len(shell_faces), 3, dtype=np.int64), shell_faces)
        ).ravel()
        shell_surface = pv.PolyData(shell_vertices, face_array)

        plotter.subplot(row, 0)
        plotter.add_mesh(
            voxel_surface,
            color="#c7c7c7",
            show_edges=True,
            edge_color="#333333",
            line_width=0.35,
            smooth_shading=False,
        )
        plotter.add_text(panel_names[2 * row], position="upper_left", font_size=18, color="black")
        plotter.camera_position = camera_position
        plotter.enable_parallel_projection()

        plotter.subplot(row, 1)
        plotter.add_mesh(
            shell_surface,
            color="#d9d9d9",
            show_edges=True,
            edge_color="#555555",
            line_width=0.25,
            smooth_shading=False,
        )
        plotter.add_text(panel_names[2 * row + 1], position="upper_left", font_size=18, color="black")
        plotter.camera_position = camera_position
        plotter.enable_parallel_projection()

    plotter.screenshot(str(output_dir / "voxel_vs_body_fitted_mesh.png"))
    plotter.close()


def build_decision_summary(
    hourglass_rows: List[Dict[str, object]], selected: Dict[str, Dict[str, object]]
) -> Dict[str, object]:
    reproduction: List[Dict[str, object]] = []
    for row in hourglass_rows:
        comparison_path = (
            OUTPUT_ROOT
            / "voxel"
            / f"{row['case_id']}_n{row['n']}"
            / "stored_reference_comparison.json"
        )
        if comparison_path.exists():
            comparison = read_json(comparison_path)
            reproduction.append(
                {
                    "case_id": row["case_id"],
                    "n": row["n"],
                    "E_eff_difference_pct": comparison["E_eff_MPa"]["difference_pct"],
                    "sigma_hp_difference_pct": comparison["sigma_hp"]["difference_pct"],
                    "passes_0p5pct": comparison["passes_0p5pct_E_and_sigma"],
                }
            )

    energy_cases = []
    for row in hourglass_rows:
        stable_pct = 100.0 * float(row["ALLAE_over_ALLSE_stable_max"])
        energy_cases.append(
            {
                "case_id": row["case_id"],
                "n": row["n"],
                "ALLAE_over_ALLSE_stable_max_pct": stable_pct,
                "passes_5pct": stable_pct <= ENERGY_FLAG_PCT,
            }
        )

    convergence_path = OUTPUT_ROOT / "shell" / "A_shell_convergence.json"
    convergence = read_json(convergence_path)
    comparisons = {}
    for case_id in ("A", "B"):
        item = selected[case_id]
        e_difference = float(item["E_eff_difference_pct"])
        sigma_difference = float(item["sigma_hp_difference_pct"])
        comparisons[case_id] = {
            "shell_resolution": item["shell_resolution"],
            "density_difference_pct": item["rho_difference_pct"],
            "E_eff_difference_pct": e_difference,
            "sigma_hp_difference_pct": sigma_difference,
            "passes_E_5pct": abs(e_difference) <= 5.0,
            "passes_sigma_hp_10pct": abs(sigma_difference) <= 10.0,
        }

    return {
        "stored_voxel_reproduction": reproduction,
        "stored_voxel_reproduction_all_pass": all(item["passes_0p5pct"] for item in reproduction),
        "hourglass_energy": energy_cases,
        "hourglass_all_pass_5pct": all(item["passes_5pct"] for item in energy_cases),
        "full_integration_follow_up_required": any(not item["passes_5pct"] for item in energy_cases),
        "shell_convergence": {
            **convergence,
            "E_eff_tolerance_pct": SHELL_E_CONV_TOL_PCT,
            "sigma_hp_tolerance_pct": SHELL_SIGMA_CONV_TOL_PCT,
        },
        "body_fitted_comparison": comparisons,
        "interpretation_guardrail": (
            "A failed the internal body-fitted equivalence targets despite converged S3 meshes; "
            "do not attribute its difference solely to voxel staircasing without further topology/thickness checks."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hourglass_rows = collect_hourglass_rows()
    body_rows, selected = collect_body_comparison_rows()
    mesh_rows = collect_mesh_audit_rows()

    write_csv(output_dir / "hourglass_energy_summary.csv", hourglass_rows, hourglass_rows[0].keys())
    write_csv(output_dir / "body_fitted_comparison.csv", body_rows, body_rows[0].keys())
    if mesh_rows:
        write_csv(output_dir / "mesh_quality_audit.csv", mesh_rows, mesh_rows[0].keys())

    write_json(output_dir / "validation_decision_summary.json", build_decision_summary(hourglass_rows, selected))

    plot_main_figure(hourglass_rows, output_dir)
    plot_slice_comparison(output_dir)
    plot_mesh_appearance(output_dir)
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
