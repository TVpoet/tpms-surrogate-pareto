# -*- coding: utf-8 -*-
"""Run the reviewer-requested C3D8R/hourglass and body-fitted-mesh checks.

This script is intentionally isolated from the production/training pipeline.  It
reuses the production voxel generator without changing its material, element,
geometry, or periodic boundary conditions, and only adds whole-model energy
history output.  The independent comparison model is a periodic, body-fitted
S3 shell mesh generated from the phi=0 midsurface.

Examples
--------
Generate, solve, and extract all planned cases::

    python scripts/run_discretization_validation.py --stage all --cpus 4 --resume

Generate inputs and mesh audits without starting Abaqus::

    python scripts/run_discretization_validation.py --stage all --generate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from string import Template
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import E_SOLID, L, NU, WALL_THICKNESS_MM, get_thickness_parameter
from scripts import batch_simulate_extract as bse


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "discretization_validation"
JOB_NAME = "tpms_mesh"
PERCENTILE = 99.9
ENERGY_FLAG_PCT = 5.0
REFERENCE_TOL_PCT = 0.5
SHELL_E_CONV_TOL_PCT = 2.0
SHELL_SIGMA_CONV_TOL_PCT = 5.0
SHELL_DENSITY_TOL_PCT = 0.5
N_THICKNESS_BINS = 20
SHELL_MESH_GENERATOR_VERSION = "surface_nets_v3"
BOUNDARY_SNAP_FRACTION = 1.5e-2


CASES: Dict[str, Dict[str, object]] = {
    "A": {
        "description": "Fig. 5(c) mesh-convergence hybrid",
        "alpha": [0.25, 0.50, 0.10, 0.15],
        "displacement_m": [2.0e-6, 2.0e-6, 2.0e-6],
        "voxel_resolutions": [60, 80, 120],
    },
    "B": {
        "description": "Triaxial-1-2-2 Pareto P1 low-stress hybrid",
        "alpha": [
            4.883814807129439e-05,
            0.26822872470354847,
            0.158362638787156,
            0.5733597983612242,
        ],
        "displacement_m": [1.0e-6, 2.0e-6, 2.0e-6],
        "voxel_resolutions": [80],
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sample_phi_on_phase(alpha: Sequence[float], phase: np.ndarray) -> np.ndarray:
    """Evaluate the four-function hybrid field at prescribed phase coordinates."""
    s = np.sin(phase)
    c = np.cos(phase)

    sx, sy, sz = s[:, None, None], s[None, :, None], s[None, None, :]
    cx, cy, cz = c[:, None, None], c[None, :, None], c[None, None, :]

    f1 = sx + sy + sz
    f2 = cx * sx + cy * sy + cz * sz
    f3 = cx * sy + cy * sz + cz * sx
    f4 = cx * sz + cy * sx + cz * sy
    phi = alpha[0] * f1 + alpha[1] * f2 + alpha[2] * f3 + alpha[3] * f4
    return np.asarray(phi, dtype=np.float32)


def sample_phi(alpha: Sequence[float], n_grid: int) -> np.ndarray:
    """Evaluate the four-function hybrid field on a periodic structured grid."""
    phase = np.linspace(0.0, 2.0 * np.pi, n_grid, dtype=np.float64)
    return sample_phi_on_phase(alpha, phase)


def sample_phi_with_ghost_layer(alpha: Sequence[float], n_grid: int) -> np.ndarray:
    """Evaluate one periodic grid layer beyond every physical cell boundary.

    The physical planes are interior marching-cubes cuts rather than the outer
    boundary of the sampled array.  Consequently, opposite cut faces inherit
    identical periodic triangulations and can be paired node-for-node.
    """
    phase_step = 2.0 * np.pi / float(n_grid - 1)
    phase = np.arange(-1, n_grid + 1, dtype=np.float64) * phase_step
    return sample_phi_on_phase(alpha, phase)


def surface_nets(
    scalar_field: np.ndarray,
    spacing_mm: float,
    origin_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a triangle mesh with the dual Surface Nets construction.

    One vertex is placed at the average of all zero crossings in every active
    grid cell.  Quads dual to sign-changing grid edges are split along the
    diagonal that maximises the worse of the two triangle-shape scores.
    Compared with marching-cubes edge vertices, this avoids
    arbitrarily short edges when the zero level passes very near a grid node,
    which is important for Abaqus S3 element-quality limits.
    """
    scalar_field = np.asarray(scalar_field, dtype=np.float64)
    if scalar_field.ndim != 3 or min(scalar_field.shape) < 3:
        raise ValueError("Surface Nets requires a three-dimensional scalar grid")

    negative = scalar_field < 0.0
    cell_negative = [
        negative[ix : negative.shape[0] - 1 + ix, iy : negative.shape[1] - 1 + iy, iz : negative.shape[2] - 1 + iz]
        for ix in (0, 1)
        for iy in (0, 1)
        for iz in (0, 1)
    ]
    all_negative = np.logical_and.reduce(cell_negative)
    all_positive = np.logical_and.reduce([~values for values in cell_negative])
    active = ~(all_negative | all_positive)
    active_cells = np.argwhere(active)
    if len(active_cells) == 0:
        raise ValueError("Surface Nets found no zero-crossing cells")

    vertex_sums = np.zeros((len(active_cells), 3), dtype=np.float64)
    crossing_counts = np.zeros(len(active_cells), dtype=np.int16)
    edge_starts: List[Tuple[int, int, int, int]] = []
    for axis in range(3):
        transverse = [value for value in range(3) if value != axis]
        for first in (0, 1):
            for second in (0, 1):
                offset = [0, 0, 0]
                offset[transverse[0]] = first
                offset[transverse[1]] = second
                edge_starts.append((axis, offset[0], offset[1], offset[2]))

    for axis, ox, oy, oz in edge_starts:
        starts = active_cells + np.array([ox, oy, oz], dtype=np.int64)
        ends = starts.copy()
        ends[:, axis] += 1
        start_values = scalar_field[starts[:, 0], starts[:, 1], starts[:, 2]]
        end_values = scalar_field[ends[:, 0], ends[:, 1], ends[:, 2]]
        crosses = (start_values < 0.0) != (end_values < 0.0)
        selected = np.where(crosses)[0]
        if len(selected) == 0:
            continue
        denominator = start_values[selected] - end_values[selected]
        fraction = start_values[selected] / denominator
        points = starts[selected].astype(np.float64)
        points[:, axis] += fraction
        points = origin_mm + spacing_mm * points
        vertex_sums[selected] += points
        crossing_counts[selected] += 1

    if np.any(crossing_counts == 0):
        raise ValueError("An active Surface Nets cell has no sign-changing edge")
    vertices = vertex_sums / crossing_counts[:, None]

    cell_ids = np.full(active.shape, -1, dtype=np.int64)
    cell_ids[active_cells[:, 0], active_cells[:, 1], active_cells[:, 2]] = np.arange(
        len(active_cells), dtype=np.int64
    )
    face_blocks: List[np.ndarray] = []

    def append_quads(quads: np.ndarray, forward: np.ndarray) -> None:
        if len(quads) == 0:
            return
        valid = np.all(quads >= 0, axis=1)
        quads = quads[valid]
        forward = forward[valid]
        if len(quads) == 0:
            return
        reversed_quads = quads[:, [0, 3, 2, 1]]
        quads = np.where(forward[:, None], quads, reversed_quads)
        def triangle_shape_score(triangles: np.ndarray) -> np.ndarray:
            a = vertices[triangles[:, 0]]
            b = vertices[triangles[:, 1]]
            c = vertices[triangles[:, 2]]
            area_twice = np.linalg.norm(np.cross(b - a, c - a), axis=1)
            squared_edges = (
                np.sum((b - a) ** 2, axis=1)
                + np.sum((c - b) ** 2, axis=1)
                + np.sum((a - c) ** 2, axis=1)
            )
            return 2.0 * np.sqrt(3.0) * area_twice / np.maximum(squared_edges, 1.0e-30)

        option_02_a = quads[:, [0, 1, 2]]
        option_02_b = quads[:, [0, 2, 3]]
        option_13_a = quads[:, [0, 1, 3]]
        option_13_b = quads[:, [1, 2, 3]]
        score_02 = np.minimum(
            triangle_shape_score(option_02_a), triangle_shape_score(option_02_b)
        )
        score_13 = np.minimum(
            triangle_shape_score(option_13_a), triangle_shape_score(option_13_b)
        )
        use_02 = score_02 >= score_13
        faces_02 = np.vstack(
            (
                quads[use_02][:, [0, 1, 2]],
                quads[use_02][:, [0, 2, 3]],
            )
        )
        faces_13 = np.vstack(
            (
                quads[~use_02][:, [0, 1, 3]],
                quads[~use_02][:, [1, 2, 3]],
            )
        )
        if len(faces_02):
            face_blocks.append(faces_02)
        if len(faces_13):
            face_blocks.append(faces_13)

    # x-directed grid edges; quad order has a +x normal.
    crossing = negative[:-1, 1:-1, 1:-1] != negative[1:, 1:-1, 1:-1]
    i, j0, k0 = np.where(crossing)
    quads = np.column_stack(
        (
            cell_ids[i, j0, k0],
            cell_ids[i, j0 + 1, k0],
            cell_ids[i, j0 + 1, k0 + 1],
            cell_ids[i, j0, k0 + 1],
        )
    )
    append_quads(quads, scalar_field[i, j0 + 1, k0 + 1] < scalar_field[i + 1, j0 + 1, k0 + 1])

    # y-directed grid edges; quad order has a +y normal.
    crossing = negative[1:-1, :-1, 1:-1] != negative[1:-1, 1:, 1:-1]
    i0, j, k0 = np.where(crossing)
    quads = np.column_stack(
        (
            cell_ids[i0, j, k0],
            cell_ids[i0, j, k0 + 1],
            cell_ids[i0 + 1, j, k0 + 1],
            cell_ids[i0 + 1, j, k0],
        )
    )
    append_quads(quads, scalar_field[i0 + 1, j, k0 + 1] < scalar_field[i0 + 1, j + 1, k0 + 1])

    # z-directed grid edges; quad order has a +z normal.
    crossing = negative[1:-1, 1:-1, :-1] != negative[1:-1, 1:-1, 1:]
    i0, j0, k = np.where(crossing)
    quads = np.column_stack(
        (
            cell_ids[i0, j0, k],
            cell_ids[i0 + 1, j0, k],
            cell_ids[i0 + 1, j0 + 1, k],
            cell_ids[i0, j0 + 1, k],
        )
    )
    append_quads(quads, scalar_field[i0 + 1, j0 + 1, k] < scalar_field[i0 + 1, j0 + 1, k + 1])

    if not face_blocks:
        raise ValueError("Surface Nets generated no faces")
    faces = np.concatenate(face_blocks, axis=0).astype(np.int64, copy=False)
    return vertices, faces


def _clip_polygon_to_axis_plane(
    polygon: List[np.ndarray],
    axis: int,
    bound: float,
    keep_greater: bool,
    tolerance: float,
) -> List[np.ndarray]:
    """Clip one convex polygon against one axis-aligned half-space."""
    if not polygon:
        return []

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return bool(point[axis] >= bound - tolerance)
        return bool(point[axis] <= bound + tolerance)

    clipped: List[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = float(current[axis] - previous[axis])
            if abs(denominator) > 1.0e-30:
                fraction = (bound - float(previous[axis])) / denominator
                intersection = previous + fraction * (current - previous)
                intersection = np.asarray(intersection, dtype=np.float64)
                intersection[axis] = bound
                clipped.append(intersection)
        if current_inside:
            kept = np.asarray(current, dtype=np.float64).copy()
            if abs(float(kept[axis]) - bound) <= tolerance:
                kept[axis] = bound
            clipped.append(kept)
        previous = current
        previous_inside = current_inside
    return clipped


def clip_triangular_surface_to_cell(
    vertices: np.ndarray,
    faces: np.ndarray,
    cell_size_mm: float,
    spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip a ghost-layer triangular surface to the closed physical unit cell.

    A small coordinate-key tolerance merges the identical intersection points
    produced independently by adjacent triangles without collapsing legitimate
    nearby marching-cubes vertices.
    """
    # skimage returns float32 vertices; after conversion to float64, nominal
    # grid planes can be displaced by a few 1e-7 mm.  Snap those coordinates
    # before clipping, but retain a strict half-space test so outside triangles
    # are not accidentally kept as finite-area boundary triangles.
    # Project a Surface-Nets vertex to the cut plane when it lies within 1.5%
    # of one grid spacing.  This removes boundary sliver triangles while the
    # maximum geometric adjustment remains negligible relative to the cell.
    snap_tolerance = max(1.0e-8, spacing_mm * BOUNDARY_SNAP_FRACTION)
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    vertices[np.abs(vertices) <= snap_tolerance] = 0.0
    vertices[np.abs(vertices - cell_size_mm) <= snap_tolerance] = cell_size_mm
    plane_tolerance = max(1.0e-12, spacing_mm * 1.0e-10)
    merge_tolerance = max(1.0e-11, spacing_mm * 1.0e-8)
    output_vertices: List[np.ndarray] = []
    output_faces: List[Tuple[int, int, int]] = []
    vertex_lookup: Dict[Tuple[int, int, int], int] = {}

    def register(point: np.ndarray) -> int:
        point = np.asarray(point, dtype=np.float64).copy()
        point[np.abs(point) <= plane_tolerance] = 0.0
        point[np.abs(point - cell_size_mm) <= plane_tolerance] = cell_size_mm
        key = tuple(int(value) for value in np.rint(point / merge_tolerance))
        existing = vertex_lookup.get(key)
        if existing is not None:
            return existing
        index = len(output_vertices)
        vertex_lookup[key] = index
        output_vertices.append(point)
        return index

    for face in faces:
        triangle = vertices[np.asarray(face, dtype=np.int64)]
        if np.any(np.max(triangle, axis=0) < -plane_tolerance):
            continue
        if np.any(np.min(triangle, axis=0) > cell_size_mm + plane_tolerance):
            continue
        polygon = [triangle[0], triangle[1], triangle[2]]
        for axis in range(3):
            polygon = _clip_polygon_to_axis_plane(
                polygon, axis, 0.0, True, plane_tolerance
            )
            polygon = _clip_polygon_to_axis_plane(
                polygon, axis, cell_size_mm, False, plane_tolerance
            )
            if len(polygon) < 3:
                break
        if len(polygon) < 3:
            continue

        root = register(polygon[0])
        for offset in range(1, len(polygon) - 1):
            triangle_ids = (root, register(polygon[offset]), register(polygon[offset + 1]))
            if len(set(triangle_ids)) < 3:
                continue
            a, b, c = (output_vertices[index] for index in triangle_ids)
            area_twice = float(np.linalg.norm(np.cross(b - a, c - a)))
            if area_twice > 1.0e-12:
                output_faces.append(triangle_ids)

    if not output_faces:
        raise ValueError("Ghost-layer surface clipping generated no triangles")
    return np.asarray(output_vertices, dtype=np.float64), np.asarray(output_faces, dtype=np.int64)


def voxel_density(alpha: Sequence[float], n_grid: int = 80) -> float:
    """Reproduce the production centre-sampling definition of relative density."""
    phase_nodes = np.linspace(0.0, 2.0 * np.pi, n_grid, dtype=np.float64)
    phase = 0.5 * (phase_nodes[:-1] + phase_nodes[1:])
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
    mask = np.abs(phi) <= (get_thickness_parameter() / 2.0)
    return float(np.mean(mask))


def gradient_norm_at_points(points_mm: np.ndarray, alpha: Sequence[float]) -> np.ndarray:
    """Analytic |grad(phi)| with coordinates in metres (result unit 1/m)."""
    points_m = points_mm / 1000.0
    k = 2.0 * np.pi / L
    x, y, z = (k * points_m[:, i] for i in range(3))
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)

    df1x, df1y, df1z = k * cx, k * cy, k * cz
    df2x, df2y, df2z = k * np.cos(2.0 * x), k * np.cos(2.0 * y), k * np.cos(2.0 * z)
    df3x = k * (-sx * sy + cz * cx)
    df3y = k * (cx * cy - sy * sz)
    df3z = k * (cy * cz - sz * sx)
    df4x = k * (-sx * sz + cy * cx)
    df4y = k * (-sy * sx + cz * cy)
    df4z = k * (cx * cz - sz * sy)

    gx = alpha[0] * df1x + alpha[1] * df2x + alpha[2] * df3x + alpha[3] * df4x
    gy = alpha[0] * df1y + alpha[1] * df2y + alpha[2] * df3y + alpha[3] * df4y
    gz = alpha[0] * df1z + alpha[1] * df2z + alpha[2] * df3z + alpha[3] * df4z
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, np.ndarray]:
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def angle(opposite: np.ndarray, side1: np.ndarray, side2: np.ndarray) -> np.ndarray:
        cosine = (side1 * side1 + side2 * side2 - opposite * opposite) / (
            2.0 * side1 * side2 + 1.0e-30
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = np.vstack((angle(bc, ab, ca), angle(ca, ab, bc), angle(ab, bc, ca))).T
    min_angles = np.min(angles, axis=1)
    semiperimeter = 0.5 * (ab + bc + ca)
    inradius = areas / np.maximum(semiperimeter, 1.0e-30)
    aspect = np.maximum.reduce((ab, bc, ca)) / np.maximum(2.0 * inradius, 1.0e-30)
    return {"areas": areas, "min_angles": min_angles, "aspect_ratios": aspect}


def assign_thickness_bins(
    areas: np.ndarray,
    raw_thickness_mm: np.ndarray,
    target_volume_mm3: float,
    n_bins: int = N_THICKNESS_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-average local shell thickness into equal-count bins and mass match."""
    if len(raw_thickness_mm) < n_bins:
        n_bins = max(1, len(raw_thickness_mm))
    order = np.argsort(raw_thickness_mm)
    chunks = np.array_split(order, n_bins)
    bin_ids = np.zeros(len(raw_thickness_mm), dtype=np.int32)
    bin_values = np.zeros(len(chunks), dtype=np.float64)
    for bin_id, element_ids in enumerate(chunks):
        weights = areas[element_ids]
        bin_values[bin_id] = float(np.average(raw_thickness_mm[element_ids], weights=weights))
        bin_ids[element_ids] = bin_id

    represented_volume = float(np.sum(areas * bin_values[bin_ids]))
    if represented_volume <= 0.0:
        raise ValueError("Non-positive represented shell volume")
    correction = target_volume_mm3 / represented_volume
    bin_values *= correction
    element_thickness = bin_values[bin_ids]
    return bin_ids, bin_values, element_thickness


def periodic_equivalence_relations(
    vertices: np.ndarray, cell_size_mm: float, spacing_mm: float
) -> Tuple[List[Tuple[int, int, Tuple[int, int, int]]], Dict[str, object]]:
    """Create independent periodic relations from exact opposite-face pairings.

    Quantising all canonical boundary coordinates at once is unsafe because a
    marching-cubes surface can contain two distinct same-face vertices that are
    much closer than the nominal grid spacing.  Instead, pair each pair of
    opposite faces one-to-one in its transverse plane, then merge the three sets
    of pairings with a union-find structure.  Each resulting equivalence class
    yields the minimum number of independent equations, including at cube edges
    and corners.
    """
    # Clipping writes the six physical planes exactly.  Keep this tolerance
    # much smaller than a voxel so near-boundary interior vertices are not
    # mistaken for periodic boundary nodes.
    boundary_tol = max(1.0e-9, spacing_mm * 1.0e-7)
    pairing_tol = max(1.0e-8, spacing_mm * 2.0e-5)
    snapped = vertices.copy()
    snapped[np.abs(snapped) <= boundary_tol] = 0.0
    snapped[np.abs(snapped - cell_size_mm) <= boundary_tol] = cell_size_mm

    low = np.isclose(snapped, 0.0, atol=boundary_tol, rtol=0.0)
    high = np.isclose(snapped, cell_size_mm, atol=boundary_tol, rtol=0.0)
    boundary = np.any(low | high, axis=1)
    boundary_ids = np.where(boundary)[0]

    parent = np.arange(len(vertices), dtype=np.int64)

    def find(node: int) -> int:
        root = int(node)
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    face_pair_counts: Dict[str, int] = {}
    face_pair_max_mismatch: Dict[str, float] = {}
    for axis, axis_name in enumerate("xyz"):
        min_ids = np.where(low[:, axis])[0]
        max_ids = np.where(high[:, axis])[0]
        if len(min_ids) != len(max_ids):
            raise ValueError(
                f"Periodic {axis_name}-face node counts differ: min={len(min_ids)}, max={len(max_ids)}"
            )
        transverse = [value for value in range(3) if value != axis]
        tree = cKDTree(snapped[max_ids][:, transverse])
        distances, neighbours = tree.query(snapped[min_ids][:, transverse], k=1)
        if len(set(int(value) for value in neighbours)) != len(neighbours):
            raise ValueError(f"Periodic {axis_name}-face nearest-neighbour matching is not one-to-one")
        maximum = float(np.max(distances)) if len(distances) else 0.0
        if maximum > pairing_tol:
            raise ValueError(
                f"Periodic {axis_name}-face mismatch {maximum:.6g} mm exceeds tolerance {pairing_tol:.6g} mm"
            )
        for local_index, neighbour in enumerate(neighbours):
            union(int(min_ids[local_index]), int(max_ids[int(neighbour)]))
        face_pair_counts[axis_name] = int(len(min_ids))
        face_pair_max_mismatch[axis_name] = maximum

    groups: Dict[int, List[int]] = defaultdict(list)
    for node_id in boundary_ids:
        groups[find(int(node_id))].append(int(node_id))
    singleton_groups = [ids for ids in groups.values() if len(ids) < 2]
    if singleton_groups:
        raise ValueError(
            "Periodic shell boundary contains unmatched nodes; "
            f"{len(singleton_groups)} singleton classes"
        )

    relations: List[Tuple[int, int, Tuple[int, int, int]]] = []
    max_canonical_mismatch = 0.0
    shift_counts = {"x": 0, "y": 0, "z": 0, "xy": 0, "xz": 0, "yz": 0, "xyz": 0}
    for ids in groups.values():
        ids_sorted = sorted(ids, key=lambda idx: (int(np.sum(high[idx])), idx))
        master = ids_sorted[0]
        master_canonical = snapped[master].copy()
        master_canonical[
            np.isclose(master_canonical, cell_size_mm, atol=boundary_tol, rtol=0.0)
        ] = 0.0
        for slave in ids_sorted[1:]:
            slave_canonical = snapped[slave].copy()
            slave_canonical[
                np.isclose(slave_canonical, cell_size_mm, atol=boundary_tol, rtol=0.0)
            ] = 0.0
            mismatch = float(np.linalg.norm(slave_canonical - master_canonical))
            max_canonical_mismatch = max(max_canonical_mismatch, mismatch)
            shift = np.rint((snapped[slave] - snapped[master]) / cell_size_mm).astype(int)
            if np.any((shift < 0) | (shift > 1)):
                raise ValueError(f"Invalid periodic shift {shift.tolist()} for nodes {master + 1}, {slave + 1}")
            shift_tuple = tuple(int(v) for v in shift)
            if shift_tuple == (0, 0, 0):
                raise ValueError(f"Duplicate nodes in periodic class: {master + 1}, {slave + 1}")
            axes = "".join(axis for axis, active in zip("xyz", shift_tuple) if active)
            shift_counts[axes] += 1
            relations.append((slave + 1, master + 1, shift_tuple))

    if max_canonical_mismatch > pairing_tol * 2.0:
        raise ValueError(
            f"Periodic transverse-coordinate mismatch {max_canonical_mismatch:.6g} mm "
            f"exceeds tolerance {pairing_tol * 2.0:.6g} mm"
        )

    vertices[:, :] = snapped
    audit = {
        "boundary_node_count": int(len(boundary_ids)),
        "periodic_class_count": int(len(groups)),
        "periodic_relation_count": int(len(relations)),
        "max_periodic_coordinate_mismatch_mm": max_canonical_mismatch,
        "boundary_tolerance_mm": boundary_tol,
        "pairing_tolerance_mm": pairing_tol,
        "face_pair_counts": face_pair_counts,
        "face_pair_max_mismatch_mm": face_pair_max_mismatch,
        "periodic_shift_counts": shift_counts,
    }
    return relations, audit


def write_id_list(stream, ids: Iterable[int], width: int = 16) -> None:
    values = list(ids)
    for start in range(0, len(values), width):
        stream.write(", ".join(str(v) for v in values[start : start + width]) + "\n")


def write_equation(
    stream,
    slave: int,
    master: int,
    dof: int,
    rp_id: int,
    macro_dof: int | None,
) -> None:
    terms = [(slave, dof, 1.0), (master, dof, -1.0)]
    if macro_dof is not None:
        terms.append((rp_id, macro_dof, -1.0))
    stream.write("*Equation\n")
    stream.write(f"{len(terms)}\n")
    for node, term_dof, coefficient in terms:
        stream.write(f"TPMS-1.{node}, {term_dof}, {coefficient:.1f}\n")


def generate_shell_input(
    case_id: str,
    resolution: int,
    case_dir: Path,
) -> Dict[str, object]:
    case = CASES[case_id]
    alpha = [float(v) for v in case["alpha"]]
    displacement_m = [float(v) for v in case["displacement_m"]]
    cell_size_mm = L * 1000.0
    spacing_mm = cell_size_mm / float(resolution - 1)
    phi = sample_phi_with_ghost_layer(alpha, resolution)
    vertices, faces = surface_nets(phi, spacing_mm, -spacing_mm)
    del phi
    vertices, faces = clip_triangular_surface_to_cell(
        vertices, faces, cell_size_mm, spacing_mm
    )

    quality = triangle_quality(vertices, faces)
    valid = quality["areas"] > 1.0e-12
    if not np.all(valid):
        faces = faces[valid]
        quality = triangle_quality(vertices, faces)
    if len(faces) == 0:
        raise ValueError("Marching cubes generated no valid shell elements")

    relations, periodic_audit = periodic_equivalence_relations(vertices, cell_size_mm, spacing_mm)

    centroids = np.mean(vertices[faces], axis=1)
    grad_norm = gradient_norm_at_points(centroids, alpha)
    if np.any(~np.isfinite(grad_norm)) or float(np.min(grad_norm)) <= 1.0e-9:
        raise ValueError("The selected midsurface contains a zero/invalid implicit-field gradient")
    raw_thickness_mm = get_thickness_parameter() / grad_norm * 1000.0
    if np.any(~np.isfinite(raw_thickness_mm)) or np.any(raw_thickness_mm <= 0.0):
        raise ValueError("Invalid local shell thickness derived from the implicit band")
    if float(np.max(raw_thickness_mm)) > cell_size_mm / 2.0:
        raise ValueError("Derived local shell thickness is unphysically large")

    target_rho = voxel_density(alpha, n_grid=80)
    target_volume_mm3 = target_rho * cell_size_mm**3
    initial_volume = float(np.sum(quality["areas"] * raw_thickness_mm))
    raw_thickness_mm *= target_volume_mm3 / initial_volume
    bin_ids, bin_values, element_thickness = assign_thickness_bins(
        quality["areas"], raw_thickness_mm, target_volume_mm3, N_THICKNESS_BINS
    )
    shell_volume_mm3 = float(np.sum(quality["areas"] * element_thickness))
    shell_rho = shell_volume_mm3 / cell_size_mm**3
    density_error_pct = (shell_rho - target_rho) / target_rho * 100.0
    if abs(density_error_pct) > SHELL_DENSITY_TOL_PCT:
        raise ValueError(f"Shell density matching failed: {density_error_pct:+.4f}%")

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for tri in faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge_counts[tuple(sorted((int(tri[i]), int(tri[j]))))] += 1
    boundary_edge_count = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edge_count = sum(1 for count in edge_counts.values() if count > 2)

    centre = np.array([cell_size_mm / 2.0] * 3)
    boundary_mask = np.any(
        np.isclose(
            vertices,
            0.0,
            atol=periodic_audit["boundary_tolerance_mm"],
            rtol=0.0,
        )
        | np.isclose(
            vertices,
            cell_size_mm,
            atol=periodic_audit["boundary_tolerance_mm"],
            rtol=0.0,
        ),
        axis=1,
    )
    interior_ids = np.where(~boundary_mask)[0]
    anchor_zero_based = int(
        interior_ids[np.argmin(np.linalg.norm(vertices[interior_ids] - centre, axis=1))]
        if len(interior_ids)
        else np.argmin(np.linalg.norm(vertices - centre, axis=1))
    )
    anchor_id = anchor_zero_based + 1
    rp_id = len(vertices) + 1

    case_dir.mkdir(parents=True, exist_ok=True)
    inp_path = case_dir / f"{JOB_NAME}.inp"
    with inp_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("*Heading\n")
        stream.write("** Body-fitted TPMS S3 validation model\n")
        stream.write(f"** Case: {case_id}; surface resolution: {resolution}\n")
        stream.write("** Units: mm-tonne-s-MPa\n")
        stream.write("*Material, name=Aluminum\n")
        stream.write("*Elastic\n")
        stream.write(f"{E_SOLID / 1.0e6:.10g}, {NU:.10g}\n")
        stream.write("*Part, name=TPMS_Shell\n")
        stream.write("*Node\n")
        for node_id, xyz in enumerate(vertices, start=1):
            stream.write(f"{node_id}, {xyz[0]:.9f}, {xyz[1]:.9f}, {xyz[2]:.9f}\n")
        stream.write(f"{rp_id}, {2.0 * cell_size_mm:.9f}, {2.0 * cell_size_mm:.9f}, {2.0 * cell_size_mm:.9f}\n")
        stream.write("*Element, type=S3\n")
        for element_id, tri in enumerate(faces, start=1):
            stream.write(f"{element_id}, {tri[0] + 1}, {tri[1] + 1}, {tri[2] + 1}\n")
        stream.write("*Elset, elset=ALL_ELEMENTS\n")
        write_id_list(stream, range(1, len(faces) + 1))
        for bin_id, thickness in enumerate(bin_values):
            element_ids = np.where(bin_ids == bin_id)[0] + 1
            set_name = f"THICK_BIN_{bin_id + 1:02d}"
            stream.write(f"*Elset, elset={set_name}\n")
            write_id_list(stream, element_ids.tolist())
            stream.write(f"*Shell Section, elset={set_name}, material=Aluminum\n")
            stream.write(f"{thickness:.9f}, 5\n")
        stream.write("*Nset, nset=ANCHOR\n")
        stream.write(f"{anchor_id}\n")
        stream.write("*Nset, nset=RP_MACRO\n")
        stream.write(f"{rp_id}\n")
        stream.write("*End Part\n")
        stream.write("*Assembly, name=Assembly\n")
        stream.write("*Instance, name=TPMS-1, part=TPMS_Shell\n")
        stream.write("*End Instance\n")
        stream.write("*Nset, nset=ANCHOR, instance=TPMS-1\n")
        stream.write(f"{anchor_id}\n")
        stream.write("*Nset, nset=RP_MACRO, instance=TPMS-1\n")
        stream.write(f"{rp_id}\n")
        for slave, master, shift in relations:
            for dof in range(1, 4):
                macro_dof = dof if shift[dof - 1] else None
                write_equation(stream, slave, master, dof, rp_id, macro_dof)
            for dof in range(4, 7):
                write_equation(stream, slave, master, dof, rp_id, None)
        stream.write("*End Assembly\n")
        stream.write("*Step, name=Loading, nlgeom=NO\n")
        stream.write("*Static\n")
        stream.write("0.1, 1., 1e-06, 0.1\n")
        stream.write("*Boundary\n")
        stream.write("ANCHOR, 1, 6, 0.\n")
        for dof, displacement in enumerate(displacement_m, start=1):
            stream.write(f"RP_MACRO, {dof}, {dof}, {displacement * 1000.0:.9f}\n")
        stream.write("*Output, field, frequency=1\n")
        stream.write("*Node Output\n")
        stream.write("U, RF\n")
        stream.write("*Element Output, directions=YES\n")
        stream.write("S, E\n")
        stream.write("*Output, history, frequency=1\n")
        stream.write("*Energy Output\n")
        stream.write("ALLAE, ALLSE, ALLIE\n")
        stream.write("*Node Output, nset=RP_MACRO\n")
        stream.write("U, RF\n")
        stream.write("*End Step\n")

    voxel_size_mm = cell_size_mm / 79.0
    thickness_voxels = element_thickness / voxel_size_mm
    metrics: Dict[str, object] = {
        "case_id": case_id,
        "mesh_generator_version": SHELL_MESH_GENERATOR_VERSION,
        "mesh_family": "body_fitted_shell",
        "element_type": "S3",
        "surface_resolution": resolution,
        "node_count": int(len(vertices) + 1),
        "element_count": int(len(faces)),
        "anchor_node_label": anchor_id,
        "rp_node_label": rp_id,
        "target_voxel_density_n80": target_rho,
        "shell_density": shell_rho,
        "density_match_error_pct": density_error_pct,
        "surface_area_mm2": float(np.sum(quality["areas"])),
        "shell_volume_mm3": shell_volume_mm3,
        "thickness_min_mm": float(np.min(element_thickness)),
        "thickness_p01_mm": float(np.percentile(element_thickness, 1.0)),
        "thickness_mean_mm": float(np.average(element_thickness, weights=quality["areas"])),
        "thickness_max_mm": float(np.max(element_thickness)),
        "thickness_p01_in_n80_voxels": float(np.percentile(thickness_voxels, 1.0)),
        "thickness_median_in_n80_voxels": float(np.median(thickness_voxels)),
        "surface_fraction_below_2_voxels": float(np.mean(thickness_voxels < 2.0)),
        "surface_fraction_below_3_voxels": float(np.mean(thickness_voxels < 3.0)),
        "triangle_min_angle_min_deg": float(np.min(quality["min_angles"])),
        "triangle_min_angle_p01_deg": float(np.percentile(quality["min_angles"], 1.0)),
        "triangle_aspect_p99": float(np.percentile(quality["aspect_ratios"], 99.0)),
        "triangle_aspect_max": float(np.max(quality["aspect_ratios"])),
        "boundary_edge_count": int(boundary_edge_count),
        "nonmanifold_edge_count": int(nonmanifold_edge_count),
        **periodic_audit,
    }
    write_json(case_dir / "mesh_metrics.json", metrics)
    manifest = {
        "case_id": case_id,
        "description": case["description"],
        "alpha": alpha,
        "displacement_m": displacement_m,
        "cell_size_mm": cell_size_mm,
        "mesh_family": "body_fitted_shell",
        "mesh_generator_version": SHELL_MESH_GENERATOR_VERSION,
        "element_type": "S3",
        "resolution": resolution,
        "rho_rel": shell_rho,
        "rp_node_label": rp_id,
        "percentile": PERCENTILE,
        "inp_path": str(inp_path),
    }
    write_json(case_dir / "case_manifest.json", manifest)
    np.savez_compressed(
        case_dir / "shell_mesh_preview.npz",
        vertices=vertices,
        faces=faces,
        element_thickness_mm=element_thickness,
    )
    return metrics


def inject_energy_output(inp_path: Path) -> None:
    text = inp_path.read_text(encoding="utf-8")
    if "ALLAE, ALLSE, ALLIE" in text:
        return
    marker = "*End Step"
    if marker not in text:
        raise ValueError(f"Cannot find {marker!r} in {inp_path}")
    addition = (
        "** Reviewer-requested whole-model energy audit; default section controls retained\n"
        "*Output, history, frequency=1\n"
        "*Energy Output\n"
        "ALLAE, ALLSE, ALLIE\n"
    )
    text = text.replace(marker, addition + marker, 1)
    inp_path.write_text(text, encoding="utf-8", newline="\n")


def generate_voxel_input(case_id: str, n_grid: int, case_dir: Path, element_type: str = "C3D8R") -> None:
    case = CASES[case_id]
    case_dir.mkdir(parents=True, exist_ok=True)
    inp_path = case_dir / f"{JOB_NAME}.inp"
    if not inp_path.exists():
        ok = bse.generate_inp(
            0,
            [float(v) for v in case["alpha"]],
            [float(v) for v in case["displacement_m"]],
            str(case_dir),
            n_grid=n_grid,
        )
        if not ok or not inp_path.exists():
            raise RuntimeError(f"Failed to generate voxel input for case {case_id}, n={n_grid}")
    if element_type == "C3D8":
        text = inp_path.read_text(encoding="utf-8")
        text = text.replace("*Element, type=C3D8R", "*Element, type=C3D8", 1)
        inp_path.write_text(text, encoding="utf-8", newline="\n")
    inject_energy_output(inp_path)
    manifest = {
        "case_id": case_id,
        "description": case["description"],
        "alpha": [float(v) for v in case["alpha"]],
        "displacement_m": [float(v) for v in case["displacement_m"]],
        "cell_size_mm": L * 1000.0,
        "mesh_family": "voxel",
        "element_type": element_type,
        "resolution": n_grid,
        "rho_rel": voxel_density(case["alpha"], n_grid=n_grid),
        "rp_node_label": None,
        "percentile": PERCENTILE,
        "inp_path": str(inp_path),
    }
    write_json(case_dir / "case_manifest.json", manifest)


def run_abaqus(case_dir: Path, abaqus_cmd: str, cpus: int, timeout: int, resume: bool) -> None:
    odb_path = case_dir / f"{JOB_NAME}.odb"
    if resume and odb_path.exists():
        log(f"  [resume] ODB exists: {odb_path}")
        return
    if odb_path.exists() and not resume:
        raise FileExistsError(f"ODB already exists; use --resume or a clean output directory: {odb_path}")

    command = [abaqus_cmd, f"job={JOB_NAME}", f"input={JOB_NAME}.inp", f"cpus={cpus}", "interactive"]
    log_path = case_dir / "abaqus_stdout.log"
    log(f"  [solve] {' '.join(command)}")
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        process = subprocess.Popen(
            command,
            cwd=str(case_dir),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            raise TimeoutError(f"Abaqus timed out after {timeout}s in {case_dir}")
    if return_code != 0 or not odb_path.exists():
        raise RuntimeError(f"Abaqus failed in {case_dir}; see {log_path}")


ODB_EXTRACTOR = r'''# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import math
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL


def last_step(odb):
    names = list(odb.steps.keys())
    return odb.steps[names[-1]]


def history_data(step, variable):
    for region in step.historyRegions.values():
        if variable in region.historyOutputs:
            return [(float(t), float(v)) for t, v in region.historyOutputs[variable].data]
    return []


def ratio_summary(numerator, denominator):
    den = dict((round(t, 12), v) for t, v in denominator)
    common = []
    for t, value in numerator:
        key = round(t, 12)
        if key in den and abs(den[key]) > 1.0e-30:
            common.append((t, value / den[key]))
    if not common:
        return {"final": None, "stable_max": None, "history": []}
    end_time = common[-1][0]
    stable = [ratio for t, ratio in common if t >= 0.1 * end_time]
    return {
        "final": float(common[-1][1]),
        "stable_max": float(max(abs(value) for value in stable)) if stable else None,
        "history": [[float(t), float(value)] for t, value in common],
    }


def extract(odb_path, manifest_path, output_path):
    with open(manifest_path, "r") as stream:
        manifest = json.load(stream)
    odb = openOdb(path=odb_path, readOnly=True)
    step = last_step(odb)
    frame = step.frames[-1]
    instance_name = list(odb.rootAssembly.instances.keys())[0]
    instance = odb.rootAssembly.instances[instance_name]

    stress = frame.fieldOutputs["S"].getSubset(position=ELEMENT_NODAL)
    mises = []
    for value in stress.values:
        try:
            candidate = float(value.mises)
            if not math.isnan(candidate) and not math.isinf(candidate):
                mises.append(candidate)
        except Exception:
            pass
    mises.sort()
    if not mises:
        raise RuntimeError("No finite ELEMENT_NODAL Mises stress values found")
    index = int(len(mises) * float(manifest.get("percentile", 99.9)) / 100.0)
    if index >= len(mises):
        index = len(mises) - 1
    sigma_hp = float(mises[index])
    sigma_max = float(mises[-1])

    try:
        rf_field = frame.fieldOutputs["RF"]
    except KeyError:
        rf_field = None
    dx, dy, dz = [float(v) * 1000.0 for v in manifest["displacement_m"]]
    cell_size_mm = float(manifest["cell_size_mm"])
    forces = [0.0, 0.0, 0.0]
    rp_label = manifest.get("rp_node_label", None)
    if rf_field is not None and rp_label is not None:
        for value in rf_field.values:
            if value.nodeLabel == int(rp_label):
                forces = [float(value.data[i]) for i in range(3)]
                break
    elif rf_field is not None:
        coords = dict((node.label, node.coordinates) for node in instance.nodes)
        xmax = max(value[0] for value in coords.values())
        ymax = max(value[1] for value in coords.values())
        zmax = max(value[2] for value in coords.values())
        tol = cell_size_mm * 0.001
        for value in rf_field.values:
            xyz = coords.get(value.nodeLabel, None)
            if xyz is None:
                continue
            if abs(xyz[0] - xmax) < tol:
                forces[0] += float(value.data[0])
            if abs(xyz[1] - ymax) < tol:
                forces[1] += float(value.data[1])
            if abs(xyz[2] - zmax) < tol:
                forces[2] += float(value.data[2])

    area = cell_size_mm * cell_size_mm
    stresses = [force / area for force in forces]
    d_eff = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d_eff > 1.0e-20:
        direction = [dx / d_eff, dy / d_eff, dz / d_eff]
        sigma_eff = sum(stresses[i] * direction[i] for i in range(3))
        epsilon_eff = d_eff / cell_size_mm
        e_eff = abs(sigma_eff) / epsilon_eff
    else:
        sigma_eff = 0.0
        epsilon_eff = 0.0
        e_eff = 0.0

    allae = history_data(step, "ALLAE")
    allse = history_data(step, "ALLSE")
    allie = history_data(step, "ALLIE")
    result = {
        "case_id": manifest["case_id"],
        "mesh_family": manifest["mesh_family"],
        "element_type": manifest["element_type"],
        "resolution": manifest["resolution"],
        "element_count": int(len(instance.elements)),
        "node_count": int(len(instance.nodes)),
        "rho_rel": float(manifest["rho_rel"]),
        "sigma_hp": sigma_hp,
        "sigma_max": sigma_max,
        "percentile": float(manifest.get("percentile", 99.9)),
        "n_mises_points": int(len(mises)),
        "reaction_force_components_N": forces,
        "macro_stress_components_MPa": stresses,
        "sigma_eff_MPa": float(sigma_eff),
        "epsilon_eff": float(epsilon_eff),
        "E_eff_MPa": float(e_eff),
        "energy": {
            "ALLAE": allae,
            "ALLSE": allse,
            "ALLIE": allie,
            "ALLAE_over_ALLSE": ratio_summary(allae, allse),
            "ALLAE_over_ALLIE": ratio_summary(allae, allie),
        },
    }
    odb.close()
    with open(output_path, "w") as stream:
        json.dump(result, stream, indent=2)
    print("OK")


extract(r"$odb_path", r"$manifest_path", r"$output_path")
'''


def extract_validation(case_dir: Path, abaqus_cmd: str, timeout: int, resume: bool) -> Dict[str, object]:
    output_path = case_dir / "validation_summary.json"
    if resume and output_path.exists():
        return read_json(output_path)
    extractor_path = case_dir / "_extract_validation.py"
    script = Template(ODB_EXTRACTOR).substitute(
        odb_path=str(case_dir / f"{JOB_NAME}.odb"),
        manifest_path=str(case_dir / "case_manifest.json"),
        output_path=str(output_path),
    )
    extractor_path.write_text(script, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [abaqus_cmd, "python", str(extractor_path)],
        cwd=str(case_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (case_dir / "extract_stdout.log").write_text(
        (result.stdout or "") + ("\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"ODB extraction failed in {case_dir}; see extract_stdout.log")
    return read_json(output_path)


def compare_with_stored_reference(case_id: str, resolution: int, summary: Dict[str, object]) -> Dict[str, object] | None:
    if case_id == "A":
        path = PROJECT_ROOT / "outputs" / "mesh_convergence" / f"n{resolution}" / "case_summary.json"
    elif case_id == "B" and resolution == 80:
        path = PROJECT_ROOT / "outputs" / "pareto_validation_dirichlet" / "Triaxial-1-2-2__P1" / "case_summary.json"
    else:
        return None
    if not path.exists():
        return None
    reference = read_json(path)
    comparisons = {}
    for new_key, old_key in (("E_eff_MPa", "E_eff"), ("sigma_hp", "sigma_d"), ("rho_rel", "rho_rel")):
        new_value = float(summary[new_key])
        old_value = float(reference[old_key])
        difference = (new_value - old_value) / old_value * 100.0
        comparisons[new_key] = {
            "new": new_value,
            "stored": old_value,
            "difference_pct": difference,
        }
    comparisons["passes_0p5pct_E_and_sigma"] = bool(
        abs(comparisons["E_eff_MPa"]["difference_pct"]) <= REFERENCE_TOL_PCT
        and abs(comparisons["sigma_hp"]["difference_pct"]) <= REFERENCE_TOL_PCT
    )
    return comparisons


def run_voxel_case(
    case_id: str,
    resolution: int,
    abaqus_cmd: str,
    cpus: int,
    timeout: int,
    resume: bool,
    generate_only: bool,
    element_type: str = "C3D8R",
) -> Dict[str, object] | None:
    suffix = f"{case_id}_n{resolution}" if element_type == "C3D8R" else f"{case_id}_n{resolution}_{element_type}"
    case_dir = OUTPUT_ROOT / "voxel" / suffix
    log(f"\n[voxel] case={case_id}, n={resolution}, element={element_type}")
    generate_voxel_input(case_id, resolution, case_dir, element_type=element_type)
    if generate_only:
        return None
    run_abaqus(case_dir, abaqus_cmd, cpus, timeout, resume)
    summary = extract_validation(case_dir, abaqus_cmd, timeout, resume)
    comparison = compare_with_stored_reference(case_id, resolution, summary)
    if comparison is not None:
        write_json(case_dir / "stored_reference_comparison.json", comparison)
    return summary


def run_shell_case(
    case_id: str,
    resolution: int,
    abaqus_cmd: str,
    cpus: int,
    timeout: int,
    resume: bool,
    generate_only: bool,
) -> Dict[str, object] | None:
    case_dir = OUTPUT_ROOT / "shell" / f"{case_id}_s{resolution}"
    log(f"\n[shell] case={case_id}, surface resolution={resolution}")
    inp_path = case_dir / f"{JOB_NAME}.inp"
    metrics_path = case_dir / "mesh_metrics.json"
    reusable_input = False
    if resume and inp_path.exists() and metrics_path.exists():
        try:
            reusable_input = (
                read_json(metrics_path).get("mesh_generator_version")
                == SHELL_MESH_GENERATOR_VERSION
            )
        except Exception:
            reusable_input = False
    if not reusable_input:
        generate_shell_input(case_id, resolution, case_dir)
    if generate_only:
        return None
    run_abaqus(case_dir, abaqus_cmd, cpus, timeout, resume)
    return extract_validation(case_dir, abaqus_cmd, timeout, resume)


def relative_difference(new: float, reference: float) -> float:
    return (new - reference) / reference * 100.0


def run_hourglass_stage(args, abaqus_cmd: str) -> Dict[Tuple[str, int], Dict[str, object]]:
    results: Dict[Tuple[str, int], Dict[str, object]] = {}
    for case_id in ("A", "B"):
        for resolution in CASES[case_id]["voxel_resolutions"]:
            result = run_voxel_case(
                case_id,
                int(resolution),
                abaqus_cmd,
                args.cpus,
                args.timeout,
                args.resume,
                args.generate_only,
            )
            if result is not None:
                results[(case_id, int(resolution))] = result

    if args.generate_only:
        return results

    failed_reference = []
    for (case_id, resolution), result in results.items():
        comparison_path = OUTPUT_ROOT / "voxel" / f"{case_id}_n{resolution}" / "stored_reference_comparison.json"
        if comparison_path.exists():
            comparison = read_json(comparison_path)
            if not comparison["passes_0p5pct_E_and_sigma"]:
                failed_reference.append((case_id, resolution, comparison))
    if failed_reference:
        raise RuntimeError(
            "New voxel results do not reproduce stored E_eff/sigma_hp within 0.5%; "
            "shell comparison has been stopped. Details: " + repr(failed_reference)
        )

    trigger_full_integration = False
    for result in results.values():
        ratio = result["energy"]["ALLAE_over_ALLSE"]["stable_max"]
        if ratio is not None and abs(float(ratio)) * 100.0 > ENERGY_FLAG_PCT:
            trigger_full_integration = True
    if trigger_full_integration:
        log("[diagnostic] ALLAE/ALLSE exceeded 5%; adding case A n=80 C3D8 full-integration check")
        run_voxel_case(
            "A",
            80,
            abaqus_cmd,
            args.cpus,
            args.timeout,
            args.resume,
            False,
            element_type="C3D8",
        )
    return results


def run_shell_stage(args, abaqus_cmd: str) -> None:
    result_120 = run_shell_case("A", 120, abaqus_cmd, args.cpus, args.timeout, args.resume, args.generate_only)
    result_160 = run_shell_case("A", 160, abaqus_cmd, args.cpus, args.timeout, args.resume, args.generate_only)
    chosen_resolution = 160
    if not args.generate_only and result_120 is not None and result_160 is not None:
        e_change = abs(relative_difference(float(result_160["E_eff_MPa"]), float(result_120["E_eff_MPa"])))
        s_change = abs(relative_difference(float(result_160["sigma_hp"]), float(result_120["sigma_hp"])))
        convergence = {
            "from_resolution": 120,
            "to_resolution": 160,
            "E_eff_change_pct": e_change,
            "sigma_hp_change_pct": s_change,
            "passes": bool(e_change <= SHELL_E_CONV_TOL_PCT and s_change <= SHELL_SIGMA_CONV_TOL_PCT),
        }
        write_json(OUTPUT_ROOT / "shell" / "A_shell_convergence.json", convergence)
        if not convergence["passes"]:
            chosen_resolution = 200
            log("[diagnostic] S3 mesh did not converge from 120 to 160; adding case A s200")
            result_200 = run_shell_case("A", 200, abaqus_cmd, args.cpus, args.timeout, args.resume, False)
            e_change_200 = abs(relative_difference(float(result_200["E_eff_MPa"]), float(result_160["E_eff_MPa"])))
            s_change_200 = abs(relative_difference(float(result_200["sigma_hp"]), float(result_160["sigma_hp"])))
            convergence["follow_up"] = {
                "from_resolution": 160,
                "to_resolution": 200,
                "E_eff_change_pct": e_change_200,
                "sigma_hp_change_pct": s_change_200,
                "passes": bool(e_change_200 <= SHELL_E_CONV_TOL_PCT and s_change_200 <= SHELL_SIGMA_CONV_TOL_PCT),
            }
            write_json(OUTPUT_ROOT / "shell" / "A_shell_convergence.json", convergence)
    run_shell_case("B", chosen_resolution, abaqus_cmd, args.cpus, args.timeout, args.resume, args.generate_only)
    write_json(OUTPUT_ROOT / "selected_shell_resolution.json", {"resolution": chosen_resolution})


def write_run_manifest(args, abaqus_cmd: str) -> None:
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "abaqus_command": abaqus_cmd,
        "stage": args.stage,
        "cpus_per_job": args.cpus,
        "timeout_s": args.timeout,
        "resume": bool(args.resume),
        "generate_only": bool(args.generate_only),
        "cases": CASES,
        "thresholds": {
            "stored_reference_tolerance_pct": REFERENCE_TOL_PCT,
            "energy_flag_pct": ENERGY_FLAG_PCT,
            "shell_E_convergence_pct": SHELL_E_CONV_TOL_PCT,
            "shell_sigma_convergence_pct": SHELL_SIGMA_CONV_TOL_PCT,
            "shell_density_match_pct": SHELL_DENSITY_TOL_PCT,
        },
    }
    write_json(OUTPUT_ROOT / "run_manifest.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("hourglass", "shell", "all"), default="all")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200, help="per-job timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="reuse existing INP/ODB/summary files")
    parser.add_argument("--generate-only", action="store_true", help="generate and audit inputs without running Abaqus")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    abaqus_cmd = bse.check_abaqus_available()
    if not abaqus_cmd:
        raise RuntimeError("Abaqus executable was not found")
    write_run_manifest(args, abaqus_cmd)

    if args.stage in ("hourglass", "all"):
        run_hourglass_stage(args, abaqus_cmd)
    if args.stage in ("shell", "all"):
        run_shell_stage(args, abaqus_cmd)
    log("\nValidation run stage completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
