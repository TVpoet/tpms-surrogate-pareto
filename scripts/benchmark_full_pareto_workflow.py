# -*- coding: utf-8 -*-
"""Benchmark the complete online surrogate-assisted Pareto-search workflow.

The timed region starts after both trained networks have been loaded and a GPU
warm-up has completed.  It includes, for one prescribed loading case:

1. Dirichlet candidate generation;
2. predictions by the mechanical-response and density networks;
3. feasibility filtering and specific-modulus calculation; and
4. non-dominated sorting.

No plotting, model loading, or result-file writing is included in the measured
wall-clock time.  Raw repetitions and a machine-readable summary are written
to ``outputs/pareto_timing_evidence`` by default.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import os
import platform
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from surrogate_model.density_model import DensityModel  # noqa: E402
from surrogate_model.model import SurrogateModel, device  # noqa: E402
from surrogate_model.pareto_utils import find_pareto_front  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pareto_timing_evidence"
DEFAULT_SURROGATE_PATH = PROJECT_ROOT / "outputs" / "surrogate_model" / "best_model.pth"
DEFAULT_DENSITY_PATH = PROJECT_ROOT / "outputs" / "surrogate_model" / "density_model.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the complete online 5-million-candidate Pareto workflow."
    )
    parser.add_argument("--n-candidates", type=int, default=5_000_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--warmup-candidates", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--delta-um",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Prescribed displacement vector in micrometres.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surrogate-path", type=Path, default=DEFAULT_SURROGATE_PATH)
    parser.add_argument("--density-path", type=Path, default=DEFAULT_DENSITY_PATH)
    args = parser.parse_args()

    if args.n_candidates <= 0:
        parser.error("--n-candidates must be positive")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup_candidates <= 0:
        parser.error("--warmup-candidates must be positive")
    return args


def sync_device() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            pass
    name = platform.processor().strip()
    if name:
        return name
    return platform.machine() or "Unavailable"


def physical_core_count() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except (ImportError, OSError):
        return None


def hardware_metadata() -> dict[str, Any]:
    gpu_name = None
    gpu_memory_gib = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        gpu_memory_gib = props.total_memory / (1024**3)

    return {
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "physical_cpu_cores": physical_core_count(),
        "logical_cpu_cores": os.cpu_count(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu_name,
        "gpu_memory_gib": gpu_memory_gib,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def warm_up(
    surrogate: SurrogateModel,
    density: DensityModel,
    delta_m: np.ndarray,
    n_candidates: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    alpha = rng.dirichlet(np.ones(4), size=n_candidates)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    delta_t = torch.tensor(delta_m, dtype=torch.float32, device=device)
    delta_batch = delta_t.unsqueeze(0).expand(n_candidates, -1)
    x = torch.cat((alpha_t, delta_batch), dim=1)
    with torch.no_grad():
        _ = surrogate(x)
        _ = density(alpha_t)
    sync_device()


def benchmark_once(
    surrogate: SurrogateModel,
    density: DensityModel,
    delta_m: np.ndarray,
    n_candidates: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    sync_device()
    total_start = time.perf_counter()

    phase_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    alpha_all = rng.dirichlet(np.ones(4), size=n_candidates)
    sampling_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    delta_t = torch.tensor(delta_m, dtype=torch.float32, device=device)
    sigma_parts: list[np.ndarray] = []
    modulus_parts: list[np.ndarray] = []
    density_parts: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_candidates, batch_size):
            stop = min(start + batch_size, n_candidates)
            alpha_t = torch.tensor(
                alpha_all[start:stop], dtype=torch.float32, device=device
            )
            delta_batch = delta_t.unsqueeze(0).expand(stop - start, -1)
            x = torch.cat((alpha_t, delta_batch), dim=1)
            response = surrogate(x)
            rho = density(alpha_t).squeeze(-1)
            sigma_parts.append(response[:, 0].cpu().numpy())
            modulus_parts.append(torch.abs(response[:, 1]).cpu().numpy())
            density_parts.append(rho.cpu().numpy())

    sync_device()
    sigma = np.concatenate(sigma_parts)
    modulus = np.concatenate(modulus_parts)
    rho = np.concatenate(density_parts)
    inference_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    valid = (sigma > 0.0) & (modulus > 0.0) & (rho > 0.0)
    alpha_valid = alpha_all[valid]
    sigma_valid = sigma[valid]
    modulus_valid = modulus[valid]
    rho_valid = rho[valid]
    specific_modulus = modulus_valid / rho_valid
    filtering_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    costs = np.column_stack((sigma_valid, -specific_modulus))
    pareto_mask = find_pareto_front(costs)
    sorting_seconds = time.perf_counter() - phase_start

    total_seconds = time.perf_counter() - total_start
    phase_sum_seconds = (
        sampling_seconds
        + inference_seconds
        + filtering_seconds
        + sorting_seconds
    )

    # Keep alpha_valid alive through the timed region to reproduce the memory
    # behavior of the production search, which retains valid candidate designs.
    return {
        "seed": seed,
        "n_candidates": int(n_candidates),
        "n_valid": int(alpha_valid.shape[0]),
        "n_pareto": int(pareto_mask.sum()),
        "sampling_seconds": sampling_seconds,
        "dual_network_inference_seconds": inference_seconds,
        "filtering_seconds": filtering_seconds,
        "pareto_sorting_seconds": sorting_seconds,
        "phase_sum_seconds": phase_sum_seconds,
        "total_seconds": total_seconds,
        "unattributed_seconds": total_seconds - phase_sum_seconds,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean_seconds": float(arr.mean()),
        "median_seconds": float(np.median(arr)),
        "std_seconds": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "minimum_seconds": float(arr.min()),
        "maximum_seconds": float(arr.max()),
        "p95_seconds": float(np.percentile(arr, 95)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_evidence_figure(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, float],
    config: dict[str, Any],
    hardware: dict[str, Any],
) -> None:
    width, height = 3600, 1440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
        if bold and italic:
            names = ("timesbi.ttf", "timesbd.ttf", "times.ttf")
        elif bold:
            names = ("timesbd.ttf", "times.ttf")
        elif italic:
            names = ("timesi.ttf", "times.ttf")
        else:
            names = ("times.ttf",)
        for name in names:
            candidate = Path("C:/Windows/Fonts") / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    font_tick = font(36)
    font_axis = font(43)
    font_legend = font(31)
    font_heading = font(47, bold=True)
    font_label = font(31, bold=True)
    font_value = font(31)
    font_note = font(27)

    chart_left, chart_top = 220, 220
    chart_right, chart_bottom = 2170, 1250
    plot_width = chart_right - chart_left
    plot_height = chart_bottom - chart_top
    run_index = np.arange(1, len(rows) + 1)
    phase_specs = [
        ("sampling_seconds", "Candidate generation", "#4C78A8"),
        ("dual_network_inference_seconds", "Dual-network prediction", "#F58518"),
        ("filtering_seconds", "Filtering", "#54A24B"),
        ("pareto_sorting_seconds", "Non-dominated sorting", "#E45756"),
        ("unattributed_seconds", "Timer overhead", "#B8B8B8"),
    ]
    totals = np.asarray([float(row["total_seconds"]) for row in rows])
    median = summary["median_seconds"]
    raw_max = max(float(totals.max()), median, 1.0e-9) * 1.15
    exponent = math.floor(math.log10(raw_max))
    fraction = raw_max / (10**exponent)
    nice_fraction = next(value for value in (1.0, 2.0, 5.0, 10.0) if fraction <= value)
    y_max = nice_fraction * (10**exponent)

    def map_y(value: float) -> int:
        return int(chart_bottom - (value / y_max) * plot_height)

    # Light horizontal grid and left-side inward ticks.  The outer rectangle
    # keeps all four axes visible; top and right axes intentionally have no ticks.
    for tick_index in range(6):
        tick_value = y_max * tick_index / 5
        y = map_y(tick_value)
        draw.line((chart_left, y, chart_right, y), fill="#D9D9D9", width=2)
        draw.line((chart_left, y, chart_left + 16, y), fill="black", width=3)
        label = f"{tick_value:.3g}"
        box = draw.textbbox((0, 0), label, font=font_tick)
        draw.text(
            (chart_left - 22 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
            label,
            font=font_tick,
            fill="black",
        )

    draw.rectangle(
        (chart_left, chart_top, chart_right, chart_bottom),
        outline="black",
        width=3,
    )

    slot = plot_width / max(len(rows), 1)
    bar_width = min(210.0, slot * 0.58)
    for run_position, row in enumerate(rows, start=1):
        center_x = chart_left + slot * (run_position - 0.5)
        bottom_value = 0.0
        for key, _, color in phase_specs:
            value = max(float(row[key]), 0.0)
            y0 = map_y(bottom_value)
            y1 = map_y(bottom_value + value)
            draw.rectangle(
                (center_x - bar_width / 2, y1, center_x + bar_width / 2, y0),
                fill=color,
                outline="black",
                width=2,
            )
            bottom_value += value
        draw.line(
            (center_x, chart_bottom, center_x, chart_bottom - 16),
            fill="black",
            width=3,
        )
        label = str(run_position)
        box = draw.textbbox((0, 0), label, font=font_tick)
        draw.text(
            (center_x - (box[2] - box[0]) / 2, chart_bottom + 18),
            label,
            font=font_tick,
            fill="black",
        )

    median_y = map_y(median)
    dash, gap = 22, 14
    x = chart_left
    while x < chart_right:
        draw.line((x, median_y, min(x + dash, chart_right), median_y), fill="black", width=4)
        x += dash + gap

    x_label = "Run"
    box = draw.textbbox((0, 0), x_label, font=font_axis)
    draw.text(
        ((chart_left + chart_right - (box[2] - box[0])) / 2, chart_bottom + 68),
        x_label,
        font=font_axis,
        fill="black",
    )
    y_label = "Wall-clock time (s)"
    y_box = draw.textbbox((0, 0), y_label, font=font_axis)
    y_image = Image.new(
        "RGBA", (y_box[2] - y_box[0] + 24, y_box[3] - y_box[1] + 24), (255, 255, 255, 0)
    )
    y_draw = ImageDraw.Draw(y_image)
    y_draw.text((12, 6), y_label, font=font_axis, fill="black")
    y_image = y_image.rotate(90, expand=True)
    image.paste(
        y_image,
        (
            25,
            int((chart_top + chart_bottom - y_image.height) / 2),
        ),
        y_image,
    )

    legend_items = [(label, color) for _, label, color in phase_specs]
    legend_items.append((f"Median = {median:.3f} s", None))
    legend_x, legend_y = chart_left, 60
    for item_index, (label, color) in enumerate(legend_items):
        if item_index == 3:
            legend_x = chart_left
            legend_y = 120
        if color is None:
            line_y = legend_y + 17
            draw.line((legend_x, line_y, legend_x + 70, line_y), fill="black", width=4)
            draw.line((legend_x + 12, line_y, legend_x + 34, line_y), fill="white", width=4)
            draw.line((legend_x + 48, line_y, legend_x + 60, line_y), fill="white", width=4)
        else:
            draw.rectangle((legend_x, legend_y + 2, legend_x + 40, legend_y + 34), fill=color, outline="black", width=2)
        draw.text((legend_x + 55, legend_y), label, font=font_legend, fill="black")
        text_width = draw.textlength(label, font=font_legend)
        legend_x += int(text_width + 125)

    delta_um = config["delta_um"]
    gpu = hardware.get("gpu") or "CPU execution"
    physical = hardware.get("physical_cpu_cores")
    logical = hardware.get("logical_cpu_cores")
    core_text = (
        f"{physical} physical / {logical} logical cores"
        if physical is not None
        else f"{logical} logical cores"
    )
    details = [
        ("Measured scope", "Complete online workflow after model loading"),
        ("Candidates", f"{config['n_candidates']:,} per run"),
        ("Loading vector", f"({delta_um[0]:g}, {delta_um[1]:g}, {delta_um[2]:g}) μm"),
        ("Batch size", f"{config['batch_size']:,}"),
        ("Repetitions", str(config["runs"])),
        ("CPU", f"{hardware['cpu']} ({core_text})"),
        ("GPU", str(gpu)),
        (
            "PyTorch / CUDA",
            f"{hardware['pytorch']} / "
            f"{hardware.get('cuda_runtime') or ('available' if hardware.get('cuda_available') else 'unavailable')}",
        ),
        ("Median", f"{summary['median_seconds']:.3f} s"),
        ("Mean ± SD", f"{summary['mean_seconds']:.3f} ± {summary['std_seconds']:.3f} s"),
        ("Range", f"{summary['minimum_seconds']:.3f}-{summary['maximum_seconds']:.3f} s"),
    ]

    info_x = 2330
    y = 90
    draw.text((info_x, y), "Timing record", font=font_heading, fill="black")
    y += 92
    for label, value in details:
        draw.text((info_x, y), label, font=font_label, fill="black")
        value_x = 2820
        wrapped = textwrap.wrap(str(value), width=43) or [""]
        for line_index, line in enumerate(wrapped):
            draw.text((value_x, y + line_index * 36), line, font=font_value, fill="black")
        y += max(76, len(wrapped) * 36 + 28)

    included_note = (
        "Included: Dirichlet sampling, both surrogate networks, feasibility filtering, "
        "specific-modulus calculation, and non-dominated sorting."
    )
    excluded_note = "Excluded: model loading, plotting, and file writing."
    note = "\n".join(textwrap.wrap(included_note, width=82))
    note += "\n" + "\n".join(textwrap.wrap(excluded_note, width=82))
    note_y = 1250
    draw.multiline_text(
        (info_x, note_y),
        note,
        font=font_note,
        fill="#333333",
        spacing=10,
    )
    image.save(path, dpi=(300, 300))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.surrogate_path.is_file():
        raise FileNotFoundError(f"Mechanical-response model not found: {args.surrogate_path}")
    if not args.density_path.is_file():
        raise FileNotFoundError(f"Density model not found: {args.density_path}")

    hardware = hardware_metadata()
    delta_um = np.asarray(args.delta_um, dtype=float)
    delta_m = delta_um * 1.0e-6

    log_lines: list[str] = []

    def log(message: str = "") -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("=" * 82)
    log("Complete Online Pareto-Search Timing Benchmark")
    log("=" * 82)
    log(f"Timestamp        : {datetime.now().astimezone().isoformat(timespec='seconds')}")
    log(f"CPU              : {hardware['cpu']}")
    log(
        "CPU cores        : "
        f"{hardware['physical_cpu_cores']} physical / {hardware['logical_cpu_cores']} logical"
    )
    log(f"GPU              : {hardware.get('gpu')}")
    cuda_display = hardware.get("cuda_runtime") or (
        "available" if hardware.get("cuda_available") else "unavailable"
    )
    log(f"PyTorch / CUDA   : {hardware['pytorch']} / {cuda_display}")
    log(f"Candidates       : {args.n_candidates:,} per run")
    log(f"Loading vector   : ({delta_um[0]:g}, {delta_um[1]:g}, {delta_um[2]:g}) um")
    log(f"Batch size       : {args.batch_size:,}")
    log(f"Measured runs    : {args.runs}")
    log(f"Fixed seed       : {args.seed}")
    log("Timing boundary  : after model loading; before result-file writing")
    log(
        "Included stages  : Dirichlet generation, both networks, filtering, "
        "specific modulus, non-dominated sorting"
    )
    log("Excluded stages  : model loading, plotting, result-file writing")
    log("=" * 82)

    # Model constructors print architecture details, including Chinese labels in
    # the density model.  Suppress those internal messages so the saved timing
    # record remains clean and encoding-independent.
    with contextlib.redirect_stdout(io.StringIO()):
        surrogate = SurrogateModel.load(str(args.surrogate_path))
        density = DensityModel.load(str(args.density_path))
    surrogate.eval()
    density.eval()
    log(f"Mechanical model : {sum(p.numel() for p in surrogate.parameters()):,} parameters")
    log(f"Density model    : {sum(p.numel() for p in density.parameters()):,} parameters")

    log(f"Warm-up          : {args.warmup_candidates:,} candidates (not timed)")
    warm_up(
        surrogate,
        density,
        delta_m,
        n_candidates=args.warmup_candidates,
        seed=args.seed + 10_000,
    )
    log("Warm-up complete.")
    log("")

    rows: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        gc.collect()
        row = benchmark_once(
            surrogate,
            density,
            delta_m,
            n_candidates=args.n_candidates,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        row = {"run": run_index, **row}
        rows.append(row)
        log(
            f"Run {run_index:02d}: total={row['total_seconds']:.3f} s | "
            f"sampling={row['sampling_seconds']:.3f} s | "
            f"dual-NN={row['dual_network_inference_seconds']:.3f} s | "
            f"filtering={row['filtering_seconds']:.3f} s | "
            f"sorting={row['pareto_sorting_seconds']:.3f} s | "
            f"valid={row['n_valid']:,} | Pareto={row['n_pareto']:,}"
        )

    timing_summary = summarize([float(row["total_seconds"]) for row in rows])
    log("")
    log("Summary")
    log("-" * 82)
    log(f"Mean             : {timing_summary['mean_seconds']:.3f} s")
    log(f"Median           : {timing_summary['median_seconds']:.3f} s")
    log(f"Sample SD        : {timing_summary['std_seconds']:.3f} s")
    log(
        f"Range            : {timing_summary['minimum_seconds']:.3f}-"
        f"{timing_summary['maximum_seconds']:.3f} s"
    )
    log(f"P95              : {timing_summary['p95_seconds']:.3f} s")

    config = {
        "n_candidates": args.n_candidates,
        "runs": args.runs,
        "batch_size": args.batch_size,
        "warmup_candidates": args.warmup_candidates,
        "seed": args.seed,
        "delta_um": delta_um.tolist(),
        "delta_m": delta_m.tolist(),
        "surrogate_path": str(args.surrogate_path.resolve()),
        "density_path": str(args.density_path.resolve()),
        "timing_scope_included": [
            "Dirichlet candidate generation",
            "mechanical-response network prediction",
            "density-network prediction",
            "feasibility filtering",
            "specific-modulus calculation",
            "non-dominated sorting",
        ],
        "timing_scope_excluded": [
            "model loading",
            "plotting",
            "result-file writing",
        ],
    }
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config,
        "hardware": hardware,
        "summary": timing_summary,
        "runs": rows,
    }

    csv_path = args.output_dir / "timing_runs.csv"
    json_path = args.output_dir / "timing_summary.json"
    log_path = args.output_dir / "benchmark_console.log"
    figure_path = args.output_dir / "benchmark_timing_evidence.png"

    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    log("")
    log(f"CSV              : {csv_path}")
    log(f"JSON             : {json_path}")
    log(f"Evidence figure  : {figure_path}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Console log      : {log_path}", flush=True)
    make_evidence_figure(figure_path, rows, timing_summary, config, hardware)


if __name__ == "__main__":
    main()
