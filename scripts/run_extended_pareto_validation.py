# -*- coding: utf-8 -*-
"""Run the locked 13-case extended Pareto FEM validation campaign.

The selection manifest is produced by select_extended_pareto_validation.py.
Nine existing P1/P2/P3 FEM summaries are verified and reused; only rows whose
``fem_action`` is ``run`` are submitted to Abaqus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.batch_simulate_extract as bse  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pareto_validation_extended"
DEFAULT_MANIFEST = OUTPUT_ROOT / "selection_manifest.csv"
RUN_METADATA = OUTPUT_ROOT / "run_metadata.json"
CASES_ROOT = OUTPUT_ROOT / "cases"

DEFAULT_N_GRID = 80
DEFAULT_CPUS = 4
DEFAULT_TIMEOUT = 1800
DEFAULT_RETRIES = 2

KEEP_AFTER_SUCCESS = {
    "case_summary.json",
    "boundary_conditions.txt",
    "fem_node_stress.csv",
    "run_record.json",
}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "load_case",
        "point_id",
        "fem_action",
        "alpha1",
        "alpha2",
        "alpha3",
        "alpha4",
        "delta_x",
        "delta_y",
        "delta_z",
        "fem_result_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if len(df) != 22:
        raise ValueError(f"Expected 22 manifest rows, found {len(df)}")
    if int((df["fem_action"] == "reuse").sum()) != 9:
        raise ValueError("Expected exactly 9 reused rows")
    if int((df["fem_action"] == "run").sum()) != 13:
        raise ValueError("Expected exactly 13 new FEM rows")
    return df


def _load_metadata(manifest_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    manifest_digest = _manifest_hash(manifest_path)
    if RUN_METADATA.exists():
        with RUN_METADATA.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        old_digest = metadata.get("manifest_sha256")
        if old_digest and old_digest != manifest_digest:
            raise RuntimeError(
                "Selection manifest changed after the FEM run metadata was created. "
                "Refusing to mix results from different selections."
            )
    else:
        metadata = {
            "created_at": _timestamp(),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_digest,
            "configuration": {
                "n_grid": args.n_grid,
                "cpus_per_job": args.cpus,
                "timeout_seconds": args.timeout,
                "retries": args.retries,
                "execution": "sequential",
            },
            "cases": {},
        }
    metadata["updated_at"] = _timestamp()
    return metadata


def _save_metadata(metadata: Dict[str, object]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata["updated_at"] = _timestamp()
    temporary = RUN_METADATA.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    temporary.replace(RUN_METADATA)


def _assert_generated_case_path(path: Path) -> None:
    resolved_root = CASES_ROOT.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"New FEM path must remain under {resolved_root}: {resolved_path}") from exc


def _clear_incomplete_case(case_dir: Path) -> None:
    """Remove stale generated files only from a validated incomplete case dir."""
    _assert_generated_case_path(case_dir)
    if not case_dir.exists():
        return
    for entry in case_dir.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _cleanup_after_success(case_dir: Path) -> List[str]:
    """Delete reproducible Abaqus intermediates and return deleted names."""
    _assert_generated_case_path(case_dir)
    deleted: List[str] = []
    for entry in list(case_dir.iterdir()):
        if entry.name in KEEP_AFTER_SUCCESS:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        deleted.append(entry.name)
    return sorted(deleted)


def _read_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    required = {"sigma_d", "E_eff", "rho_rel"}
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"Invalid FEM summary {path}: missing {missing}")
    return summary


def _case_key(row: pd.Series) -> str:
    return f"{row['load_case']}__{row['point_id']}"


def _run_one(
    row: pd.Series,
    abaqus_cmd: str,
    args: argparse.Namespace,
    metadata: Dict[str, object],
) -> None:
    key = _case_key(row)
    summary_path = Path(row["fem_result_path"])
    case_dir = summary_path.parent
    _assert_generated_case_path(case_dir)

    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"FEM result already exists for {key}. Re-run with --resume to reuse it."
            )
        summary = _read_summary(summary_path)
        metadata["cases"][key] = {
            "status": "completed_cached",
            "updated_at": _timestamp(),
            "result_path": str(summary_path.resolve()),
            "sigma_d": float(summary["sigma_d"]),
            "E_eff": float(summary["E_eff"]),
            "rho_rel": float(summary["rho_rel"]),
        }
        _save_metadata(metadata)
        _log(f"[SKIP] {key}: cached case_summary.json")
        return

    case_dir.mkdir(parents=True, exist_ok=True)
    existing_odb = case_dir / "tpms_mesh.odb"
    if not existing_odb.exists():
        _clear_incomplete_case(case_dir)
    alpha = [float(row[f"alpha{i}"]) for i in range(1, 5)]
    delta = [float(row[name]) for name in ("delta_x", "delta_y", "delta_z")]
    started = time.time()
    record = {
        "case": key,
        "status": "running",
        "started_at": _timestamp(),
        "alpha": alpha,
        "delta_m": delta,
        "n_grid": args.n_grid,
        "cpus": args.cpus,
        "timeout_seconds": args.timeout,
        "attempts": 0,
    }
    metadata["cases"][key] = record
    _save_metadata(metadata)

    if existing_odb.exists():
        _log(f"[1/3] {key}: existing ODB found; preserving completed solver result")
    else:
        _log(f"[1/3] {key}: generating n={args.n_grid} voxel INP")
        if not bse.generate_inp(0, alpha, delta, str(case_dir), n_grid=args.n_grid):
            raise RuntimeError(f"{key}: INP generation failed")

    simulation_ok = False
    for attempt in range(1, args.retries + 2):
        record["attempts"] = attempt
        _save_metadata(metadata)
        _log(f"[2/3] {key}: Abaqus attempt {attempt}/{args.retries + 1}, CPUs={args.cpus}")
        if bse.run_simulation(
            str(case_dir), abaqus_cmd, cpus=args.cpus, timeout=args.timeout
        ):
            simulation_ok = True
            break
        if attempt <= args.retries:
            _log(f"[RETRY] {key}: cleaning Abaqus processes before retry")
            bse.kill_abaqus_processes()
            time.sleep(3)
    if not simulation_ok:
        raise RuntimeError(f"{key}: Abaqus failed after {args.retries + 1} attempts")

    _log(f"[3/3] {key}: extracting case summary")
    if not bse.extract_summary_data(
        str(case_dir), abaqus_cmd, delta, n_grid=args.n_grid
    ):
        raise RuntimeError(f"{key}: result extraction failed")
    summary = _read_summary(summary_path)

    elapsed = time.time() - started
    run_record = {
        **record,
        "status": "completed",
        "finished_at": _timestamp(),
        "elapsed_seconds": elapsed,
        "sigma_d": float(summary["sigma_d"]),
        "E_eff": float(summary["E_eff"]),
        "rho_rel": float(summary["rho_rel"]),
    }
    with (case_dir / "run_record.json").open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2, ensure_ascii=False)
    deleted = _cleanup_after_success(case_dir)
    run_record["deleted_intermediate_files"] = deleted
    with (case_dir / "run_record.json").open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2, ensure_ascii=False)
    metadata["cases"][key] = run_record
    _save_metadata(metadata)
    _log(
        f"[OK] {key}: sigma_hp={float(summary['sigma_d']):.3f} MPa, "
        f"E_eff={abs(float(summary['E_eff'])):.3f} MPa, "
        f"rho_rel={float(summary['rho_rel']):.6f}, elapsed={elapsed / 60:.1f} min"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    metadata = _load_metadata(manifest_path, args)

    reused = manifest[manifest["fem_action"] == "reuse"]
    missing_reused = [
        str(Path(row["fem_result_path"]))
        for _, row in reused.iterrows()
        if not Path(row["fem_result_path"]).exists()
    ]
    if missing_reused:
        raise FileNotFoundError("Missing reused FEM summaries:\n" + "\n".join(missing_reused))
    for _, row in reused.iterrows():
        summary = _read_summary(Path(row["fem_result_path"]))
        key = _case_key(row)
        metadata["cases"][key] = {
            "status": "reused_existing",
            "updated_at": _timestamp(),
            "result_path": str(Path(row["fem_result_path"]).resolve()),
            "sigma_d": float(summary["sigma_d"]),
            "E_eff": float(summary["E_eff"]),
            "rho_rel": float(summary["rho_rel"]),
        }
    _save_metadata(metadata)

    pending = manifest[manifest["fem_action"] == "run"]
    print("=" * 88)
    print("Extended Pareto FEM validation")
    print(f"Manifest: {manifest_path}")
    print(f"Reused:   {len(reused)} existing summaries")
    print(f"New FEM:  {len(pending)} cases")
    print(
        f"Settings: n_grid={args.n_grid}, CPUs={args.cpus}, timeout={args.timeout}s, "
        f"retries={args.retries}, sequential"
    )
    print("=" * 88)
    for _, row in pending.iterrows():
        exists = Path(row["fem_result_path"]).exists()
        print(f"  {_case_key(row):<30} {'cached' if exists else 'pending'}")

    if args.dry_run:
        print("\nDry run complete; Abaqus was not launched.")
        return

    abaqus_cmd = bse.check_abaqus_available()
    if not abaqus_cmd:
        raise RuntimeError("Abaqus executable was not found")
    _log(f"Abaqus command: {abaqus_cmd}")

    failures = []
    start_all = time.time()
    for ordinal, (_, row) in enumerate(pending.iterrows(), start=1):
        key = _case_key(row)
        _log(f"\n--- [{ordinal}/{len(pending)}] {key} ---")
        try:
            _run_one(row, abaqus_cmd, args, metadata)
        except Exception as exc:
            failures.append((key, str(exc)))
            metadata["cases"][key] = {
                **metadata["cases"].get(key, {}),
                "status": "failed",
                "failed_at": _timestamp(),
                "error": str(exc),
            }
            _save_metadata(metadata)
            _log(f"[ERROR] {key}: {exc}")

    metadata["campaign_elapsed_seconds"] = time.time() - start_all
    metadata["campaign_status"] = "completed" if not failures else "completed_with_failures"
    metadata["failures"] = [{"case": key, "error": error} for key, error in failures]
    _save_metadata(metadata)

    print("\n" + "=" * 88)
    print(f"Campaign elapsed: {(time.time() - start_all) / 60:.1f} min")
    print(f"Succeeded: {len(pending) - len(failures)}/{len(pending)}")
    if failures:
        for key, error in failures:
            print(f"  FAILED {key}: {error}")
        raise SystemExit(1)
    print(f"Run metadata: {RUN_METADATA}")
    print("=" * 88)


if __name__ == "__main__":
    main()
