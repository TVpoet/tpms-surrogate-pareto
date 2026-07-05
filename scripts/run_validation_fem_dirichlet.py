# -*- coding: utf-8 -*-
"""
跑 Dirichlet Pareto 候选的 FEM 闭环验证 + 对比 NN 预测。

论文主线为 Biaxial-XY、Triaxial-1-1-2、Triaxial-1-2-2 三工况 × P1/P2/P3。
如果输入 JSON 包含 Uniaxial-X 等诊断工况，脚本也会一并处理；README 表格只汇总论文主线三工况。

输入: outputs/validation_candidates_dirichlet.json
     （由 scripts/pick_three_dirichlet.py 生成）
输出:
  - outputs/pareto_validation_dirichlet/{case}__{P1,P2,P3}/case_summary.json
  - outputs/validation_comparison_dirichlet.json
  - 终端打印对比表

每个 case 跑完会清理中间文件（.inp/.odb/.com/.dat/.msg/.prt/.sim/.sta/.lck/.stt）。

体素分辨率: 80×80×80（与 lattice 验证案例一致，对比公平）
"""

import os
import sys
import json
import shutil
import time
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import scripts.batch_simulate_extract as bse


# ============================================================================
# 配置
# ============================================================================

INPUT_JSON = os.path.join(PROJECT_ROOT, 'outputs', 'validation_candidates_dirichlet.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'pareto_validation_dirichlet')
COMPARISON_JSON = os.path.join(PROJECT_ROOT, 'outputs', 'validation_comparison_dirichlet.json')

CPUS = 16            # 物理核数 (Ryzen 9 9950X)
N_GRID = 80          # 体素分辨率，和 lattice 验证一致
RETRY = 2

POINT_KEYS = [
    ('A_min_sigma',     'P1'),   # 最小 σ_hp（左）
    ('C_p3',            'P2'),   # 中间
    ('B_max_stiff',     'P3'),   # 最大 E/ρ（右，高刚度）
]

KEEP_FILES = {'case_summary.json', 'boundary_conditions.txt', 'fem_node_stress.csv'}


# ============================================================================
# 工具
# ============================================================================

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def cleanup_case_dir(case_dir):
    """删除 case 目录里的中间文件，保留 JSON 摘要等。"""
    if not os.path.isdir(case_dir):
        return
    for name in os.listdir(case_dir):
        if name in KEEP_FILES:
            continue
        path = os.path.join(case_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception as e:
            log(f"    [WARN] cleanup failed for {name}: {e}")


def run_single_fem(case_name, point_tag, alpha, delta_m, abaqus_cmd):
    """跑单个 FEM case，返回提取的 case_summary 字典。"""
    case_dir = os.path.join(OUTPUT_DIR, f'{case_name}__{point_tag}')
    os.makedirs(case_dir, exist_ok=True)
    json_path = os.path.join(case_dir, 'case_summary.json')

    if os.path.exists(json_path):
        log(f"  [SKIP] {case_name}__{point_tag} already done, reading cache")
        with open(json_path, 'r') as f:
            return json.load(f)

    log(f"  [1/3] Generating INP (n_grid={N_GRID})...")
    if not bse.generate_inp(0, alpha, delta_m, case_dir, n_grid=N_GRID):
        raise RuntimeError(f'INP generation failed')

    log(f"  [2/3] Running Abaqus (CPUs={CPUS}, retry={RETRY})...")
    sim_ok = False
    for attempt in range(RETRY + 1):
        if bse.run_simulation(case_dir, abaqus_cmd, cpus=CPUS):
            sim_ok = True
            break
        if attempt < RETRY:
            log(f"    Retry {attempt + 1}/{RETRY}...")
            bse.kill_abaqus_processes()
            time.sleep(5)
    if not sim_ok:
        raise RuntimeError(f'Abaqus simulation failed after {RETRY} retries')

    log(f"  [3/3] Extracting summary...")
    if not bse.extract_summary_data(case_dir, abaqus_cmd, delta_m, n_grid=N_GRID):
        raise RuntimeError(f'Summary extraction failed')

    with open(json_path, 'r') as f:
        result = json.load(f)

    cleanup_case_dir(case_dir)
    log(f"  [OK] sigma_d={result['sigma_d']:.2f} MPa, E_eff={result['E_eff']:.0f} MPa, rho={result['rho_rel']:.4f}")

    return result


# ============================================================================
# 主流程
# ============================================================================

def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"找不到候选 JSON: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print(f"Dirichlet 采样 Pareto 闭环 FEM 验证：{len(candidates)} 工况 × 3 候选 = {total} 次 Abaqus")
    print("=" * 80)
    print(f"Input:      {INPUT_JSON}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Comparison: {COMPARISON_JSON}")
    print(f"CPUs:       {CPUS}")
    print(f"n_grid:     {N_GRID} (=80x80x80 voxels)")
    print("=" * 80)

    abaqus_cmd = bse.check_abaqus_available()
    if not abaqus_cmd:
        log("[ERROR] Abaqus not found")
        return
    log(f"Abaqus: {abaqus_cmd}")

    comparison = OrderedDict()
    total = sum(3 for _ in candidates)
    count = 0
    t_start = time.time()

    for case_name, case_data in candidates.items():
        delta_m = case_data['delta_m']
        delta_um = case_data['delta_um']
        case_results = OrderedDict()

        log(f"\n{'#' * 70}")
        log(f"# {case_name}  delta = [{delta_um[0]:.1f}, {delta_um[1]:.1f}, {delta_um[2]:.1f}] um")
        log(f"{'#' * 70}")

        for key, tag in POINT_KEYS:
            count += 1
            cand = case_data[key]
            alpha = cand['alpha']

            elapsed_min = (time.time() - t_start) / 60
            log(f"\n--- [{count}/{total}] {case_name} {tag} ({key}) | elapsed: {elapsed_min:.1f} min ---")
            log(f"  alpha = [{alpha[0]:.4f}, {alpha[1]:.4f}, {alpha[2]:.4f}, {alpha[3]:.4f}]")
            log(f"  delta = {delta_um} um")
            log(f"  min alpha = {min(alpha):.6f}")

            try:
                fem = run_single_fem(case_name, tag, alpha, delta_m, abaqus_cmd)

                err_sigma = (cand['pred_sigma_d'] - fem['sigma_d']) / fem['sigma_d'] * 100
                fem_ss = abs(fem['E_eff']) / fem['rho_rel'] if fem['rho_rel'] > 0 else 0
                err_ss = (cand['pred_specific_stiffness'] - fem_ss) / fem_ss * 100 if fem_ss > 0 else float('nan')
                err_rho = (cand['pred_rho_rel'] - fem['rho_rel']) / fem['rho_rel'] * 100

                case_results[tag] = OrderedDict([
                    ('alpha', alpha),
                    ('pred_sigma_d', cand['pred_sigma_d']),
                    ('fem_sigma_d', float(fem['sigma_d'])),
                    ('err_sigma_pct', float(err_sigma)),
                    ('pred_E_eff_per_rho', cand['pred_specific_stiffness']),
                    ('fem_E_eff_per_rho', float(fem_ss)),
                    ('err_E_per_rho_pct', float(err_ss)),
                    ('pred_rho_rel', cand['pred_rho_rel']),
                    ('fem_rho_rel', float(fem['rho_rel'])),
                    ('err_rho_pct', float(err_rho)),
                ])

                log(f"  COMPARE sigma_d:   NN={cand['pred_sigma_d']:.2f}  FEM={fem['sigma_d']:.2f}  err={err_sigma:+.2f}%")
                log(f"  COMPARE E/rho:     NN={cand['pred_specific_stiffness']:.0f}  FEM={fem_ss:.0f}  err={err_ss:+.2f}%")
                log(f"  COMPARE rho_rel:   NN={cand['pred_rho_rel']:.4f}  FEM={fem['rho_rel']:.4f}  err={err_rho:+.2f}%")

            except Exception as e:
                log(f"  [ERROR] {e}")
                case_results[tag] = {'error': str(e), 'alpha': alpha}

            comparison[case_name] = case_results
            with open(COMPARISON_JSON, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)

    elapsed_total = (time.time() - t_start) / 60
    print("\n" + "=" * 105)
    print(f"All done in {elapsed_total:.1f} min")
    print("=" * 105)

    header = f"{'Case':<18} {'Pt':<4} {'NN sig':>8} {'FEM sig':>8} {'sig err%':>9} {'NN E/r':>9} {'FEM E/r':>9} {'E/r err%':>9} {'rho err%':>9}"
    print(header)
    print("-" * len(header))
    max_abs_err = 0.0
    n_pass = 0
    n_total = 0
    for case_name, case_results in comparison.items():
        for tag in ('P1', 'P2', 'P3'):
            r = case_results.get(tag, {})
            if 'error' in r:
                print(f"{case_name:<18} {tag:<4} ERROR: {r['error']}")
                continue
            n_total += 1
            err_max = max(abs(r['err_sigma_pct']), abs(r['err_E_per_rho_pct']))
            max_abs_err = max(max_abs_err, err_max)
            if err_max < 10:
                n_pass += 1
            print(f"{case_name:<18} {tag:<4} "
                  f"{r['pred_sigma_d']:>8.2f} {r['fem_sigma_d']:>8.2f} {r['err_sigma_pct']:>+8.2f}% "
                  f"{r['pred_E_eff_per_rho']:>9.0f} {r['fem_E_eff_per_rho']:>9.0f} {r['err_E_per_rho_pct']:>+8.2f}% "
                  f"{r['err_rho_pct']:>+8.2f}%")
    print("-" * len(header))
    print(f"Pass rate (max |err| < 10%): {n_pass}/{n_total}")
    print(f"Worst |err|: {max_abs_err:.2f}%")
    print("=" * 105)


if __name__ == '__main__':
    main()
