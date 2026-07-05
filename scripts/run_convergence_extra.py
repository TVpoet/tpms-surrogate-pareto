# -*- coding: utf-8 -*-
"""
网格收敛性算例：复用 batch_simulate_extract 的四步管线，
对固定构型在指定网格分辨率下跑单个 FEM 算例。

默认工况（可用 --alpha / --out_base 覆盖）：
  - alpha = [0.25, 0.50, 0.10, 0.15]  (rho≈0.365)
  - displacement = (2, 2, 2) um 等三轴
  - Solid 体素网格, C3D8R

用法:
    python scripts/run_convergence_extra.py --n_grids 60 70 80
    python scripts/run_convergence_extra.py --n_grids 120 --alpha 0.25 0.25 0.25 0.25 --timeout 3600

输出到 {out_base}/n{N}/ (保留 INP 供 convergence_report.py 按单元数重算 rho_rel)
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import scripts.batch_simulate_extract as bse

# 默认工况（论文收敛性研究构型）
ALPHA = [0.25, 0.50, 0.10, 0.15]
DISPLACEMENT = [2e-6, 2e-6, 2e-6]  # m, 等三轴 2 um
OUT_BASE = os.path.join(PROJECT_ROOT, 'outputs', 'mesh_convergence')


def _clean_sim_files(case_dir):
    """删除上一次失败留下的求解器中间文件，便于重试时干净重跑。"""
    import glob, shutil
    for f in glob.glob(os.path.join(case_dir, 'tpms_mesh.*')):
        if f.endswith('.inp'):
            continue  # 保留 INP，避免重复生成
        try:
            os.remove(f)
        except (OSError, PermissionError):
            pass
    sd = os.path.join(case_dir, 'tpms_mesh.simdir')
    if os.path.isdir(sd):
        shutil.rmtree(sd, ignore_errors=True)


def run_one(n_grid: int, cpus: int, timeout: int, retries: int) -> bool:
    import json
    case_dir = os.path.join(OUT_BASE, f'n{n_grid}')
    os.makedirs(case_dir, exist_ok=True)

    # 每个仿真的超时（超过即 kill），统一覆盖模块级常量
    bse.SIMULATION_TIMEOUT = timeout

    abaqus_cmd = bse.check_abaqus_available()
    if not abaqus_cmd:
        print("[ERROR] 未找到 Abaqus")
        return False

    print(f"\n{'='*60}\nn_grid = {n_grid}  ->  {case_dir}\n{'='*60}")
    print(f"  alpha = {ALPHA}")
    print(f"  disp  = {[d*1000 for d in DISPLACEMENT]} mm (等三轴)")
    print(f"  per-sim timeout = {timeout}s, 最多重试 {retries} 次\n")

    json_path = os.path.join(case_dir, 'case_summary.json')
    odb_path = os.path.join(case_dir, 'tpms_mesh.odb')
    inp_path = os.path.join(case_dir, 'tpms_mesh.inp')

    # Step 1: 生成 INP（只需一次）
    if not os.path.exists(inp_path):
        print("[1/4] 生成 INP ...")
        if not bse.generate_inp(0, ALPHA, DISPLACEMENT, case_dir, n_grid):
            print("[ERROR] INP 生成失败")
            return False
    else:
        print("[1/4] INP 已存在，跳过生成")

    # Step 2-4: 仿真+提取，带超时重试（实现"超 N 分钟 kill 重跑"）
    for attempt in range(1, retries + 1):
        if not os.path.exists(odb_path):
            print(f"[2/4] 运行 Abaqus 仿真 (第 {attempt}/{retries} 次, 超时 {timeout}s) ...")
            ok = bse.run_simulation(case_dir, abaqus_cmd, cpus, timeout=timeout)
            if not ok:
                print(f"[WARN] 第 {attempt} 次仿真失败/超时, kill 并清理后重试")
                bse.kill_abaqus_processes()
                _clean_sim_files(case_dir)
                continue
        else:
            print("[2/4] ODB 已存在，跳过仿真")

        print("[3/4] 提取应力场 ...")
        if not bse.extract_stress_field(case_dir, abaqus_cmd):
            print(f"[WARN] 第 {attempt} 次应力提取失败, 清理后重试")
            bse.kill_abaqus_processes()
            _clean_sim_files(case_dir)
            continue

        print("[4/4] 提取汇总 (sigma_d/sigma_max/E_eff/rho_rel) ...")
        if os.path.exists(json_path):
            os.remove(json_path)  # 强制重算，保证 n_grid 口径正确
        if not bse.extract_summary_data(case_dir, abaqus_cmd, DISPLACEMENT, n_grid=n_grid):
            print(f"[WARN] 第 {attempt} 次汇总提取失败, 清理后重试")
            bse.kill_abaqus_processes()
            _clean_sim_files(case_dir)
            continue

        with open(json_path) as f:
            s = json.load(f)
        print(f"\n[OK] n={n_grid}: sigma_d={s['sigma_d']:.4f}  "
              f"sigma_max={s['sigma_max']:.4f}  E_eff={s['E_eff']:.1f}  "
              f"rho_rel={s['rho_rel']:.6f}")
        bse.kill_abaqus_processes()
        return True

    print(f"[ERROR] n={n_grid} 重试 {retries} 次仍失败")
    return False


def main():
    global ALPHA, OUT_BASE
    p = argparse.ArgumentParser()
    p.add_argument('--n_grids', type=int, nargs='+', required=True)
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--timeout', type=int, default=600)  # 单仿真 10 分钟
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--alpha', type=float, nargs=4, default=ALPHA,
                   help='TPMS 混合系数 [P,G,D,I-WP]，和须为 1')
    p.add_argument('--out_base', type=str, default=OUT_BASE,
                   help='输出根目录（默认 outputs/mesh_convergence）')
    args = p.parse_args()

    ALPHA = list(args.alpha)
    OUT_BASE = args.out_base if os.path.isabs(args.out_base) \
        else os.path.join(PROJECT_ROOT, args.out_base)
    print(f"ALPHA   = {ALPHA}")
    print(f"OUT_BASE = {OUT_BASE}")

    results = {}
    for n in args.n_grids:
        results[n] = run_one(n, args.cpus, args.timeout, args.retries)

    print("\n" + "=" * 50)
    print("汇总:")
    for n, ok in results.items():
        print(f"  n={n}: {'OK' if ok else 'FAILED'}")
    print("=" * 50)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
