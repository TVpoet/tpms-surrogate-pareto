# -*- coding: utf-8 -*-
"""
通用网格收敛性报告：给定算例目录与网格列表，
1) 汇总 case_summary.json -> {base}/convergence_table.csv
2) 画合并多 Y 轴图(sigma_hp 红 / E_eff 蓝 / rho_rel 绿)-> 指定 png

用法:
    python scripts/convergence_report.py \
        --base outputs/mesh_convergence_uniform \
        --grids 60 70 80 90 100 110 120 \
        --out outputs/mesh_convergence/convergence_plot.png
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_SIG = 'red'
C_E = 'blue'
C_RHO = '#27ae60'


def count_elements_from_inp(inp_path):
    in_elements = False
    count = 0
    with open(inp_path, 'r') as f:
        for line in f:
            if line.startswith('*Element, type=C3D8R'):
                in_elements = True
                continue
            if in_elements:
                if line.startswith('*'):
                    break
                count += 1
    return count


def build_table(base, grids):
    import json
    rows = []
    for n in grids:
        cdir = os.path.join(base, f'n{n}')
        with open(os.path.join(cdir, 'case_summary.json')) as f:
            s = json.load(f)
        n_nodes = len(pd.read_csv(os.path.join(cdir, 'fem_node_stress.csv')))
        inp = os.path.join(cdir, 'tpms_mesh.inp')
        csvp = os.path.join(base, 'convergence_table.csv')
        if os.path.exists(inp):
            n_elem = count_elements_from_inp(inp)
        else:
            prev = pd.read_csv(csvp)
            n_elem = int(prev[prev['n_grid'] == n].iloc[0]['n_elements'])
        rows.append({
            'n_grid': n, 'n_nodes': n_nodes, 'n_elements': n_elem,
            'sigma_d': s['sigma_d'], 'sigma_max': s['sigma_max'],
            'E_eff': s['E_eff'], 'rho_rel': n_elem / (n - 1) ** 3,
        })
    df = pd.DataFrame(rows).sort_values('n_grid')
    out_csv = os.path.join(base, 'convergence_table.csv')
    df.to_csv(out_csv, index=False)
    print(f'Table saved: {out_csv}')
    print(df.to_string(index=False))
    return df


def plot_combined(df, out_png, sig_max, e_max, rho_max, xticks, sig_min=0, e_min=0, rho_min=0):
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'Microsoft YaHei'],
        'font.serif': ['Times New Roman'],
        'font.sans-serif': ['Microsoft YaHei'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'axes.labelsize': 26,
        'xtick.labelsize': 26, 'ytick.labelsize': 26,
    })
    n = df['n_grid'].values
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.95)
    ax2 = ax1.twinx()

    ax1.plot(n, df['sigma_d'], 'o-', color=C_SIG, lw=2, ms=8)
    ax2.plot(n, df['E_eff'] / 1000.0, 's-', color=C_E, lw=2, ms=8)

    ax1.set_ylim(sig_min, sig_max)
    ax2.set_ylim(e_min, e_max)
    ax1.set_xlim(n.min() - 4, n.max() + 4)
    ax1.set_xticks(xticks)

    ax1.set_xlabel(r'Grid resolution $n$')
    ax1.set_ylabel(r'$\sigma_{\mathrm{hp}}$ (MPa)', color=C_SIG)
    ax2.set_ylabel(r'$E_{\mathrm{eff}}$ (GPa)', color=C_E)

    ax1.spines['left'].set_color(C_SIG)
    ax1.tick_params(axis='y', which='both', direction='in', colors=C_SIG)
    ax2.spines['right'].set_color(C_E)
    ax2.tick_params(axis='y', which='both', direction='in', colors=C_E)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    ax1.tick_params(axis='x', which='both', direction='in', top=False, labeltop=False)
    for side in ('top', 'bottom'):
        ax1.spines[side].set_visible(True)
        ax1.spines[side].set_color('black')
    ax1.spines['right'].set_visible(False)

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_svg = os.path.splitext(out_png)[0] + '.svg'
    plt.savefig(out_svg, bbox_inches='tight')
    print(f'Plot saved: {out_png}')
    print(f'Plot saved: {out_svg}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--grids', type=int, nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sig_min', type=float, default=52)
    p.add_argument('--sig_max', type=float, default=65)
    p.add_argument('--e_min', type=float, default=21.2)
    p.add_argument('--e_max', type=float, default=22.5)
    p.add_argument('--rho_min', type=float, default=0.36)
    p.add_argument('--rho_max', type=float, default=0.37)
    p.add_argument('--xticks', type=int, nargs='+', default=[60, 80, 100, 120])
    args = p.parse_args()

    base = args.base if os.path.isabs(args.base) else os.path.join(PROJECT_ROOT, args.base)
    out = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    df = build_table(base, args.grids)
    plot_combined(df, out, args.sig_max, args.e_max, args.rho_max, args.xticks,
                  args.sig_min, args.e_min, args.rho_min)


if __name__ == '__main__':
    main()
