# -*- coding: utf-8 -*-
"""
基于 Dirichlet(1,1,1,1) 采样的 Pareto 搜索 + 挑 P1/P2/P3 候选

设计思路:
  - 训练数据是 seed=42 的 Dirichlet 采样
  - Pareto 搜索改用 seed=43 的 Dirichlet 采样（5M 样本），避免评估泄露
  - 不需要 alpha_min 约束——Dirichlet 连续分布自动不到 αᵢ=0 边界
  - 训练分布和推理分布严格一致

输出:
  - outputs/validation_candidates_dirichlet.json  9 个新候选
  - outputs/surrogate_model/pareto_dirichlet_*.png  3 张标注图
"""

import os
import sys
import json
from collections import OrderedDict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': ['Times New Roman', 'Microsoft YaHei'],
    'font.serif': ['Times New Roman'],
    'font.sans-serif': ['Microsoft YaHei'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from surrogate_model.model import SurrogateModel, device
from surrogate_model.density_model import DensityModel
from surrogate_model.pareto_utils import (
    evaluate_pure_topologies, find_pareto_front,
    PURE_MARKERS, PURE_COLORS, TPMS_NAMES,
)


# ============================================================================
# 配置
# ============================================================================

LOAD_CASES = OrderedDict([
    # delta = strain × L 的位移目标 (m)；p3_method = 'arc' (弧长中点) 或 'knee' (拐点)
    # T-1-1-2 Pareto 在 σ~28 一带有不同拓扑路径，特意用 knee 取 Primitive 主导那条
    ('Biaxial-XY',     {'delta': np.array([2e-6, 2e-6, 0.0]),  'p3_method': 'arc'}),
    ('Triaxial-1-1-2', {'delta': np.array([1e-6, 1e-6, 2e-6]), 'p3_method': 'knee'}),
    ('Triaxial-1-2-2', {'delta': np.array([1e-6, 2e-6, 2e-6]), 'p3_method': 'arc'}),
    ('Uniaxial-X',     {'delta': np.array([2e-6, 0.0, 0.0]),   'p3_method': 'arc'}),  # 单轴 X，不进 paper 主线
])

N_SAMPLES = 5_000_000      # Pareto 搜索的 Dirichlet 样本数
SEED_PARETO = 43           # 关键：不同于训练的 seed=42，避免重叠
BATCH_SIZE = 200_000       # GPU 推理 batch 大小

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'validation_candidates_dirichlet.json')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'surrogate_model')

MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'surrogate_model')
SURROGATE_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
DENSITY_PATH = os.path.join(MODEL_DIR, 'density_model.pth')


# ============================================================================
# 核心计算
# ============================================================================

def compute_pareto_dirichlet(surr, dens, delta, n_samples=N_SAMPLES, seed=SEED_PARETO):
    """用 Dirichlet 采样做 Pareto 搜索."""
    print(f"  生成 {n_samples:,} 个 Dirichlet(1,1,1,1) 样本 (seed={seed})...")
    rng = np.random.default_rng(seed)
    alpha_all = rng.dirichlet([1.0, 1.0, 1.0, 1.0], n_samples)

    # GPU batch 推理
    print(f"  NN 推理 (batch={BATCH_SIZE:,})...")
    delta_t = torch.tensor(delta, dtype=torch.float32, device=device)

    sigma_list, E_list, rho_list = [], [], []
    for i in range(0, n_samples, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n_samples)
        alpha_batch = torch.tensor(alpha_all[i:end], dtype=torch.float32, device=device)
        delta_batch = delta_t.unsqueeze(0).expand(end - i, -1)
        x = torch.cat([alpha_batch, delta_batch], dim=1)
        with torch.no_grad():
            y = surr(x)
            rho = dens(alpha_batch).squeeze(-1)
        sigma_list.append(y[:, 0].cpu().numpy())
        E_list.append(torch.abs(y[:, 1]).cpu().numpy())
        rho_list.append(rho.cpu().numpy())

    sigma = np.concatenate(sigma_list)
    E = np.concatenate(E_list)
    rho = np.concatenate(rho_list)

    valid = (sigma > 0) & (E > 0) & (rho > 0)
    print(f"  有效样本: {int(valid.sum()):,} / {n_samples:,}")

    alpha = alpha_all[valid]
    sigma = sigma[valid]
    E = E[valid]
    rho = rho[valid]
    ss = E / rho

    costs = np.column_stack([sigma, -ss])
    pareto_mask = find_pareto_front(costs)
    print(f"  Pareto 前沿: {int(pareto_mask.sum()):,} 个")

    return {
        'alpha': alpha,
        'sigma_d': sigma,
        'E_eff': E,
        'rho_rel': rho,
        'specific_stiffness': ss,
    }, pareto_mask


def pick_three(sigma, ss, method='arc'):
    """挑 A (min sigma), B (max ss), C (中间点/P2, by `method`).

    method:
      - 'arc'  : 弧长中点（normalize 两轴后沿 Pareto 走半弧长位置）—— 平滑 Pareto 视觉居中
      - 'knee' : 拐点（normalize 两轴后到前沿端点弦的最大垂直距离）—— L 形 Pareto 视觉卡角

    两种方法都：
      - 返回真实 Pareto 点索引，可造、可 FEM 验证
      - 对间断 Pareto 鲁棒（argmin/argmax 即可）
    """
    idx_A = int(np.argmin(sigma))
    idx_B = int(np.argmax(ss))

    # 按 σ 升序排序，使 Pareto 点沿曲线从低应力端排到高比刚度端
    order = np.argsort(sigma)
    σ_sorted = sigma[order]
    E_sorted = ss[order]

    # 归一化两轴到 [0, 1]
    σ_range = σ_sorted.max() - σ_sorted.min()
    E_range = E_sorted.max() - E_sorted.min()
    σ_norm = (σ_sorted - σ_sorted.min()) / σ_range if σ_range > 0 else np.zeros_like(σ_sorted)
    E_norm = (E_sorted - E_sorted.min()) / E_range if E_range > 0 else np.zeros_like(E_sorted)

    if method == 'knee':
        # 弦：从归一化低应力端到高比刚度端；距弦垂直距离最大点 = knee
        x1, y1 = σ_norm[0],  E_norm[0]
        x2, y2 = σ_norm[-1], E_norm[-1]
        denom = float(np.hypot(y2 - y1, x2 - x1))
        if denom > 0:
            dist = np.abs((y2 - y1) * σ_norm - (x2 - x1) * E_norm + x2 * y1 - y2 * x1) / denom
        else:
            dist = np.zeros_like(σ_norm)
        k_in_sorted = int(np.argmax(dist))
    else:  # 'arc'
        # 相邻段长，累积弧长；半弧长位置最近点
        dx = np.diff(σ_norm)
        dy = np.diff(E_norm)
        seg = np.sqrt(dx ** 2 + dy ** 2)
        L = np.concatenate([[0.0], np.cumsum(seg)])
        L_half = L[-1] / 2.0
        k_in_sorted = int(np.argmin(np.abs(L - L_half)))

    idx_C = int(order[k_in_sorted])
    return idx_A, idx_B, idx_C


def pack_candidate(alpha, sigma, E, rho):
    return {
        'alpha': [float(a) for a in alpha],
        'pred_sigma_d': float(sigma),
        'pred_E_eff': float(E),
        'pred_rho_rel': float(rho),
        'pred_specific_stiffness': float(E / rho) if rho > 0 else 0.0,
    }


# ============================================================================
# 绘图（复用 paper 风格）
# ============================================================================

def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, labeltop=False, labelright=False)


TAG_LAYOUTS = {
    'Biaxial-XY': {
        'P1': {'offset': (-18, -6), 'ha': 'right', 'va': 'center'},
        'P2': {'offset': (10, -18), 'ha': 'left',  'va': 'top'},
        'P3': {'offset': (18, 6),   'ha': 'left',  'va': 'bottom'},
    },
    'Triaxial-1-1-2': {
        'P1': {'offset': (-18, -6), 'ha': 'right', 'va': 'center'},
        'P2': {'offset': (-22, 0),  'ha': 'right', 'va': 'center'},
        'P3': {'offset': (18, -10), 'ha': 'left',  'va': 'top'},
    },
    'Triaxial-1-2-2': {
        'P1': {'offset': (-18, -6), 'ha': 'right', 'va': 'center'},
        'P2': {'offset': (-22, 0),  'ha': 'right', 'va': 'center'},
        'P3': {'offset': (18, -10), 'ha': 'left',  'va': 'top'},
    },
    'Uniaxial-X': {
        'P1': {'offset': (-18, -6), 'ha': 'right', 'va': 'center'},
        'P2': {'offset': (-22, 0),  'ha': 'right', 'va': 'center'},
        'P3': {'offset': (18, -10), 'ha': 'left',  'va': 'top'},
    },
}


def annotate_pareto_point(ax, sigma, ss, tag, layout):
    ax.scatter(sigma, ss, s=60, marker='o',
               facecolors='white', edgecolors='black', linewidths=1.2, zorder=15)
    ox, oy = layout['offset']
    ax.annotate(
        tag, xy=(sigma, ss), xytext=(ox, oy),
        textcoords='offset points', fontsize=10, color='black',
        fontweight='bold', ha=layout['ha'], va=layout['va'],
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.92),
        arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                       shrinkA=2, shrinkB=5, mutation_scale=8),
        zorder=16,
    )


def plot_pareto(case_name, delta, results_all, pareto_mask, idx_A, idx_B, idx_C,
                save_path, surrogate_model, density_model):
    sigma_all = results_all['sigma_d']
    ss_all = results_all['specific_stiffness']
    p_sigma = sigma_all[pareto_mask]
    p_ss = ss_all[pareto_mask]

    fig, ax = plt.subplots(figsize=(6, 5))

    # 灰色背景（采样 50k 防止图太重）
    n_show = min(len(sigma_all[~pareto_mask]), 50_000)
    bg_idx = np.random.RandomState(0).choice(np.sum(~pareto_mask), n_show, replace=False)
    bg_sigma = sigma_all[~pareto_mask][bg_idx]
    bg_ss = ss_all[~pareto_mask][bg_idx]
    ax.scatter(bg_sigma, bg_ss, c='lightgray', alpha=0.3, s=3, zorder=1, rasterized=True)

    # Pareto 前沿
    ax.scatter(p_sigma, p_ss, c='red', s=10, edgecolors='black', linewidths=0.3,
               label='Pareto front', zorder=10)

    # 纯构型 FEM baseline（若无缓存则退回 NN 预测，避免触发新 FEM 跑）
    try:
        pure_results = evaluate_pure_topologies(
            surrogate_model, density_model, delta,
            baseline_source='fem', load_case_name=case_name,
        )
        label_suffix = 'FEM'
    except Exception as e:
        print(f"  [WARN] FEM baseline 不可用 ({e})，回退到 NN 预测")
        pure_results = evaluate_pure_topologies(
            surrogate_model, density_model, delta,
            baseline_source='predict',
        )
        label_suffix = 'NN'
    for i, r in enumerate(pure_results):
        ax.scatter(r['sigma_d'], r['specific_stiffness'],
                   marker=PURE_MARKERS[i], c=PURE_COLORS[i],
                   s=40, edgecolors='black', linewidths=0.9,
                   label=f"{TPMS_NAMES[i]} ({label_suffix})", zorder=20)

    # P1 = 低应力端，P2 = 中间点，P3 = 高比刚度端
    layouts = TAG_LAYOUTS.get(case_name, TAG_LAYOUTS['Biaxial-XY'])
    for tag, idx in [('P1', idx_A), ('P2', idx_C), ('P3', idx_B)]:
        annotate_pareto_point(ax, p_sigma[idx], p_ss[idx], tag, layouts[tag])

    ax.set_xlabel(r'$\sigma_{\mathrm{hp}}$ (MPa)', fontsize=11)
    ax.set_ylabel(r'$E_{\mathrm{eff}} / \rho_{\mathrm{rel}}$ (MPa)', fontsize=11)
    ax.tick_params(labelsize=9)
    delta_um = delta * 1e6
    ax.set_title(
        rf'{case_name} (Dirichlet, seed={SEED_PARETO}): '
        rf'$\delta$ = [{delta_um[0]:.1f}, {delta_um[1]:.1f}, {delta_um[2]:.1f}] $\mu\mathrm{{m}}$',
        fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=0.9)
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot: {save_path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 70)
    print("Pareto 闭环 Dirichlet 采样版本")
    print("=" * 70)
    print(f"  采样:       Dirichlet(1,1,1,1)")
    print(f"  样本数:     {N_SAMPLES:,}")
    print(f"  Pareto seed: {SEED_PARETO} (训练用 42, 互相独立)")
    print(f"  alpha_min:  无（Dirichlet 自动远离边界）")
    print("=" * 70)

    surr = SurrogateModel.load(SURROGATE_PATH)
    dens = DensityModel.load(DENSITY_PATH)

    results_dict = OrderedDict()

    for name, cfg in LOAD_CASES.items():
        delta = cfg['delta']
        p3_method = cfg['p3_method']
        print(f"\n--- {name}  (P2 method: {p3_method}) ---")
        delta_um = delta * 1e6
        print(f"  delta = [{delta_um[0]:.1f}, {delta_um[1]:.1f}, {delta_um[2]:.1f}] um")

        results, mask = compute_pareto_dirichlet(surr, dens, delta)
        p_alpha = results['alpha'][mask]
        p_sigma = results['sigma_d'][mask]
        p_E = results['E_eff'][mask]
        p_rho = results['rho_rel'][mask]
        p_ss = p_E / p_rho

        idx_A, idx_B, idx_C = pick_three(p_sigma, p_ss, method=p3_method)

        c_record = pack_candidate(p_alpha[idx_C], p_sigma[idx_C], p_E[idx_C], p_rho[idx_C])
        # 保留历史字段名以兼容既有结果；C_p3 当前映射为显示标签 P2。
        c_record['p3_method'] = p3_method
        case_record = OrderedDict([
            ('delta_um', [float(x) for x in delta_um]),
            ('delta_m', [float(x) for x in delta]),
            ('pareto_size', int(mask.sum())),
            ('p3_method', p3_method),
            ('A_min_sigma', pack_candidate(p_alpha[idx_A], p_sigma[idx_A], p_E[idx_A], p_rho[idx_A])),
            ('B_max_stiff', pack_candidate(p_alpha[idx_B], p_sigma[idx_B], p_E[idx_B], p_rho[idx_B])),
            ('C_p3',        c_record),
        ])
        results_dict[name] = case_record

        # 打印
        for tag, idx, rec in [
            ('P1 (min σ)',          idx_A, case_record['A_min_sigma']),
            (f'P2 ({p3_method})',   idx_C, case_record['C_p3']),
            ('P3 (max E/ρ)',        idx_B, case_record['B_max_stiff']),
        ]:
            a = rec['alpha']
            print(f"    [{tag:<15}] α=[{a[0]:.4f},{a[1]:.4f},{a[2]:.4f},{a[3]:.4f}] "
                  f"σ={rec['pred_sigma_d']:.2f}  E/ρ={rec['pred_specific_stiffness']:.0f}  "
                  f"ρ={rec['pred_rho_rel']:.4f}  min α={min(a):.4f}")

        # 画图
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = os.path.join(PLOTS_DIR, f'pareto_dirichlet_{name}.png')
        plot_pareto(name, delta, results, mask, idx_A, idx_B, idx_C, plot_path, surr, dens)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] JSON: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
