# -*- coding: utf-8 -*-
"""Pareto 搜索与纯构型基准的共享工具。"""

import json
import os

import numpy as np
import torch

from surrogate_model.model import device


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'verification')
BASELINE_FEM_DIR = os.path.join(VERIFY_DIR, 'pure_baselines')

# 4 种 TPMS 基函数名称（对应 geometry.py 中的 f1-f4）
TPMS_NAMES = ['$f_1$', '$f_2$', '$f_3$', '$f_4$']
PURE_LABELS = ['f1', 'f2', 'f3', 'f4']

PURE_ALPHAS = [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 1.0]),
]

PURE_MARKERS = ['D', 's', '^', 'v']
PURE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']


def predict_rho_rel(density_model, alpha: np.ndarray) -> float:
    """使用密度网络预测相对密度。"""
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        rho = density_model(alpha_tensor)
    return rho.item()


def format_delta_tag(delta_target: np.ndarray) -> str:
    """将载荷向量格式化为稳定的文件名标签。"""
    delta_um = np.asarray(delta_target, dtype=float) * 1e6
    parts = []
    for axis, value in zip(('dx', 'dy', 'dz'), delta_um):
        parts.append(f"{axis}{value:.2f}".replace('-', 'm').replace('.', 'p'))
    return '_'.join(parts) + '_um'


def get_baseline_case_dir(pure_label: str,
                          delta_target: np.ndarray,
                          load_case_name: str = None) -> str:
    """返回纯构型基准 FEM 缓存目录。"""
    if load_case_name:
        case_tag = load_case_name.replace(' ', '_')
    else:
        case_tag = format_delta_tag(delta_target)
    case_name = f'baseline_{case_tag}_{pure_label}'
    return os.path.join(BASELINE_FEM_DIR, case_name)


def load_or_run_baseline_fem(alpha: np.ndarray,
                             pure_label: str,
                             delta_target: np.ndarray,
                             load_case_name: str = None,
                             cpus: int = 4,
                             n_grid: int = 80,
                             timeout: int = 1800) -> dict:
    """读取或补跑纯构型基准 FEM 结果。"""
    import scripts.batch_simulate_extract as batch_extract

    case_dir = get_baseline_case_dir(pure_label, delta_target, load_case_name)
    os.makedirs(case_dir, exist_ok=True)
    json_path = os.path.join(case_dir, 'case_summary.json')

    if not os.path.exists(json_path):
        batch_extract.SIMULATION_TIMEOUT = timeout
        abaqus_cmd = batch_extract.check_abaqus_available()
        if not abaqus_cmd:
            raise RuntimeError('Abaqus is not available, cannot compute FEM baselines.')
        print(f"  [FEM baseline] {pure_label} -> {case_dir}")
        if not batch_extract.generate_inp(0, alpha.tolist(), delta_target.tolist(), case_dir, n_grid=n_grid):
            raise RuntimeError(f'Failed to generate INP for baseline {pure_label}.')
        if not batch_extract.run_simulation(case_dir, abaqus_cmd, cpus=cpus, timeout=timeout):
            raise RuntimeError(f'Failed to run FEM simulation for baseline {pure_label}.')
        if not batch_extract.extract_summary_data(case_dir, abaqus_cmd, delta_target.tolist(), n_grid=n_grid):
            raise RuntimeError(f'Failed to extract FEM summary for baseline {pure_label}.')

    with open(json_path, 'r', encoding='utf-8') as f:
        fem = json.load(f)

    rho_rel = float(fem['rho_rel'])
    e_eff = abs(float(fem['E_eff']))
    return {
        'sigma_d': float(fem['sigma_d']),
        'E_eff': e_eff,
        'rho_rel': rho_rel,
        'specific_stiffness': e_eff / rho_rel if rho_rel > 0 else 0.0,
        'case_dir': case_dir,
    }


def find_pareto_front(costs: np.ndarray) -> np.ndarray:
    """找到 2D Pareto 前沿（两个目标都最小化），O(n log n)。"""
    sorted_idx = np.lexsort((costs[:, 1], costs[:, 0]))
    pareto_mask = np.zeros(len(costs), dtype=bool)
    min_second = np.inf

    for i in sorted_idx:
        if costs[i, 1] < min_second:
            pareto_mask[i] = True
            min_second = costs[i, 1]

    return pareto_mask


def evaluate_pure_topologies(model, density_model, delta_target,
                             baseline_source: str = 'predict',
                             load_case_name: str = None):
    """评估 4 个纯 TPMS 构型在给定载荷下的性能。"""
    results = []

    if baseline_source == 'predict':
        model.eval()
        density_model.eval()
        delta_tensor = torch.tensor(delta_target, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            for i, alpha in enumerate(PURE_ALPHAS):
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device).unsqueeze(0)
                x = torch.cat([alpha_tensor, delta_tensor], dim=1)
                y = model(x)

                sigma_d = y[0, 0].item()
                e_eff = abs(y[0, 1].item())
                rho_rel = predict_rho_rel(density_model, alpha)
                specific_stiffness = e_eff / rho_rel if rho_rel > 0 else 0.0

                results.append({
                    'name': TPMS_NAMES[i],
                    'alpha': alpha,
                    'sigma_d': sigma_d,
                    'E_eff': e_eff,
                    'rho_rel': rho_rel,
                    'specific_stiffness': specific_stiffness,
                    'source': 'predict',
                })
    elif baseline_source == 'fem':
        for i, alpha in enumerate(PURE_ALPHAS):
            fem = load_or_run_baseline_fem(
                alpha, PURE_LABELS[i], delta_target, load_case_name=load_case_name
            )
            results.append({
                'name': TPMS_NAMES[i],
                'alpha': alpha,
                'sigma_d': fem['sigma_d'],
                'E_eff': fem['E_eff'],
                'rho_rel': fem['rho_rel'],
                'specific_stiffness': fem['specific_stiffness'],
                'source': 'fem',
                'case_dir': fem['case_dir'],
            })
    else:
        raise ValueError(f"Unsupported baseline_source: {baseline_source}")

    return results
