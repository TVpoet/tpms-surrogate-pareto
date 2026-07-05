# -*- coding: utf-8 -*-
"""
Pareto 前沿 FEM 验证脚本

对代理模型预测的 Pareto 最优解进行 FEM 仿真验证
对比代理模型预测和 FEM 真实结果

使用方法:
    python scripts/verify_pareto.py --alpha "0.937,0.017,0.334,0.102" --delta "0.01,0.01,0.01"
    
    或验证多个解:
    python scripts/verify_pareto.py --file pareto_solutions.csv
"""

# 解决 OpenMP 库冲突问题（必须在导入 numpy/torch 之前设置）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import json
import time
import subprocess
import numpy as np
import pandas as pd
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import L, WALL_THICKNESS_MM
from surrogate_model.model import SurrogateModel, device
from surrogate_model.density_model import DensityModel

# 输出目录
VERIFY_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'verification')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'surrogate_model')

# Abaqus 配置
ABAQUS_CMD = r"D:\ABAQUS2022\commands\abaqus.bat"
SIMULATION_TIMEOUT = 600

# 失效包络面配置
SIGMA_YIELD = 280.0  # 铝合金屈服强度 (MPa)


def plot_failure_envelope_3d(envelope_points, alpha, delta_mm, save_path):
    """绘制带当前加载点的失效包络面 3D 图"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 过滤合理范围内的点（转换为 μm 便于显示）
    max_range = 0.1  # mm
    mask = (np.abs(envelope_points).max(axis=1) < max_range)
    points = envelope_points[mask]
    
    if len(points) > 0:
        # 绘制包络面点云
        scatter = ax.scatter(
            points[:, 0] * 1000,  # mm → μm
            points[:, 1] * 1000, 
            points[:, 2] * 1000, 
            c=np.linalg.norm(points, axis=1) * 1000, 
            cmap='viridis', 
            alpha=0.3, 
            s=10,
            label='Failure Envelope'
        )
        plt.colorbar(scatter, ax=ax, label='Distance (μm)', shrink=0.6)
    
    # 标记当前加载点
    ax.scatter(
        [delta_mm[0] * 1000], 
        [delta_mm[1] * 1000], 
        [delta_mm[2] * 1000],
        c='red', 
        s=200, 
        marker='*',
        edgecolors='black',
        linewidths=2,
        label=f'Current Load δ = {delta_mm} mm',
        zorder=10
    )
    
    # 绘制从原点到当前加载点的连线
    ax.plot([0, delta_mm[0] * 1000], 
            [0, delta_mm[1] * 1000], 
            [0, delta_mm[2] * 1000], 
            'r--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('δx (μm)', fontsize=12)
    ax.set_ylabel('δy (μm)', fontsize=12)
    ax.set_zlabel('δz (μm)', fontsize=12)
    
    title = f'Failure Envelope (σ_yield = {SIGMA_YIELD} MPa)\n'
    title += f'α = [{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}, {alpha[3]:.3f}]'
    ax.set_title(title, fontsize=14)
    
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  包络面图保存: {save_path}")
    plt.close()


def load_models():
    """加载代理模型和密度模型"""
    print("加载模型...")
    
    # 代理模型
    model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到代理模型: {model_path}")
    surrogate = SurrogateModel.load(model_path)
    
    # 密度模型
    density_path = os.path.join(MODEL_DIR, 'density_model.pth')
    if not os.path.exists(density_path):
        raise FileNotFoundError(f"找不到密度模型: {density_path}")
    density = DensityModel.load(density_path)
    
    return surrogate, density


def predict_with_model(surrogate, density, alpha, delta):
    """使用代理模型预测"""
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device).unsqueeze(0)
    delta_t = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # 代理模型预测
        x = torch.cat([alpha_t, delta_t], dim=1)
        y = surrogate(x)
        sigma_d = y[0, 0].item()
        E_eff = abs(y[0, 1].item())
        
        # 密度模型预测
        rho_rel = density(alpha_t).item()
    
    return {
        'sigma_d': sigma_d,
        'E_eff': E_eff,
        'rho_rel': rho_rel,
        'specific_stiffness': E_eff / rho_rel if rho_rel > 0 else 0
    }


def run_fem_simulation(alpha, delta, case_name):
    """运行单次 FEM 仿真"""
    from scripts.batch_simulate_extract import (
        generate_inp, run_simulation, extract_summary_data, check_abaqus_available
    )
    import shutil
    
    # 创建验证目录（如果已存在则删除重建）
    case_dir = os.path.join(VERIFY_DIR, case_name)
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)  # 删除旧目录
    os.makedirs(case_dir)
    
    # 检查 Abaqus
    abaqus_cmd = check_abaqus_available()
    if not abaqus_cmd:
        print("[ERROR] Abaqus 不可用")
        return None
    
    print(f"  生成 INP 文件...")
    if not generate_inp(0, alpha, delta, case_dir, n_grid=80):
        print("[ERROR] INP 文件生成失败")
        return None
    
    print(f"  运行 Abaqus 仿真...")
    if not run_simulation(case_dir, abaqus_cmd, cpus=4, timeout=SIMULATION_TIMEOUT):
        print("[ERROR] 仿真失败")
        return None
    
    print(f"  提取结果...")
    if not extract_summary_data(case_dir, abaqus_cmd, delta, n_grid=80):
        print("[ERROR] 数据提取失败")
        return None
    
    # 读取结果
    json_path = os.path.join(case_dir, 'case_summary.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    
    return None


def compare_results(predicted, fem_result):
    """对比预测结果和 FEM 结果"""
    comparison = {
        'sigma_d': {
            'predicted': predicted['sigma_d'],
            'fem': fem_result['sigma_d'],
            'error': abs(predicted['sigma_d'] - fem_result['sigma_d']) / fem_result['sigma_d'] * 100
        },
        'E_eff': {
            'predicted': predicted['E_eff'],
            'fem': abs(fem_result['E_eff']),
            'error': abs(predicted['E_eff'] - abs(fem_result['E_eff'])) / abs(fem_result['E_eff']) * 100
        },
        'rho_rel': {
            'predicted': predicted['rho_rel'],
            'fem': fem_result['rho_rel'],
            'error': abs(predicted['rho_rel'] - fem_result['rho_rel']) / fem_result['rho_rel'] * 100
        }
    }
    return comparison


def print_comparison(alpha, delta_mm, comparison):
    """打印对比结果（单位统一为 MPa 和 mm）"""
    print("\n" + "=" * 75)
    print("验证结果")
    print("=" * 75)
    print(f"结构参数 α = {[f'{a:.4f}' for a in alpha]}")
    print(f"加载条件 δ = [{delta_mm[0]:.4f}, {delta_mm[1]:.4f}, {delta_mm[2]:.4f}] mm")
    print("-" * 75)
    print(f"{'指标':<15} {'单位':<8} {'代理模型':<14} {'FEM':<14} {'误差(%)':<10}")
    print("-" * 75)
    
    # 定义单位
    units = {'sigma_d': 'MPa', 'E_eff': 'MPa', 'rho_rel': '-'}
    
    for key, vals in comparison.items():
        unit = units.get(key, '')
        print(f"{key:<15} {unit:<8} {vals['predicted']:<14.4f} {vals['fem']:<14.4f} {vals['error']:<10.2f}")
    
    # 判断是否通过
    all_errors = [vals['error'] for vals in comparison.values()]
    max_error = max(all_errors)
    avg_error = sum(all_errors) / len(all_errors)
    
    print("-" * 70)
    print(f"平均误差: {avg_error:.2f}%")
    print(f"最大误差: {max_error:.2f}%")
    
    if max_error < 10:
        print("[PASS] 验证通过 (误差 < 10%)")
    elif max_error < 20:
        print("[WARN] 验证一般 (10% < 误差 < 20%)")
    else:
        print("[FAIL] 验证失败 (误差 > 20%)")
    
    return comparison


def verify_single(alpha, delta, case_name="verify"):
    """验证单个 Pareto 解
    
    Args:
        alpha: 结构参数 [4]
        delta: 加载条件（米），用于代理模型和 FEM
        case_name: 验证 case 名称
    """
    delta_mm = [d * 1000 for d in delta]  # m → mm（仅用于显示）
    
    print("\n" + "=" * 70)
    print(f"验证: α = {[f'{a:.4f}' for a in alpha]}")
    print(f"      δ = {delta_mm} mm = {delta} m")
    print("=" * 70)
    
    # 加载模型
    surrogate, density = load_models()
    
    # 代理模型预测（使用米）
    print("\n[1/2] 代理模型预测...")
    predicted = predict_with_model(surrogate, density, alpha, delta)
    print(f"  σ_d = {predicted['sigma_d']:.4f} MPa")
    print(f"  E_eff = {predicted['E_eff']:.4f} MPa")
    print(f"  ρ_rel = {predicted['rho_rel']:.4f}")
    
    # FEM 仿真（也使用米，与 export_inp 一致）
    print("\n[2/2] FEM 仿真...")
    fem_result = run_fem_simulation(alpha, delta, case_name)  # delta 是米
    
    if fem_result is None:
        print("[ERROR] FEM 仿真失败，无法验证")
        return None
    
    print(f"  σ_d = {fem_result['sigma_d']:.4f} MPa")
    print(f"  E_eff = {abs(fem_result['E_eff']):.4f} MPa")
    print(f"  ρ_rel = {fem_result['rho_rel']:.4f}")
    
    # 对比
    comparison = compare_results(predicted, fem_result)
    print_comparison(alpha, delta_mm, comparison)
    
    # 保存验证报告
    report = {
        'alpha': alpha,
        'delta': delta,
        'predicted': predicted,
        'fem': fem_result,
        'comparison': comparison
    }
    
    report_path = os.path.join(VERIFY_DIR, f'{case_name}_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告保存: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Pareto 前沿 FEM 验证')
    parser.add_argument('--alpha', type=str, required=True,
                        help='结构参数 α，格式: "α1,α2,α3,α4"')
    parser.add_argument('--delta', type=str, required=True,
                        help='加载条件 δ，格式: "δx,δy,δz" (米)')
    parser.add_argument('--name', type=str, default='verify',
                        help='验证 case 名称')
    
    args = parser.parse_args()
    
    # 解析参数
    alpha = [float(x) for x in args.alpha.split(',')]
    delta = [float(x) for x in args.delta.split(',')]
    
    if len(alpha) != 4:
        print("[ERROR] α 必须是 4 个值")
        return
    if len(delta) != 3:
        print("[ERROR] δ 必须是 3 个值")
        return
    
    # 归一化 α，确保和精确为 1（避免浮点精度问题）
    alpha_sum = sum(alpha)
    alpha = [a / alpha_sum for a in alpha]
    print(f"归一化后 α = {alpha} (和 = {sum(alpha):.6f})")
    
    # δ 单位转换：用户输入毫米，模型需要米
    delta_m = [d / 1000 for d in delta]  # mm → m
    print(f"δ 输入: {delta} mm → 转换为: {delta_m} m")
    
    os.makedirs(VERIFY_DIR, exist_ok=True)
    
    # 运行验证（代理模型和 FEM 都用米）
    verify_single(alpha, delta_m, args.name)


if __name__ == '__main__':
    main()
