# -*- coding: utf-8 -*-
"""
批量仿真 + 提取应力场 (一键脚本)

功能:
1. 生成 (α, BC) 组合
2. 为每个组合生成 INP 文件
3. 运行 Abaqus 仿真
4. 从 ODB 提取应力场数据
5. 保存到 fem_data/parameterized/

使用方法:
    python scripts/batch_simulate_extract.py --n_cases 50 --cpus 4

参考: 本项目的批量 Abaqus 仿真模式
"""

import os
import sys
import argparse
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import L, E_SOLID, NU, DISPLACEMENT_RANGE, WALL_THICKNESS_MM

# ====================================================================================
# 配置
# ====================================================================================

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'fem_data', 'parameterized')

# Abaqus 命令 (使用完整路径)
ABAQUS_CMD = r'D:\ABAQUS2022\commands\abaqus.bat'

# 默认参数
DEFAULT_N_CASES = 50      # 默认生成 50 组
DEFAULT_CPUS = 4          # 每个仿真的 CPU 数
DEFAULT_N_GRID = 80       # 网格分辨率 (降低以加快仿真)
DEFAULT_RETRY = 2         # 失败重试次数（提高可靠性）
CLEANUP_BATCH_SIZE = 50   # 每多少个 case 执行一次清理

# 仿真超时 (秒)
SIMULATION_TIMEOUT = 600  # 10分钟

# 日志文件
LOG_FILE = os.path.join(PROJECT_ROOT, 'batch_simulation.log')

# CSV 文件路径
TRAINING_DATA_CSV = os.path.join(OUTPUT_DIR, 'training_data.csv')


# ====================================================================================
# 工具函数
# ====================================================================================

def log(msg, also_print=True):
    """记录日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')
    if also_print:
        print(msg)
        sys.stdout.flush()


def kill_abaqus_processes():
    """杀死所有 Abaqus 进程"""
    try:
        # Windows
        subprocess.run(['taskkill', '/F', '/IM', 'standard.exe'], 
                      capture_output=True, timeout=30)
        subprocess.run(['taskkill', '/F', '/IM', 'ABQcaeK.exe'], 
                      capture_output=True, timeout=30)
        subprocess.run(['taskkill', '/F', '/IM', 'pre.exe'], 
                      capture_output=True, timeout=30)
    except:
        pass


def check_abaqus_available():
    """检查 Abaqus 是否可用"""
    
    # 首先检查环境变量
    abaqus_path = os.environ.get('ABAQUS_PATH')
    if abaqus_path:
        for ext in ['.bat', '.exe', '']:
            cmd_path = os.path.join(abaqus_path, f'abaqus{ext}')
            if os.path.exists(cmd_path):
                log(f"  Abaqus 可用 (环境变量): {cmd_path}")
                return cmd_path
    
    # 检查常见安装路径
    common_paths = [
        r"D:\ABAQUS2022\commands",         # 用户实际安装路径
        r"E:\abaqus2023\commands",         # 备用安装路径
        r"C:\SIMULIA\Commands",
        r"D:\SIMULIA\Commands",
        r"E:\SIMULIA\Commands",
        r"C:\SIMULIA\EstProducts\2022\win_b64\code\bin",
        r"D:\SIMULIA\EstProducts\2022\win_b64\code\bin",
        r"C:\SIMULIA\EstProducts\2023\win_b64\code\bin",
        r"D:\SIMULIA\EstProducts\2023\win_b64\code\bin",
        r"E:\SIMULIA\EstProducts\2023\win_b64\code\bin",
        r"C:\Program Files\Dassault Systemes\SimulationServices\V6R2022x\win_b64\code\bin",
        r"D:\Program Files\Dassault Systemes\SimulationServices\V6R2022x\win_b64\code\bin",
    ]
    
    for base_path in common_paths:
        for ext in ['.bat', '.exe', '']:
            cmd_path = os.path.join(base_path, f'abaqus{ext}')
            if os.path.exists(cmd_path):
                log(f"  Abaqus 可用 (自动检测): {cmd_path}")
                return cmd_path
    
    # 检查 PATH 中的命令
    for cmd in ['abaqus', 'abq2024', 'abq2023', 'abq2022', 'abq2021', 'abq2020']:
        try:
            result = subprocess.run(
                [cmd, 'information=release'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                log(f"  Abaqus 可用 (PATH): {cmd}")
                return cmd
        except:
            continue
    
    return None


# ====================================================================================
# (α, BC) 组合生成
# ====================================================================================

def append_case_to_csv(case_dir: str, case_id: int, alpha: list, displacement: list) -> bool:
    """
    将单个 case 的汇总数据追加到 training_data.csv
    
    Returns:
        True 如果成功追加
    """
    import pandas as pd
    import json
    
    json_path = os.path.join(case_dir, 'case_summary.json')
    if not os.path.exists(json_path):
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 构建一行数据
        row = {
            'case_id': f'case_{case_id:05d}',
            'alpha1': alpha[0],
            'alpha2': alpha[1],
            'alpha3': alpha[2],
            'alpha4': alpha[3],
            'delta_x': displacement[0],
            'delta_y': displacement[1],
            'delta_z': displacement[2],
            'sigma_d': data.get('sigma_d', 0),
            'sigma_max': data.get('sigma_max', 0),
            'E_eff': data.get('E_eff', 0),
            'rho_rel': data.get('rho_rel', 0),
            'mesh_volume': data.get('mesh_volume', 0)
        }
        
        # 追加到 CSV
        df_new = pd.DataFrame([row])
        
        if os.path.exists(TRAINING_DATA_CSV):
            # 检查是否已存在
            df_existing = pd.read_csv(TRAINING_DATA_CSV)
            if row['case_id'] in df_existing['case_id'].values:
                return True  # 已存在，跳过
            df_new.to_csv(TRAINING_DATA_CSV, mode='a', header=False, index=False)
        else:
            df_new.to_csv(TRAINING_DATA_CSV, index=False)
        
        return True
    except Exception as e:
        log(f"    [ERROR] CSV 追加失败: {e}")
        return False


def cleanup_case_dir(case_dir: str) -> bool:
    """
    删除 case 目录
    
    Returns:
        True 如果成功删除
    """
    import shutil
    try:
        shutil.rmtree(case_dir)
        return True
    except Exception as e:
        log(f"    [ERROR] 删除失败: {e}")
        return False


def batch_cleanup(successful_cases: list, combinations: np.ndarray):
    """
    批量清理：追加到 CSV 并删除成功的 case 目录
    
    Args:
        successful_cases: [(case_id, case_dir), ...]
        combinations: 所有组合的数组
    """
    if not successful_cases:
        return
    
    log(f"\n    [CLEANUP] 正在处理 {len(successful_cases)} 个成功的 case...")
    
    appended = 0
    cleaned = 0
    
    for case_id, case_dir in successful_cases:
        combo = combinations[case_id]
        alpha = combo[:4].tolist()
        displacement = combo[4:].tolist()
        
        # 追加到 CSV
        if append_case_to_csv(case_dir, case_id, alpha, displacement):
            appended += 1
            # 只有追加成功才删除
            if cleanup_case_dir(case_dir):
                cleaned += 1
    
    log(f"    [CLEANUP] 完成: {appended} 条追加到 CSV, {cleaned} 个目录已删除")



def generate_combinations(n_cases: int, 
                          displacement_range: tuple = DISPLACEMENT_RANGE,
                          seed: int = 42) -> np.ndarray:
    """
    生成 (α, BC) 组合
    
    Args:
        n_cases: 组合数量
        displacement_range: 位移范围 (min, max)
        seed: 随机种子
    
    Returns:
        combinations: [n_cases, 7] 数组，格式 [α1, α2, α3, α4, δx, δy, δz]
    """
    np.random.seed(seed)
    
    # α 参数: Dirichlet 分布 (和为1)
    alphas = np.random.dirichlet([1, 1, 1, 1], n_cases)
    
    # BC 参数: 三轴加载 (三个方向都随机采样)
    delta_x = np.random.uniform(displacement_range[0], displacement_range[1], n_cases)
    delta_y = np.random.uniform(displacement_range[0], displacement_range[1], n_cases)
    delta_z = np.random.uniform(displacement_range[0], displacement_range[1], n_cases)
    deltas = np.column_stack([delta_x, delta_y, delta_z])
    
    combinations = np.column_stack([alphas, deltas])
    
    return combinations


def save_metadata(combinations: np.ndarray, output_dir: str):
    """保存 metadata.csv"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'case_id': [f'case_{i:05d}' for i in range(len(combinations))],
        'alpha1': combinations[:, 0],
        'alpha2': combinations[:, 1],
        'alpha3': combinations[:, 2],
        'alpha4': combinations[:, 3],
        'delta_x': combinations[:, 4],
        'delta_y': combinations[:, 5],
        'delta_z': combinations[:, 6],
    })
    
    csv_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(csv_path, index=False)
    log(f"  保存 metadata: {csv_path}")
    
    return csv_path


# ====================================================================================
# INP 文件生成
# ====================================================================================

def generate_inp(case_id: int, alpha: list, displacement: list, 
                 case_dir: str, n_grid: int = DEFAULT_N_GRID) -> bool:
    """
    为单个 case 生成 INP 文件
    
    Returns:
        True 如果成功
    """
    try:
        from scripts.abaqus_export import TPMSMeshGenerator
        
        generator = TPMSMeshGenerator(L=L, E_solid=E_SOLID, nu=NU)
        
        # 静默模式
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            result = generator.export_inp(
                alpha=alpha,
                displacement=displacement,
                n_grid=n_grid,
                output_dir=case_dir,
                mesh_mode='solid',
                solid_type='C3D8R'
            )
        
        # 检查 INP 文件是否生成
        inp_path = os.path.join(case_dir, 'tpms_mesh.inp')
        return os.path.exists(inp_path)
        
    except Exception as e:
        log(f"    [ERROR] 生成 INP 失败: {e}")
        return False


# ====================================================================================
# Abaqus 仿真
# ====================================================================================

def run_simulation(case_dir: str, abaqus_cmd: str, cpus: int, 
                   timeout: int = SIMULATION_TIMEOUT) -> bool:
    """
    运行单个 Abaqus 仿真
    
    Returns:
        True 如果成功
    """
    job_name = 'tpms_mesh'
    odb_path = os.path.join(case_dir, f'{job_name}.odb')
    
    # 检查是否已完成
    if os.path.exists(odb_path):
        return True
    
    # 删除旧文件
    for ext in ['.odb', '.lck', '.sta', '.msg', '.dat', '.com', '.prt', '.sim', '.stt']:
        old_file = os.path.join(case_dir, f'{job_name}{ext}')
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
            except:
                pass
    
    # 构建命令
    cmd = [
        abaqus_cmd,
        f'job={job_name}',
        f'input={job_name}.inp',
        f'cpus={cpus}',
        'interactive'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return os.path.exists(odb_path)
        
    except subprocess.TimeoutExpired:
        log(f"    [TIMEOUT] 仿真超时")
        kill_abaqus_processes()
        return False
    except Exception as e:
        log(f"    [ERROR] 仿真错误: {e}")
        return False


# ====================================================================================
# 应力场提取
# ====================================================================================

def extract_stress_field(case_dir: str, abaqus_cmd: str) -> bool:
    """
    从 ODB 提取应力场数据
    
    由于需要 odbAccess，必须用 abaqus python 运行
    
    Returns:
        True 如果成功
    """
    odb_path = os.path.join(case_dir, 'tpms_mesh.odb')
    csv_path = os.path.join(case_dir, 'fem_node_stress.csv')
    
    # 检查是否已提取
    if os.path.exists(csv_path):
        return True
    
    # 检查 ODB 是否存在
    if not os.path.exists(odb_path):
        return False
    
    # 创建提取脚本
    extract_script = os.path.join(case_dir, '_extract_stress.py')
    
    # 使用 string.Template 替代 .format() 避免花括号冲突
    script_content = '''# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

def extract(odb_path, output_csv):
    from odbAccess import openOdb
    from abaqusConstants import ELEMENT_NODAL
    
    odb = openOdb(path=odb_path, readOnly=True)
    
    # Get last step and frame
    step_name = odb.steps.keys()[-1]
    step = odb.steps[step_name]
    frame = step.frames[-1]
    
    # Get instance
    instance_name = odb.rootAssembly.instances.keys()[0]
    instance = odb.rootAssembly.instances[instance_name]
    
    # Extract nodes
    node_coords = {}
    for node in instance.nodes:
        node_coords[node.label] = node.coordinates
    
    # Extract stress at nodes
    stress_field = frame.fieldOutputs['S']
    stress_at_nodes = stress_field.getSubset(position=ELEMENT_NODAL)
    
    node_stress = {}
    for value in stress_at_nodes.values:
        node_label = value.nodeLabel
        if node_label not in node_stress:
            node_stress[node_label] = {'S11': [], 'S22': [], 'S33': [], 'S12': [], 'S13': [], 'S23': [], 'Mises': []}
        
        data = value.data
        node_stress[node_label]['S11'].append(data[0])
        node_stress[node_label]['S22'].append(data[1])
        node_stress[node_label]['S33'].append(data[2])
        node_stress[node_label]['S12'].append(data[3])
        node_stress[node_label]['S13'].append(data[4] if len(data) >= 6 else 0.0)
        node_stress[node_label]['S23'].append(data[5] if len(data) >= 6 else 0.0)
        node_stress[node_label]['Mises'].append(value.mises)
    
    # Average and save
    with open(output_csv, 'w') as f:
        f.write("NodeID,X,Y,Z,S11,S22,S33,S12,S13,S23,Mises\\n")
        for node_label in sorted(node_coords.keys()):
            if node_label in node_stress:
                coords = node_coords[node_label]
                stress = node_stress[node_label]
                line = "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\\n" % (
                    node_label,
                    coords[0], coords[1], coords[2],
                    sum(stress['S11'])/len(stress['S11']),
                    sum(stress['S22'])/len(stress['S22']),
                    sum(stress['S33'])/len(stress['S33']),
                    sum(stress['S12'])/len(stress['S12']),
                    sum(stress['S13'])/len(stress['S13']),
                    sum(stress['S23'])/len(stress['S23']),
                    sum(stress['Mises'])/len(stress['Mises'])
                )
                f.write(line)
    
    odb.close()
    print("OK")

# Run
odb_path = r"$odb_path"
output_csv = r"$csv_path"
extract(odb_path, output_csv)
'''
    # 使用 Template 替换变量
    from string import Template
    script_content = Template(script_content).substitute(odb_path=odb_path, csv_path=csv_path)
    
    with open(extract_script, 'w') as f:
        f.write(script_content)
    
    # 运行提取脚本
    try:
        result = subprocess.run(
            [abaqus_cmd, 'python', extract_script],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=SIMULATION_TIMEOUT  # 使用统一的超时配置
        )
        
        # 清理临时脚本
        try:
            os.remove(extract_script)
        except:
            pass
        
        return os.path.exists(csv_path)
        
    except Exception as e:
        log(f"    [ERROR] 提取失败: {e}")
        return False


# ====================================================================================
# 汇总数据提取 (σ_d, E_eff, ρ_rel)
# ====================================================================================

def extract_summary_data(case_dir: str, abaqus_cmd: str, displacement: list, n_grid: int = DEFAULT_N_GRID) -> bool:
    """
    从 ODB 提取汇总数据：σ_d, E_eff, ρ_rel

    Args:
        case_dir: case 目录路径
        abaqus_cmd: Abaqus 命令
        displacement: 边界位移 [δx, δy, δz] (mm)
        n_grid: 网格分辨率（节点数/边），用于体素计数法计算 ρ_rel

    Returns:
        True 如果成功
    """
    odb_path = os.path.join(case_dir, 'tpms_mesh.odb')
    json_path = os.path.join(case_dir, 'case_summary.json')
    
    # 检查是否已提取
    if os.path.exists(json_path):
        return True
    
    # 检查 ODB 是否存在
    if not os.path.exists(odb_path):
        return False
    
    # 创建提取脚本
    extract_script = os.path.join(case_dir, '_extract_summary.py')
    
    # 单胞尺寸 (mm)
    L_mm = L * 1000  # config.L 是 m，转为 mm
    
    script_content = '''# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import json

def extract_summary(odb_path, output_json, displacement, L_mm, n_grid):
    from odbAccess import openOdb
    from abaqusConstants import ELEMENT_NODAL
    
    odb = openOdb(path=odb_path, readOnly=True)
    
    # Get last step and frame
    step_name = odb.steps.keys()[-1]
    step = odb.steps[step_name]
    frame = step.frames[-1]
    
    # Get instance
    instance_name = odb.rootAssembly.instances.keys()[0]
    instance = odb.rootAssembly.instances[instance_name]
    
    # ============================================
    # 1. 提取 Mises 应力，计算 sigma_d
    # ============================================
    stress_field = frame.fieldOutputs['S']
    stress_at_nodes = stress_field.getSubset(position=ELEMENT_NODAL)
    
    mises_list = []
    for value in stress_at_nodes.values:
        mises_list.append(value.mises)
    
    # 99.9th percentile 危险应力
    mises_sorted = sorted(mises_list)  # 升序
    n = len(mises_sorted)
    idx_99_9 = int(n * 0.999)  # 99.9 百分位索引
    sigma_d = mises_sorted[idx_99_9] if idx_99_9 < n else mises_sorted[-1]
    sigma_max = mises_sorted[-1]  # 最大值
    
    # ============================================
    # 2. 计算 ρ_rel（体素计数法）
    # ============================================
    n_elements = len(instance.elements)
    n_cells = (n_grid - 1) ** 3
    rho_rel = float(n_elements) / float(n_cells)
    mesh_volume = rho_rel * (L_mm ** 3)
    
    # ============================================
    # 3. 计算 E_eff (从反力 - 只取正面节点)
    # ============================================
    # 获取反力 (Abaqus Repository 不支持 .get())
    try:
        rf_field = frame.fieldOutputs['RF']
    except KeyError:
        rf_field = None
    E_eff = 0.0
    
    if rf_field is not None:
        # 注意：displacement 传入时是 m 单位，需要转换为 mm 与 L_mm 一致
        import math
        dx, dy, dz = [d * 1000 for d in displacement]  # m -> mm
        
        d_eff = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if d_eff > 1e-10:
            # ---- 构建节点坐标字典 ----
            node_coords_rf = {}
            for node in instance.nodes:
                node_coords_rf[node.label] = node.coordinates

            # ---- 确定正面节点 (坐标 ≈ L_mm) ----
            all_x = [c[0] for c in node_coords_rf.values()]
            all_y = [c[1] for c in node_coords_rf.values()]
            all_z = [c[2] for c in node_coords_rf.values()]
            x_max_val = max(all_x)
            y_max_val = max(all_y)
            z_max_val = max(all_z)

            tol = L_mm * 0.001  # 容差: 边长的 0.1%

            x_pos_nodes = set()  # +x 面节点
            y_pos_nodes = set()  # +y 面节点
            z_pos_nodes = set()  # +z 面节点
            for nid, coords in node_coords_rf.items():
                if abs(coords[0] - x_max_val) < tol:
                    x_pos_nodes.add(nid)
                if abs(coords[1] - y_max_val) < tol:
                    y_pos_nodes.add(nid)
                if abs(coords[2] - z_max_val) < tol:
                    z_pos_nodes.add(nid)

            # ---- 分方向求和: 只在对应正面上求该方向的反力 ----
            rf_x_pos = 0.0  # +x 面上 RF_x 的总和
            rf_y_pos = 0.0  # +y 面上 RF_y 的总和
            rf_z_pos = 0.0  # +z 面上 RF_z 的总和
            for value in rf_field.values:
                nid = value.nodeLabel
                if nid in x_pos_nodes:
                    rf_x_pos += value.data[0]
                if nid in y_pos_nodes:
                    rf_y_pos += value.data[1]
                if nid in z_pos_nodes:
                    rf_z_pos += value.data[2]

            # ---- 计算各方向均质化应力 ----
            A = L_mm * L_mm  # 面积
            sigma_x = rf_x_pos / A  # MPa (N/mm^2 在 mm-tonne-s 单位制)
            sigma_y = rf_y_pos / A
            sigma_z = rf_z_pos / A
            
            # ---- 沿加载方向的等效刚度 ----
            # 应变分量
            eps_x = dx / L_mm
            eps_y = dy / L_mm
            eps_z = dz / L_mm
            
            # 加载方向单位向量
            nx = dx / d_eff
            ny = dy / d_eff
            nz = dz / d_eff
            
            # 应力投影到加载方向
            sigma_eff = sigma_x * nx + sigma_y * ny + sigma_z * nz
            
            # 应变模
            epsilon_eff = d_eff / L_mm
            
            # 等效刚度 (取绝对值确保 E > 0)
            E_eff = abs(sigma_eff) / epsilon_eff
    
    odb.close()
    
    # ============================================
    # 4. 保存结果
    # ============================================
    summary = {
        'sigma_d': sigma_d,
        'sigma_max': sigma_max,
        'E_eff': E_eff,
        'rho_rel': rho_rel,
        'mesh_volume': mesh_volume,
        'n_mises_points': len(mises_list),
        'percentile': 99.9
    }
    
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("OK")

# Run
odb_path = r"$odb_path"
output_json = r"$json_path"
displacement = $displacement
L_mm = $L_mm
n_grid = $n_grid
extract_summary(odb_path, output_json, displacement, L_mm, n_grid)
'''
    # 使用 Template 替换变量
    from string import Template
    script_content = Template(script_content).substitute(
        odb_path=odb_path,
        json_path=json_path,
        displacement=str(displacement),
        L_mm=L_mm,
        n_grid=n_grid
    )
    
    with open(extract_script, 'w') as f:
        f.write(script_content)
    
    # 运行提取脚本
    try:
        result = subprocess.run(
            [abaqus_cmd, 'python', extract_script],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=SIMULATION_TIMEOUT
        )
        
        # 检查是否成功
        if not os.path.exists(json_path):
            # 打印错误信息便于调试
            if result.stderr:
                log(f"    [STDERR] {result.stderr[:500]}")
            if result.returncode != 0:
                log(f"    [ERROR] Abaqus 返回码: {result.returncode}")
        
        # 清理临时脚本
        try:
            os.remove(extract_script)
        except:
            pass
        
        return os.path.exists(json_path)
        
    except Exception as e:
        log(f"    [ERROR] 汇总提取失败: {e}")
        return False


# ====================================================================================
# 主流程
# ====================================================================================

def process_single_case(case_id: int, alpha: list, displacement: list,
                        output_dir: str, abaqus_cmd: str, cpus: int,
                        n_grid: int, max_retry: int) -> dict:
    """
    处理单个 case: INP生成 → 仿真 → 提取
    
    Returns:
        {'case_id': int, 'status': str, 'message': str}
    """
    case_name = f'case_{case_id:05d}'
    case_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    # 关键文件路径
    json_path = os.path.join(case_dir, 'case_summary.json')
    odb_path = os.path.join(case_dir, 'tpms_mesh.odb')
    inp_path = os.path.join(case_dir, 'tpms_mesh.inp')
    
    # ===== 智能检测逻辑 =====
    # 以 case_summary.json 为准：有 json 才算完成
    if os.path.exists(json_path):
        return {'case_id': case_id, 'status': 'already_done', 'message': '已完成'}
    
    # 有 ODB 但无 json：只需重新提取
    need_simulation = not os.path.exists(odb_path)
    
    if not need_simulation:
        log(f"    [INFO] 有 ODB 但无汇总数据，重新提取...")
    
    # Step 1: 生成 INP（如果需要仿真且无 INP）
    if need_simulation and not os.path.exists(inp_path):
        if not generate_inp(case_id, alpha, displacement, case_dir, n_grid):
            return {'case_id': case_id, 'status': 'failed', 'message': 'INP生成失败'}
    
    # Step 2: 仿真 (只有需要时才执行)
    if need_simulation:
        sim_success = False
        for attempt in range(max_retry + 1):
            if run_simulation(case_dir, abaqus_cmd, cpus):
                sim_success = True
                break
            elif attempt < max_retry:
                log(f"    重试 {attempt + 1}/{max_retry}...")
                kill_abaqus_processes()
                time.sleep(5)
        
        if not sim_success:
            return {'case_id': case_id, 'status': 'failed', 'message': '仿真失败'}
    
    # Step 3: 提取应力场 (带重试)
    extract_success = False
    for attempt in range(max_retry + 1):
        # 每次重试前清理环境
        if attempt > 0:
            log(f"    提取重试 {attempt}/{max_retry}...")
            kill_abaqus_processes()
            # 删除可能损坏的临时脚本
            extract_script = os.path.join(case_dir, '_extract_stress.py')
            if os.path.exists(extract_script):
                try:
                    os.remove(extract_script)
                except:
                    pass
            # 等待许可证释放
            time.sleep(10)
        
        if extract_stress_field(case_dir, abaqus_cmd):
            extract_success = True
            break
    
    if not extract_success:
        return {'case_id': case_id, 'status': 'failed', 'message': '应力场提取失败'}
    
    # Step 4: 提取汇总数据 (σ_d, E_eff, ρ_rel)
    if not extract_summary_data(case_dir, abaqus_cmd, displacement):
        log(f"    [WARN] 汇总数据提取失败，但应力场已保存")
        return {'case_id': case_id, 'status': 'partial', 'message': '汇总数据提取失败'}
    
    return {'case_id': case_id, 'status': 'success', 'message': 'OK'}


def main():
    parser = argparse.ArgumentParser(description='批量仿真 + 提取应力场')
    parser.add_argument('--n_cases', type=int, default=DEFAULT_N_CASES,
                       help=f'组合数量 (默认 {DEFAULT_N_CASES})')
    parser.add_argument('--cpus', type=int, default=DEFAULT_CPUS,
                       help=f'每个仿真的 CPU 数 (默认 {DEFAULT_CPUS})')
    parser.add_argument('--n_grid', type=int, default=DEFAULT_N_GRID,
                       help=f'网格分辨率 (默认 {DEFAULT_N_GRID})')
    parser.add_argument('--retry', type=int, default=DEFAULT_RETRY,
                       help=f'失败重试次数 (默认 {DEFAULT_RETRY})')
    parser.add_argument('--start', type=int, default=0,
                       help='起始 case ID')
    parser.add_argument('--end', type=int, default=None,
                       help='结束 case ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 清空日志
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('')
    
    print("\n" + "=" * 70)
    print("批量仿真 + 提取应力场")
    print("=" * 70)
    
    # 检查 Abaqus
    log("\n[1/5] 检查 Abaqus...")
    abaqus_cmd = check_abaqus_available()
    if not abaqus_cmd:
        log("  [ERROR] 未找到 Abaqus！请确保 Abaqus 已安装并在 PATH 中")
        return
    
    # 生成组合
    log(f"\n[2/5] 生成 {args.n_cases} 个 (α, BC) 组合...")
    combinations = generate_combinations(args.n_cases, seed=args.seed)
    
    # 保存 metadata
    save_metadata(combinations, OUTPUT_DIR)
    
    # 筛选范围
    end_id = args.end if args.end is not None else args.n_cases
    case_range = range(args.start, end_id)
    log(f"  处理范围: case_{args.start:05d} ~ case_{end_id-1:05d}")
    
    # 统计
    log(f"\n[3/5] 配置:")
    log(f"  输出目录: {OUTPUT_DIR}")
    log(f"  网格分辨率: {args.n_grid}^3")
    log(f"  CPU 数: {args.cpus}")
    log(f"  重试次数: {args.retry}")
    
    # 开始处理
    log(f"\n[4/5] 开始处理...")
    
    # 读取 CSV 中已有的 case_id（以 CSV 为唯一真相来源）
    existing_case_ids = set()
    if os.path.exists(TRAINING_DATA_CSV):
        import pandas as pd
        df_existing = pd.read_csv(TRAINING_DATA_CSV)
        existing_case_ids = set(df_existing['case_id'].tolist())
        log(f"  已有 {len(existing_case_ids)} 个 case 在 CSV 中")
    
    results = []
    start_time = time.time()
    successful_cases = []  # 用于批量清理: [(case_id, case_dir), ...]
    
    for i, case_id in enumerate(case_range):
        combo = combinations[case_id]
        alpha = combo[:4].tolist()
        displacement = combo[4:].tolist()
        case_name = f'case_{case_id:05d}'
        case_dir = os.path.join(OUTPUT_DIR, case_name)
        
        log(f"\n  Case {case_id} ({i+1}/{len(case_range)}):")
        log(f"    α = [{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}, {alpha[3]:.3f}]")
        log(f"    δ = [{displacement[0]*1000:.4f}, {displacement[1]*1000:.4f}, {displacement[2]*1000:.4f}] mm")
        
        # 检查是否已在 CSV 中（直接跳过，不检查文件）
        if case_name in existing_case_ids:
            log(f"    → 已在 CSV 中 (完全跳过)")
            results.append({'case_id': case_id, 'status': 'in_csv', 'message': '已在CSV中'})
            continue
        
        result = process_single_case(
            case_id, alpha, displacement,
            OUTPUT_DIR, abaqus_cmd, args.cpus,
            args.n_grid, args.retry
        )
        
        results.append(result)
        
        # 显示状态
        if result['status'] == 'success':
            log(f"    → OK")
            successful_cases.append((case_id, case_dir))
        elif result['status'] == 'already_done':
            log(f"    → 已存在 (跳过仿真，加入清理队列)")
            successful_cases.append((case_id, case_dir))  # 也加入清理列表
        else:
            log(f"    → FAILED: {result['message']}")
        
        # 清理 Abaqus 进程
        kill_abaqus_processes()
        
        # 每 CLEANUP_BATCH_SIZE 个成功的 case 执行一次批量清理
        if len(successful_cases) >= CLEANUP_BATCH_SIZE:
            batch_cleanup(successful_cases, combinations)
            successful_cases = []  # 清空列表
        
        # 间隔（增加等待时间避免许可证竞争）
        time.sleep(5)
    
    # 处理剩余的成功 case
    if successful_cases:
        batch_cleanup(successful_cases, combinations)
    
    # 统计结果
    elapsed = time.time() - start_time
    success = sum(1 for r in results if r['status'] == 'success')
    already = sum(1 for r in results if r['status'] == 'already_done')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    log(f"\n[5/5] 完成!")
    print("\n" + "=" * 70)
    print("统计:")
    print(f"  成功: {success}")
    print(f"  已存在: {already}")
    print(f"  失败: {failed}")
    print(f"  总耗时: {elapsed/60:.1f} 分钟")
    print("=" * 70)
    print(f"\n数据位置: {OUTPUT_DIR}")
    print(f"下一步: 使用 surrogate_model/train.py 训练代理模型")


if __name__ == '__main__':
    main()



