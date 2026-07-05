# -*- coding: utf-8 -*-
"""
汇总所有 case 数据到 training_data.csv

读取所有 case_summary.json + metadata.csv
生成用于训练代理模型的数据集

使用方法:
    python scripts/aggregate_data.py
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'fem_data', 'parameterized')


def main():
    """主函数"""
    # 读取 metadata
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"[ERROR] 找不到 metadata: {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    n_cases = len(df)
    print(f"找到 {n_cases} 个 case")
    
    # 汇总数据
    data = []
    missing_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=n_cases, desc="汇总数据"):
        case_id = row['case_id']
        # 兼容 'case_00000' 和 0 两种格式
        if isinstance(case_id, str) and case_id.startswith('case_'):
            case_dir = os.path.join(OUTPUT_DIR, case_id)
        else:
            case_dir = os.path.join(OUTPUT_DIR, f'case_{int(case_id):05d}')
        json_path = os.path.join(case_dir, 'case_summary.json')
        
        if not os.path.exists(json_path):
            missing_count += 1
            continue
        
        # 读取汇总数据
        with open(json_path, 'r') as f:
            summary = json.load(f)
        
        # 合并数据
        item = {
            'case_id': case_id,
            'alpha1': row['alpha1'],
            'alpha2': row['alpha2'],
            'alpha3': row['alpha3'],
            'alpha4': row['alpha4'],
            'delta_x': row['delta_x'],
            'delta_y': row['delta_y'],
            'delta_z': row['delta_z'],
            'sigma_d': summary.get('sigma_d', 0),
            'sigma_max': summary.get('sigma_max', 0),
            'E_eff': summary.get('E_eff', 0),
            'rho_rel': summary.get('rho_rel', 0),
            'mesh_volume': summary.get('mesh_volume', 0),
        }
        data.append(item)
    
    if missing_count > 0:
        print(f"警告: {missing_count} 个 case 缺少 case_summary.json")
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, 'training_data.csv')
    df_out = pd.DataFrame(data)
    df_out.to_csv(output_path, index=False)
    print(f"保存到: {output_path}")
    print(f"共 {len(data)} 条数据")
    
    # 打印统计
    print("\n数据统计:")
    print(df_out[['sigma_d', 'sigma_max', 'E_eff', 'rho_rel']].describe())


if __name__ == '__main__':
    main()
