# -*- coding: utf-8 -*-
"""
代理模型数据加载

从 training_data.csv 加载数据用于训练
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'fem_data', 'parameterized')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_training_data(data_path: str = None):
    """
    加载训练数据
    
    Args:
        data_path: CSV 文件路径 (默认 fem_data/parameterized/training_data.csv)
        
    Returns:
        df: pandas DataFrame
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'training_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到训练数据: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # 修复 E_eff 负值问题（物理上模量应为正值）
    n_negative = (df['E_eff'] < 0).sum()
    if n_negative > 0:
        print(f"  警告: {n_negative} 条数据 E_eff 为负值，已取绝对值")
        df['E_eff'] = df['E_eff'].abs()
    
    print(f"加载训练数据: {len(df)} 条记录")
    
    return df


class SurrogateDataset(Dataset):
    """
    代理模型数据集
    
    输入: [α₁, α₂, α₃, α₄, δx, δy, δz]
    输出: [σ_d, E_eff]
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 包含训练数据的 DataFrame
        """
        # 输入特征
        self.X = torch.tensor(df[[
            'alpha1', 'alpha2', 'alpha3', 'alpha4',
            'delta_x', 'delta_y', 'delta_z'
        ]].values, dtype=torch.float32, device=device)
        
        # 输出目标
        self.Y = torch.tensor(df[[
            'sigma_d', 'E_eff'
        ]].values, dtype=torch.float32, device=device)
        
        # 附加信息 (用于比强度/比刚度计算)
        self.rho_rel = torch.tensor(df['rho_rel'].values, 
                                     dtype=torch.float32, device=device)
        
        # 计算归一化参数
        self.input_mean = self.X.mean(dim=0)
        self.input_std = self.X.std(dim=0)
        self.output_mean = self.Y.mean(dim=0)
        self.output_std = self.Y.std(dim=0)
        
        print(f"数据集: {len(self)} 条")
        print(f"  输入范围: α ∈ [{self.X[:, :4].min():.3f}, {self.X[:, :4].max():.3f}]")
        # δ 原始单位是米，转换为毫米显示
        delta_mm = self.X[:, 4:] * 1000  # m -> mm
        print(f"           δ ∈ [{delta_mm.min():.4f}, {delta_mm.max():.4f}] mm")
        print(f"  输出范围: σ_d ∈ [{self.Y[:, 0].min():.1f}, {self.Y[:, 0].max():.1f}] MPa")
        print(f"           E_eff ∈ [{self.Y[:, 1].min():.1f}, {self.Y[:, 1].max():.1f}] MPa")
        print(f"           ρ_rel ∈ [{self.rho_rel.min():.3f}, {self.rho_rel.max():.3f}]")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_normalization_params(self):
        """返回归一化参数"""
        return (
            self.input_mean.cpu().numpy(),
            self.input_std.cpu().numpy(),
            self.output_mean.cpu().numpy(),
            self.output_std.cpu().numpy()
        )


def create_data_loaders(df: pd.DataFrame, 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        batch_size: int = 32,
                        seed: int = 42):
    """
    创建训练/验证/测试数据加载器
    
    Args:
        df: 数据 DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
        seed: 随机种子
        
    Returns:
        train_loader, val_loader, test_loader, full_dataset
    """
    np.random.seed(seed)
    
    n = len(df)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    
    # 先创建训练集获取归一化参数
    train_dataset = SurrogateDataset(train_df)
    
    # 验证集和测试集也使用相同的数据结构
    val_dataset = SurrogateDataset(val_df)
    test_dataset = SurrogateDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n数据划分: 训练 {len(train_df)} / 验证 {len(val_df)} / 测试 {len(test_df)}")
    
    return train_loader, val_loader, test_loader, train_dataset

