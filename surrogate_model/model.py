# -*- coding: utf-8 -*-
"""
代理模型神经网络定义

输入: (α₁, α₂, α₃, α₄, δx, δy, δz) = 7维
输出: (σ_d, E_eff) = 2维

注意: ρ_rel 由独立的 DensityModel 预测，见 density_model.py
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SurrogateModel(nn.Module):
    """
    代理模型：预测 (α, δ) → (σ_d, E_eff)

    特点:
    1. MLP 架构，6 层 × 256 维（实际配置以 train.py 与已存 checkpoint 为准）
    2. 包含残差连接
    3. 输入输出归一化
    """
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 n_layers: int = 6,
                 input_dim: int = 7,
                 output_dim: int = 2,
                 dropout: float = 0.0):
        """
        Args:
            hidden_dim: 隐藏层维度
            n_layers: 隐藏层数量
            input_dim: 输入维度 (默认 7: α₁-₄ + δx-z)
            output_dim: 输出维度 (默认 2: σ_d + E_eff)
            dropout: Dropout 概率 (0 表示不使用)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        
        # 输入层
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)
        ])
        
        # Dropout 层 (正则化)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # 初始化
        self._init_weights()
        
        # 归一化参数 (训练时设置)
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))
        
        # 统计
        total_params = sum(p.numel() for p in self.parameters())
        print(f'SurrogateModel: {total_params:,} params')
        print(f'  Input: {input_dim}D (alpha + delta)')
        print(f'  Hidden: {n_layers} x {hidden_dim}')
        print(f'  Output: {output_dim}D (sigma_d + E_eff)')
        if dropout > 0:
            print(f'  Dropout: {dropout}')
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def set_normalization(self, input_mean, input_std, output_mean, output_std):
        """设置归一化参数"""
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32, device=device)
        self.input_std = torch.tensor(input_std, dtype=torch.float32, device=device)
        self.output_mean = torch.tensor(output_mean, dtype=torch.float32, device=device)
        self.output_std = torch.tensor(output_std, dtype=torch.float32, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, 7] (已归一化或原始)
            
        Returns:
            y: 输出预测 [batch, 2] (归一化后)
        """
        # 输入归一化
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        # 前向传播
        h = self.activation(self.fc_in(x_norm))
        if self.dropout is not None:
            h = self.dropout(h)
        
        for layer in self.hidden_layers:
            h = h + self.activation(layer(h))  # 残差连接
            if self.dropout is not None:
                h = self.dropout(h)
        
        y_norm = self.fc_out(h)
        
        # 输出反归一化
        y = y_norm * self.output_std + self.output_mean
        
        return y
    
    def predict(self, alpha: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        方便的预测接口
        
        Args:
            alpha: [batch, 4] or [4]
            delta: [batch, 3] or [3]
            
        Returns:
            (sigma_d, E_eff): 两个 [batch] tensor
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        
        x = torch.cat([alpha, delta], dim=1)
        y = self.forward(x)
        
        return y[:, 0], y[:, 1]  # sigma_d, E_eff
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout_rate,  # 保存 dropout
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            hidden_dim=checkpoint.get('hidden_dim', 128),
            n_layers=checkpoint.get('n_layers', 4),
            input_dim=checkpoint.get('input_dim', 7),
            output_dim=checkpoint.get('output_dim', 2),
            dropout=checkpoint.get('dropout', 0.0),  # 读取 dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
