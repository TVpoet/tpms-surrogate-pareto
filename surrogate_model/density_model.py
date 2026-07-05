# -*- coding: utf-8 -*-
"""
密度网络模型

输入: α = (α₁, α₂, α₃, α₄) 4维
输出: ρ_rel 1维

相对密度是纯几何量，只与结构参数有关，与加载无关。
"""

import os
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DensityModel(nn.Module):
    """
    密度预测网络: α (4维) → ρ_rel (1维)
    
    使用简单的 MLP，因为这是一个纯几何映射。
    """
    
    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, dropout: float = 0.0):
        """
        Args:
            hidden_dim: 隐藏层维度
            n_layers: 隐藏层数量
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.input_dim = 4
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        
        # 输入层
        self.fc_in = nn.Linear(4, hidden_dim)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, 1)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # 归一化参数
        self.register_buffer('alpha_mean', torch.zeros(4))
        self.register_buffer('alpha_std', torch.ones(4))
        self.register_buffer('rho_mean', torch.tensor(0.0))
        self.register_buffer('rho_std', torch.tensor(1.0))
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        print(f'DensityModel: {total_params:,} params')
        print(f'  输入: 4维 (alpha1, alpha2, alpha3, alpha4)')
        print(f'  隐藏层: {n_layers} x {hidden_dim}')
        print(f'  输出: 1维 (rho_rel)')
        if dropout > 0:
            print(f'  Dropout: {dropout}')
    
    def set_normalization(self, alpha_mean, alpha_std, rho_mean, rho_std):
        """设置归一化参数"""
        self.alpha_mean = torch.tensor(alpha_mean, dtype=torch.float32, device=device)
        self.alpha_std = torch.tensor(alpha_std, dtype=torch.float32, device=device)
        self.rho_mean = torch.tensor(rho_mean, dtype=torch.float32, device=device)
        self.rho_std = torch.tensor(rho_std, dtype=torch.float32, device=device)
    
    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            alpha: [batch, 4] 结构参数
            
        Returns:
            rho_rel: [batch, 1] 相对密度
        """
        # 输入归一化
        alpha_norm = (alpha - self.alpha_mean) / (self.alpha_std + 1e-8)
        
        # 前向传播
        h = self.activation(self.fc_in(alpha_norm))
        if self.dropout is not None:
            h = self.dropout(h)
        
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
            if self.dropout is not None:
                h = self.dropout(h)
        
        rho_norm = self.fc_out(h)
        
        # 反归一化
        rho = rho_norm * self.rho_std + self.rho_mean
        
        return rho
    
    def predict(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        便捷预测接口
        
        Args:
            alpha: [batch, 4] or [4]
            
        Returns:
            rho_rel: [batch] or scalar
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        
        rho = self.forward(alpha)
        return rho.squeeze(-1)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout_rate,
        }, path)
        print(f"密度模型已保存: {path}")
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            hidden_dim=checkpoint.get('hidden_dim', 32),
            n_layers=checkpoint.get('n_layers', 2),
            dropout=checkpoint.get('dropout', 0.0),
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
