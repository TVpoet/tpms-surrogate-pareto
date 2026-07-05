# -*- coding: utf-8 -*-
"""几何相关：TPMSGeometry"""
import torch
import numpy as np

class TPMSGeometry:
    """TPMS几何定义 (从归档文件独立实现)"""
    
    def __init__(self, L: float = 0.01):
        """
        Args:
            L: 单胞尺寸 (m)
        """
        self.L = L
        self.k = 2 * np.pi / L
    
    def phi(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
            alpha: list) -> torch.Tensor:
        """
        计算 TPMS 隐式函数
        
        φ(x,y,z) = α₁·f₁ + α₂·f₂ + α₃·f₃ + α₄·f₄
        
        其中:
        - f₁ = sin(kx) + sin(ky) + sin(kz)
        - f₂ = cos(kx)sin(kx) + cos(ky)sin(ky) + cos(kz)sin(kz)
        - f₃ = cos(kx)sin(ky) + cos(ky)sin(kz) + cos(kz)sin(kx)
        - f₄ = cos(kx)sin(kz) + cos(ky)sin(kx) + cos(kz)sin(ky)
        """
        k = self.k
        xn, yn, zn = k * x, k * y, k * z
        
        f1 = torch.sin(xn) + torch.sin(yn) + torch.sin(zn)
        f2 = torch.cos(xn)*torch.sin(xn) + torch.cos(yn)*torch.sin(yn) + torch.cos(zn)*torch.sin(zn)
        f3 = torch.cos(xn)*torch.sin(yn) + torch.cos(yn)*torch.sin(zn) + torch.cos(zn)*torch.sin(xn)
        f4 = torch.cos(xn)*torch.sin(zn) + torch.cos(yn)*torch.sin(xn) + torch.cos(zn)*torch.sin(yn)
        
        # 支持 list 或 tensor 格式的 alpha
        if isinstance(alpha, (list, tuple)):
            return alpha[0]*f1 + alpha[1]*f2 + alpha[2]*f3 + alpha[3]*f4
        else:
            # tensor 格式
            if alpha.dim() == 1:
                return alpha[0]*f1 + alpha[1]*f2 + alpha[2]*f3 + alpha[3]*f4
            else:
                # [N, 4] 格式
                return alpha[:, 0:1]*f1 + alpha[:, 1:2]*f2 + alpha[:, 2:3]*f3 + alpha[:, 3:4]*f4
    
    def phi_tensor(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
                   alpha) -> torch.Tensor:
        """
        phi_tensor 是 phi 的别名，保持向后兼容
        
        用于兼容 abaqus_export.py 等调用 phi_tensor 的代码
        """
        return self.phi(x, y, z, alpha)
    
    def gradient_phi(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     alpha: list) -> tuple:
        """计算隐式函数的梯度 (用于法向量计算)"""
        from torch.autograd import grad
        
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        
        phi = self.phi(x, y, z, alpha)
        
        dphi_dx = grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        dphi_dy = grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        dphi_dz = grad(phi, z, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        
        return dphi_dx, dphi_dy, dphi_dz
    
    def normal(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
               alpha: list) -> tuple:
        """计算表面法向量 n = ∇φ / |∇φ|"""
        dphi_dx, dphi_dy, dphi_dz = self.gradient_phi(x, y, z, alpha)
        norm = torch.sqrt(dphi_dx**2 + dphi_dy**2 + dphi_dz**2 + 1e-10)
        return dphi_dx/norm, dphi_dy/norm, dphi_dz/norm


__all__ = ["TPMSGeometry"]
