# -*- coding: utf-8 -*-
"""通用工具：设备等"""
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["device"]
