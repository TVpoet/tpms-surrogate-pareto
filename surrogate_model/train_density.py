# -*- coding: utf-8 -*-
"""
密度网络训练脚本

训练 α → ρ_rel 的映射关系
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': ['Times New Roman', 'Microsoft YaHei'],
    'font.serif': ['Times New Roman'],
    'font.sans-serif': ['Microsoft YaHei'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    # 源图 12″ 宽在 Word 6.3″ 下缩到 ~0.53，显式放大字号到 ~2× 补偿
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})
from sklearn.metrics import r2_score

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from surrogate_model.density_model import DensityModel, device
from surrogate_model.data import load_training_data

# 训练参数 (针对高 ρ 区不准的问题做了三项调整)
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
PATIENCE = 80
WEIGHT_DECAY = 1e-4

# 模型架构 — 措施 3 + 4：去 dropout，加大网络
HIDDEN_DIM = 128  # 64 → 128 (增加容量)
N_LAYERS = 4      # 3 → 4 (加深)
DROPOUT = 0.0     # 0.1 → 0.0 (去除正则化，避免预测向均值收缩)


def relative_mse_loss(y_pred, y_true, eps=1e-6):
    """措施 2：相对误差 MSE 损失

    标准 MSE 偏向中间数值（abs err 大的样本主导）；
    相对 MSE 让每个样本按"相对误差%"贡献，强制网络在高 ρ 区也精准。
    """
    return ((y_pred - y_true) / (y_true + eps)).pow(2).mean()

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'surrogate_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _style_axes(ax):
    """应用统一绘图样式：四轴显示、刻度朝内、上右轴无刻度。"""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, labeltop=False, labelright=False)


def train_density_model():
    """训练密度网络"""
    print("=" * 60)
    print("密度网络训练 (α → ρ_rel)")
    print("=" * 60)
    
    # 加载数据
    df = load_training_data()
    
    # 提取 α 和 ρ_rel（去重，因为同一个 α 可能有多个 δ）
    alpha_cols = ['alpha1', 'alpha2', 'alpha3', 'alpha4']
    
    # 按 α 分组取平均 ρ_rel（理论上同一 α 的 ρ_rel 应该相同）
    df_grouped = df.groupby(alpha_cols)['rho_rel'].mean().reset_index()
    print(f"唯一 α 组合数: {len(df_grouped)}")
    
    # 转为 numpy
    X = df_grouped[alpha_cols].values
    Y = df_grouped['rho_rel'].values
    
    # 划分数据
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f"数据划分: 训练 {len(train_idx)} / 验证 {len(val_idx)} / 测试 {len(test_idx)}")
    print(f"ρ_rel 范围: [{Y.min():.4f}, {Y.max():.4f}]")
    
    # 计算归一化参数
    alpha_mean = X_train.mean(axis=0)
    alpha_std = X_train.std(axis=0)
    rho_mean = Y_train.mean()
    rho_std = Y_train.std()
    
    # 转为 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device).unsqueeze(-1)
    
    # 创建模型
    model = DensityModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT).to(device)
    model.set_normalization(alpha_mean, alpha_std, rho_mean, rho_std)
    
    # 优化器 (带 L2 正则化)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    # 训练
    print(f"\n开始训练 (epochs={EPOCHS}, patience={PATIENCE}, loss=relative_MSE)...")

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    n_train = len(X_train_t)

    for epoch in range(EPOCHS):
        model.train()

        # ---- bug 修复：每 epoch 遍历全训练集 ----
        perm = torch.randperm(n_train)
        epoch_losses = []
        for batch_start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[batch_start:batch_start + BATCH_SIZE]
            X_batch = X_train_t[batch_idx]
            Y_batch = Y_train_t[batch_idx]

            optimizer.zero_grad()
            Y_pred = model(X_batch)

            # 措施 2：相对误差损失（直接用 raw 值，不归一化）
            loss = relative_mse_loss(Y_pred, Y_batch)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses))

        # 验证（同样用相对 MSE）
        model.eval()
        with torch.no_grad():
            Y_val_pred = model(X_val_t)
            val_loss = relative_mse_loss(Y_val_pred, Y_val_t).item()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            model.save(os.path.join(OUTPUT_DIR, 'density_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\n早停触发 (patience={PATIENCE})")
            break
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d} | Train: {avg_train_loss:.4e} | Val: {val_loss:.4e} | LR: {lr:.2e}")
    
    # 加载最佳模型
    model = DensityModel.load(os.path.join(OUTPUT_DIR, 'density_model.pth'))
    
    # 测试评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        Y_test_pred = model(X_test_t).squeeze(-1).cpu().numpy()
    
    r2 = r2_score(Y_test, Y_test_pred)
    mae = np.mean(np.abs(Y_test - Y_test_pred))
    
    print("rho_rel prediction:")
    print(f"  R2 = {r2:.4f}")
    print(f"  MAE = {mae:.6f}")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 预测 vs 真实
    ax1 = axes[0]
    ax1.scatter(Y_test, Y_test_pred, alpha=0.7, s=50)
    ax1.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 
             'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel(r'True $\rho_{\mathrm{rel}}$')
    ax1.set_ylabel(r'Predicted $\rho_{\mathrm{rel}}$')
    ax1.set_title(rf'Density Prediction ($R^2$ = {r2:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    _style_axes(ax1)
    
    # 损失曲线
    ax2 = axes[1]
    ax2.plot(history['train_loss'], label='Train')
    ax2.plot(history['val_loss'], label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (normalized)')
    ax2.set_title('Training History')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _style_axes(ax2)
    
    plt.tight_layout()
    DIAGRAMS_DIR = os.path.join(PROJECT_ROOT, 'diagrams')
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    save_path = os.path.join(DIAGRAMS_DIR, 'density_model_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果图保存: {save_path}")
    plt.close()
    
    return model


if __name__ == '__main__':
    train_density_model()
