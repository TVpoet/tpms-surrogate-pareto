# -*- coding: utf-8 -*-
"""
代理模型训练脚本

训练 (α, δ) → (σ_d, E_eff) 的 MLP 映射

使用方法:
    python surrogate_model/train.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免QThread崩溃
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
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from surrogate_model.model import SurrogateModel, device
from surrogate_model.data import load_training_data, create_data_loaders


# 配置
EPOCHS = 1000
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
PATIENCE = 150

# 模型架构
HIDDEN_DIM = 256
N_LAYERS = 6
DROPOUT = 0.1

# 正则化
WEIGHT_DECAY = 1e-4


# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'surrogate_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _style_axes(ax):
    """应用统一绘图样式：四轴显示、刻度朝内、上右轴无刻度。"""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, labeltop=False, labelright=False)


def train():
    """主训练函数"""
    print("="*60)
    print("代理模型训练")
    print("="*60)
    
    # 加载数据
    df = load_training_data()
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
        df, train_ratio=0.8, val_ratio=0.1, batch_size=BATCH_SIZE
    )
    
    # 创建模型（使用配置参数）
    model = SurrogateModel(
        hidden_dim=HIDDEN_DIM, 
        n_layers=N_LAYERS,
        dropout=DROPOUT  # Dropout 正则化
    ).to(device)
    
    # 设置归一化参数
    norm_params = train_dataset.get_normalization_params()
    model.set_normalization(*norm_params)
    
    # 优化器（带 L2 正则化）
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY  # L2 正则化
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30  # 更慢的衰减
    )
    
    # 损失函数
    mse_loss = nn.MSELoss()
    
    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_sigma': [], 'val_sigma': [],
        'train_E': [], 'val_E': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    print(f"\n开始训练 (epochs={EPOCHS}, patience={PATIENCE})...")
    for epoch in range(EPOCHS):
        # ================================
        # 训练阶段
        # ================================
        model.train()
        train_losses = []
        train_sigma_losses = []
        train_E_losses = []

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()

            Y_pred = model(X_batch)

            # 数据损失（使用归一化后的值，避免量级差异）
            Y_pred_norm = (Y_pred - model.output_mean) / (model.output_std + 1e-8)
            Y_batch_norm = (Y_batch - model.output_mean) / (model.output_std + 1e-8)

            loss_sigma = mse_loss(Y_pred_norm[:, 0], Y_batch_norm[:, 0])
            loss_E = mse_loss(Y_pred_norm[:, 1], Y_batch_norm[:, 1])
            loss = loss_sigma + loss_E

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_sigma_losses.append(loss_sigma.item())
            train_E_losses.append(loss_E.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_train_sigma = np.mean(train_sigma_losses)
        avg_train_E = np.mean(train_E_losses)
        
        # ================================
        # 验证阶段
        # ================================
        model.eval()
        val_losses = []
        val_sigma_losses = []
        val_E_losses = []
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                Y_pred = model(X_batch)

                # 使用归一化后的值计算损失（与训练一致）
                Y_pred_norm = (Y_pred - model.output_mean) / (model.output_std + 1e-8)
                Y_batch_norm = (Y_batch - model.output_mean) / (model.output_std + 1e-8)

                loss_sigma = mse_loss(Y_pred_norm[:, 0], Y_batch_norm[:, 0])
                loss_E = mse_loss(Y_pred_norm[:, 1], Y_batch_norm[:, 1])
                loss = loss_sigma + loss_E

                val_losses.append(loss.item())
                val_sigma_losses.append(loss_sigma.item())
                val_E_losses.append(loss_E.item())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_sigma = np.mean(val_sigma_losses)
        avg_val_E = np.mean(val_E_losses)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_sigma'].append(avg_train_sigma)
        history['val_sigma'].append(avg_val_sigma)
        history['train_E'].append(avg_train_E)
        history['val_E'].append(avg_val_E)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            model.save(os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d} | Train: {avg_train_loss:.4e} | "
                  f"Val: {avg_val_loss:.4e} | LR: {lr:.2e}")
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f"\n早停触发 (patience={PATIENCE})")
            break
    
    # ================================
    # 测试评估
    # ================================
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)
    
    model = SurrogateModel.load(os.path.join(OUTPUT_DIR, 'best_model.pth'))
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            Y_pred = model(X_batch)
            all_y_true.append(Y_batch.cpu().numpy())
            all_y_pred.append(Y_pred.cpu().numpy())
    
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    
    # 计算 R²
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
    
    r2_sigma = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_E = r2_score(y_true[:, 1], y_pred[:, 1])
    
    print(f"sigma_d: R2 = {r2_sigma:.4f}")
    print(f"E_eff:   R2 = {r2_E:.4f}")
    
    # ================================
    # 绘图
    # ================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].semilogy(history['train_loss'], label='Train')
    axes[0, 0].semilogy(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    _style_axes(axes[0, 0])
    
    # σ_d 预测 vs 真实
    axes[0, 1].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5, s=10)
    axes[0, 1].plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                    [y_true[:, 0].min(), y_true[:, 0].max()], 'r--')
    axes[0, 1].set_xlabel(r'FEM $\sigma_{\mathrm{hp}}$ (MPa)')
    axes[0, 1].set_ylabel(r'Predicted $\sigma_{\mathrm{hp}}$ (MPa)')
    axes[0, 1].set_title(rf'$\sigma_{{\mathrm{{hp}}}}$: $R^2$ = {r2_sigma:.4f}')
    axes[0, 1].grid(True)
    _style_axes(axes[0, 1])
    
    # E_eff 预测 vs 真实
    axes[1, 0].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5, s=10)
    axes[1, 0].plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                    [y_true[:, 1].min(), y_true[:, 1].max()], 'r--')
    axes[1, 0].set_xlabel(r'FEM $E_{\mathrm{eff}}$ (MPa)')
    axes[1, 0].set_ylabel(r'Predicted $E_{\mathrm{eff}}$ (MPa)')
    axes[1, 0].set_title(rf'$E_{{\mathrm{{eff}}}}$: $R^2$ = {r2_E:.4f}')
    axes[1, 0].grid(True)
    _style_axes(axes[1, 0])
    
    # 物理约束损失
    axes[1, 1].semilogy(history['train_sigma'], label='σ_d')
    axes[1, 1].semilogy(history['train_E'], label='E_eff')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Per-Target Training Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    _style_axes(axes[1, 1])
    
    plt.tight_layout()
    DIAGRAMS_DIR = os.path.join(PROJECT_ROOT, 'diagrams')
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    plt.savefig(os.path.join(DIAGRAMS_DIR, 'training_results.png'), dpi=150)
    plt.close()
    
    print(f"\n结果保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    train()
