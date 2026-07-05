# TPMS 点阵 Pareto 前沿快速搜索：代理模型辅助的多轴多目标设计

## 项目概述

本项目针对**三周期极小曲面（TPMS）点阵**，提出一种**代理模型辅助的高通量 Pareto 前沿搜索框架**。通过双神经网络代理模型替代 FEM 响应评估，在 4 维混合系数单纯形上完成百万量级候选点的多目标 Pareto 非支配提取，并以独立 FEM 闭环复核证实框架的工程可信度。

**核心价值**：把 TPMS 多轴多目标 Pareto 搜索从 FEM 直算不可行（500 万候选同规模需约 2.4 年（16 核并行））变成秒级可完成工作流，单点加速比约 **$3 \times 10^5$** 倍（500 万 Dirichlet 候选搜索的总耗时比 $\sim 2 \times 10^7$）。

### 关键指标（实测 AMD Ryzen 9 9950X + NVIDIA RTX 4070 SUPER）

| 指标 | 数值 |
|---|---|
| 输入维度 | 7 维（α₁–α₄ + δₓ/δᵧ/δ_z）|
| 输出维度 | 2 维（σ_hp, E_eff）+ 密度预测 ρ_rel |
| 训练数据 | 2000 组 FEM 仿真 |
| 代理精度（测试集 R²） | σ_hp 0.932 / E_eff 0.970 / ρ_rel 0.995 |
| **单次推理（batch=1, GPU）** | **0.226 ms** |
| **500 万 Dirichlet 候选 Pareto 搜索（GPU）** | **3.3 s** |
| 单次 FEM 算例耗时（典型）| ~1 min（80³ C3D8R）|
| **对 FEM 单点加速比** | **约 $3 \times 10^5$** |
| **同规模对比（FEM 需 ~2.4 年（16 核并行） vs 代理 3.3 s）** | **$\sim 2 \times 10^7$** |
| FEM 闭环验证误差（9 点 Pareto 代表）| 平均 0.4–8.6%（典型 ~5%）|

---

## 研究背景

### TPMS 混合参数化

用 4 个基函数的加权组合定义 TPMS 隐式曲面：

```
phi(x, y, z; alpha) = alpha1 * f1 + alpha2 * f2 + alpha3 * f3 + alpha4 * f4

约束：alpha1 + alpha2 + alpha3 + alpha4 = 1, alpha_i >= 0
```

其中 f1–f4 均满足周期性、立方对称性、光滑性和中心对称性（Wang et al., 2022, CMAME）。α 参数控制构型混合比例，形成连续的 4 维单纯形设计空间。

### 力学性能指标

> **符号约定**：本文用 **σ_hp**（high-percentile stress indicator，高分位应力指标）表示**未平均的单元-节点（ELEMENT_NODAL）von Mises 应力的 99.9% 分位**（统一由 `extract_summary_data` 提取，训练与验证同口径）。在源码、CSV 列、JSON 键中其字面量仍为历史命名 `sigma_d`，二者一一对应（σ_hp ≡ code 中的 `sigma_d`）。

| 指标 | 定义 | 用途 |
|---|---|---|
| σ_hp | 99.9% 分位 von Mises 应力 | 高分位应力指标（比 max 更稳健，作为 max 的稳健代理）|
| E_eff | 等效杨氏模量 | 结构刚度 |
| ρ_rel | 相对密度 = V_mesh / L³ | 轻量化指标 |
| E_eff / ρ_rel | 比刚度 | 单位密度承载效率 |

### 材料与几何

- 材料：铝合金（E = 70 GPa, ν = 0.33）
- 单胞尺寸：L = 10 mm
- 壁厚：0.6 mm（隐式场带域 |φ| ≤ t/2）
- 加载：位移控制，|δ| ≤ 3 μm（约 0.03% 应变，线弹性）

### 文献对位

- **Wang et al. (2022, CMAME)**：基函数来源与代理模型逆向设计先驱（NN + GA，单目标曲线拟合）。本文沿用其 4 基函数但**目标问题**（双目标 Pareto）、**搜索方法**（穷举 + 非支配排序）、**加载**（多轴）均不同。
- **Abueidda et al. (2016, Mech. Materials)**：TPMS 均匀化方法支撑，与本文 E_eff 定义口径一致。
- **Abueidda et al. (2019, Mater. Design)**：4×4×4 Gyroid 实验 + 周期边界 FEM 双向对比，作为本文 E_eff 的外部基准。
- **Lu et al. (2021, JMRT)**：AlSi10Mg Gyroid 打印试件实验，作为 ρ_rel 的外部对齐参照。

---

## 核心结论

**1. 双代理框架将百万规模 Pareto 搜索从 FEM 不可行变为秒级可行，并以 FEM 闭环复核保障精度。**
代理模型在 RTX 4070 SUPER 上单样本推理中位耗时 0.226 ms，500 万 Dirichlet 候选的完整 Pareto 搜索仅 3.3 s；与单次 FEM ~1 min 相比，单点加速 ~$3 \times 10^5$，同规模耗时比 ~$2 \times 10^7$。9 点 Pareto 代表的独立 FEM 复核平均误差控制在 ~9% 以内（典型约 5%，前沿极端解最高约 13%），足以支撑工程级快速筛选与相对排序。

**2. 多工况 Pareto 前沿刻画揭示加载路径依赖性与前沿非连通性。**
在双轴 (2,2,0) 与两类三轴 (1,1,2)、(1,2,2) 工况下，对 500 万 Dirichlet 候选点完成非支配排序：三工况前沿在 σ_hp–E/ρ 平面占据**显著不同区域**（加载路径依赖性），且部分工况前沿呈现**间断**（disconnected Pareto front）——反映 4 基函数主导区对应不同拓扑家族。在所有工况下，4 个纯基函数 FEM 基线均位于 Pareto 前沿之下，表明**连续混合构型可实现优于纯构型的应力—刚度权衡**。该框架为多轴服役条件提供了 load-aware 设计图。

---

## 技术路线

```
Phase 1: FEM 批量仿真
  随机采样 2000 组 (alpha, delta) 组合
  Abaqus 求解 -> 提取 sigma_d, E_eff, rho_rel
                    |
                    v
Phase 2: 数据汇总
  各 case 的 case_summary.json -> training_data.csv
                    |
                    v
Phase 3: 代理模型训练
  主模型: MLP (alpha, delta) -> (sigma_d, E_eff)    [6×256, GELU, 残差, Dropout]
  密度模型: MLP alpha -> rho_rel                    [4×128, GELU]
                    |
                    v
Phase 4: 逆向设计（Pareto 搜索）
  给定载荷 delta_target:
    目标 1: min sigma_d(alpha, delta_target)
    目标 2: max E_eff(alpha, delta_target) / rho_rel(alpha)
  500 万 Dirichlet 候选采样 + O(n log n) 非支配排序 -> Pareto 前沿
                    |
                    v
Phase 5: FEM 闭环验证
  按 Pareto 前沿 σ_hp 分位选取 P1/P2/P3 三点跑真实 FEM
  对比代理预测 vs FEM 真实值 -> 量化代理精度边界
```

---

## 核心结果

### 代理模型精度

纯 MLP 架构（主模型 6 层 × 256 维，残差连接 + GELU + Dropout）：

| 指标 | R² | 说明 |
|---|---|---|
| σ_hp | 0.932 | 99.9% 分位 von Mises 应力 |
| E_eff | 0.970 | 等效杨氏模量 |
| ρ_rel | 0.995 | 相对密度（独立密度模型）|

### 多工况 Pareto 前沿

3 种代表性多轴载荷工况下对 500 万 Dirichlet 候选点完成非支配提取（GPU 约 3.3 s）：

| 工况 | δ (μm) | 类型 |
|---|---|---|
| Biaxial-XY | (2, 2, 0) | 双轴等比 |
| Triaxial-1:1:2 | (1, 1, 2) | 三轴非等比 |
| Triaxial-1:2:2 | (1, 2, 2) | 三轴非等比 |

主要观察：

- **加载路径依赖性**：三工况的 Pareto 前沿在 σ_hp–E/ρ 平面中占据显著不同区域，不存在通用最优混合构型
- **前沿非连通性**：部分工况前沿呈现间断（disconnected Pareto front），反映不同基函数主导区对应不同拓扑家族；500 万 Dirichlet(1,1,1,1) 候选点已足够密集，可排除采样不足导致的伪间断
- **混合优于纯构型**：在所有工况下，4 个纯基函数 FEM 基线均位于 Pareto 前沿之下；连续混合可实现优于纯 TPMS 架构的应力—刚度权衡

### FEM 闭环验证

对论文主线 3 个工况在 Pareto 前沿上选取 P1/P2/P3 三点独立 FEM 复核，其中 P1 为低应力端，P2 为中间折中点，P3 为高比刚度端：

| 工况 | 代表点 | α | σ_hp 误差 | E/ρ 误差 | 平均误差 |
|---|---|---|---|---|---|
| Biaxial-XY | P1 | (0.000, 0.001, 0.859, 0.140) | 0.0% | −3.3% | 1.7% |
| Biaxial-XY | P2 | (0.001, 0.173, 0.382, 0.444) | −9.6% | −4.9% | 7.2% |
| Biaxial-XY | P3 | (0.046, 0.259, 0.357, 0.337) | +3.1% | +8.4% | 5.7% |
| Triaxial-1:1:2 | P1 | (0.000, 0.243, 0.138, 0.619) | +1.3% | +4.0% | 2.7% |
| Triaxial-1:1:2 | P2 | (0.253, 0.001, 0.366, 0.380) | −0.2% | −3.8% | 2.0% |
| Triaxial-1:1:2 | P3 | (0.024, 0.270, 0.366, 0.340) | −2.2% | +4.6% | 3.4% |
| Triaxial-1:2:2 | P1 | (0.000, 0.268, 0.158, 0.573) | −2.1% | +3.3% | 2.7% |
| Triaxial-1:2:2 | P2 | (0.202, 0.000, 0.404, 0.393) | −1.7% | +2.4% | 2.0% |
| Triaxial-1:2:2 | P3 | (0.000, 0.294, 0.384, 0.322) | −2.1% | +5.6% | 3.8% |

> σ_hp、E_eff、ρ_rel 的 FEM 实测均由 `extract_summary_data` 统一提取（σ_hp = 未平均的 ELEMENT_NODAL von Mises 99.9 分位，与训练数据同口径）。

9 点平均误差 1.7–7.2%（典型约 3–4%）。最大 |σ_hp| 误差为 9.6%，最大 |E/ρ| 误差为 8.4%，9 个代表点均保持在 10% 以内。该精度足以支撑高通量筛选与候选设计的相对排序。

### 网格收敛性

代表性工况（α=[0.25, 0.50, 0.10, 0.15], δ=[2,2,2] μm, ρ_rel≈0.365）体素网格收敛（n=60–120）：

| n | σ_hp (MPa) | E_eff (MPa) | ρ_rel |
|---|---|---|---|
| 60 | 63.71 | 21,533 | 0.3651 |
| 70 | 61.26 | 21,665 | 0.3645 |
| 80 | 56.44 | 22,095 | 0.3652 |
| 90 | 56.34 | 22,151 | 0.3651 |
| 100 | 55.63 | 22,259 | 0.3648 |
| 110 | 55.26 | 22,310 | 0.3649 |
| 120 | 54.91 | 22,369 | 0.3650 |

n80 → n120：σ_hp 2.8%、E_eff 1.2%、ρ_rel 0.06%，收敛良好。训练数据采用 n=80 已具备足够精度。

### 应力百分位敏感性

- 99.9 / 99.5 / 99 三种阈值下构型相对排序保持一致
- 99.9th percentile 兼顾应力集中敏感性与统计稳健性
- σ_max 在网格加密过程中无规律波动（78–95 MPa 间起伏），验证了分位数指标必要性

---

## 使用指南

### 环境配置

#### 测试硬件

| 组件 | 配置 |
|---|---|
| CPU | AMD Ryzen 9 9950X |
| GPU | NVIDIA RTX 4070 SUPER（12 GB VRAM）|
| 操作系统 | Windows 11 |

> **GPU 可选**：训练与推理在 CPU 上也能完整运行（推理速度差异见性能表）。500 万 Dirichlet 候选的 Pareto 搜索使用 GPU 约 3.3 s，CPU 也可在数十秒内完成。

#### 软件依赖

```bash
conda create -n tpms python=3.10
conda activate tpms
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) 列出全部核心依赖（torch、numpy、scipy、matplotlib、pandas、tqdm、scikit-learn、scikit-image、pyvista、Pillow）。

另需 Abaqus 2022（**仅 FEM 仿真需要**；代理推理与 Pareto 搜索不依赖 Abaqus）。

### 运行步骤

```bash
# 1. 批量 FEM 仿真（需 Abaqus，约数小时）
python scripts/batch_simulate_extract.py --n_cases 2000

# 2. 汇总数据
python scripts/aggregate_data.py

# 3. 训练代理模型
python surrogate_model/train.py
python surrogate_model/train_density.py

# 4. 逆向设计（多工况 Pareto 搜索，Dirichlet 500 万采样）
python scripts/pick_three_dirichlet.py
python scripts/save_full_front.py

# 5. FEM 闭环验证（读取 validation_candidates_dirichlet.json；论文表格汇总主线 3 工况）
python scripts/run_validation_fem_dirichlet.py

# 6. 代理推理速度 benchmark（可选）
python scripts/benchmark_inference.py

# 7. 可选：手动验证一个自定义候选点（需 Abaqus）
python scripts/verify_pareto.py --alpha "0.045,0.227,0.697,0.031" --delta "0.002,0.002,0.002" --name pareto_v1

# 8. 网格收敛性研究（需 Abaqus；默认构型 α=[0.25,0.50,0.10,0.15]，n=60–120）
python scripts/run_convergence_extra.py --n_grids 60 70 80 90 100 110 120
python scripts/convergence_report.py --base outputs/mesh_convergence \
    --grids 60 70 80 90 100 110 120 --out outputs/mesh_convergence/convergence_plot.png
```

---

## 文件结构

```
TPMS/
|-- config.py                           # 全局配置（几何、材料、训练参数）
|
|-- tpms/                               # 核心模块
|   |-- geometry.py                     # TPMS 隐式函数 (f1-f4)
|   +-- utils.py
|
|-- surrogate_model/                    # 代理模型
|   |-- model.py                        # SurrogateModel (7->2 MLP, 6x256)
|   |-- density_model.py                # DensityModel (4->1 MLP, 4x128)
|   |-- data.py                         # SurrogateDataset
|   |-- train.py                        # 主模型训练
|   |-- train_density.py                # 密度模型训练
|   +-- pareto_utils.py                 # Pareto 搜索 + 纯构型基线共享工具
|
|-- scripts/                            # 辅助脚本
|   |-- batch_simulate_extract.py       # 批量 FEM 仿真 + 数据提取
|   |-- aggregate_data.py               # 数据汇总 -> training_data.csv
|   |-- pick_three_dirichlet.py         # Dirichlet 采样 Pareto 搜索 + 代表点选取
|   |-- save_full_front.py              # 保存论文 3 工况完整 Pareto 前沿
|   |-- run_validation_fem_dirichlet.py # FEM 闭环验证 Dirichlet 代表点
|   |-- verify_pareto.py                # 可选：手动验证单个自定义候选点
|   |-- benchmark_inference.py          # 代理推理速度 benchmark
|   |-- abaqus_export.py                # INP 文件生成
|   |-- run_convergence_extra.py        # 网格收敛 FEM 批量运行
|   |-- convergence_report.py           # 网格收敛建表 + 绘图
|   +-- stress_percentile_sensitivity.py# 应力百分位敏感性
|
|-- fem_data/parameterized/             # FEM 数据
|   |-- metadata.csv
|   +-- training_data.csv               # 训练数据（2000 样本）
|
|-- outputs/
|   |-- surrogate_model/                # 训练好的权重
|   |   |-- best_model.pth              #   主模型 (256x6)
|   |   +-- density_model.pth           #   密度模型 (128x4)
|   |-- benchmark_comparison/           # Pareto 前沿 front_*.json
|   |-- pareto_validation_dirichlet/    # FEM 闭环验证 9 点 (Fig.13)
|   |-- pareto_stress_strain/           # 代表点应力-应变曲线
|   |-- mesh_convergence/               # 网格收敛数据
|   |-- verification/                   # 纯构型基线 + 应力百分位 (Table2)
|   +-- validation_*_dirichlet.json     # Pareto 候选与验证对比
|
+-- README.md                           # 本文件