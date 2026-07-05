# -*- coding: utf-8 -*-
"""
ABAQUS INP 文件导出模块

功能:
  1. 生成TPMS几何网格（Shell或Solid）
  2. 导出ABAQUS INP格式文件
  3. 设置RVE周期性边界条件
  4. 网格质量改进和验证

注意: 此模块不依赖训练好的模型，只需要几何参数即可生成INP文件

使用方法:
  from abaqus_export import TPMSMeshGenerator
  
  generator = TPMSMeshGenerator(L=0.01)  # 10mm单胞
  generator.export_inp(
      alpha=[0.25, 0.25, 0.25, 0.25],
      displacement=[0.0, 0.0, 0.00001],
      output_dir='abaqus_export'
  )

作者: GitHub Copilot
日期: 2025-12-01
"""

import os
import numpy as np
import torch
from collections import defaultdict

# 导入配置和几何
from config import L, E_SOLID, NU, WALL_THICKNESS_MM

# ====================================================================================
# 默认导出配置 (修改此处可影响默认行为)
# ====================================================================================

# 默认TPMS混合系数和位移工况
DEFAULT_ALPHA = [0.25, 0.25, 0.25, 0.25]  # [P, G, D, I-WP]
DEFAULT_DISPLACEMENT = [0.0, 0.0, 0.00001]  # [δx, δy, δz] (m)

# 网格分辨率
DEFAULT_N_GRID = 100

# Shell单元配置
DEFAULT_SHELL_TYPE = 'MIXED'  # 'S3', 'S4R', 'MIXED'
DEFAULT_SHELL_THICKNESS = 0.05  # mm

# Solid单元配置
DEFAULT_MESH_MODE = 'solid'  # 'shell' or 'solid'
DEFAULT_SOLID_TYPE = 'C3D8R'
DEFAULT_WALL_THICKNESS = 0.6  # mm (solid模式的壁厚，需与config.WALL_THICKNESS_MM一致！)

# 边界拟合配置
BOUNDARY_WARP = True
WARP_BAND_FACTOR = 0.4
WARP_STEP_LIMIT_RATIO = 0.45


# ====================================================================================
# TPMS网格生成器类
# ====================================================================================

class TPMSMeshGenerator:
    """
    TPMS网格生成器 - 用于生成ABAQUS INP文件
    
    此类不依赖代理模型，可独立使用
    """
    
    def __init__(self, L: float = None, E_solid: float = None, nu: float = None):
        """
        初始化网格生成器
        
        Args:
            L: 单胞尺寸 (m)，默认使用config.L
            E_solid: 杨氏模量 (Pa)，默认使用config.E_SOLID
            nu: 泊松比，默认使用config.NU
        """
        from config import L as CONFIG_L
        self.L = L if L is not None else CONFIG_L
        self.E_solid = E_solid if E_solid is not None else E_SOLID
        self.nu = nu if nu is not None else NU
        
        # 导入TPMSGeometry（从重构后的模块）
        from tpms import TPMSGeometry, device
        self.TPMSGeometry = TPMSGeometry
        self.device = device
        
        print(f"[TPMSMeshGenerator] 初始化完成")
        print(f"  单胞尺寸: L = {self.L*1000:.2f} mm")
        print(f"  材料: E = {self.E_solid/1e9:.1f} GPa, nu = {self.nu}")
    
    def export_inp(self, alpha: list, displacement: list,
                   n_grid: int = None,
                   output_dir: str = 'abaqus_export',
                   mesh_mode: str = None,
                   shell_type: str = None,
                   shell_thickness: float = None,
                   solid_type: str = None) -> dict:
        """
        导出TPMS几何为ABAQUS INP文件
        
        Args:
            alpha: TPMS混合系数 [α₁, α₂, α₃, α₄]，总和必须为1
            displacement: 位移 [δx, δy, δz]，单位: m
            n_grid: 网格分辨率，默认100
            output_dir: 输出目录
            mesh_mode: 'shell' 或 'solid'
            shell_type: Shell单元类型 ('S3', 'S4R', 'MIXED')
            shell_thickness: Shell厚度 (mm)
            solid_type: Solid单元类型 ('C3D8R')
        
        Returns:
            生成结果信息字典
        """
        from skimage import measure
        
        # 使用默认配置
        if n_grid is None:
            n_grid = DEFAULT_N_GRID
        if mesh_mode is None:
            mesh_mode = DEFAULT_MESH_MODE
        if shell_type is None:
            shell_type = DEFAULT_SHELL_TYPE
        if shell_thickness is None:
            shell_thickness = DEFAULT_SHELL_THICKNESS
        if solid_type is None:
            solid_type = DEFAULT_SOLID_TYPE
        
        print("\n" + "="*80)
        print("ABAQUS INP 导出")
        print("="*80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 验证输入
        if len(alpha) != 4 or abs(sum(alpha) - 1.0) > 1e-6:
            raise ValueError("alpha必须是4个元素且总和为1")
        
        # 打印配置信息
        L_mm = self.L * 1000
        print(f"\n[CONFIG] 配置:")
        print(f"  alpha = {alpha}")
        print(f"  displacement = [{displacement[0]*1000:.6f}mm, {displacement[1]*1000:.6f}mm, {displacement[2]*1000:.6f}mm]")
        strain_pcts = [(displacement[i] / self.L) * 100 for i in range(3)]
        print(f"  strain = [{strain_pcts[0]:.4f}%, {strain_pcts[1]:.4f}%, {strain_pcts[2]:.4f}%]")
        print(f"  网格分辨率: {n_grid}^3")
        print(f"  Mesh模式: {mesh_mode}")
        if mesh_mode == 'shell':
            print(f"  Shell类型: {shell_type}")
            print(f"  Shell厚度: {shell_thickness} mm")
        else:
            print(f"  Solid类型: {solid_type}")
        print(f"  单胞尺寸: L = {L_mm:.2f} mm")
        print(f"  输出目录: {output_dir}/")
        
        # 生成TPMS隐式函数
        print(f"\n[1/2] 生成TPMS几何...")
        x = np.linspace(0, L_mm, n_grid)
        y = np.linspace(0, L_mm, n_grid)
        z = np.linspace(0, L_mm, n_grid)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        tpms_geo = self.TPMSGeometry(L=self.L)
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        x_t = torch.tensor((X/1000.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
        y_t = torch.tensor((Y/1000.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
        z_t = torch.tensor((Z/1000.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
        phi = tpms_geo.phi_tensor(x_t, y_t, z_t, alpha_t).detach().cpu().numpy().reshape(n_grid, n_grid, n_grid)
        
        inp_path = os.path.join(output_dir, 'tpms_mesh.inp')
        result = {'status': 'success', 'inp_path': inp_path}
        
        if mesh_mode == 'shell':
            try:
                verts, faces, normals, _ = measure.marching_cubes(
                    phi, level=0.0, spacing=(L_mm/n_grid, L_mm/n_grid, L_mm/n_grid)
                )
                print(f"  [OK] 壳表面网格生成成功")
                print(f"    顶点数: {len(verts):,}")
                print(f"    面片数: {len(faces):,}")
                result['n_vertices'] = len(verts)
                result['n_faces'] = len(faces)
            except Exception as e:
                print(f"  [ERROR] 网格生成失败: {e}")
                result['status'] = 'failed'
                result['error'] = str(e)
                return result
            
            self._write_inp_shell(inp_path, verts, faces, L_mm, alpha, displacement,
                                  shell_type=shell_type, shell_thickness=shell_thickness)
        else:
            # 使用本文件的DEFAULT_WALL_THICKNESS，而不是config的WALL_THICKNESS_MM
            mm = DEFAULT_WALL_THICKNESS
            thick_param = mm / ((self.L / (2 * np.pi)) * 1000)
            xc = (x[:-1] + x[1:]) / 2.0
            yc = (y[:-1] + y[1:]) / 2.0
            zc = (z[:-1] + z[1:]) / 2.0
            Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
            
            x_tc = torch.tensor((Xc/1000.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
            y_tc = torch.tensor((Yc/1000.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
            z_tc = torch.tensor((Zc/1000.0).reshape(-1, 1), dtype=torch.float32, device=self.device)
            phi_c = tpms_geo.phi_tensor(x_tc, y_tc, z_tc, alpha_t).detach().cpu().numpy().reshape(n_grid-1, n_grid-1, n_grid-1)
            mask_center = np.abs(phi_c) <= (thick_param/2.0)
            
            nodes_coords, elems_conn = self._generate_voxel_mesh(L_mm, n_grid, mask_center)
            print(f"  [OK] 实体体素网格生成成功")
            print(f"    节点数: {len(nodes_coords):,}")
            print(f"    单元数: {len(elems_conn):,}")
            result['n_nodes'] = len(nodes_coords)
            result['n_elements'] = len(elems_conn)
            
            if BOUNDARY_WARP and len(nodes_coords) > 0:
                nodes_coords = self._warp_boundary_nodes(nodes_coords, L_mm, n_grid, thick_param, alpha)
                print("  已执行边界拟合")
            
            self._write_inp_solid(inp_path, nodes_coords, elems_conn, L_mm, alpha, displacement, solid_type)
        
        print(f"\n[2/2] INP文件已保存: {inp_path}")
        
        # 生成边界条件说明文件
        bc_path = os.path.join(output_dir, 'boundary_conditions.txt')
        self._write_bc_readme(bc_path, L_mm, alpha, displacement, mesh_mode, shell_type, shell_thickness, solid_type)
        
        print("\n" + "="*80)
        print("导出完成！")
        print("="*80)
        print(f"\n已生成文件:")
        print(f"  1. {inp_path}")
        print(f"     → ABAQUS完整模型（INP格式）")
        print(f"  2. {bc_path}")
        print(f"     → 边界条件说明")
        print(f"\n下一步:")
        print(f"  1. 打开ABAQUS CAE")
        print(f"  2. File → Import → Model → 选择 {inp_path}")
        print(f"  3. Job → Create → Submit")
        print("="*80)
        
        return result
    
    def _write_bc_readme(self, filepath: str, L_mm: float, alpha: list, 
                         displacement: list, mesh_mode: str,
                         shell_type: str, shell_thickness: float, solid_type: str):
        """写入边界条件说明文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ABAQUS有限元分析 - 边界条件设置\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. 几何参数\n")
            f.write(f"   - 尺寸: {L_mm} mm × {L_mm} mm × {L_mm} mm\n")
            f.write(f"   - TPMS α: [{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}, {alpha[3]:.3f}]\n\n")
            
            f.write("2. 材料属性\n")
            f.write(f"   - 弹性模量: E = {self.E_solid/1e9} GPa\n")
            f.write(f"   - 泊松比: nu = {self.nu}\n\n")
            
            f.write("3. 边界条件 (RVE周期性边界)\n")
            f.write(f"   - X方向: u(L,y,z) - u(0,y,z) = {displacement[0]*1000:.6f} mm\n")
            f.write(f"   - Y方向: v(x,L,z) - v(x,0,z) = {displacement[1]*1000:.6f} mm\n")
            f.write(f"   - Z方向: w(x,y,L) - w(x,y,0) = {displacement[2]*1000:.6f} mm\n")
            f.write(f"   - CORNER点固定消除刚体运动\n\n")
            
            f.write("4. 网格信息\n")
            if mesh_mode == 'shell':
                f.write(f"   - 网格模式: Shell\n")
                f.write(f"   - 单元类型: {shell_type}\n")
                f.write(f"   - Shell厚度: {shell_thickness} mm\n")
            else:
                f.write(f"   - 网格模式: Solid\n")
                f.write(f"   - 单元类型: {solid_type}\n")
    
    # ========== 网格生成函数 ==========
    
    def _generate_voxel_mesh(self, L_mm: float, n_grid: int, mask_center: np.ndarray):
        """生成体素网格"""
        xg = np.linspace(0, L_mm, n_grid)
        yg = np.linspace(0, L_mm, n_grid)
        zg = np.linspace(0, L_mm, n_grid)
        node_id_map = {}
        nodes = []
        elems = []
        
        def nid(i, j, k):
            key = (i, j, k)
            if key in node_id_map:
                return node_id_map[key]
            idx = len(nodes) + 1
            node_id_map[key] = idx
            nodes.append([xg[i], yg[j], zg[k]])
            return idx
        
        nx, ny, nz = n_grid-1, n_grid-1, n_grid-1
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not mask_center[i, j, k]:
                        continue
                    n1 = nid(i, j, k)
                    n2 = nid(i+1, j, k)
                    n3 = nid(i+1, j+1, k)
                    n4 = nid(i, j+1, k)
                    n5 = nid(i, j, k+1)
                    n6 = nid(i+1, j, k+1)
                    n7 = nid(i+1, j+1, k+1)
                    n8 = nid(i, j+1, k+1)
                    elems.append([n1, n2, n3, n4, n5, n6, n7, n8])
        
        return np.array(nodes, dtype=np.float64), np.array(elems, dtype=np.int64)
    
    def _warp_boundary_nodes(self, nodes: np.ndarray, L_mm: float, n_grid: int, 
                             thick_param: float, alpha: list) -> np.ndarray:
        """边界节点拟合到TPMS表面"""
        spacing = L_mm / n_grid
        tol = max(1e-6, 1e-6 * L_mm)
        x = nodes[:, 0]
        y = nodes[:, 1]
        z = nodes[:, 2]
        
        edge_x = (x <= x.min()+tol) | (x >= x.max()-tol)
        edge_y = (y <= y.min()+tol) | (y >= y.max()-tol)
        edge_z = (z <= z.min()+tol) | (z >= z.max()-tol)
        freeze = edge_x | edge_y | edge_z
        
        xm = torch.tensor(x/1000.0, dtype=torch.float32, device=self.device).reshape(-1, 1)
        ym = torch.tensor(y/1000.0, dtype=torch.float32, device=self.device).reshape(-1, 1)
        zm = torch.tensor(z/1000.0, dtype=torch.float32, device=self.device).reshape(-1, 1)
        xm.requires_grad_(True)
        ym.requires_grad_(True)
        zm.requires_grad_(True)
        
        tp = self.TPMSGeometry(L=self.L)
        at = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        phi = tp.phi_tensor(xm, ym, zm, at)
        
        gx, gy, gz = torch.autograd.grad(phi, [xm, ym, zm], 
                                          grad_outputs=torch.ones_like(phi),
                                          create_graph=False, retain_graph=False)
        gnorm = torch.sqrt(gx*gx + gy*gy + gz*gz) + 1e-12
        
        t = thick_param / 2.0
        dplus = torch.abs(phi - t)
        dminus = torch.abs(phi + t)
        target = torch.where(dplus < dminus, torch.full_like(phi, t), torch.full_like(phi, -t))
        
        band = WARP_BAND_FACTOR * thick_param
        near = torch.min(dplus, dminus) <= band
        mask = near.detach().cpu().numpy().reshape(-1)
        mask = mask & (~freeze)
        
        if not np.any(mask):
            return nodes
        
        delta = ((phi - target) / gnorm).detach().cpu().numpy().reshape(-1)
        nx = (gx / gnorm).detach().cpu().numpy().reshape(-1)
        ny = (gy / gnorm).detach().cpu().numpy().reshape(-1)
        nz = (gz / gnorm).detach().cpu().numpy().reshape(-1)
        
        step_mm = delta * 1000.0
        limit = spacing * WARP_STEP_LIMIT_RATIO
        step_mm = np.clip(step_mm, -limit, limit)
        
        x_new = x.copy()
        y_new = y.copy()
        z_new = z.copy()
        x_new[mask] = x[mask] - step_mm[mask] * nx[mask]
        y_new[mask] = y[mask] - step_mm[mask] * ny[mask]
        z_new[mask] = z[mask] - step_mm[mask] * nz[mask]
        
        out = nodes.copy()
        out[:, 0] = x_new
        out[:, 1] = y_new
        out[:, 2] = z_new
        return out
    
    # ========== INP写入函数 (Solid) ==========
    
    def _write_inp_solid(self, filepath: str, nodes: np.ndarray, elems: np.ndarray,
                         L_mm: float, alpha: list, displacement: list,
                         solid_type: str = 'C3D8R'):
        """写入Solid单元INP文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("*Heading\n")
            f.write("** TPMS Solid - Generated by TPMSMeshGenerator\n")
            f.write(f"** Alpha: [{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}, {alpha[3]:.3f}]\n")
            f.write(f"** Size: {L_mm:.1f} x {L_mm:.1f} x {L_mm:.1f} mm\n")
            f.write("** Units: mm-tonne-s-MPa\n")
            f.write("*Material, name=Aluminum\n")
            f.write("*Elastic\n")
            f.write(f"{self.E_solid/1e6:.1f}, {self.nu}\n")
            f.write("*Part, name=TPMS_Solid\n")
            f.write("*Node\n")
            
            for i, v in enumerate(nodes, start=1):
                f.write(f"{i}, {v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}\n")
            
            # 不再使用孤立参考点节点（它们会被ABAQUS视为不活跃）
            # 改用网格中实际存在的角落节点
            
            f.write(f"*Element, type={solid_type}\n")
            for eid, conn in enumerate(elems, start=1):
                f.write(f"{eid}, {conn[0]}, {conn[1]}, {conn[2]}, {conn[3]}, {conn[4]}, {conn[5]}, {conn[6]}, {conn[7]}\n")
            
            total_elements = len(elems)
            xs = nodes[:, 0]
            ys = nodes[:, 1]
            zs = nodes[:, 2]
            
            tol_x = max(1e-3, 1e-3*(xs.max()-xs.min()))
            tol_y = max(1e-3, 1e-3*(ys.max()-ys.min()))
            tol_z = max(1e-3, 1e-3*(zs.max()-zs.min()))
            
            x_min_idx = np.where(xs <= xs.min()+tol_x)[0]
            x_max_idx = np.where(xs >= xs.max()-tol_x)[0]
            y_min_idx = np.where(ys <= ys.min()+tol_y)[0]
            y_max_idx = np.where(ys >= ys.max()-tol_y)[0]
            z_min_idx = np.where(zs <= zs.min()+tol_z)[0]
            z_max_idx = np.where(zs >= zs.max()-tol_z)[0]
            
            # 节点集（只写入非空的节点集）
            for name, idx in [('X_MIN', x_min_idx), ('X_MAX', x_max_idx),
                             ('Y_MIN', y_min_idx), ('Y_MAX', y_max_idx),
                             ('Z_MIN', z_min_idx), ('Z_MAX', z_max_idx)]:
                if len(idx) > 0:
                    f.write(f"*Nset, nset={name}\n")
                    for i in range(0, len(idx), 16):
                        chunk = (idx[i:i+16]+1).tolist()
                        f.write(", ".join(map(str, chunk)) + "\n")
            
            f.write("*Elset, elset=ALL_ELEMENTS\n")
            all_elements = list(range(1, total_elements+1))
            for i in range(0, len(all_elements), 16):
                chunk = all_elements[i:i+16]
                f.write(", ".join(map(str, chunk)) + "\n")
            
            f.write("*Solid Section, elset=ALL_ELEMENTS, material=Aluminum\n")
            
            # 使用实际网格角落节点作为参考点（避免孤立节点不活跃问题）
            def find_corner_node(target_coord):
                """找到最接近目标坐标的节点"""
                dists = np.linalg.norm(nodes - target_coord, axis=1)
                return int(np.argmin(dists)) + 1
            
            x_min_val, x_max_val = nodes[:,0].min(), nodes[:,0].max()
            y_min_val, y_max_val = nodes[:,1].min(), nodes[:,1].max()
            z_min_val, z_max_val = nodes[:,2].min(), nodes[:,2].max()
            
            # 所有8个角落节点（使用实际网格节点）
            corner_000_id = find_corner_node(np.array([x_min_val, y_min_val, z_min_val]))  # (0,0,0)
            corner_L00_id = find_corner_node(np.array([x_max_val, y_min_val, z_min_val]))  # (L,0,0)
            corner_0L0_id = find_corner_node(np.array([x_min_val, y_max_val, z_min_val]))  # (0,L,0)
            corner_00L_id = find_corner_node(np.array([x_min_val, y_min_val, z_max_val]))  # (0,0,L)
            corner_LL0_id = find_corner_node(np.array([x_max_val, y_max_val, z_min_val]))  # (L,L,0)
            corner_L0L_id = find_corner_node(np.array([x_max_val, y_min_val, z_max_val]))  # (L,0,L)
            corner_0LL_id = find_corner_node(np.array([x_min_val, y_max_val, z_max_val]))  # (0,L,L)
            corner_LLL_id = find_corner_node(np.array([x_max_val, y_max_val, z_max_val]))  # (L,L,L)
            
            # 在 Part 内定义所有8个角落节点集
            for name, nid in [('CORNER_000', corner_000_id), ('CORNER_L00', corner_L00_id),
                              ('CORNER_0L0', corner_0L0_id), ('CORNER_00L', corner_00L_id),
                              ('CORNER_LL0', corner_LL0_id), ('CORNER_L0L', corner_L0L_id),
                              ('CORNER_0LL', corner_0LL_id), ('CORNER_LLL', corner_LLL_id)]:
                f.write(f"*Nset, nset={name}\n")
                f.write(f"{nid}\n")
            
            f.write("*End Part\n")
            f.write("*Assembly, name=Assembly\n")
            f.write("*Instance, name=TPMS-1, part=TPMS_Solid\n")
            f.write("*End Instance\n")
            
            # 在 Assembly 级别定义所有8个角落节点集
            for name, nid in [('CORNER_000', corner_000_id), ('CORNER_L00', corner_L00_id),
                              ('CORNER_0L0', corner_0L0_id), ('CORNER_00L', corner_00L_id),
                              ('CORNER_LL0', corner_LL0_id), ('CORNER_L0L', corner_L0L_id),
                              ('CORNER_0LL', corner_0LL_id), ('CORNER_LLL', corner_LLL_id)]:
                f.write(f"*Nset, nset={name}, instance=TPMS-1\n")
                f.write(f"{nid}\n")
            
            # 周期性边界条件 (Equation) - 移入 Assembly 内部
            corner_nodes = {
                '000': corner_000_id, 'L00': corner_L00_id,
                '0L0': corner_0L0_id, '00L': corner_00L_id,
                'LL0': corner_LL0_id, 'L0L': corner_L0L_id,
                '0LL': corner_0LL_id, 'LLL': corner_LLL_id
            }
            self._write_periodic_equations(f, nodes, x_min_idx, x_max_idx, 
                                           y_min_idx, y_max_idx, z_min_idx, z_max_idx,
                                           corner_nodes)
            
            f.write("*End Assembly\n")
            
            # 分析步
            f.write("*Step, name=Loading, nlgeom=NO\n")
            f.write("*Static\n")
            f.write("0.1, 1., 1e-06, 0.1\n")
            
            disp_x = displacement[0] * 1000
            disp_y = displacement[1] * 1000
            disp_z = displacement[2] * 1000
            
            # 边界条件：所有8个角落节点的完整位移约束
            # 周期性要求：u(x+L) - u(x) = δx，v(y+L) - v(y) = δy，w(z+L) - w(z) = δz
            f.write("*Boundary\n")
            # CORNER_000 (0,0,0): 原点固定
            f.write("CORNER_000, 1, 1, 0.\n")
            f.write("CORNER_000, 2, 2, 0.\n")
            f.write("CORNER_000, 3, 3, 0.\n")
            # CORNER_L00 (L,0,0): X偏移
            f.write(f"CORNER_L00, 1, 1, {disp_x:.6f}\n")
            f.write("CORNER_L00, 2, 2, 0.\n")
            f.write("CORNER_L00, 3, 3, 0.\n")
            # CORNER_0L0 (0,L,0): Y偏移
            f.write("CORNER_0L0, 1, 1, 0.\n")
            f.write(f"CORNER_0L0, 2, 2, {disp_y:.6f}\n")
            f.write("CORNER_0L0, 3, 3, 0.\n")
            # CORNER_00L (0,0,L): Z偏移
            f.write("CORNER_00L, 1, 1, 0.\n")
            f.write("CORNER_00L, 2, 2, 0.\n")
            f.write(f"CORNER_00L, 3, 3, {disp_z:.6f}\n")
            # CORNER_LL0 (L,L,0): X+Y偏移
            f.write(f"CORNER_LL0, 1, 1, {disp_x:.6f}\n")
            f.write(f"CORNER_LL0, 2, 2, {disp_y:.6f}\n")
            f.write("CORNER_LL0, 3, 3, 0.\n")
            # CORNER_L0L (L,0,L): X+Z偏移
            f.write(f"CORNER_L0L, 1, 1, {disp_x:.6f}\n")
            f.write("CORNER_L0L, 2, 2, 0.\n")
            f.write(f"CORNER_L0L, 3, 3, {disp_z:.6f}\n")
            # CORNER_0LL (0,L,L): Y+Z偏移
            f.write("CORNER_0LL, 1, 1, 0.\n")
            f.write(f"CORNER_0LL, 2, 2, {disp_y:.6f}\n")
            f.write(f"CORNER_0LL, 3, 3, {disp_z:.6f}\n")
            # CORNER_LLL (L,L,L): X+Y+Z偏移
            f.write(f"CORNER_LLL, 1, 1, {disp_x:.6f}\n")
            f.write(f"CORNER_LLL, 2, 2, {disp_y:.6f}\n")
            f.write(f"CORNER_LLL, 3, 3, {disp_z:.6f}\n")
            
            f.write("*Output, field, variable=PRESELECT\n")
            f.write("*Node Output\n")
            f.write("U, RF\n")
            f.write("*Element Output\n")
            f.write("S, E, MISES, PEEQ\n")
            f.write("*End Step\n")
    
    def _pair_boundary_nodes(self, coords_min: np.ndarray, coords_max: np.ndarray,
                              idx_min: np.ndarray, idx_max: np.ndarray, 
                              direction: str) -> list:
        """使用KD-Tree快速配对边界节点，确保一对一映射"""
        from scipy.spatial import cKDTree
        
        if len(idx_min) == 0 or len(idx_max) == 0:
            return []
        
        n_min, n_max = len(idx_min), len(idx_max)
        if n_min != n_max:
            print(f"  [WARNING] {direction}方向边界节点数不匹配: MIN={n_min}, MAX={n_max}")
        
        # 构建 KD-Tree 对较大一侧
        # 从较小一侧查询，并确保每个max节点只被配对一次
        if n_min <= n_max:
            tree = cKDTree(coords_max)
            # 查询多个候选以便去重
            k = min(5, n_max)
            dists, indices = tree.query(coords_min, k=k)
            
            used_max = set()
            pairs = []
            for i in range(n_min):
                # 找到第一个未被使用的最近邻
                candidates = indices[i] if k > 1 else [indices[i]]
                for j in (candidates if isinstance(candidates, np.ndarray) else [candidates]):
                    if j not in used_max:
                        used_max.add(j)
                        pairs.append((int(idx_min[i])+1, int(idx_max[j])+1))
                        break
        else:
            tree = cKDTree(coords_min)
            k = min(5, n_min)
            dists, indices = tree.query(coords_max, k=k)
            
            used_min = set()
            pairs = []
            for j in range(n_max):
                candidates = indices[j] if k > 1 else [indices[j]]
                for i in (candidates if isinstance(candidates, np.ndarray) else [candidates]):
                    if i not in used_min:
                        used_min.add(i)
                        pairs.append((int(idx_min[i])+1, int(idx_max[j])+1))
                        break
        
        print(f"  [INFO] {direction}方向: 成功配对 {len(pairs)} 个节点对")
        return pairs
    
    def _write_periodic_equations(self, f, nodes: np.ndarray,
                                   x_min_idx, x_max_idx, y_min_idx, y_max_idx, 
                                   z_min_idx, z_max_idx, corner_nodes: dict):
        """写入周期性边界条件方程
        
        使用实际网格角落节点作为参考，约束形式：
        u(x_max) - u(x_min) - u(corner_L00) + u(corner_000) = 0
        即 u(x_max) - u(x_min) = u(corner_L00) - u(corner_000)
        
        注意：为避免边缘/角落节点被多次约束，需要排除重叠节点
        """
        c000 = corner_nodes['000']
        cL00 = corner_nodes['L00']
        c0L0 = corner_nodes['0L0']
        c00L = corner_nodes['00L']
        
        # 转换为集合便于快速查找，并排除角落节点
        corner_set = {c000-1, cL00-1, c0L0-1, c00L-1}  # -1 因为节点ID从1开始
        x_boundary_set = set(x_min_idx.tolist()) | set(x_max_idx.tolist())
        y_boundary_set = set(y_min_idx.tolist()) | set(y_max_idx.tolist())
        
        # X方向配对 - 排除角落节点
        x_min_filtered = np.array([i for i in x_min_idx if i not in corner_set])
        x_max_filtered = np.array([i for i in x_max_idx if i not in corner_set])
        if x_min_filtered.size > 0 and x_max_filtered.size > 0:
            yz_min = nodes[x_min_filtered][:, 1:3]
            yz_max = nodes[x_max_filtered][:, 1:3]
            pairs_x = self._pair_boundary_nodes(yz_min, yz_max, x_min_filtered, x_max_filtered, 'X')
        else:
            pairs_x = []
        
        # Y方向配对 - 排除 X 边界和角落节点
        y_min_filtered = np.array([i for i in y_min_idx if i not in x_boundary_set and i not in corner_set])
        y_max_filtered = np.array([i for i in y_max_idx if i not in x_boundary_set and i not in corner_set])
        if y_min_filtered.size > 0 and y_max_filtered.size > 0:
            xz_min = nodes[y_min_filtered][:, [0, 2]]
            xz_max = nodes[y_max_filtered][:, [0, 2]]
            pairs_y = self._pair_boundary_nodes(xz_min, xz_max, y_min_filtered, y_max_filtered, 'Y')
        else:
            pairs_y = []
        
        # Z方向配对 - 排除 X/Y 边界和角落节点
        xy_boundary_set = x_boundary_set | y_boundary_set
        z_min_filtered = np.array([i for i in z_min_idx if i not in xy_boundary_set and i not in corner_set])
        z_max_filtered = np.array([i for i in z_max_idx if i not in xy_boundary_set and i not in corner_set])
        if z_min_filtered.size > 0 and z_max_filtered.size > 0:
            xy_min = nodes[z_min_filtered][:, 0:2]
            xy_max = nodes[z_max_filtered][:, 0:2]
            pairs_z = self._pair_boundary_nodes(xy_min, xy_max, z_min_filtered, z_max_filtered, 'Z')
        else:
            pairs_z = []
        
        # 写入X方向约束: u(x_max) - u(x_min) - u(corner_L00) + u(corner_000) = 0
        for a, b in pairs_x:
            f.write("*Equation\n4\n")
            f.write(f"TPMS-1.{b}, 1, 1.\n")
            f.write(f"TPMS-1.{a}, 1, -1.\n")
            f.write(f"TPMS-1.{cL00}, 1, -1.\n")
            f.write(f"TPMS-1.{c000}, 1, 1.\n")
            # Y, Z 方向保持周期性
            f.write("*Equation\n2\n")
            f.write(f"TPMS-1.{b}, 2, 1.\n")
            f.write(f"TPMS-1.{a}, 2, -1.\n")
            f.write("*Equation\n2\n")
            f.write(f"TPMS-1.{b}, 3, 1.\n")
            f.write(f"TPMS-1.{a}, 3, -1.\n")
        
        # 写入Y方向约束: v(y_max) - v(y_min) - v(corner_0L0) + v(corner_000) = 0
        for a, b in pairs_y:
            f.write("*Equation\n4\n")
            f.write(f"TPMS-1.{b}, 2, 1.\n")
            f.write(f"TPMS-1.{a}, 2, -1.\n")
            f.write(f"TPMS-1.{c0L0}, 2, -1.\n")
            f.write(f"TPMS-1.{c000}, 2, 1.\n")
            f.write("*Equation\n2\n")
            f.write(f"TPMS-1.{b}, 1, 1.\n")
            f.write(f"TPMS-1.{a}, 1, -1.\n")
            f.write("*Equation\n2\n")
            f.write(f"TPMS-1.{b}, 3, 1.\n")
            f.write(f"TPMS-1.{a}, 3, -1.\n")
        
        # 写入Z方向约束: w(z_max) - w(z_min) - w(corner_00L) + w(corner_000) = 0
        for a, b in pairs_z:
            f.write("*Equation\n4\n")
            f.write(f"TPMS-1.{b}, 3, 1.\n")
            f.write(f"TPMS-1.{a}, 3, -1.\n")
            f.write(f"TPMS-1.{c00L}, 3, -1.\n")
            f.write(f"TPMS-1.{c000}, 3, 1.\n")
            f.write("*Equation\n2\n")
            f.write(f"TPMS-1.{b}, 1, 1.\n")
            f.write(f"TPMS-1.{a}, 1, -1.\n")
            f.write("*Equation\n2\n")
            f.write(f"TPMS-1.{b}, 2, 1.\n")
            f.write(f"TPMS-1.{a}, 2, -1.\n")
    
    # ========== INP写入函数 (Shell) ==========
    
    def _write_inp_shell(self, filepath: str, vertices: np.ndarray, faces: np.ndarray,
                         L_mm: float, alpha: list, displacement: list,
                         shell_type: str = 'S3', shell_thickness: float = 2.0):
        """写入Shell单元INP文件"""
        # 网格质量改进
        print(f"  [INFO] 原始网格: {len(vertices)} 节点, {len(faces)} 面片")
        quality_metrics = self._compute_mesh_quality(vertices, faces)
        improved_verts, improved_faces = self._improve_mesh_quality(vertices, faces, quality_metrics)
        print(f"  [INFO] 改进后: {len(improved_verts)} 节点, {len(improved_faces)} 面片")
        
        # 三角形配对
        if shell_type in ['S4R', 'MIXED']:
            print(f"  [INFO] 三角形配对生成四边形单元...")
            quad_elements, tri_elements = self._pair_triangles_to_quads(improved_faces, improved_verts)
            print(f"    - 四边形(S4R): {len(quad_elements)} 个")
            print(f"    - 三角形(S3):  {len(tri_elements)} 个")
        else:
            quad_elements = []
            tri_elements = improved_faces
        
        # 计算边界节点
        min_x, max_x = improved_verts[:, 0].min(), improved_verts[:, 0].max()
        min_y, max_y = improved_verts[:, 1].min(), improved_verts[:, 1].max()
        min_z, max_z = improved_verts[:, 2].min(), improved_verts[:, 2].max()
        
        tol_x = max(1e-3, 1e-3 * (max_x - min_x))
        tol_y = max(1e-3, 1e-3 * (max_y - min_y))
        tol_z = max(1e-3, 1e-3 * (max_z - min_z))
        
        x_min_idx = np.where(improved_verts[:, 0] <= min_x + tol_x)[0]
        x_max_idx = np.where(improved_verts[:, 0] >= max_x - tol_x)[0]
        y_min_idx = np.where(improved_verts[:, 1] <= min_y + tol_y)[0]
        y_max_idx = np.where(improved_verts[:, 1] >= max_y - tol_y)[0]
        z_min_idx = np.where(improved_verts[:, 2] <= min_z + tol_z)[0]
        z_max_idx = np.where(improved_verts[:, 2] >= max_z - tol_z)[0]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("*Heading\n")
            f.write("** TPMS Shell - Generated by TPMSMeshGenerator\n")
            f.write(f"** Alpha: [{alpha[0]:.3f}, {alpha[1]:.3f}, {alpha[2]:.3f}, {alpha[3]:.3f}]\n")
            f.write(f"** Size: {L_mm:.1f} x {L_mm:.1f} x {L_mm:.1f} mm\n")
            f.write("** Units: mm-tonne-s-MPa\n")
            f.write("**\n")
            
            f.write("*Part, name=TPMS_Part\n")
            f.write("*Node\n")
            for i, v in enumerate(improved_verts, start=1):
                f.write(f"{i}, {v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}\n")
            
            # 不再使用孤立参考点节点，改用网格中实际存在的角落节点
            
            # 单元
            elem_id = 1
            if len(quad_elements) > 0:
                f.write("*Element, type=S4R\n")
                for quad in quad_elements:
                    f.write(f"{elem_id}, {quad[0]+1}, {quad[1]+1}, {quad[2]+1}, {quad[3]+1}\n")
                    elem_id += 1
            
            if len(tri_elements) > 0:
                f.write("*Element, type=S3\n")
                for tri in tri_elements:
                    f.write(f"{elem_id}, {tri[0]+1}, {tri[1]+1}, {tri[2]+1}\n")
                    elem_id += 1
            
            total_elements = len(quad_elements) + len(tri_elements)
            
            # 节点集
            for name, idx in [('X_MIN', x_min_idx), ('X_MAX', x_max_idx),
                             ('Y_MIN', y_min_idx), ('Y_MAX', y_max_idx),
                             ('Z_MIN', z_min_idx), ('Z_MAX', z_max_idx)]:
                if idx.size > 0:
                    f.write(f"*Nset, nset={name}\n")
                    nodes_list = (idx + 1).tolist()
                    for i in range(0, len(nodes_list), 16):
                        chunk = nodes_list[i:i+16]
                        f.write(", ".join(map(str, chunk)) + "\n")
            
            # 先定义单元集，再引用
            f.write("*Elset, elset=ALL_ELEMENTS\n")
            all_elements = list(range(1, total_elements + 1))
            for i in range(0, len(all_elements), 16):
                chunk = all_elements[i:i+16]
                f.write(", ".join(map(str, chunk)) + "\n")
            
            f.write(f"*Shell Section, elset=ALL_ELEMENTS, material=Aluminum\n")
            f.write(f"{shell_thickness}, 5\n")
            
            # 使用实际网格角落节点作为参考点（避免孤立节点不活跃问题）
            def find_corner_node_shell(target_coord):
                """找到最接近目标坐标的节点"""
                dists = np.linalg.norm(improved_verts - target_coord, axis=1)
                return int(np.argmin(dists)) + 1
            
            # 所有8个角落节点（使用实际网格节点）
            corner_000_id = find_corner_node_shell(np.array([min_x, min_y, min_z]))  # (0,0,0)
            corner_L00_id = find_corner_node_shell(np.array([max_x, min_y, min_z]))  # (L,0,0)
            corner_0L0_id = find_corner_node_shell(np.array([min_x, max_y, min_z]))  # (0,L,0)
            corner_00L_id = find_corner_node_shell(np.array([min_x, min_y, max_z]))  # (0,0,L)
            corner_LL0_id = find_corner_node_shell(np.array([max_x, max_y, min_z]))  # (L,L,0)
            corner_L0L_id = find_corner_node_shell(np.array([max_x, min_y, max_z]))  # (L,0,L)
            corner_0LL_id = find_corner_node_shell(np.array([min_x, max_y, max_z]))  # (0,L,L)
            corner_LLL_id = find_corner_node_shell(np.array([max_x, max_y, max_z]))  # (L,L,L)
            
            # 在 Part 内定义所有8个角落节点集
            for name, nid in [('CORNER_000', corner_000_id), ('CORNER_L00', corner_L00_id),
                              ('CORNER_0L0', corner_0L0_id), ('CORNER_00L', corner_00L_id),
                              ('CORNER_LL0', corner_LL0_id), ('CORNER_L0L', corner_L0L_id),
                              ('CORNER_0LL', corner_0LL_id), ('CORNER_LLL', corner_LLL_id)]:
                f.write(f"*Nset, nset={name}\n")
                f.write(f"{nid}\n")
            
            f.write("*End Part\n")
            
            f.write("**\n")
            f.write("*Assembly, name=Assembly\n")
            f.write("*Instance, name=TPMS-1, part=TPMS_Part\n")
            f.write("*End Instance\n")
            
            # 在 Assembly 级别定义所有8个角落节点集
            for name, nid in [('CORNER_000', corner_000_id), ('CORNER_L00', corner_L00_id),
                              ('CORNER_0L0', corner_0L0_id), ('CORNER_00L', corner_00L_id),
                              ('CORNER_LL0', corner_LL0_id), ('CORNER_L0L', corner_L0L_id),
                              ('CORNER_0LL', corner_0LL_id), ('CORNER_LLL', corner_LLL_id)]:
                f.write(f"*Nset, nset={name}, instance=TPMS-1\n")
                f.write(f"{nid}\n")
            
            # 周期性边界条件 - 移入 Assembly 内部
            corner_nodes = {
                '000': corner_000_id, 'L00': corner_L00_id,
                '0L0': corner_0L0_id, '00L': corner_00L_id,
                'LL0': corner_LL0_id, 'L0L': corner_L0L_id,
                '0LL': corner_0LL_id, 'LLL': corner_LLL_id
            }
            self._write_periodic_equations(f, improved_verts, x_min_idx, x_max_idx,
                                           y_min_idx, y_max_idx, z_min_idx, z_max_idx,
                                           corner_nodes)
            
            f.write("*End Assembly\n")
            
            f.write("**\n")
            f.write("*Material, name=Aluminum\n")
            f.write("*Elastic\n")
            f.write(f"{self.E_solid/1e6:.1f}, {self.nu}\n")
            
            f.write("**\n")
            f.write("*Step, name=Loading, nlgeom=NO\n")
            f.write("*Static\n")
            f.write("0.1, 1., 1e-06, 0.1\n")
            
            disp_x = displacement[0] * 1000
            disp_y = displacement[1] * 1000
            disp_z = displacement[2] * 1000
            
            # 边界条件：所有8个角落节点的完整位移约束
            # 周期性要求：u(x+L) - u(x) = δx，v(y+L) - v(y) = δy，w(z+L) - w(z) = δz
            f.write("*Boundary\n")
            # CORNER_000 (0,0,0): 原点固定
            f.write("CORNER_000, 1, 1, 0.\n")
            f.write("CORNER_000, 2, 2, 0.\n")
            f.write("CORNER_000, 3, 3, 0.\n")
            # CORNER_L00 (L,0,0): X偏移
            f.write(f"CORNER_L00, 1, 1, {disp_x:.6f}\n")
            f.write("CORNER_L00, 2, 2, 0.\n")
            f.write("CORNER_L00, 3, 3, 0.\n")
            # CORNER_0L0 (0,L,0): Y偏移
            f.write("CORNER_0L0, 1, 1, 0.\n")
            f.write(f"CORNER_0L0, 2, 2, {disp_y:.6f}\n")
            f.write("CORNER_0L0, 3, 3, 0.\n")
            # CORNER_00L (0,0,L): Z偏移
            f.write("CORNER_00L, 1, 1, 0.\n")
            f.write("CORNER_00L, 2, 2, 0.\n")
            f.write(f"CORNER_00L, 3, 3, {disp_z:.6f}\n")
            # CORNER_LL0 (L,L,0): X+Y偏移
            f.write(f"CORNER_LL0, 1, 1, {disp_x:.6f}\n")
            f.write(f"CORNER_LL0, 2, 2, {disp_y:.6f}\n")
            f.write("CORNER_LL0, 3, 3, 0.\n")
            # CORNER_L0L (L,0,L): X+Z偏移
            f.write(f"CORNER_L0L, 1, 1, {disp_x:.6f}\n")
            f.write("CORNER_L0L, 2, 2, 0.\n")
            f.write(f"CORNER_L0L, 3, 3, {disp_z:.6f}\n")
            # CORNER_0LL (0,L,L): Y+Z偏移
            f.write("CORNER_0LL, 1, 1, 0.\n")
            f.write(f"CORNER_0LL, 2, 2, {disp_y:.6f}\n")
            f.write(f"CORNER_0LL, 3, 3, {disp_z:.6f}\n")
            # CORNER_LLL (L,L,L): X+Y+Z偏移
            f.write(f"CORNER_LLL, 1, 1, {disp_x:.6f}\n")
            f.write(f"CORNER_LLL, 2, 2, {disp_y:.6f}\n")
            f.write(f"CORNER_LLL, 3, 3, {disp_z:.6f}\n")
            
            f.write("**\n")
            f.write("*Output, field, variable=PRESELECT\n")
            f.write("*Node Output\n")
            f.write("U, RF\n")
            f.write("*Element Output\n")
            f.write("S, E, MISES, PEEQ\n")
            f.write("*End Step\n")
    
    # ========== 网格质量函数 ==========
    
    def _compute_mesh_quality(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """计算网格质量指标"""
        try:
            a = vertices[faces[:, 0]]
            b = vertices[faces[:, 1]]
            c = vertices[faces[:, 2]]
            
            cross_prod = np.cross(b - a, c - a)
            areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
            
            edge_ab = np.linalg.norm(b - a, axis=1)
            edge_bc = np.linalg.norm(c - b, axis=1)
            edge_ca = np.linalg.norm(a - c, axis=1)
            
            perimeter = edge_ab + edge_bc + edge_ca
            inradius = areas / (0.5 * perimeter + 1e-10)
            max_edge = np.maximum(np.maximum(edge_ab, edge_bc), edge_ca)
            aspect_ratios = max_edge / (2 * inradius + 1e-10)
            
            cos_A = (edge_ab**2 + edge_ca**2 - edge_bc**2) / (2 * edge_ab * edge_ca + 1e-10)
            cos_B = (edge_ab**2 + edge_bc**2 - edge_ca**2) / (2 * edge_ab * edge_bc + 1e-10)
            cos_C = (edge_bc**2 + edge_ca**2 - edge_ab**2) / (2 * edge_bc * edge_ca + 1e-10)
            
            cos_A = np.clip(cos_A, -1, 1)
            cos_B = np.clip(cos_B, -1, 1)
            cos_C = np.clip(cos_C, -1, 1)
            
            angles_A = np.arccos(cos_A) * 180 / np.pi
            angles_B = np.arccos(cos_B) * 180 / np.pi
            angles_C = np.arccos(cos_C) * 180 / np.pi
            
            min_angles = np.minimum(np.minimum(angles_A, angles_B), angles_C)
            
            return {
                'areas': areas,
                'aspect_ratios': aspect_ratios,
                'min_angles': min_angles
            }
        except Exception as e:
            n_faces = len(faces)
            return {
                'areas': np.ones(n_faces),
                'aspect_ratios': np.ones(n_faces),
                'min_angles': np.full(n_faces, 60.0)
            }
    
    def _improve_mesh_quality(self, vertices: np.ndarray, faces: np.ndarray,
                              quality_metrics: dict) -> tuple:
        """智能网格质量改进"""
        area_tol = 1e-6
        aspect_tol = 200
        angle_tol = 5.0
        
        areas = quality_metrics['areas']
        aspect_ratios = quality_metrics['aspect_ratios']
        min_angles = quality_metrics['min_angles']
        
        bad_area = areas <= area_tol
        bad_aspect = aspect_ratios >= aspect_tol
        bad_angle = min_angles <= angle_tol
        
        problem_faces = bad_area | bad_aspect | bad_angle
        
        if np.sum(problem_faces) == 0:
            return vertices, faces
        
        # 拉普拉斯平滑
        improved_vertices = self._laplacian_smoothing(vertices, faces, iterations=2)
        
        # 重新计算质量
        improved_quality = self._compute_mesh_quality(improved_vertices, faces)
        
        # 删除严重退化面片
        severe_degenerate = (improved_quality['areas'] <= 1e-8) | \
                           (improved_quality['aspect_ratios'] >= 1000) | \
                           (improved_quality['min_angles'] <= 0.1)
        
        if np.any(severe_degenerate):
            valid_faces = faces[~severe_degenerate]
            used_nodes = np.unique(valid_faces)
            old_to_new = -np.ones(len(improved_vertices), dtype=int)
            old_to_new[used_nodes] = np.arange(len(used_nodes))
            final_vertices = improved_vertices[used_nodes]
            final_faces = old_to_new[valid_faces]
        else:
            final_vertices = improved_vertices
            final_faces = faces
        
        return final_vertices, final_faces
    
    def _laplacian_smoothing(self, vertices: np.ndarray, faces: np.ndarray,
                             iterations: int = 2, alpha: float = 0.1) -> np.ndarray:
        """拉普拉斯平滑"""
        neighbors = defaultdict(set)
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                neighbors[v1].add(v2)
                neighbors[v2].add(v1)
        
        smoothed_vertices = vertices.copy()
        
        for iteration in range(iterations):
            new_vertices = smoothed_vertices.copy()
            for v_idx in range(len(vertices)):
                if v_idx in neighbors and len(neighbors[v_idx]) > 0:
                    neighbor_coords = smoothed_vertices[list(neighbors[v_idx])]
                    centroid = np.mean(neighbor_coords, axis=0)
                    new_vertices[v_idx] = (1 - alpha) * smoothed_vertices[v_idx] + alpha * centroid
            smoothed_vertices = new_vertices
        
        return smoothed_vertices
    
    def _pair_triangles_to_quads(self, triangles: np.ndarray, vertices: np.ndarray) -> tuple:
        """三角形配对为四边形"""
        edge_to_tris = defaultdict(list)
        
        for tri_idx, tri in enumerate(triangles):
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                edge_to_tris[edge].append(tri_idx)
        
        paired = set()
        quads = []
        
        for edge, tri_indices in edge_to_tris.items():
            if len(tri_indices) == 2:
                tri_idx1, tri_idx2 = tri_indices
                if tri_idx1 in paired or tri_idx2 in paired:
                    continue
                
                tri1 = triangles[tri_idx1]
                tri2 = triangles[tri_idx2]
                
                shared_edge = set(edge)
                vert1_unique = [v for v in tri1 if v not in shared_edge][0]
                vert2_unique = [v for v in tri2 if v not in shared_edge][0]
                
                edge_list = list(edge)
                quad = [edge_list[0], vert1_unique, edge_list[1], vert2_unique]
                
                if self._is_valid_quad(quad, vertices):
                    quads.append(quad)
                    paired.add(tri_idx1)
                    paired.add(tri_idx2)
        
        remaining_tris = [triangles[i] for i in range(len(triangles)) if i not in paired]
        
        return np.array(quads, dtype=int) if quads else np.array([], dtype=int).reshape(0, 4), \
               np.array(remaining_tris, dtype=int) if remaining_tris else np.array([], dtype=int).reshape(0, 3)
    
    def _is_valid_quad(self, quad: list, vertices: np.ndarray) -> bool:
        """检查四边形有效性"""
        if len(set(quad)) != 4:
            return False
        
        try:
            coords = vertices[quad]
            area1 = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            area2 = 0.5 * np.linalg.norm(np.cross(coords[2] - coords[0], coords[3] - coords[0]))
            total_area = area1 + area2
            
            if total_area < 1e-8:
                return False
            
            edges = [
                np.linalg.norm(coords[1] - coords[0]),
                np.linalg.norm(coords[2] - coords[1]),
                np.linalg.norm(coords[3] - coords[2]),
                np.linalg.norm(coords[0] - coords[3])
            ]
            
            max_edge = max(edges)
            min_edge = min(edges)
            
            if min_edge < 1e-8 or max_edge / min_edge > 50:
                return False
            
            return True
        except:
            return False


# ====================================================================================
# 命令行接口
# ====================================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TPMS网格生成与ABAQUS INP导出')
    parser.add_argument('--alpha', type=float, nargs=4, default=DEFAULT_ALPHA,
                       help='TPMS混合系数 [P, G, D, I-WP]')
    parser.add_argument('--disp', type=float, nargs=3, default=DEFAULT_DISPLACEMENT,
                       help='位移 [δx, δy, δz] (m)')
    parser.add_argument('--n_grid', type=int, default=DEFAULT_N_GRID, help='网格分辨率')
    parser.add_argument('--output', type=str, default='abaqus_export', help='输出目录')
    parser.add_argument('--mode', type=str, default=DEFAULT_MESH_MODE, choices=['shell', 'solid'],
                       help='网格模式')
    
    args = parser.parse_args()
    
    generator = TPMSMeshGenerator()
    generator.export_inp(
        alpha=args.alpha,
        displacement=args.disp,
        n_grid=args.n_grid,
        output_dir=args.output,
        mesh_mode=args.mode
    )


if __name__ == "__main__":
    main()
