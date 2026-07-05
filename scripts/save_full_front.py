# -*- coding: utf-8 -*-
"""
存完整 Pareto 前沿坐标(图2用),纯代理推理(500万 Dirichlet),3 个主工况。
输出: outputs/benchmark_comparison/front_<load>.json
  front_sigma_hp / front_E_over_rho : 完整非支配前沿
  cloud_sigma_hp / cloud_E_over_rho : 候选点云(抽样 4 万,画灰底用)
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from surrogate_model.model import SurrogateModel
from surrogate_model.density_model import DensityModel
from scripts.pick_three_dirichlet import (
    compute_pareto_dirichlet, LOAD_CASES, SURROGATE_PATH, DENSITY_PATH, N_SAMPLES,
)

OUT = os.path.join(PROJECT_ROOT, "outputs", "benchmark_comparison")
os.makedirs(OUT, exist_ok=True)
MAIN = ["Biaxial-XY", "Triaxial-1-1-2", "Triaxial-1-2-2"]

surr = SurrogateModel.load(SURROGATE_PATH)
dens = DensityModel.load(DENSITY_PATH)

for case in MAIN:
    delta = LOAD_CASES[case]["delta"]
    res, mask = compute_pareto_dirichlet(surr, dens, delta, n_samples=N_SAMPLES)
    sig, ss = res["sigma_d"], res["specific_stiffness"]
    rng = np.random.RandomState(0)
    idx = rng.choice(len(sig), min(40000, len(sig)), replace=False)
    out = {
        "load": case,
        "delta_um": [float(x * 1e6) for x in delta],
        "pareto_size": int(mask.sum()),
        "front_sigma_hp": sig[mask].tolist(),
        "front_E_over_rho": ss[mask].tolist(),
        "cloud_sigma_hp": sig[idx].tolist(),
        "cloud_E_over_rho": ss[idx].tolist(),
    }
    json.dump(out, open(os.path.join(OUT, f"front_{case}.json"), "w"))
    print(f"{case}: front={int(mask.sum())}  sigma_hp[{sig[mask].min():.1f},{sig[mask].max():.1f}]  "
          f"E/rho[{ss[mask].min():.0f},{ss[mask].max():.0f}]")
print("Saved fronts to", OUT)
