# -*- coding: utf-8 -*-
"""
对验证算例做 Mises stress percentile 敏感性分析

输出每个 case 的:
- sigma_max
- p99.9, p99.5, p99.0, p95.0
- top 0.1%, 0.5%, 1.0%, 5.0% 平均值
"""

from __future__ import print_function
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIF = os.path.join(PROJECT_ROOT, 'outputs', 'verification')

CASES = [
    {'name': 'pareto_v1',         'odb': os.path.join(VERIF, 'pareto_v1', 'tpms_mesh.odb')},
    {'name': 'pareto_uniaxial_x', 'odb': os.path.join(VERIF, 'pareto_uniaxial_x', 'tpms_mesh.odb')},
    {'name': 'pareto_v3',         'odb': os.path.join(VERIF, 'pareto_v3', 'tpms_mesh.odb')},
]

OUTPUT_JSON = os.path.join(VERIF, 'stress_percentile_sensitivity.json')


def percentile_value(sorted_vals, pct):
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = int(n * pct / 100.0)
    if idx >= n:
        idx = n - 1
    return sorted_vals[idx]


def top_fraction_mean(sorted_vals, frac):
    n = len(sorted_vals)
    if n == 0:
        return None
    k = int(n * frac)
    if k < 1:
        k = 1
    subset = sorted_vals[-k:]
    return sum(subset) / float(len(subset))


def analyze_case(odb_path):
    from odbAccess import openOdb
    from abaqusConstants import ELEMENT_NODAL

    odb = openOdb(path=odb_path, readOnly=True)
    step_name = odb.steps.keys()[-1]
    step = odb.steps[step_name]
    frame = step.frames[-1]

    stress_field = frame.fieldOutputs['S']
    stress_at_nodes = stress_field.getSubset(position=ELEMENT_NODAL)

    mises = []
    for value in stress_at_nodes.values:
        mises.append(float(value.mises))

    odb.close()

    mises_sorted = sorted(mises)

    result = {
        'n_mises_points': len(mises_sorted),
        'sigma_max': mises_sorted[-1],
        'percentile_99_9': percentile_value(mises_sorted, 99.9),
        'percentile_99_5': percentile_value(mises_sorted, 99.5),
        'percentile_99_0': percentile_value(mises_sorted, 99.0),
        'percentile_95_0': percentile_value(mises_sorted, 95.0),
        'top_0_1_mean': top_fraction_mean(mises_sorted, 0.001),
        'top_0_5_mean': top_fraction_mean(mises_sorted, 0.005),
        'top_1_0_mean': top_fraction_mean(mises_sorted, 0.01),
        'top_5_0_mean': top_fraction_mean(mises_sorted, 0.05),
    }
    return result


def main():
    all_results = {}

    print('=' * 80)
    print('Mises Stress Percentile Sensitivity Analysis')
    print('=' * 80)

    for case in CASES:
        print('\n[Case] {}'.format(case['name']))
        print('ODB: {}'.format(case['odb']))
        if not os.path.exists(case['odb']):
            print('  ERROR: ODB not found')
            continue

        res = analyze_case(case['odb'])
        all_results[case['name']] = res

        print('  n_points      = {}'.format(res['n_mises_points']))
        print('  sigma_max     = {:.6f}'.format(res['sigma_max']))
        print('  p99.9         = {:.6f}'.format(res['percentile_99_9']))
        print('  p99.5         = {:.6f}'.format(res['percentile_99_5']))
        print('  p99.0         = {:.6f}'.format(res['percentile_99_0']))
        print('  p95.0         = {:.6f}'.format(res['percentile_95_0']))
        print('  top 0.1% mean = {:.6f}'.format(res['top_0_1_mean']))
        print('  top 0.5% mean = {:.6f}'.format(res['top_0_5_mean']))
        print('  top 1.0% mean = {:.6f}'.format(res['top_1_0_mean']))
        print('  top 5.0% mean = {:.6f}'.format(res['top_5_0_mean']))

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)

    print('\nSaved to: {}'.format(OUTPUT_JSON))


if __name__ == '__main__':
    main()
