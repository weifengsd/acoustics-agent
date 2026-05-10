"""
Deep analysis of AT vs PyBellhop differences.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyacoustics.config import ConfigLoader
from pyacoustics.simulation import Simulation
from tests.run_all_comparisons import read_at_shd, generate_at_bellhop_env
import subprocess
from scipy.ndimage import zoom

project_root = Path(__file__).resolve().parent.parent

envs = [
    ('pekeris', 'Pekeris (半空间)'),
    ('deep_water', 'Deep Water (深水)'),
    ('Munk', 'Munk (深海声道)'),
    ('shallow_water', 'Shallow Water (浅水)'),
    ('iso', 'Isovelocity (等声速)'),
    ('sduct', 'Surface Duct (表面声道)'),
]

fig, axes = plt.subplots(len(envs), 3, figsize=(20, 4*len(envs)))

bellhop_exe = project_root.parent / 'at' / 'bin' / 'bellhop.exe'
results = []

for row, (env_name, label) in enumerate(envs):
    yaml_path = project_root / 'tests' / env_name
    yaml_file = yaml_path / ('munk.yaml' if env_name == 'Munk' else f'{env_name}.yaml')
    if not yaml_file.exists():
        continue
    
    config = ConfigLoader.load_yaml(str(yaml_file))
    if config.solver.type != 'bellhop':
        continue

    # AT Bellhop
    env_file = yaml_path / f'{env_name}_at.env'
    generate_at_bellhop_env(config, env_file)
    subprocess.run([str(bellhop_exe), env_file.stem], cwd=str(yaml_path),
                   check=True, capture_output=True)
    
    shd_file = yaml_path / f'{env_file.stem}.shd'
    P_at, rz_at, rr_at = read_at_shd(shd_file)
    TL_at_full = -20 * np.log10(np.abs(P_at) + 1e-20)  # shape: (nrz, nrr)
    
    # PyBellhop coherent
    sim = Simulation(config)
    num_r, num_z = 200, 100
    TL_py = sim.compute_coherent_tl(num_r=num_r, num_z=num_z)  # shape: (num_r, num_z)
    _, r_grid_py, z_grid_py = sim._coherent_tl_cache
    
    # Resample AT onto PyBellhop grid via interpolation
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (rz_at.astype(np.float64), rr_at.astype(np.float64)),
        TL_at_full.astype(np.float64),
        method='linear', bounds_error=False, fill_value=120.0
    )
    # Build meshgrid matching TL_py layout: TL_py[ir, iz]
    rr_mesh, zz_mesh = np.meshgrid(r_grid_py, z_grid_py, indexing='ij')
    pts = np.column_stack([zz_mesh.ravel(), rr_mesh.ravel()])
    TL_at_rs = interp(pts).reshape(num_r, num_z)  # same shape as TL_py
    
    r_km = r_grid_py / 1000.0
    src_depth = config.geometry.source.depths[0]
    
    # Panel 1: TL at source depth
    iz_src = np.argmin(np.abs(z_grid_py - src_depth))
    tl_at_line = TL_at_rs[:, iz_src]
    tl_py_line = TL_py[:, iz_src]
    
    ax1 = axes[row, 0]
    ax1.plot(r_km, tl_at_line, 'b-', linewidth=1, label='AT', alpha=0.8)
    ax1.plot(r_km, tl_py_line, 'r--', linewidth=1, label='Py', alpha=0.8)
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('TL (dB)')
    ax1.set_title(f'{label}\nTL @ z={src_depth}m (source depth)')
    ax1.legend(fontsize=7)
    ylo = max(30, min(np.percentile(tl_at_line, 1), np.percentile(tl_py_line, 1)) - 5)
    yhi = min(120, max(np.percentile(tl_at_line, 99), np.percentile(tl_py_line, 99)) + 5)
    ax1.set_ylim(yhi, ylo)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TL at mid-depth
    iz_mid = num_z // 2
    z_mid = z_grid_py[iz_mid]
    tl_at_mid = TL_at_rs[:, iz_mid]
    tl_py_mid = TL_py[:, iz_mid]
    
    ax2 = axes[row, 1]
    ax2.plot(r_km, tl_at_mid, 'b-', linewidth=1, label='AT', alpha=0.8)
    ax2.plot(r_km, tl_py_mid, 'r--', linewidth=1, label='Py', alpha=0.8)
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('TL (dB)')
    ax2.set_title(f'TL @ z={z_mid:.0f}m (mid-depth)')
    ax2.legend(fontsize=7)
    ylo2 = max(30, min(np.percentile(tl_at_mid, 1), np.percentile(tl_py_mid, 1)) - 5)
    yhi2 = min(120, max(np.percentile(tl_at_mid, 99), np.percentile(tl_py_mid, 99)) + 5)
    ax2.set_ylim(yhi2, ylo2)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Error histogram
    diff = TL_py - TL_at_rs  # positive = PyBellhop predicts more loss
    valid = (TL_at_rs < 115) & (TL_py < 115)
    
    ax3 = axes[row, 2]
    if np.any(valid):
        d = diff[valid]
        mean_bias = np.mean(d)
        std_diff = np.std(d)
        mae = np.mean(np.abs(d))
        median_bias = np.median(d)
        
        ax3.hist(d, bins=60, color='steelblue', alpha=0.7, density=True,
                 range=(max(-40, d.min()), min(40, d.max())))
        ax3.axvline(mean_bias, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_bias:+.1f}')
        ax3.axvline(median_bias, color='orange', linestyle=':', linewidth=2,
                    label=f'Median: {median_bias:+.1f}')
        ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('TL_py - TL_at (dB)')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Error: MAE={mae:.1f}, σ={std_diff:.1f}, bias={mean_bias:+.1f}')
        ax3.legend(fontsize=7)
        
        results.append({
            'env': env_name, 'label': label,
            'mae': mae, 'bias': mean_bias, 'std': std_diff, 'median': median_bias
        })
        print(f"[{env_name:15s}] MAE={mae:6.2f}  Bias={mean_bias:+7.2f}  Std={std_diff:6.2f}  Median={median_bias:+7.2f}")
    else:
        ax3.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax3.transAxes)

plt.tight_layout()
plt.savefig(project_root / 'tests' / 'diff_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved diff_analysis.png")
