import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import struct
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pyacoustics.simulation import Simulation
from pyacoustics.config import ConfigLoader

def generate_at_bellhop_env(config, env_file_path):
    out = f"'{config.project}'\n"
    out += f"{config.frequency}\n"
    out += "1\n"
    
    ssp_type = config.environment.ssp.type
    opt1 = 'S' if ssp_type == 'spline' else 'C'
    out += f"'{opt1}VW'\n"
    
    depths = [d.depth for d in config.environment.ssp.data]
    cs = [d.c for d in config.environment.ssp.data]
    n_depths = len(depths)
    z_bottom = config.environment.bottom.depth
    out += f"{n_depths} 0.0 {z_bottom}\n"
    for d, c in zip(depths, cs):
        out += f"  {d} {c} /\n"
        
    out += "'A' 0.0\n"
    bot_c = config.environment.bottom.c_p if hasattr(config.environment.bottom, 'c_p') and config.environment.bottom.c_p else 1600.0
    bot_rho = config.environment.bottom.density if hasattr(config.environment.bottom, 'density') and config.environment.bottom.density else 1.8
    bot_alpha = config.environment.bottom.attenuation_p if hasattr(config.environment.bottom, 'attenuation_p') and config.environment.bottom.attenuation_p else 0.0
    out += f"{z_bottom} {bot_c} 0.0 {bot_rho} {bot_alpha} /\n"
    
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    
    out += "501\n"
    rd_min, rd_max = min(config.geometry.receivers.depths), max(config.geometry.receivers.depths)
    out += f"{rd_min} {rd_max} /\n"
    
    out += "1001\n"
    rr_min, rr_max = min(config.geometry.receivers.ranges)/1000.0, max(config.geometry.receivers.ranges)/1000.0
    out += f"{rr_min} {rr_max} /\n"
    
    out += "'CG'\n"
    
    num_beams = 2000
    config.solver.num_beams = num_beams
    out += f"{num_beams}\n"
    angles = config.solver.angles
    out += f"{angles[0]} {angles[1]} /\n"
    
    step = config.solver.step_size if config.solver.step_size > 0 else 50.0
    zbox = z_bottom * 1.1
    rbox = rr_max * 1.01
    out += f"{step} {zbox} {rbox}\n"
    
    with open(env_file_path, 'w') as f:
        f.write(out)

def read_at_shd(filename):
    with open(filename, 'rb') as f:
        lrecl_data = f.read(4)
        LRecl = struct.unpack('i', lrecl_data)[0]
        rec_size = LRecl * 4
        
        def read_rec(rec_num):
            f.seek((rec_num - 1) * rec_size)
            return f.read(rec_size)
            
        rec3 = read_rec(3)
        nfreq, ntheta, nsx, nsy, nsz, nrz, nrr = struct.unpack('7i', rec3[:28])
        
        rec9 = read_rec(9)
        rz_vec = np.frombuffer(rec9[:4*nrz], dtype=np.float32)
        
        rec10 = read_rec(10)
        rr_vec = np.frombuffer(rec10[:8*nrr], dtype=np.float64)
        
        P = np.zeros((nrz, nrr), dtype=np.complex64)
        for irz in range(nrz):
            rec = read_rec(11 + irz)
            row = np.frombuffer(rec[:8*nrr], dtype=np.complex64)
            P[irz, :] = row
            
        return P, rz_vec, rr_vec

def compute_tl_field_custom(config, ray_paths, rz_at, rr_at):
    from scipy.ndimage import gaussian_filter
    r_all = np.concatenate([r for r, z, a in ray_paths])
    z_all = np.concatenate([z for r, z, a in ray_paths])
    amp_all = np.concatenate([a for r, z, a in ray_paths])
    num_r = len(rr_at)
    num_z = len(rz_at)
    r_max = rr_at[-1]
    z_max = rz_at[-1]
    r_bins = np.linspace(0, r_max, num_r + 1)
    z_bins = np.linspace(0, z_max, num_z + 1)
    weights = np.abs(amp_all)**2
    H, _, _ = np.histogram2d(r_all, z_all, bins=[r_bins, z_bins], weights=weights)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    r_centers[r_centers == 0] = 1.0
    Intensity = H / r_centers[:, None]
    Intensity = gaussian_filter(Intensity, sigma=2.0)
    Intensity = np.maximum(Intensity, 1e-12)
    TL = -10 * np.log10(Intensity)
    TL = TL - np.min(TL) + 50
    return TL, r_bins, z_bins

def process_environment(yaml_path):
    base_dir = yaml_path.parent
    env_name = base_dir.name
    print(f"\n[{env_name}] Processing...")
    
    # Load config
    config = ConfigLoader.load_yaml(str(yaml_path))
    if config.solver.type != "bellhop":
        print(f"[{env_name}] Skipping, solver type is {config.solver.type}")
        return False
        
    bellhop_exe = project_root.parent / 'at' / 'bin' / 'bellhop.exe'
    
    # 1. AT Bellhop
    env_file = base_dir / f"{env_name}_at.env"
    generate_at_bellhop_env(config, env_file)
    print(f"[{env_name}] Running AT Bellhop...")
    try:
        subprocess.run([str(bellhop_exe), env_file.stem], cwd=str(base_dir), check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[{env_name}] AT Bellhop failed!")
        return False
        
    shd_file = base_dir / f"{env_file.stem}.shd"
    if not shd_file.exists():
        print(f"[{env_name}] Error: .shd was not generated.")
        return False
        
    P_at, rz_at, rr_at = read_at_shd(shd_file)
    TL_at = -20 * np.log10(np.abs(P_at) + 1e-20)
    
    # 2. PyBellhop (Coherent Gaussian Beam)
    print(f"[{env_name}] Running PyBellhop (Coherent)...")
    sim = Simulation(config)
    
    try:
        # Use exact AT Bellhop grid for 1:1 comparison
        TL_py = sim.compute_coherent_tl(r_grid=rr_at, z_grid=rz_at)
    except Exception as e:
        print(f"[{env_name}] PyBellhop coherent TL failed: {e}")
        import traceback; traceback.print_exc()
        return False
    
    # Compare basic stats directly on 1:1 matching grids
    diff = np.abs(TL_at - TL_py.T)
    valid = TL_at < 120
    mean_diff = np.mean(diff[valid]) if np.any(valid) else float('nan')
    print(f"[{env_name}] Mean absolute difference (dB): {mean_diff:.2f}")
    
    # 3. Plot Comparison using environment-specific colorbar ranges (clim) from AT runtests
    CLIM_MAP = {
        'munk': (50.0, 100.0),
        'leaky_munk': (50.0, 100.0),
        'iso': (60.0, 80.0),
        'pekeris': (40.0, 80.0),
        'rigid_pekeris': (40.0, 80.0),
        'shallow_water': (40.0, 80.0),
        'arctic': (50.0, 100.0),
        'gulf': (70.0, 100.0),
        'dickins': (70.0, 120.0),
        'sduct': (50.0, 100.0),
        'sbcx': (40.0, 80.0),
        'sed_atten': (40.0, 80.0),
        'free_space': (60.0, 80.0),
        'thorp': (50.0, 100.0),
        'lloyd': (40.0, 80.0),
        'shaded_source': (50.0, 100.0),
        'bottom_gradient': (40.0, 80.0),
        'rap_water': (40.0, 80.0),
        'head_wave': (40.0, 80.0)
    }
    vmin, vmax = CLIM_MAP.get(env_name.lower(), (50.0, 100.0))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    r_min_km_at = rr_at[0] / 1000.0
    r_max_km_at = rr_at[-1] / 1000.0
    
    im1 = ax1.imshow(
        TL_at, 
        extent=[r_min_km_at, r_max_km_at, rz_at[-1], rz_at[0]],
        aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax
    )
    ax1.set_title(f'AT Bellhop Coherent TL ({env_name})')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Depth (m)')
    plt.colorbar(im1, ax=ax1, label='TL (dB)')
    
    im2 = ax2.imshow(
        TL_py.T, 
        extent=[r_min_km_at, r_max_km_at, rz_at[-1], rz_at[0]],
        aspect='auto', cmap='jet_r', vmin=vmin, vmax=vmax
    )
    ax2.set_title(f'PyBellhop Coherent TL ({env_name})')
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Depth (m)')
    plt.colorbar(im2, ax=ax2, label='TL (dB)')
    
    plt.tight_layout()
    save_path = base_dir / f'{env_name}_bellhop_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[{env_name}] Plot saved to {save_path.name}")
    return True

def main():
    tests_dir = project_root / 'tests'
    success_count = 0
    total_count = 0
    for subdir in tests_dir.iterdir():
        if subdir.is_dir():
            for yaml_file in subdir.glob("*.yaml"):
                # We only process one yaml per directory (the main one)
                if yaml_file.name == f"{subdir.name}.yaml":
                    total_count += 1
                    if process_environment(yaml_file):
                        success_count += 1
    print(f"\nProcessed {success_count}/{total_count} environments successfully.")

if __name__ == '__main__':
    main()
