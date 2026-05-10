import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import struct
from pathlib import Path
import shutil

# Add project root to sys.path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pyacoustics.simulation import Simulation

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
    """
    Computes the approximate TL grid matching the AT grid points.
    """
    from scipy.ndimage import gaussian_filter
    
    r_all = np.concatenate([r for r, z in ray_paths])
    z_all = np.concatenate([z for r, z in ray_paths])
    
    num_r = len(rr_at) - 1
    num_z = len(rz_at) - 1
    
    # AT uses ranges in meters, but rr_at is usually in meters (Wait, Bellhop rr output is meters or km?)
    # Bellhop shd reader rr_vec is in meters. The env file input is in km, but .shd writes meters.
    r_max = rr_at[-1]
    z_max = rz_at[-1]
    
    r_bins = np.linspace(0, r_max, num_r + 1)
    z_bins = np.linspace(0, z_max, num_z + 1)
    
    H, _, _ = np.histogram2d(r_all, z_all, bins=[r_bins, z_bins])
    
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    r_centers[r_centers == 0] = 1.0
    Intensity = H / r_centers[:, None]
    
    # Smooth
    Intensity = gaussian_filter(Intensity, sigma=2.0)
    
    Intensity = np.maximum(Intensity, 1e-12)
    TL = -10 * np.log10(Intensity)
    
    # Normalization (Baseline)
    TL = TL - np.min(TL) + 50
    return TL, r_bins, z_bins

def main():
    base_dir = Path(__file__).parent
    
    # 1. Prepare AT Bellhop Env
    ref_env = project_root.parent / 'at' / 'tests' / 'Munk' / 'MunkB_Coh.env'
    env_file = base_dir / 'munk_bellhop.env'
    shutil.copy(ref_env, env_file)
    
    # Run AT Bellhop
    bellhop_exe = project_root.parent / 'at' / 'bin' / 'bellhop.exe'
    print(f"Running {bellhop_exe}...")
    subprocess.run([str(bellhop_exe), 'munk_bellhop'], cwd=str(base_dir), check=True)
    
    shd_file = base_dir / 'munk_bellhop.shd'
    if not shd_file.exists():
        print(f"Error: {shd_file} was not generated.")
        return
        
    P_at, rz_at, rr_at = read_at_shd(shd_file)
    TL_at = -20 * np.log10(np.abs(P_at) + 1e-20)
    
    # 2. Run pyacoustics Bellhop
    print("Running pyacoustics Bellhop...")
    config_path = base_dir / "munk.yaml"
    sim = Simulation(config_path)
    res = sim.run()
    
    config = sim.config
    TL_py, r_bins_py, z_bins_py = compute_tl_field_custom(config, res, rz_at, rr_at)
    
    # 3. Plot Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    vmin, vmax = 50.0, 100.0
    
    r_min_km_at = rr_at[0] / 1000.0
    r_max_km_at = rr_at[-1] / 1000.0
    
    im1 = ax1.imshow(
        TL_at, 
        extent=[r_min_km_at, r_max_km_at, rz_at[-1], rz_at[0]],
        aspect='auto', 
        cmap='jet_r',
        vmin=vmin, 
        vmax=vmax
    )
    ax1.set_title('AT Bellhop Coherent TL (MunkB_Coh.env)')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Depth (m)')
    plt.colorbar(im1, ax=ax1, label='TL (dB)')
    
    im2 = ax2.imshow(
        TL_py.T, 
        extent=[r_min_km_at, r_max_km_at, rz_at[-1], rz_at[0]],
        aspect='auto', 
        cmap='jet_r',
        vmin=vmin, 
        vmax=vmax
    )
    ax2.set_title('acoustics-agent pybellhop Approximate TL')
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Depth (m)')
    plt.colorbar(im2, ax=ax2, label='TL (dB)')
    
    plt.tight_layout()
    save_path = base_dir / 'munk_bellhop_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    main()
