import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from pyacoustics.schema import SimulationConfig

def get_auto_clim(TL: np.ndarray) -> Tuple[float, float]:
    """
    Calculates automatic color bar limits (vmin, vmax) based on the TL field.
    Uses a standard 60dB dynamic range starting from the minimum loss.
    """
    # Filter out invalid values (e.g., shadow zone caps)
    valid_tl = TL[TL < 119.0]
    if len(valid_tl) == 0:
        return 0.0, 100.0
    
    # Start vmin at a clean 5dB boundary near the 1st percentile
    vmin = np.floor(np.percentile(valid_tl, 1) / 5.0) * 5.0
    vmax = vmin + 60.0
    
    return vmin, vmax

def plot_rays(config: SimulationConfig, ray_paths: List[Tuple[np.ndarray, np.ndarray]], save_path: str = None):
    """
    Plots the Sound Speed Profile on the left and Ray Trajectories on the right.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 4]})
    
    # 1. Plot SSP
    z = [layer.depth for layer in config.environment.ssp.data]
    c = [layer.c for layer in config.environment.ssp.data]
    
    # Bottom depth
    z_bottom = config.environment.bottom.depth
    
    ax1.plot(c, z, 'b-', linewidth=2)
    ax1.set_ylim(z_bottom, 0) # Invert y axis
    ax1.set_xlabel('Sound Speed (m/s)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('SSP')
    ax1.grid(True, linestyle='--')
    
    # 2. Plot Rays
    for r_path, z_path, amp_path in ray_paths:
        # Convert range to km for easier viewing
        ax2.plot(r_path / 1000.0, z_path, 'k-', linewidth=0.5, alpha=0.6)
        
    ax2.set_ylim(z_bottom, 0) # Invert y axis
    ax2.set_xlim(0, max(config.geometry.receivers.ranges) / 1000.0)
    ax2.set_xlabel('Range (km)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title(f'Ray Trajectories ({config.project})')
    
    # Fill bottom
    ax2.fill_between(ax2.get_xlim(), z_bottom, z_bottom * 1.1, color='saddlebrown', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def compute_tl_field(config: SimulationConfig, ray_paths: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the approximate TL grid and returns (TL_grid, r_bins, z_bins).
    Uses the square of the amplitude path as weights to account for boundary losses.
    """
    from scipy.ndimage import gaussian_filter
    
    # Flatten all ray points and amplitudes
    r_all = np.concatenate([r for r, z, a in ray_paths])
    z_all = np.concatenate([z for r, z, a in ray_paths])
    amp_all = np.concatenate([a for r, z, a in ray_paths])
    weights = np.abs(amp_all)**2
    
    # Define grid
    r_max = max(config.geometry.receivers.ranges)
    z_max = config.environment.bottom.depth
    
    num_r = 800
    num_z = 400
    r_bins = np.linspace(0, r_max, num_r + 1)
    z_bins = np.linspace(0, z_max, num_z + 1)
    
    # 2. Histogram with intensity weights
    H, _, _ = np.histogram2d(r_all, z_all, bins=[r_bins, z_bins], weights=weights)
    
    # Calculate Intensity
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
    r_centers[r_centers == 0] = 1.0
    Intensity = H / r_centers[:, None]
    
    # Smooth
    Intensity = gaussian_filter(Intensity, sigma=2.0)
    
    # Convert to dB
    Intensity = np.maximum(Intensity, 1e-12)
    TL = -10 * np.log10(Intensity)
    
    # Normalization (Baseline)
    TL = TL - np.min(TL) + 50
    return TL, r_bins, z_bins

def plot_tl(config: SimulationConfig, ray_paths: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], save_path: str = None, **kwargs):
    """
    Plots the approximate incoherent Transmission Loss (TL) Field using Ray Density.
    """
    TL, r_bins, z_bins = compute_tl_field(config, ray_paths)
    r_max = max(config.geometry.receivers.ranges)
    z_max = config.environment.bottom.depth

    fig, ax = plt.subplots(figsize=(12, 6))
    auto_vmin, auto_vmax = get_auto_clim(TL)
    vmin = kwargs.get('vmin', auto_vmin)
    vmax = kwargs.get('vmax', auto_vmax)
    im = ax.imshow(TL.T, extent=[0, r_max/1000.0, z_max, 0], aspect='auto', cmap='jet_r',
                   vmin=vmin, vmax=vmax)
                   
    plt.colorbar(im, label='Approx. Transmission Loss (dB)')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Transmission Loss Field - {config.project}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

