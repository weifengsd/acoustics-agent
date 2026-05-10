import numpy as np
from numba import njit
from pyacoustics.schema import SimulationConfig
from pyacoustics.solvers.bellhop.tracer import trace_beam_ray, MAX_STEPS

@njit(fastmath=True)
def find_arrivals_at_receiver(
    r_rcv: float, z_rcv: float,
    ray_r: np.ndarray, ray_z: np.ndarray,
    ray_c: np.ndarray, ray_tau: np.ndarray,
    ray_p: np.ndarray, ray_q: np.ndarray,
    ray_amp: np.ndarray,
    n_pts: int,
    freq: float,
    z_min: float, z_max: float
):
    """
    Finds all crossings of a specific receiver location by a single ray.
    Uses a simple distance threshold and interpolation.
    """
    arrivals = []
    
    # We look for segments that pass 'close' to the receiver.
    # In a 2D sense, we look for the point of closest approach.
    for i in range(n_pts - 1):
        r1, z1 = ray_r[i], ray_z[i]
        r2, z2 = ray_r[i+1], ray_z[i+1]
        
        # Check if receiver range is within this segment's range span
        r_min_seg, r_max_seg = min(r1, r2), max(r1, r2)
        if r_rcv < r_min_seg or r_rcv > r_max_seg:
            continue
            
        # Linear interpolation to find z at r_rcv
        if abs(r2 - r1) < 1e-10:
            continue
            
        frac = (r_rcv - r1) / (r2 - r1)
        z_at_r = z1 + frac * (z2 - z1)
        
        # Check if z is 'close enough' to z_rcv
        # A typical threshold is wavelength or a small fixed distance
        # For arrivals, we often look for any crossing of the horizontal line z_rcv
        # and then filter by proximity, OR use a beam-width based approach.
        # Here we use a simpler 'window' approach.
        if abs(z_at_r - z_rcv) < 10.0: # 10m window for now
            # Found an arrival!
            tau = ray_tau[i] + frac * (ray_tau[i+1] - ray_tau[i])
            
            # Amplitude (complex)
            amp = ray_amp[i] + frac * (ray_amp[i+1] - ray_amp[i])
            
            # Angle at receiver (from ray tangent)
            dz = z2 - z1
            dr = r2 - r1
            angle_rad = np.arctan2(dz, dr)
            
            # Volume absorption (Thorp) if needed could be added here, 
            # but usually it's handled in the post-processing.
            
            arrivals.append((tau, amp, np.degrees(angle_rad)))
            
    return arrivals

def compute_arrivals(config: SimulationConfig, r_rcv: float, z_rcv: float):
    """
    Orchestrates arrival extraction for all rays.
    """
    from pyacoustics.solvers.bellhop.core import PyBellhop
    solver = PyBellhop(config)
    
    # We need full beam tracing to get p, q, and tau
    beam_data = solver.run_beam_tracing()
    
    all_arrivals = []
    
    n_rays = beam_data['num_beams']
    for i in range(n_rays):
        ray_arrs = find_arrivals_at_receiver(
            r_rcv, z_rcv,
            beam_data['ray_r'][i], beam_data['ray_z'][i],
            beam_data['ray_c'][i], beam_data['ray_tau'][i],
            beam_data['ray_p'][i], beam_data['ray_q'][i],
            beam_data['ray_amp'][i],
            int(beam_data['ray_npts'][i]),
            config.frequency,
            0.0, config.environment.bottom.depth
        )
        for tau, amp, angle in ray_arrs:
            all_arrivals.append({
                'tau': tau,
                'amplitude': amp,
                'arrival_angle': angle,
                'launch_angle': np.degrees(beam_data['ray_alphas'][i])
            })
            
    # Sort by time
    all_arrivals.sort(key=lambda x: x['tau'])
    return all_arrivals
