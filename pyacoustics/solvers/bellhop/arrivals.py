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
    z_min: float, z_max: float,
    d_alpha: float
):
    """
    Finds all crossings of a specific receiver location by a single ray.
    Returns (tau, amp, angle, dist_to_rcv)
    """
    arrivals = []
    
    dz_spacing = r_rcv * d_alpha
    window = max(dz_spacing * 1.5, 5.0) 
    
    for i in range(n_pts - 1):
        r1, z1 = ray_r[i], ray_z[i]
        r2, z2 = ray_r[i+1], ray_z[i+1]
        
        r_min_seg, r_max_seg = min(r1, r2), max(r1, r2)
        if r_rcv < r_min_seg or r_rcv > r_max_seg:
            continue
            
        dr = r2 - r1
        if abs(dr) < 1e-10:
            continue
            
        frac = (r_rcv - r1) / dr
        z_at_r = z1 + frac * (z2 - z1)
        
        dist = abs(z_at_r - z_rcv)
        if dist < window:
            tau = ray_tau[i] + frac * (ray_tau[i+1] - ray_tau[i])
            amp = ray_amp[i] + frac * (ray_amp[i+1] - ray_amp[i])
            dz = z2 - z1
            angle_rad = np.arctan2(dz, dr)
            
            arrivals.append((tau, amp, np.degrees(angle_rad), dist))
            
    return arrivals

def compute_arrivals(config: SimulationConfig, r_rcv: float, z_rcv: float):
    """
    Orchestrates arrival extraction for all rays with de-duplication.
    """
    from pyacoustics.solvers.bellhop.core import PyBellhop
    solver = PyBellhop(config)
    
    beam_data = solver.run_beam_tracing()
    
    raw_arrivals = []
    n_rays = beam_data['num_beams']
    alphas = beam_data['ray_alphas']
    if len(alphas) > 1:
        d_alpha = np.abs(np.mean(np.diff(alphas)))
    else:
        d_alpha = 0.01
    
    for i in range(n_rays):
        ray_arrs = find_arrivals_at_receiver(
            r_rcv, z_rcv,
            beam_data['ray_r'][i], beam_data['ray_z'][i],
            beam_data['ray_c'][i], beam_data['ray_tau'][i],
            beam_data['ray_p'][i], beam_data['ray_q'][i],
            beam_data['ray_amp'][i],
            int(beam_data['ray_npts'][i]),
            config.frequency,
            0.0, config.environment.bottom.depth,
            d_alpha
        )
        for tau, amp, angle, dist in ray_arrs:
            raw_arrivals.append({
                'tau': tau,
                'amplitude': amp,
                'arrival_angle': angle,
                'launch_angle': np.degrees(alphas[i]),
                'dist': dist
            })
            
    if not raw_arrivals:
        return []

    # Sort by time to facilitate de-duplication
    raw_arrivals.sort(key=lambda x: x['tau'])
    
    # De-duplication logic:
    # If multiple adjacent rays contribute to the "same" path (very close tau),
    # pick the one that passed closest to the receiver center (minimum dist).
    final_arrivals = []
    if raw_arrivals:
        current_cluster = [raw_arrivals[0]]
        
        # Tau threshold for "same arrival"
        # Since adjacent rays can differ by several milliseconds at long ranges,
        # we increase this to 10ms (0.01s) to properly cluster them.
        tau_eps = 0.01 
        
        for i in range(1, len(raw_arrivals)):
            # If travel time is very similar, they are likely the same physical path
            if (raw_arrivals[i]['tau'] - current_cluster[-1]['tau']) < tau_eps:
                current_cluster.append(raw_arrivals[i])
            else:
                # Process cluster: pick the ray that passes closest to the receiver
                best = min(current_cluster, key=lambda x: x['dist'])
                final_arrivals.append(best)
                current_cluster = [raw_arrivals[i]]
        
        # Process last cluster
        if current_cluster:
            best = min(current_cluster, key=lambda x: x['dist'])
            final_arrivals.append(best)

    # Sort final list by time
    final_arrivals.sort(key=lambda x: x['tau'])
    return final_arrivals



