import numpy as np
import struct
from pathlib import Path

def generate_at_bellhop_env(config, env_file_path):
    """
    Converts a SimulationConfig object into a legacy Bellhop .env file.
    """
    out = f"'{config.project}'\n"
    out += f"{config.frequency}\n"
    out += "1\n" # Number of frequencies
    
    # SSP Options
    ssp_type = config.environment.ssp.type
    # 'S' for spline, 'C' for C-linear
    opt1 = 'S' if ssp_type == 'spline' else 'C'
    # 'V' for Vacuum, 'W' for Water (standard)
    out += f"'{opt1}VW'\n"
    
    depths = [d.depth for d in config.environment.ssp.data]
    cs = [d.c for d in config.environment.ssp.data]
    n_depths = len(depths)
    z_bottom = config.environment.bottom.depth
    out += f"{n_depths} 0.0 {z_bottom}\n"
    for d, c in zip(depths, cs):
        out += f"  {d} {c} /\n"
        
    # Bottom Options: 'A' for Acousto-elastic / Halfspace
    out += "'A' 0.0\n"
    bot_c = config.environment.bottom.c_p if hasattr(config.environment.bottom, 'c_p') and config.environment.bottom.c_p else 1600.0
    bot_rho = config.environment.bottom.density if hasattr(config.environment.bottom, 'density') and config.environment.bottom.density else 1.8
    bot_alpha = config.environment.bottom.attenuation_p if hasattr(config.environment.bottom, 'attenuation_p') and config.environment.bottom.attenuation_p else 0.5
    out += f"{z_bottom} {bot_c} 0.0 {bot_rho} {bot_alpha} /\n"
    
    # Geometry
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    
    # Receiver depths
    rd_vec = config.geometry.receivers.depths
    out += f"{len(rd_vec)}\n"
    out += " ".join(map(str, rd_vec)) + " /\n"
    
    # Receiver ranges
    rr_vec = np.array(config.geometry.receivers.ranges) / 1000.0 # Convert to km for Bellhop
    out += f"{len(rr_vec)}\n"
    out += " ".join(map(str, rr_vec)) + " /\n"
    
    # Beam Options
    # 'C' for Coherent, 'G' for Gaussian Beam
    # 'R' for Rays
    # We choose 'R' if we want ray paths, 'C' if we want TL
    # But for a general run, we often run 'R' first then 'C'
    # Here we default to 'R' to get ray paths
    out += "'R'\n"
    
    num_beams = config.solver.num_beams if config.solver.num_beams > 0 else 400
    out += f"{num_beams}\n"
    angles = config.solver.angles
    out += f"{angles[0]} {angles[1]} /\n"
    
    step = config.solver.step_size if config.solver.step_size > 0 else 0.0
    zbox = z_bottom * 1.2
    rbox = max(rr_vec) * 1.1
    out += f"{step} {zbox} {rbox}\n"
    
    with open(env_file_path, 'w') as f:
        f.write(out)

def read_at_shd(filename):
    """Reads a binary .shd file produced by Bellhop/Kraken."""
    with open(filename, 'rb') as f:
        lrecl_data = f.read(4)
        if not lrecl_data: return None, None, None
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

def read_at_ray(filename):
    """Reads a text-based .ray file produced by Bellhop."""
    rays = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not lines: return []
        
        idx = 0
        # Title, Freq
        idx += 2 
        # Ns, Ntheta, Nphi
        idx += 1
        # Nr, ...
        num_rays = int(float(lines[idx].split()[0]))
        idx += 1
        
        # Skip until we find 'rz' or 'xy' marker
        while idx < len(lines) and "'rz'" not in lines[idx] and "'xy'" not in lines[idx]:
            idx += 1
        idx += 1 # Move past 'rz'
        
        for _ in range(num_rays):
            if idx >= len(lines): break
            # Launch angle
            idx += 1 
            # Npts, ...
            npts = int(float(lines[idx].split()[0]))
            idx += 1
            
            r_coords = []
            z_coords = []
            for _ in range(npts):
                if idx >= len(lines): break
                parts = lines[idx].split()
                if len(parts) >= 2:
                    r_coords.append(float(parts[0]))
                    z_coords.append(float(parts[1]))
                idx += 1
            
            if r_coords:
                rays.append((np.array(r_coords), np.array(z_coords), np.ones(len(r_coords))))
            
    return rays


def read_at_arr(filename):
    """Reads a text-based .arr file produced by Bellhop."""
    arrivals = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not lines: return []
        
        idx = 0
        # '2D' or Title
        idx += 1
        # Freq
        idx += 1
        
        # Nsources
        num_src = int(float(lines[idx].split()[0]))
        idx += 1
        
        # Nrec_depth (Note: older versions might swap r and z, but standard is z then r)
        num_z = int(float(lines[idx].split()[0]))
        idx += 1
        
        # Nrec_range
        num_r = int(float(lines[idx].split()[0]))
        idx += 1
        
        # Max number of arrivals
        max_narr = int(float(lines[idx].split()[0]))
        idx += 1
        
        # We assume 1 source, 1 depth, 1 range for simplicity in the returned flat list,
        # or we just append all of them. Usually run_arrivals is for a specific (r, z).
        for _ in range(num_src):
            for _ in range(num_z):
                for _ in range(num_r):
                    if idx >= len(lines): break
                    while idx < len(lines) and not lines[idx].strip():
                        idx += 1
                    
                    if idx >= len(lines): break
                    num_arr = int(float(lines[idx].split()[0]))
                    idx += 1
                    
                    for _ in range(num_arr):
                        if idx >= len(lines): break
                        parts = lines[idx].split()
                        if len(parts) >= 8:
                            # Amp_mag, Phase(deg), Delay, [skip], src_angle, rcv_angle, top_ref, bot_ref
                            amp_mag = float(parts[0])
                            phase_deg = float(parts[1])
                            amp = amp_mag * np.exp(1j * np.radians(phase_deg))
                            tau = float(parts[2])
                            
                            arrivals.append({
                                'amplitude': amp,
                                'tau': tau,
                                'launch_angle': float(parts[4]),
                                'arrival_angle': float(parts[5]),

                                'top_reflections': int(float(parts[6])),
                                'bottom_reflections': int(float(parts[7]))
                            })
                        idx += 1
                        
    return arrivals


