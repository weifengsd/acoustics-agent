import numpy as np
import struct
import os

def generate_at_field_flp(config, flp_file_path):
    out = "'/'\n'RA'\n9999\n1\n"
    rr_vec = np.array(config.geometry.receivers.ranges) / 1000.0
    out += f"{rr_vec[0]} {rr_vec[-1]} /\n"
    if len(rr_vec) == 2: out += "1001\n"
    else: out += f"{len(rr_vec)}\n"
    out += f"{rr_vec[0]} {rr_vec[-1]} /\n"
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    rd_vec = np.array(config.geometry.receivers.depths)
    out += "1001\n"
    out += f"{rd_vec[0]} {rd_vec[-1]} /\n"
    out += "1001\n0.0 /\n"
    with open(flp_file_path, 'w') as f: f.write(out)

def generate_at_kraken_env(config, env_file_path):
    out = f"'{config.project}'\n{config.frequency}\n1\n'NVW'\n"
    z_bottom = config.environment.bottom.depth or config.environment.ssp.data[-1].depth
    out += f"0 0.0 {z_bottom}\n"
    for p in config.environment.ssp.data:
        rho = p.density if hasattr(p, 'density') and p.density else 1.0
        out += f"  {p.depth} {p.c} 0.0 {rho} 0.0 0.0 /\n"
    bot_bc = config.environment.bottom.type
    bot_opt = 'V' if bot_bc == "vacuum" else ('R' if bot_bc == "rigid" else 'A')
    out += f"'{bot_opt}' 0.0\n"
    bot_c = config.environment.bottom.c_p if hasattr(config.environment.bottom, 'c_p') else 1600.0
    bot_rho = config.environment.bottom.density if hasattr(config.environment.bottom, 'density') else 1.8
    bot_alpha = config.environment.bottom.attenuation_p if hasattr(config.environment.bottom, 'attenuation_p') else 0.0
    out += f"{z_bottom} {bot_c} 0.0 {bot_rho} {bot_alpha} /\n"
    c_low, c_high = 1450.0, 1650.0
    if hasattr(config.solver, 'phase_speed_limits'):
        c_low, c_high = config.solver.phase_speed_limits
        if c_low <= 0: c_low = 1450.0
        if c_high <= 0: c_high = 1800.0
    out += f"{c_low} {c_high}\n0.0\n"
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    rd_vec = config.geometry.receivers.depths
    out += "1001\n"
    out += f"{rd_vec[0]} {rd_vec[-1]} /\n"
    with open(env_file_path, 'w') as f: f.write(out)

def generate_at_bellhop_env(config, env_file_path):
    out = f"'{config.project}'\n{config.frequency}\n1\n"
    opt1 = 'S' if config.environment.ssp.type == 'spline' else 'C'
    out += f"'{opt1}VW'\n"
    depths = [d.depth for d in config.environment.ssp.data]
    cs = [d.c for d in config.environment.ssp.data]
    z_bottom = config.environment.bottom.depth
    out += f"{len(depths)} 0.0 {z_bottom}\n"
    for d, c in zip(depths, cs): out += f"  {d} {c} /\n"
    out += "'A' 0.0\n"
    bot_c = config.environment.bottom.c_p if hasattr(config.environment.bottom, 'c_p') else 1600.0
    bot_rho = config.environment.bottom.density if hasattr(config.environment.bottom, 'density') else 1.8
    bot_alpha = config.environment.bottom.attenuation_p if hasattr(config.environment.bottom, 'attenuation_p') else 0.5
    out += f"{z_bottom} {bot_c} 0.0 {bot_rho} {bot_alpha} /\n"
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    rd_vec = config.geometry.receivers.depths
    out += f"{len(rd_vec)}\n"
    out += " ".join(map(str, rd_vec)) + " /\n"
    rr_vec = np.array(config.geometry.receivers.ranges) / 1000.0
    out += f"{len(rr_vec)}\n"
    out += " ".join(map(str, rr_vec)) + " /\n"
    out += "'R'\n"
    num_beams = config.solver.num_beams if config.solver.num_beams > 0 else 400
    out += f"{num_beams}\n"
    angles = config.solver.angles
    out += f"{angles[0]} {angles[1]} /\n"
    step = config.solver.step_size if config.solver.step_size > 0 else 0.0
    out += f"{step} {z_bottom*1.2} {max(rr_vec)*1.1}\n"
    with open(env_file_path, 'w') as f: f.write(out)

def read_at_shd(filename):
    with open(filename, 'rb') as f:
        l_raw = f.read(4)
        if not l_raw: return None, None, None
        rec_size = struct.unpack('i', l_raw)[0] * 4
        def read_rec(n):
            f.seek((n-1)*rec_size); return f.read(rec_size)
        dims = struct.unpack('7i', read_rec(3)[:28])
        nrz, nrr = dims[5], dims[6]
        rz = np.frombuffer(read_rec(9)[:4*nrz], dtype=np.float32)
        rr = np.frombuffer(read_rec(10)[:8*nrr], dtype=np.float64)
        dtype = np.complex128 if rec_size >= 16*nrr else np.complex64
        item = 16 if rec_size >= 16*nrr else 8
        P = np.zeros((nrz, nrr), dtype=dtype)
        for i in range(nrz): P[i,:] = np.frombuffer(read_rec(11+i)[:item*nrr], dtype=dtype)
        return P, rz, rr

def read_at_ray(filename):
    rays = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not lines: return []
        idx = 2; idx += 1
        num_rays = int(float(lines[idx].split()[0])); idx += 1
        while idx < len(lines) and "'rz'" not in lines[idx] and "'xy'" not in lines[idx]: idx += 1
        idx += 1
        for _ in range(num_rays):
            if idx >= len(lines): break
            idx += 1; npts = int(float(lines[idx].split()[0])); idx += 1
            r_coords, z_coords = [], []
            for _ in range(npts):
                if idx >= len(lines): break
                parts = lines[idx].split()
                if len(parts) >= 2: r_coords.append(float(parts[0])); z_coords.append(float(parts[1]))
                idx += 1
            if r_coords: rays.append((np.array(r_coords), np.array(z_coords), np.ones(len(r_coords))))
    return rays

def read_at_mod(filename):
    """
    Reads a binary .mod file produced by Kraken.
    Follows the exact logic of Acoustics Toolbox read_modes_bin.m
    """
    import os
    import struct
    import numpy as np

    if not os.path.exists(filename) or os.path.getsize(filename) < 100:
        return np.array([]), np.array([])
        
    with open(filename, 'rb') as f:
        # First 4 bytes is LRecl in WORDS
        lrecl_raw = f.read(4)
        if len(lrecl_raw) < 4: return np.array([]), np.array([])
        LRecl = struct.unpack('<i', lrecl_raw)[0]
        rec_size = LRecl * 4
        
        # Record 0 (Header)
        f.seek(4) # Skip the 4-byte LRecl
        title = f.read(80).decode('ascii', errors='ignore').strip()
        nfreq, nmedia, ntot, nmat = struct.unpack('<4i', f.read(16))
        
        if ntot <= 0 or ntot > 100000 or nmat <= 0:
            return np.array([]), np.array([])
            
        # Record 4 (0-based 4, which is the 5th record) is z
        f.seek(4 * rec_size)
        z_bins = np.frombuffer(f.read(ntot * 4), dtype='<f4')
        
        # Record 5 is M (number of modes)
        f.seek(5 * rec_size)
        M = struct.unpack('<i', f.read(4))[0]
        
        if M <= 0 or M > 100000:
            return np.array([]), z_bins
            
        # Modes start at Record 7
        modes = []
        for i in range(1, M + 1):
            f.seek((6 + i) * rec_size)
            phi_raw = f.read(nmat * 8)
            if len(phi_raw) < nmat * 8:
                break
            phi_floats = np.frombuffer(phi_raw, dtype='<f4')
            phi_complex = phi_floats[0::2] + 1j * phi_floats[1::2]
            modes.append(phi_complex)
            
        return np.array(modes), z_bins
