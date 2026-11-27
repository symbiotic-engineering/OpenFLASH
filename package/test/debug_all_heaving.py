import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import capytaine as cpt

from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.multi_equations import omega
from test_capytaine_potential import ALL_CONFIGS, g 

# =====================================================================
# 1. DYNAMICALLY GENERATE THE PAIRS
# =====================================================================
# Build up similarity to `test_potential_generator.ipynb` & `test_capytaine_potential.py`
SUBSETS = [
    # 2-body, h=1.5 (Base Case)
    {"a": [0.5, 1.0], "d": [0.8, 0.2], "h": 1.5},
    # 2-body, h=10.0 (Building up h)
    {"a": [0.5, 1.0], "d": [0.8, 0.2], "h": 10.0},
    # 2-body, h=50.0 (Approaching Deep Water)
    {"a": [0.5, 1.0], "d": [0.8, 0.2], "h": 50.0},
    # 3-body, h=100.0 (Matches "tall - compound cylinder" from test_potential_generator)
    {"a": [3.0, 5.0, 10.0], "d": [29.0, 7.0, 4.0], "h": 100.0},
    # 4-body, h=100.0 (Increasing dimension to track error manifestation)
    {"a": [0.3, 0.5, 1.0, 1.2], "d": [1.1, 0.9, 0.7, 0.5], "h": 100.0}
]

for i, sub in enumerate(SUBSETS):
    a_arr = np.array(sub["a"])
    # Sort depths to create structured cases
    d_sorted_desc = np.sort(sub["d"])[::-1]
    d_sorted_asc = np.sort(sub["d"])
    h = sub["h"]
    
    n_bodies = len(a_arr)
    # Match NMK array to dimension (radii + exterior)
    nmk = [15] * (n_bodies + 1)
    heaving_map = [True] * n_bodies
    body_map = list(range(n_bodies))
    
    # Case 1: Descending depths
    ALL_CONFIGS[f"config_var{i+1}_case1"] = {
        "h": h, "a": a_arr, "d": d_sorted_desc,
        "heaving_map": heaving_map, "body_map": body_map, "m0": 1.0, "NMK": nmk, 
        "R_range": np.linspace(0.0, 2 * a_arr[-1], num=50), "Z_range": np.linspace(0, -h, num=50),
    }
    
    # Case 2: Ascending depths
    ALL_CONFIGS[f"config_var{i+1}_case2"] = {
        "h": h, "a": a_arr, "d": d_sorted_asc,
        "heaving_map": heaving_map, "body_map": body_map, "m0": 1.0, "NMK": nmk, 
        "R_range": np.linspace(0.0, 2 * a_arr[-1], num=50), "Z_range": np.linspace(0, -h, num=50),
    }

# =====================================================================
# 2. CAPYTAINE MESH GENERATION
# =====================================================================
def get_points(a, d):
    d_prime = list(d) + [0]
    pt_lst = [(0, - d[0])]
    d_idx, a_idx = 0, 0
    for i in range(len(a)):
        pt_lst.append((a[a_idx], - d_prime[d_idx]))
        d_idx +=1
        pt_lst.append((a[a_idx], - d_prime[d_idx]))
        a_idx +=1
    return pt_lst

def get_f_densities(pt_lst, total_units):
    face_lengths = np.array([])
    for i in range(len(pt_lst) - 1):
        p1, p2 = pt_lst[i], pt_lst[i + 1]
        face_lengths = np.append(face_lengths, abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]))
    total_length = sum(face_lengths)
    densities = np.vectorize(lambda x: max(1, x/total_length * total_units))(face_lengths)
    remainders = densities % 1
    densities = densities.astype(int)
    rem_units = total_units - sum(densities)
    if rem_units < 0:
        for u in range(int(-rem_units)): densities[np.argmax(densities)] -= 1
    else:
        for u in range(int(rem_units)):
            idx = np.argmax(remainders)
            densities[idx] += 1
            remainders[idx] = 0
    return densities

def make_face(p1, p2, f_density, t_density):
    zarr, rarr = np.linspace(p1[1], p2[1], f_density + 1), np.linspace(p1[0], p2[0], f_density + 1)
    xyz = np.array([np.array([x/np.sqrt(2), y/np.sqrt(2), z]) for x,y,z in zip(rarr,rarr,zarr)])
    return cpt.AxialSymmetricMesh.from_profile(xyz, nphi=t_density)

def faces_and_heaves(heaving, region, p1, p2, f_density, t_density, meshes, mask, panel_ct):
    mesh = make_face(p1, p2, f_density, t_density)
    meshes += mesh
    new_panels = f_density * t_density
    direction = [0, 0, 1] if heaving[region] else [0, 0, 0]
    mask.extend([direction] * new_panels)
    return meshes, mask, (panel_ct + new_panels)

def build_capytaine_body(a, d, heaving, t_densities, face_units):
    pts = get_points(a, d)
    f_densities = get_f_densities(pts, face_units)
    meshes = cpt.meshes.meshes.Mesh()
    mask, panel_ct = [], 0
    
    for i in range((len(pts) - 1) // 2):
        p1, p2, p3 = pts[2 * i], pts[2 * i + 1], pts[2 * i + 2]
        meshes, mask, panel_ct = faces_and_heaves(heaving, i, p1, p2, f_densities[2 * i], t_densities[i], meshes, mask, panel_ct)
        region = i if p2[1] < p3[1] else i + 1
        meshes, mask, panel_ct = faces_and_heaves(heaving, region, p2, p3, f_densities[2 * i + 1], t_densities[region], meshes, mask, panel_ct)
    
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    body = cpt.FloatingBody(meshes)
    body.dofs["Heave"] = mask
    sys.stdout = real_stdout
    
    return body

# =====================================================================
# 3. ANALYSIS ENGINES
# =====================================================================
def analyze_openflash(config_name):
    p = ALL_CONFIGS[config_name]
    m0, freq = p["m0"], omega(p["m0"], p["h"], g)
    n_bodies = len(p["a"])
    
    # Base matrix A
    active_map_base = [True] + [False] * (n_bodies - 1)
    geom_base = BasicRegionGeometry.from_vectors(
        a=p["a"], d=p["d"], h=p["h"], NMK=p["NMK"], 
        body_map=p["body_map"], heaving_map=active_map_base 
    )
    prob_base = MEEMProblem(geom_base)
    prob_base.set_frequencies(np.array([freq]))
    engine = MEEMEngine(problem_list=[prob_base])
    
    cache = engine.build_problem_cache(prob_base)
    A = engine.assemble_A_multi(prob_base, m0)
    A_inv = np.linalg.inv(A)
    
    phi_total = None
    b_total = np.zeros(A.shape[0], dtype=complex)
    
    # Track the hydro coefficient block matrix dynamically
    AM_matrix = np.zeros((n_bodies, n_bodies))
    B_matrix = np.zeros((n_bodies, n_bodies))
    
    for i in range(n_bodies):
        active_map = [False] * n_bodies
        active_map[i] = True
        
        geom_i = BasicRegionGeometry.from_vectors(
            a=p["a"], d=p["d"], h=p["h"], NMK=p["NMK"], 
            body_map=p["body_map"], heaving_map=active_map
        )
        prob_i = MEEMProblem(geom_i)
        prob_i.set_frequencies(np.array([freq]))
        engine_i = MEEMEngine(problem_list=[prob_i])
        
        engine_i.update_forcing(prob_i)
        b_i = engine_i.assemble_b_multi(prob_i, m0)
        x_i = np.linalg.solve(A, b_i)
        
        b_total += b_i
        
        pot_dict = engine_i.calculate_potentials(
            prob_i, x_i, m0, spatial_res=50, sharp=False, 
            R_range=p["R_range"], Z_range=p["Z_range"]
        )
        
        if phi_total is None:
            phi_total = pot_dict['phi']
            R, Z = pot_dict['R'], pot_dict['Z']
        else:
            phi_total += pot_dict['phi']
            
        # Get hydro coeffs dynamically 
        coeffs = engine_i.compute_hydrodynamic_coefficients(prob_i, x_i, m0, modes_to_calculate=list(range(n_bodies)), rho=1023)
        for coeff in coeffs:
            j = coeff['mode']
            AM_matrix[j, i] = coeff['real']
            B_matrix[j, i] = coeff['imag']
            
    total_AM = np.sum(AM_matrix)
    total_B = np.sum(B_matrix)
        
    return phi_total, R, Z, A, A_inv, b_total, freq, total_AM, total_B

def analyze_capytaine(config_name, R_grid, Z_grid, omega_val):
    p = ALL_CONFIGS[config_name]
    a, d, h, m0 = p["a"], p["d"], p["h"], p["m0"]
    n_bodies = len(a)
    
    body = build_capytaine_body(a, d, [True]*n_bodies, t_densities=[30]*n_bodies, face_units=60)
    
    solver = cpt.BEMSolver()
    rad_problem = cpt.RadiationProblem(body=body, wavenumber=m0, water_depth=h, rho=1023)
    result = solver.solve(rad_problem)
    
    # --- BULLETPROOF EXTRACTION ---
    # Handle both flat dicts {'Heave': 1.0} and nested dicts {'Heave': {'Heave': 1.0}}
    am_val = result.added_mass["Heave"]
    cap_am = float(am_val["Heave"] if isinstance(am_val, dict) else am_val)
    
    b_val = result.radiation_damping["Heave"]
    cap_b = float(b_val["Heave"] if isinstance(b_val, dict) else b_val)
    # ------------------------------
    
    points = np.column_stack((R_grid.ravel(), np.zeros_like(R_grid.ravel()), Z_grid.ravel()))
    phi_cap_raw = solver.compute_potential(points, result)
    
    phi_cap_conv = (phi_cap_raw.imag * (-1.0/omega_val)) + 1j * (phi_cap_raw.real * (1.0/omega_val))
    phi_cap_grid = phi_cap_conv.reshape(R_grid.shape)
    
    regions = [(R_grid <= a[0]) & (Z_grid > -d[0])]
    for i in range(1, len(a)):
        regions.append((R_grid > a[i-1]) & (R_grid <= a[i]) & (Z_grid > -d[i]))
    for reg in regions:
        phi_cap_grid[reg] = np.nan
        
    return phi_cap_grid, cap_am, cap_b

def format_indices(indices):
    if len(indices) == 0: return "None"
    ranges, start, end = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == end + 1: end = i
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = i
    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    return "[" + ", ".join(ranges) + "]"

# =====================================================================
# 4. BATCH RUNNER AND PLOTTER
# =====================================================================
def run_all_pairs():
    for i in range(len(SUBSETS)):
        c1_name = f"config_var{i+1}_case1"
        c2_name = f"config_var{i+1}_case2"
        h_val = SUBSETS[i]['h']
        dim = len(SUBSETS[i]['a'])
        
        print(f"\n{'='*60}\nAnalyzing Set {i+1} | Dim: {dim} | h={h_val}\n{'='*60}")

        # 1. Run OpenFLASH
        phi_of1, R, Z, A1, A1_inv, b1, w1, of_am1, of_b1 = analyze_openflash(c1_name)
        phi_of2, _, _, A2, A2_inv, b2, w2, of_am2, of_b2 = analyze_openflash(c2_name)
        
        # 2. Run Capytaine
        print("  > Solving Capytaine BEM (Case 1)...")
        phi_cap1, cap_am1, cap_b1 = analyze_capytaine(c1_name, R, Z, w1)
        print("  > Solving Capytaine BEM (Case 2)...")
        phi_cap2, cap_am2, cap_b2 = analyze_capytaine(c2_name, R, Z, w2)

        # 3. Print Hydro Coefficients Table
        print(f"\n  --- Hydrodynamic Coefficients ---")
        print(f"  [Case 1] AM | OF: {of_am1:>10.2f} | Cap: {cap_am1:>10.2f} | Err: {abs(of_am1-cap_am1):.2e}")
        print(f"  [Case 1] B  | OF: {of_b1:>10.2f} | Cap: {cap_b1:>10.2f} | Err: {abs(of_b1-cap_b1):.2e}")
        print(f"  [Case 2] AM | OF: {of_am2:>10.2f} | Cap: {cap_am2:>10.2f} | Err: {abs(of_am2-cap_am2):.2e}")
        print(f"  [Case 2] B  | OF: {of_b2:>10.2f} | Cap: {cap_b2:>10.2f} | Err: {abs(of_b2-cap_b2):.2e}")
        print(f"  [Shift]  AM | OF Δ: {of_am1-of_am2:>9.2e} | Cap Δ: {cap_am1-cap_am2:>9.2e}")
        print(f"  [Shift]  B  | OF Δ: {of_b1-of_b2:>9.2e} | Cap Δ: {cap_b1-cap_b2:>9.2e}\n")

        # 4. Calculate Errors
        err1 = np.abs(phi_of1 - phi_cap1)
        err2 = np.abs(phi_of2 - phi_cap2)
        str_zeros_1 = format_indices(np.where(np.abs(b1) < 1e-10)[0].tolist())
        str_zeros_2 = format_indices(np.where(np.abs(b2) < 1e-10)[0].tolist())

        # =====================================================================
        # 5. MASTER 6x3 PLOT
        # =====================================================================
        fig, axes = plt.subplots(6, 3, figsize=(18, 33))
        fig.suptitle(f"Set {i+1} Comprehensive Diagnostic\nDim={dim} | a={SUBSETS[i]['a']} | h={h_val}\nCase 1: d descending | Case 2: d ascending", fontsize=18)
        
        vmax_of = max(np.nanmax(np.abs(phi_of1)), np.nanmax(np.abs(phi_of2)))
        vmax_cap = max(np.nanmax(np.abs(phi_cap1)), np.nanmax(np.abs(phi_cap2)))
        vmax_err = max(np.nanmax(err1), np.nanmax(err2))

        # --- ROW 1: OpenFLASH ---
        im = axes[0, 0].pcolormesh(R, Z, np.abs(phi_of1), cmap='viridis', vmin=0, vmax=vmax_of, shading='auto')
        fig.colorbar(im, ax=axes[0, 0]); axes[0, 0].set_title(f"OpenFLASH |phi| (Case 1)\nAM: {of_am1:.2f} | B: {of_b1:.2f}")
        
        im = axes[0, 1].pcolormesh(R, Z, np.abs(phi_of2), cmap='viridis', vmin=0, vmax=vmax_of, shading='auto')
        fig.colorbar(im, ax=axes[0, 1]); axes[0, 1].set_title(f"OpenFLASH |phi| (Case 2)\nAM: {of_am2:.2f} | B: {of_b2:.2f}")
        
        im = axes[0, 2].pcolormesh(R, Z, np.abs(np.abs(phi_of1) - np.abs(phi_of2)), cmap='coolwarm', shading='auto')
        fig.colorbar(im, ax=axes[0, 2]); axes[0, 2].set_title("OF Change |Case 1 - Case 2|")

        # --- ROW 2: Capytaine ---
        im = axes[1, 0].pcolormesh(R, Z, np.abs(phi_cap1), cmap='viridis', vmin=0, vmax=vmax_cap, shading='auto')
        fig.colorbar(im, ax=axes[1, 0]); axes[1, 0].set_title(f"Capytaine |phi| (Case 1)\nAM: {cap_am1:.2f} | B: {cap_b1:.2f}")
        
        im = axes[1, 1].pcolormesh(R, Z, np.abs(phi_cap2), cmap='viridis', vmin=0, vmax=vmax_cap, shading='auto')
        fig.colorbar(im, ax=axes[1, 1]); axes[1, 1].set_title(f"Capytaine |phi| (Case 2)\nAM: {cap_am2:.2f} | B: {cap_b2:.2f}")
        
        im = axes[1, 2].pcolormesh(R, Z, np.abs(np.abs(phi_cap1) - np.abs(phi_cap2)), cmap='coolwarm', shading='auto')
        fig.colorbar(im, ax=axes[1, 2]); axes[1, 2].set_title("Cap Change |Case 1 - Case 2|")

        # --- ROW 3: Error |OF - Cap| ---
        im = axes[2, 0].pcolormesh(R, Z, err1, cmap='Reds', vmin=0, vmax=vmax_err, shading='auto')
        fig.colorbar(im, ax=axes[2, 0]); axes[2, 0].set_title(f"Error |OF - Cap| (Case 1)\nAM Err: {abs(of_am1-cap_am1):.2e}")
        
        im = axes[2, 1].pcolormesh(R, Z, err2, cmap='Reds', vmin=0, vmax=vmax_err, shading='auto')
        fig.colorbar(im, ax=axes[2, 1]); axes[2, 1].set_title(f"Error |OF - Cap| (Case 2)\nAM Err: {abs(of_am2-cap_am2):.2e}")
        
        im = axes[2, 2].pcolormesh(R, Z, np.abs(err1 - err2), cmap='Reds', shading='auto')
        fig.colorbar(im, ax=axes[2, 2]); axes[2, 2].set_title("Error Shift |Err1 - Err2|")

        # --- ROW 4: A Matrix Sparsity ---
        axes[3, 0].spy(np.abs(A1) > 1e-10, markersize=2, color='blue')
        axes[3, 0].set_title("Matrix A (Case 1)")
        axes[3, 1].spy(np.abs(A2) > 1e-10, markersize=2, color='blue')
        axes[3, 1].set_title("Matrix A (Case 2)")
        diff_A = np.abs(A1 - A2)
        axes[3, 2].spy(diff_A > 1e-10, markersize=2, color='red')
        axes[3, 2].set_title(f"A Matrix Structural Change\nMax Err: {np.max(diff_A):.2e}")

        # --- ROW 5: A^(-1) Matrix Sparsity ---
        axes[4, 0].spy(np.abs(A1_inv) > 1e-10, markersize=2, color='green')
        axes[4, 0].set_title("Matrix A^(-1) (Case 1)")
        axes[4, 1].spy(np.abs(A2_inv) > 1e-10, markersize=2, color='green')
        axes[4, 1].set_title("Matrix A^(-1) (Case 2)")
        diff_A_inv = np.abs(A1_inv - A2_inv)
        axes[4, 2].spy(diff_A_inv > 1e-10, markersize=2, color='darkred')
        axes[4, 2].set_title(f"A^(-1) Matrix Structural Change\nMax Err: {np.max(diff_A_inv):.2e}")

        # --- ROW 6: b Vector ---
        axes[5, 0].plot(np.abs(b1), 'bo-', markersize=4)
        axes[5, 0].set_title(f"Vector b (Case 1)\nZeros: {str_zeros_1}")
        axes[5, 0].grid(True, linestyle='--', alpha=0.6)
        
        axes[5, 1].plot(np.abs(b2), 'bo-', markersize=4)
        axes[5, 1].set_title(f"Vector b (Case 2)\nZeros: {str_zeros_2}")
        axes[5, 1].grid(True, linestyle='--', alpha=0.6)
        
        diff_b = np.abs(b1 - b2)
        axes[5, 2].plot(diff_b, 'ro-', markersize=4)
        axes[5, 2].set_title(f"Change in b Vector\nMax Err: {np.max(diff_b):.2e}")
        axes[5, 2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.97]) 
        filename = f"master_benchmark_set{i+1}_dim{dim}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  > Saved benchmark plot to {filename}")

if __name__ == "__main__":
    run_all_pairs()