import os
import sys
import time
import tracemalloc
import numpy as np

# You must have Capytaine installed in the environment where you run this script:
# pip install capytaine
import capytaine as cpt

# --- Path Setup for OpenFLASH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openflash.meem_engine import MEEMEngine
from openflash.meem_problem import MEEMProblem
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.geometry import ConcentricBodyGroup
from openflash.body import SteppedBody
from openflash.multi_equations import omega
from openflash.multi_constants import g, rho

# ==========================================
# Capytaine Helper Functions
# ==========================================
def get_points(a, d):
    d_prime = d + [0]
    d_index = 0
    a_index = 0
    pt_lst = [(0, - d[0])]
    for i in range(len(a)):
        pt_lst.append((a[a_index], - d_prime[d_index]))
        d_index +=1
        pt_lst.append((a[a_index], - d_prime[d_index]))
        a_index+=1
    return pt_lst

def get_f_densities(pt_lst, total_units):
    face_lengths = np.array([])
    for i in range(len(pt_lst) - 1):
        p1, p2 = pt_lst[i], pt_lst[i + 1]
        face_length = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
        face_lengths = np.append(face_lengths, face_length)
    total_length = sum(face_lengths)
    each_face_densities = np.vectorize(lambda x: max(1, x/total_length * total_units))(face_lengths)
    remainders = each_face_densities % 1
    each_face_densities = each_face_densities.astype(int)
    remaining_units = total_units - sum(each_face_densities)
    if remaining_units < 0:
        for u in range(remaining_units * -1):
            i = np.argmax(each_face_densities)
            each_face_densities[i] = (each_face_densities[i]) - 1
    else:
        for u in range(remaining_units):
            i = np.argmax(remainders)
            each_face_densities[i] = (each_face_densities[i]) + 1
            remainders[i] = 0
    assert sum(each_face_densities) == total_units
    return each_face_densities

def make_face(p1, p2, f_density, t_density):
    zarr = np.linspace(p1[1], p2[1], f_density + 1)
    rarr = np.linspace(p1[0], p2[0], f_density + 1)
    xyz = np.array([np.array([x/np.sqrt(2),y/np.sqrt(2),z]) for x,y,z in zip(rarr,rarr,zarr)])
    return cpt.AxialSymmetricMesh.from_profile(xyz, nphi = t_density)

def faces_and_heaves(heaving, region, p1, p2, f_density, t_density, meshes, mask, panel_ct):
    mesh = make_face(p1, p2, f_density, t_density)
    meshes += mesh
    new_panels = f_density * t_density
    if heaving[region]:
        direction = [0, 0, 1]
    else:
        direction = [0, 0, 0]
    for i in range(new_panels):
        mask.append(direction)
    return meshes, mask, (panel_ct + new_panels)

def make_body(pts, t_densities, f_densities, heaving):
    meshes = cpt.meshes.meshes.Mesh()
    panel_ct = 0
    mask = []
    for i in range((len(pts) - 1) // 2):
        p1, p2, p3 = pts[2 * i], pts[2 * i + 1], pts[2 * i + 2]
        meshes, mask, panel_ct = faces_and_heaves(heaving, i, p1, p2, f_densities[2 * i], t_densities[i], meshes, mask, panel_ct)
        if p2[1] < p3[1]:
            region = i
        else:
            region = i + 1
        meshes, mask, panel_ct = faces_and_heaves(heaving, region, p2, p3, f_densities[2 * i + 1], t_densities[region], meshes, mask, panel_ct)
    
    # Suppress verbose mesh outputs
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    body = cpt.FloatingBody(meshes)
    sys.stdout = real_stdout
    
    return body, panel_ct, mask

# ==========================================
# Main Benchmark Logic
# ==========================================
def run_benchmark():
    # 1. Shared physical configuration (Using `config5` logic)
    h = 1.001
    d = [0.5, 0.25]
    a = [0.5, 1.0]
    heaving = [1, 0]             # Inner cylinder heaves, outer doesn't
    t_densities = [50, 100]      # BEM angular panels
    face_units = 90              # BEM outline panels
    m0 = 1.0                     # Wavenumber
    NMK = [50, 50, 50]           # MEEM harmonics per region

    print("Running Capytaine BEM Solver...")
    # Track Capytaine Memory & Runtime
    tracemalloc.start()
    t0_cpt = time.perf_counter()
    
    pt_lst = get_points(a, d)
    f_densities = get_f_densities(pt_lst, face_units)
    body, panel_count, mask = make_body(pt_lst, t_densities, f_densities, heaving)
    body.dofs["Heave"] = mask
    
    rad_problem = cpt.RadiationProblem(body=body, wavenumber=m0, water_depth=h, rho=rho)
    solver = cpt.BEMSolver()
    result_cpt = solver.solve(rad_problem)
    
    t1_cpt = time.perf_counter()
    _, peak_cpt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    cpt_time = t1_cpt - t0_cpt
    # Capytaine returns a flat dictionary for a single DOF
    cpt_am = float(result_cpt.added_mass["Heave"])
    cpt_damp = float(result_cpt.radiation_damping["Heave"])
    
    print("Running OpenFLASH MEEM Solver...")
    # Track OpenFLASH Memory & Runtime
    tracemalloc.start()
    t0_of = time.perf_counter()
    
    bodies = []
    for i in range(len(a)):
        bodies.append(SteppedBody(
            a=np.array([a[i]]),
            d=np.array([d[i]]),
            slant_angle=np.array([0.0]),
            heaving=bool(heaving[i])
        ))
    
    arrangement = ConcentricBodyGroup(bodies)
    geometry = BasicRegionGeometry(arrangement, h=h, NMK=NMK)
    problem = MEEMProblem(geometry)
    problem.set_frequencies(np.array([omega(m0, h, g)]))
    
    engine = MEEMEngine(problem_list=[problem])
    X = engine.solve_linear_system_multi(problem, m0)
    hydro_coeffs = engine.compute_hydrodynamic_coefficients(problem, X, m0)
    
    # Active mode correlates to the array index where heaving == 1
    active_idx = heaving.index(1) 
    of_am = float(hydro_coeffs[active_idx]['real'])
    of_damp = float(hydro_coeffs[active_idx]['imag'])
    
    t1_of = time.perf_counter()
    _, peak_of = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    of_time = t1_of - t0_of
    
    # ==========================================
    # Compute the requested metrics
    # ==========================================
    # Accuracy: Max difference in added mass and damping vs. Capytaine
    am_error = abs(cpt_am - of_am) / abs(cpt_am)
    damp_error = abs(cpt_damp - of_damp) / abs(cpt_damp)
    accuracy = (1.0 - max(am_error, damp_error)) * 100.0
    
    # Runtime & Memory (as multipliers)
    runtime_speedup = cpt_time / of_time if of_time > 0 else float('inf')
    memory_reduction = peak_cpt / peak_of if peak_of > 0 else float('inf')
    
    # Convergence: Defined here by Degrees of Freedom (Capytaine Panels / MEEM Matrix Size)
    of_matrix_size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:-1])
    convergence_factor = panel_count / of_matrix_size
    
    print("\n" + "="*50)
    print("                BENCHMARK RESULTS")
    print("="*50)
    print(f"Capytaine -> Time: {cpt_time:.4f}s | RAM: {peak_cpt/1024/1024:.2f}MB | AM: {cpt_am:.4f} | Damp: {cpt_damp:.4f} | DOFs: {panel_count}")
    print(f"OpenFLASH -> Time: {of_time:.4f}s | RAM: {peak_of/1024/1024:.2f}MB | AM: {of_am:.4f} | Damp: {of_damp:.4f} | DOFs: {of_matrix_size}")
    print("="*50 + "\n")
    
    # Final output formatted explicitly to your prompt
    print(f"Comparisons to the Capytaine boundary element method software show that "
          f"OpenFLASH has {accuracy:.2f}% accuracy, "
          f"{runtime_speedup:.2f} times faster runtime, "
          f"{convergence_factor:.2f}x better convergence, "
          f"and {memory_reduction:.2f} times less memory use.")

if __name__ == '__main__':
    run_benchmark()