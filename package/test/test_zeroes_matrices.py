import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

ALL_CONFIGS = {
    "config0": {
        "h": 1.001, "a": np.array([0.5, 1]), "d": np.array([0.5, 0.25]),
        "heaving_map": [True, True], "body_map": [0, 1], "m0": 1.0,
        "NMK": [15]*3, "R_range": np.linspace(0.0, 2 * 1, num=50), "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config1": {
        "h": 1.5, "a": np.array([0.3, 0.5, 1, 1.2, 1.6]), "d": np.array([1.1, 0.85, 0.75, 0.4, 0.15]),
        "heaving_map": [True, True, True, True, True], "body_map": [0, 1, 2, 3, 4], "m0": 1.0,
        "NMK": [15] * 6, "R_range": np.linspace(0.0, 2 * 1.6, num=50), "Z_range": np.linspace(0, -1.5, num=50),
    },
    "config2": {
        "h": 100.0, "a": np.array([3, 5, 10]), "d": np.array([29, 7, 4]),
        "heaving_map": [True, True, True], "body_map": [0, 1, 2], "m0": 1.0,
        "NMK": [100] * 4, "R_range": np.linspace(0.0, 2 * 10, num=50), "Z_range": np.linspace(0, -100, num=50),
    },
    "config3": {
        "h": 1.9, "a": np.array([0.3, 0.5, 1, 1.2, 1.6]), "d": np.array([0.5, 0.7, 0.8, 0.2, 0.5]),
        "heaving_map": [True, True, True, True, True], "body_map": [0, 1, 2, 3, 4], "m0": 1.0,
        "NMK": [15] * 6, "R_range": np.linspace(0.0, 2 * 1.6, num=50), "Z_range": np.linspace(0, -1.9, num=50),
    },
    "config4": {
        "h": 1.001, "a": np.array([0.5, 1]), "d": np.array([0.5, 0.25]),
        "heaving_map": [False, True], "body_map": [0, 1], "m0": 1.0,
        "NMK": [15] * 3, "R_range": np.linspace(0.0, 2 * 1, num=50), "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config5": {
        "h": 1.001, "a": np.array([0.5, 1]), "d": np.array([0.5, 0.25]),
        "heaving_map": [True, False], "body_map": [0, 1], "m0": 1.0,
        "NMK": [15] * 3, "R_range": np.linspace(0.0, 2 * 1, num=50), "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config6": {
        "h": 100.0, "a": np.array([3, 5, 10]), "d": np.array([29, 7, 4]),
        "heaving_map": [False, True, True], "body_map": [0, 1, 2], "m0": 1.0,
        "NMK": [100] * 4, "R_range": np.linspace(0.0, 2 * 10, num=50), "Z_range": np.linspace(0, -100, num=50),
    },
    "config7": {
        "h": 1.001, "a": np.array([0.5, 1]), "d": np.array([0.25, 0.5]),
        "heaving_map": [True, False], "body_map": [0, 1], "m0": 1.0,
        "NMK": [10] * 3, "R_range": np.linspace(0.0, 2 * 1, num=50), "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config8": {
        "h": 1.001, "a": np.array([0.5, 1]), "d": np.array([0.25, 0.5]),
        "heaving_map": [False, True], "body_map": [0, 1], "m0": 1.0,
        "NMK": [10] * 3, "R_range": np.linspace(0.0, 2 * 1, num=50), "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config9": {
        "h": 100.0, "a": np.array([3, 5, 10]), "d": np.array([4, 7, 29]),
        "heaving_map": [True, True, True], "body_map": [0, 1, 2], "m0": 1.0,
        "NMK": [100] * 4, "R_range": np.linspace(0.0, 2 * 10, num=50), "Z_range": np.linspace(0, -100, num=50),
    },
    "config10": {
        "h": 1.5, "a": np.array([0.3, 0.5, 1, 1.2, 1.6]), "d": np.array([0.15, 0.4, 0.75, 0.85, 1.1]),
        "heaving_map": [True, True, True, True, True], "body_map": [0, 1, 2, 3, 4], "m0": 1.0,
        "NMK": [10]*6, "R_range": np.linspace(0.0, 2 * 1.6, num=50), "Z_range": np.linspace(0, -1.5, num=50),
    },
    "config11": {
        "h": 1.001, "a": np.array([0.5, 1]), "d": np.array([0.25, 0.5]),
        "heaving_map": [True, True], "body_map": [0, 1], "m0": 1.0,
        "NMK": [10] * 3, "R_range": np.linspace(0.0, 2 * 1, num=50), "Z_range": np.linspace(0, -1.001, num=50),
    },
    "config14": {
        "h": 1.9, "a": np.array([1.2, 1.6]), "d": np.array([0.2, 0.5]),
        "heaving_map": [True, True], "body_map": [0, 1], "m0": 1.0,
        "NMK": [15] * 3, "R_range": np.linspace(0.0, 2 * 1.6, num=50), "Z_range": np.linspace(0, -1.9, num=50),
    },
}

def analyze_schur(A_matrix, name):
    """
    Computes the condition number of the Schur complement of the matrix A.
    Assumes A is evenly split between potential and velocity matching.
    """
    half = A_matrix.shape[0] // 2
    P = A_matrix[:half, :half]
    Q = A_matrix[:half, half:]
    R = A_matrix[half:, :half]
    V = A_matrix[half:, half:]
    try:
        P_inv = linalg.inv(P)
        S = V - R @ P_inv @ Q
        cond_S = np.linalg.cond(S)
        print(f"  -> {name} Schur Complement (S) Condition Number: {cond_S:.2e}")
    except np.linalg.LinAlgError:
        print(f"  -> [WARNING] {name} Schur Complement could not be computed (P is singular).")

def analyze_single_config(config_name, p, output_dir):
    """
    Builds the problem, assembles A and b, identifies sparsity/zeros for A, b, and A^-1, 
    and saves side-by-side spy plots for visual analysis.
    """
    print(f"\n{'='*50}")
    print(f"=== Zeros & Sparsity Analysis: {config_name} ===")
    
    # 1. Safely isolate a single heaving body (engine only supports 1 at a time)
    test_heaving_map = [False] * len(p["heaving_map"])
    if any(p["heaving_map"]):
        test_heaving_map[p["heaving_map"].index(True)] = True
        
    geom = BasicRegionGeometry.from_vectors(
        a=p["a"], d=p["d"], h=p["h"], NMK=p["NMK"], 
        heaving_map=test_heaving_map, body_map=p["body_map"]
    )
    problem = MEEMProblem(geom)
    engine = MEEMEngine([problem])
    m0 = p["m0"]
    
    try:
        engine._ensure_m_k_and_N_k_arrays(problem, m0)
        A = engine.assemble_A_multi(problem, m0)
        b = engine.assemble_b_multi(problem, m0)
    except Exception as e:
        print(f"  [ERROR] Could not assemble matrices for {config_name}: {e}")
        return

    # 2. Analyze Vector b
    zeros_b = np.where(np.abs(b) < 1e-10)[0]
    print(f"Vector b Zeros: {len(zeros_b)} / {len(b)} elements are zero.")

    # 3. Analyze Matrix A
    zeros_A = np.where(np.abs(A) < 1e-10)
    num_zeros_A = len(zeros_A[0])
    total_A = A.size
    print(f"Matrix A Sparsity: {num_zeros_A}/{total_A} elements are zero ({num_zeros_A/total_A*100:.2f}%)")

    # Analyze Schur Complement for the base config
    analyze_schur(A, f"{config_name} Base Matrix")

    # 4. Analyze Matrix A^-1
    A_inv = None
    try:
        A_inv = linalg.inv(A)
        zeros_A_inv = np.where(np.abs(A_inv) < 1e-10)
        num_zeros_A_inv = len(zeros_A_inv[0])
        print(f"Matrix A^-1 Sparsity: {num_zeros_A_inv}/{total_A} elements are zero ({num_zeros_A_inv/total_A*100:.2f}%)")
    except np.linalg.LinAlgError:
        print("  [WARNING] Matrix A is ill-conditioned or singular. A^-1 could not be computed.")

    # 5. Spy Plots
    fig, axes = plt.subplots(1, 2 if A_inv is not None else 1, figsize=(14 if A_inv is not None else 7, 7))
    fig.suptitle(f"Sparsity Analysis: {config_name} (Tolerance: 1e-10)", fontsize=14)
    
    if A_inv is not None:
        axes[0].spy(np.abs(A) > 1e-10, markersize=1.5, color='blue')
        axes[0].set_title(f"Matrix A")
        axes[1].spy(np.abs(A_inv) > 1e-10, markersize=1.5, color='green')
        axes[1].set_title(f"Inverse Matrix A$^{{-1}}$")
    else:
        axes.spy(np.abs(A) > 1e-10, markersize=1.5, color='blue')
        axes.set_title(f"Matrix A")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"zeros_{config_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Sparsity plot saved to: {save_path}")

def analyze_geometry_flip(config_name, p, output_dir):
    """
    Forces an infinitesimal geometry flip on the first boundary of the config
    (Case 1: d0 > d1 vs. Case 2: d0 < d1) and outputs the difference matrices.
    """
    print(f"\n=== Geometry Flip Diagnostic: {config_name} ===")
    
    eps = 1e-5
    m0 = p["m0"]
    h = p["h"]
    
    # Safe heaving map
    test_heaving_map = [False] * len(p["heaving_map"])
    if any(p["heaving_map"]):
        test_heaving_map[p["heaving_map"].index(True)] = True

    # Find the depth of the second domain to use as the pivot for the flip
    if len(p["d"]) < 2:
        print("  [SKIP] Need at least 2 domains to perform a flip.")
        return
        
    target_d = p["d"][1] 
    
    # Case 1: Inner is deeper
    d_case1 = p["d"].copy()
    d_case1[0] = target_d + eps
    
    # Case 2: Inner is shallower
    d_case2 = p["d"].copy()
    d_case2[0] = target_d - eps

    def get_model(d_array):
        geom = BasicRegionGeometry.from_vectors(
            a=p["a"], d=d_array, h=h, NMK=p["NMK"], 
            heaving_map=test_heaving_map, body_map=p["body_map"]
        )
        problem = MEEMProblem(geom)
        engine = MEEMEngine([problem])
        engine._ensure_m_k_and_N_k_arrays(problem, m0)
        A = engine.assemble_A_multi(problem, m0)
        b = engine.assemble_b_multi(problem, m0)
        return A, b, engine, problem

    try:
        A1, b1, engine1, prob1 = get_model(d_case1)
        A2, b2, engine2, prob2 = get_model(d_case2)
    except Exception as e:
        print(f"  [ERROR] Flip setup failed: {e}")
        return

    # Calculate Schur Complements
    analyze_schur(A1, "Case 1 (d0 > d1)")
    analyze_schur(A2, "Case 2 (d0 < d1)")

    # --- SPY PLOTS (A1 vs A2 Difference) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Sparsity Pattern Flip: {config_name} (Tol: 1e-10)")
    axes[0].spy(np.abs(A1) > 1e-10, markersize=1)
    axes[0].set_title("Case 1 (d0 > d1)")
    axes[1].spy(np.abs(A2) > 1e-10, markersize=1)
    axes[1].set_title("Case 2 (d0 < d1)")
    delta_A = np.abs(A1 - A2)
    axes[2].spy(delta_A > 1e-5, markersize=1, color='red')
    axes[2].set_title("Difference (|A1 - A2| > 1e-5)")
    plt.tight_layout()
    spy_path = os.path.join(output_dir, f"flip_spy_{config_name}.png")
    plt.savefig(spy_path)
    plt.close()
    
    try:
        x1 = linalg.solve(A1, b1)
        x2 = linalg.solve(A2, b2)
        print(f"  -> ||x1 - x2||: {np.linalg.norm(x1 - x2):.6e}")
    except np.linalg.LinAlgError:
        print(f"  [WARNING] Could not solve linear system for flip.")
        return

    # --- SPATIAL RECONSTRUCTION ---
    pot1 = engine1.calculate_potentials(prob1, x1, m0, spatial_res=50, sharp=False)
    pot2 = engine2.calculate_potentials(prob2, x2, m0, spatial_res=50, sharp=False)

    # Change to a 3-row layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f"Flip Reconstruction Comparison: {config_name}", fontsize=16)

    # Calculate global mins and maxes for consistent colorbars
    vh_min = min(np.nanmin(pot1["phiH"].real), np.nanmin(pot2["phiH"].real))
    vh_max = max(np.nanmax(pot1["phiH"].real), np.nanmax(pot2["phiH"].real))
    vp_min = min(np.nanmin(pot1["phiP"].real), np.nanmin(pot2["phiP"].real))
    vp_max = max(np.nanmax(pot1["phiP"].real), np.nanmax(pot2["phiP"].real))
    
    # Calculate limits for the Total Potential
    vt_min = min(np.nanmin(pot1["phi"].real), np.nanmin(pot2["phi"].real))
    vt_max = max(np.nanmax(pot1["phi"].real), np.nanmax(pot2["phi"].real))

    # Row 1: Homogeneous
    axes[0, 0].pcolormesh(pot1["R"], pot1["Z"], pot1["phiH"].real, cmap='viridis', vmin=vh_min, vmax=vh_max)
    axes[0, 0].set_title(r"Case 1: $\phi_H$ (d0 > d1)")
    axes[0, 1].pcolormesh(pot2["R"], pot2["Z"], pot2["phiH"].real, cmap='viridis', vmin=vh_min, vmax=vh_max)
    axes[0, 1].set_title(r"Case 2: $\phi_H$ (d0 < d1)")

    # Row 2: Particular
    axes[1, 0].pcolormesh(pot1["R"], pot1["Z"], pot1["phiP"].real, cmap='plasma', vmin=vp_min, vmax=vp_max)
    axes[1, 0].set_title(r"Case 1: $\phi_P$ (d0 > d1)")
    axes[1, 1].pcolormesh(pot2["R"], pot2["Z"], pot2["phiP"].real, cmap='plasma', vmin=vp_min, vmax=vp_max)
    axes[1, 1].set_title(r"Case 2: $\phi_P$ (d0 < d1)")

    # Row 3: Total Potential (The Continuous Physics)
    axes[2, 0].pcolormesh(pot1["R"], pot1["Z"], pot1["phi"].real, cmap='inferno', vmin=vt_min, vmax=vt_max)
    axes[2, 0].set_title(r"Case 1: Total $\phi$ (Continuous)")
    axes[2, 1].pcolormesh(pot2["R"], pot2["Z"], pot2["phi"].real, cmap='inferno', vmin=vt_min, vmax=vt_max)
    axes[2, 1].set_title(r"Case 2: Total $\phi$ (Continuous)")

    plt.tight_layout()
    recon_path = os.path.join(output_dir, f"flip_recon_{config_name}.png")
    plt.savefig(recon_path)
    plt.close()
    print(f"  -> Flip plots saved to: {recon_path}")

if __name__ == "__main__":
    out_dir = "test_artifacts/matrices"
    os.makedirs(out_dir, exist_ok=True)
    
    # Run both diagnostics for every configuration
    for name, params in ALL_CONFIGS.items():
        analyze_single_config(name, params, out_dir)
        analyze_geometry_flip(name, params, out_dir)