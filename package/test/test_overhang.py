import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

def test_overhang_convergence():
    # config7 parameters (Mushroom shape)
    h = 1.001
    a = np.array([0.5, 1.0])
    d = np.array([0.25, 0.5]) # d0 < d1
    m0 = 1.0
    
    # Test different harmonic truncations
    NMK_list = [[10, 10, 10], [30, 30, 30], [60, 60, 60]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(r"Convergence Study: Overhang Geometry ($d_0 < d_1$) with Least Squares", fontsize=16)
    
    for idx, NMK in enumerate(NMK_list):
        geom = BasicRegionGeometry.from_vectors(
            a=a, d=d, h=h, NMK=NMK, 
            heaving_map=[True, False],
            body_map=[0, 1]  
        )
        problem = MEEMProblem(geom)
        engine = MEEMEngine([problem])
        
        # Manually assemble and solve
        engine._ensure_m_k_and_N_k_arrays(problem, m0)
        A = engine.assemble_A_multi(problem, m0)
        b = engine.assemble_b_multi(problem, m0)
        
        # --- NEW SOLVER ---
        # Using least-squares to handle the ill-conditioned high-harmonic matrix.
        # This will find the best-fit physical coefficients even if high-n rows underflow to zero.
        x = np.linalg.lstsq(A, b, rcond=1e-12)[0]
        
        # Reconstruct potentials
        pot = engine.calculate_potentials(problem, x, m0, spatial_res=150, sharp=False)
        
        # Plot Real Potential
        ax = axes[idx]
        im = ax.pcolormesh(pot["R"], pot["Z"], pot["phi"].real, cmap='viridis', shading='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Modes (NMK) = {NMK[0]}")
        
        # Draw geometry boundaries
        ax.axvline(0.5, color='white', linestyle='--', alpha=0.8)
        ax.plot([0, 0.5], [-0.25, -0.25], color='red', linewidth=2, label="Body 0")
        ax.plot([0.5, 1.0], [-0.5, -0.5], color='orange', linewidth=2, label="Body 1")
        ax.plot([0.5, 0.5], [-0.25, -0.5], color='orange', linewidth=2) # Overhang wall
        if idx == 0:
            ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_overhang_convergence()