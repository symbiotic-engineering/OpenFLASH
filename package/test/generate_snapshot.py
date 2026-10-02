# package/test/generate_snapshot.py
import numpy as np
import os
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

# Constants
h = 1.001
a1 = .5
a2 = 1
d1 = .5
d2 = .25
m0 = 1.0
N, M, K = 4, 4, 4

def generate():
    # 1. Run OpenFLASH
    geo = BasicRegionGeometry.from_vectors(
        a=np.array([a1, a2]),
        d=np.array([d1, d2]),
        h=h,
        NMK=[N, M, K],
        slant_angle=np.zeros(2)
    )
    
    prob = MEEMProblem(geo)
    engine = MEEMEngine([prob])
    
    engine._ensure_m_k_and_N_k_arrays(prob, m0)
    A = engine.assemble_A_multi(prob, m0)
    
    # 2. Ensure directory exists (THE FIX)
    # This gets the folder where this script is located (package/test)
    base_dir = os.path.dirname(__file__) 
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # 3. Save
    output_path = os.path.join(data_dir, "gold_standard_A_2region.npy")
    np.save(output_path, A)
    print(f"Snapshot saved to: {output_path}")

if __name__ == "__main__":
    generate()