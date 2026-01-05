# package/test/test_matrix_snapshot.py
import pytest
import numpy as np
import os
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

# Constants matching the snapshot generation
h = 1.001
a1 = .5
a2 = 1
d1 = .5
d2 = .25
m0 = 1.0
N, M, K = 4, 4, 4

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "data/gold_standard_A_2region.npy")

def test_matrix_assembly_regression():
    """
    Ensures the A matrix assembly remains bitwise identical to a known 'Gold Standard' 
    snapshot. This protects against accidental changes in loop logic or indexing.
    """
    if not os.path.exists(SNAPSHOT_PATH):
        pytest.fail("Gold standard snapshot not found. Run 'generate_snapshot.py' first.")
    
    # Load Gold Standard
    A_expected = np.load(SNAPSHOT_PATH)
    
    # Generate Current
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
    A_actual = engine.assemble_A_multi(prob, m0)
    
    # Assert
    # We use a very strict tolerance because this is a regression test on the same machine/logic
    np.testing.assert_allclose(
        A_actual, A_expected, 
        rtol=1e-12, atol=1e-12, 
        err_msg="Matrix assembly logic has changed! Result differs from snapshot."
    )