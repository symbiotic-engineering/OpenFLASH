# test_matrix_caching.py
import pytest
import numpy as np
import time
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

# --- Configuration from your script ---
H = 100
D = [29, 7, 4]
A = [3, 5, 10]
# Heaving [0, 1, 1] means:
# Segment 0 is Stationary (Body 0)
# Segments 1 & 2 are Heaving together (Body 1)
HEAVING_SEGMENTS = [0, 1, 1] 
NMK_COUNT = 100
RHO = 1023
M0_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def setup_problem_objects():
    """
    Helper to set up the Geometry and Problem.
    """
    # Map segments to bodies to satisfy 'Single Rigid Body' constraint per body
    # [0, 1, 1] -> Body 0 is seg 0, Body 1 is segs 1,2
    body_map = [0, 1, 1]
    heaving_map = [False, True]
    
    # NMK length must be len(A) + 1
    NMK = [NMK_COUNT] * (len(A) + 1)

    geometry = BasicRegionGeometry.from_vectors(
        a=np.array(A),
        d=np.array(D),
        h=H,
        NMK=NMK,
        slant_angle=np.zeros_like(A),
        body_map=body_map,
        heaving_map=heaving_map
    )
    
    problem = MEEMProblem(geometry)
    return problem

def solve_fresh_every_time(m0_list):
    """
    Simulates the 'Old' method: Re-create the Engine (and thus matrices) 
    from scratch for every m0.
    """
    ams = []
    dps = []
    
    start_time = time.perf_counter()
    
    for m0 in m0_list:
        # 1. New Problem & Engine every iteration (Forces full rebuild)
        prob = setup_problem_objects()
        engine = MEEMEngine([prob])
        
        # 2. Solve
        X = engine.solve_linear_system_multi(prob, m0)
        res = engine.compute_hydrodynamic_coefficients(prob, X, m0)
        
        # Extract results for Body 1 (the heaving one)
        # res is a list of dicts. We want the one where mode==1.
        # (Since Body 0 is static, it might not even return a force, 
        # or it will be 0. We specifically want Body 1).
        
        # Find result for Body 1
        body_res = next(r for r in res if r['mode'] == 1)
        # FIX: Use 'real' instead of 'nondim_real'
        ams.append(body_res['real'])
        # FIX: Use 'imag' instead of 'nondim_imag'
        dps.append(body_res['imag'])
        
    duration = time.perf_counter() - start_time
    return np.array(ams), np.array(dps), duration

def solve_cached_update(m0_list):
    """
    Simulates the 'New' method: Initialize Engine once (cache template),
    then only update m0-dependent parts in the loop.
    """
    ams = []
    dps = []
    
    start_time = time.perf_counter()
    
    # 1. Initialize Once
    prob = setup_problem_objects()
    engine = MEEMEngine([prob]) # Cache is built here
    
    for m0 in m0_list:
        # 2. Solve reusing the engine (Triggering the optimized update)
        X = engine.solve_linear_system_multi(prob, m0)
        res = engine.compute_hydrodynamic_coefficients(prob, X, m0)
        
        body_res = next(r for r in res if r['mode'] == 1)
        # FIX: Use 'real' instead of 'nondim_real'
        ams.append(body_res['real'])
        # FIX: Use 'imag' instead of 'nondim_imag'
        dps.append(body_res['imag'])

    duration = time.perf_counter() - start_time
    return np.array(ams), np.array(dps), duration

def test_caching_correctness_and_speed():
    """
    Validates that the cached matrix update method produces identical 
    results to the fresh rebuild method, and runs faster.
    """
    print("\n--- Running Fresh Rebuild Loop ---")
    am_fresh, dp_fresh, t_fresh = solve_fresh_every_time(M0_VALUES)
    
    print("\n--- Running Cached Update Loop ---")
    am_cached, dp_cached, t_cached = solve_cached_update(M0_VALUES)
    
    print(f"\nTime Fresh:  {t_fresh:.4f}s")
    print(f"Time Cached: {t_cached:.4f}s")
    print(f"Speedup:     {t_fresh/t_cached:.2f}x")

    # 1. Correctness Assertion
    # The results must be mathematically identical (or extremely close floating point tolerance)
    np.testing.assert_allclose(
        am_fresh, am_cached, rtol=1e-10, err_msg="Added Mass differs between fresh and cached methods"
    )
    np.testing.assert_allclose(
        dp_fresh, dp_cached, rtol=1e-10, err_msg="Damping differs between fresh and cached methods"
    )
    print(">> CORRECTNESS CHECK PASSED: Results match.")

    # 2. Performance Assertion
    # The cached version should strictly be faster because it skips rebuilding 
    # the m0-independent blocks (which are large in this config).
    assert t_cached < t_fresh, "Cached implementation should be faster than fresh rebuild"
    print(">> PERFORMANCE CHECK PASSED: Caching provided speedup.")