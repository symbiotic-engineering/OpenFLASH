# package/test/openflash_utils.py

import numpy as np
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine
from openflash.multi_equations import omega
from openflash.multi_constants import g

def run_openflash_case(h, d, a, heaving, NMK, m0, rho):
    """
    Runs a single OpenFLASH simulation and returns NON-DIMENSIONAL hydrodynamic coefficients.
    
    Normalization:
        Added Mass (Non-Dim) = AM_dimensional / (rho * Volume)
        Damping (Non-Dim)    = Damping_dimensional / (rho * Volume * omega)
    
    Returns
    -------
    am_norm : float
        Non-dimensional added mass.
    dp_norm : float
        Non-dimensional damping.
    phase : float
        Excitation phase (radians) of the first mode.
    """

    # 1. Build body_map and heaving_map
    body_map = []
    heaving_map = []

    if len(heaving) > 0:
        current_body_idx = 0
        body_map.append(current_body_idx)
        heaving_map.append(bool(heaving[0]))

        for i in range(1, len(heaving)):
            if heaving[i] == heaving[i - 1]:
                body_map.append(current_body_idx)
            else:
                current_body_idx += 1
                body_map.append(current_body_idx)
                heaving_map.append(bool(heaving[i]))
    else:
        raise ValueError("Heaving array must be non-empty.")

    # 2. Geometry + problem setup
    geometry = BasicRegionGeometry.from_vectors(
        a=np.asarray(a),
        d=np.asarray(d),
        h=h,
        NMK=NMK,
        slant_angle=np.zeros(len(a)),
        body_map=body_map,
        heaving_map=heaving_map
    )

    problem = MEEMProblem(geometry)
    engine = MEEMEngine([problem])

    # 3. Solve MEEM system
    X = engine.solve_linear_system_multi(problem, m0)
    
    results = engine.compute_hydrodynamic_coefficients(problem, X, m0, rho=rho)

    # 4. Extract DIMENSIONAL added mass & damping
    total_am_dimensional = 0.0
    total_dp_dimensional = 0.0
    
    # Extract phase from the first mode (assuming single body for this util)
    phase_val = results[0]["excitation_phase"] if results else 0.0

    for res in results:
        total_am_dimensional += res["real"]
        total_dp_dimensional += res["imag"]

    # 5. NORMALIZE RESULTS
    vol = 0.0
    prev_r = 0.0
    for r, draft in zip(a, d):
        area = np.pi * (r**2 - prev_r**2)
        vol += area * draft
        prev_r = r
        
    mass_displaced = vol * rho
    
    # Calculate angular frequency for damping normalization
    w = omega(m0, h, g)

    if mass_displaced > 0:
        am_norm = total_am_dimensional / mass_displaced
        if w > 1e-12:
            dp_norm = total_dp_dimensional / (mass_displaced * w)
        else:
            dp_norm = 0.0
    else:
        am_norm = np.nan
        dp_norm = np.nan
    
    # FIX: Return phase in the 3rd slot
    return am_norm, dp_norm, phase_val