# package/test/openflash_utils.py

import numpy as np
from openflash.basic_region_geometry import BasicRegionGeometry
from openflash.meem_problem import MEEMProblem
from openflash.meem_engine import MEEMEngine

def run_openflash_case(h, d, a, heaving, NMK, m0, rho):
    """
    Runs a single OpenFLASH simulation and returns TOTAL DIMENSIONAL added mass.
    
    Returns
    -------
    am_total : float
        Total dimensional added mass (kg) summing all heaving regions.
    dp_total : float
        Total dimensional damping (kg/s) summing all heaving regions.
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
    results = engine.compute_hydrodynamic_coefficients(problem, X, m0)

    # 4. Extract DIMENSIONAL added mass & damping
    total_am_dimensional = 0.0
    total_dp_dimensional = 0.0

    for res in results:
        total_am_dimensional += res["real"]
        total_dp_dimensional += res["imag"]

    # 5. UNIT FIX APPLIED:
    # The erroneous factor of h^3 has been removed from meem_engine.py.
    # Therefore, we no longer need to divide by h^3 here.
    # The result from compute_hydrodynamic_coefficients is now correctly in [kg].
    
    return total_am_dimensional, total_dp_dimensional, 0