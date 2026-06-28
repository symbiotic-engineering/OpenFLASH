import json
import os
import numpy as np

def update_notebook():
    # Find the notebook in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, 'tutorial_walk.ipynb')
    
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find notebook at {notebook_path}")
        return

    # --- Preserve Initial Cells (Header & Imports) ---
    new_cells = nb['cells'][:2]

    # --- Cell 2: Problem Setup ---
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------\n",
            "# --- 1. Problem Setup ---\n",
            "# ---------------------------------\n",
            "print(\"\\n--- 1. Setting up the Problem ---\")\n",
            "h = 1.001            # Water Depth (m)\n",
            "d_list = [0.5, 0.25]   # Step depths (m) for inner and outer bodies\n",
            "a_list = [0.5, 1.0]    # Radii (m) for inner and outer bodies\n",
            "NMK = [30, 30, 30]     # Harmonics for inner, middle, and exterior domains\n",
            "\n",
            "m0 = 1.0    # Non-dimensional wave number\n",
            "problem_omega = omega(m0, h, g)\n",
            "print(f\"Wave number (m0): {m0}, Angular frequency (omega): {problem_omega:.4f}\")"
        ]
    })

    # --- Cell 3: Helper Function ---
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def run_simulation(heaving_list, case_name):\n",
            "    \"\"\"Sets up geometry, problem, and engine for a specific heaving configuration.\"\"\"\n",
            "    print(f\"\\n--- Running Simulation: {case_name} ---\")\n",
            "    \n",
            "    # 1. Create Bodies\n",
            "    bodies = []\n",
            "    for i in range(len(a_list)):\n",
            "        body = SteppedBody(\n",
            "            a=np.array([a_list[i]]),\n",
            "            d=np.array([d_list[i]]),\n",
            "            slant_angle=np.array([0.0]), \n",
            "            heaving=heaving_list[i]\n",
            "        )\n",
            "        bodies.append(body)\n",
            "\n",
            "    # 2. Arrangement\n",
            "    arrangement = ConcentricBodyGroup(bodies)\n",
            "\n",
            "    # 3. Geometry\n",
            "    geometry = BasicRegionGeometry(\n",
            "        body_arrangement=arrangement,\n",
            "        h=h,\n",
            "        NMK=NMK\n",
            "    )\n",
            "\n",
            "    # 4. Problem\n",
            "    problem = MEEMProblem(geometry)\n",
            "    problem.set_frequencies(np.array([problem_omega]))\n",
            "    \n",
            "    # 5. Engine\n",
            "    engine = MEEMEngine(problem_list=[problem])\n",
            "    \n",
            "    # 6. Solve\n",
            "    print(f\"Solving for {case_name}...\")\n",
            "    X = engine.solve_linear_system_multi(problem, m0)\n",
            "    \n",
            "    # 7. Coefficients\n",
            "    coeffs = engine.compute_hydrodynamic_coefficients(problem, X, m0)\n",
            "    \n",
            "    return problem, X, coeffs\n"
        ]
    })

    # --- Cell 4-5: Case 1 ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Case 1: Inner Body Heaving (Mode 0)\n\nWe simulate the case where only the inner body is heaving (`heaving_list = [True, False]`)."]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "problem1, X1, coeffs1 = run_simulation([True, False], \"Inner Body Heaving\")\n",
            "\n",
            "print(\"\\nHydrodynamic Coefficients (Mode 0):\")\n",
            "if coeffs1:\n",
            "    print(pd.DataFrame(coeffs1))"
        ]
    })

    # --- Cell 6-7: Case 2 ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Case 2: Outer Body Heaving (Mode 1)\n\nNow we simulate the case where only the outer body is heaving (`heaving_list = [False, True]`).\n\n**Note:** Currently, the package calculates the diagonal terms of the hydrodynamic coefficient matrix. Off-diagonal (coupling) terms are not yet implemented, so we run separate problems for each mode."
        ]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "problem2, X2, coeffs2 = run_simulation([False, True], \"Outer Body Heaving\")\n",
            "\n",
            "print(\"\\nHydrodynamic Coefficients (Mode 1):\")\n",
            "if coeffs2:\n",
            "    print(pd.DataFrame(coeffs2))"
        ]
    })

    # --- Cell 8-9: Visualization ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Potential Field Visualization\n\nWe can visualize the potential field for Case 2 (Outer Body Heaving)."]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Re-initialize Engine for Visualization\n",
            "engine_viz = MEEMEngine([problem2])\n",
            "\n",
            "# 2. Calculate Potentials\n",
            "print(\"Calculating potentials on grid...\")\n",
            "potentials = engine_viz.calculate_potentials(\n",
            "    problem=problem2,\n",
            "    solution_vector=X2,\n",
            "    m0=m0,\n",
            "    spatial_res=100,\n",
            "    sharp=True\n",
            ")\n",
            "\n",
            "# 3. Extract Fields\n",
            "R = potentials['R']\n",
            "Z = potentials['Z']\n",
            "phi_abs = np.abs(potentials['phi'])\n",
            "\n",
            "# 4. Plot\n",
            "print(\"Plotting...\")\n",
            "fig, ax = engine_viz.visualize_potential(phi_abs, R, Z, \"Total Potential Field (Absolute Magnitude)\")\n",
            "plt.show()"
        ]
    })

    # --- Cell 10-11: Domain Analysis ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Domain Analysis\n\nWe can inspect the fluid domains created by the `BasicRegionGeometry`."
        ]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"\\n--- Inspecting Fluid Domains ---\")\n",
            "domains = problem2.geometry.domain_list\n",
            "\n",
            "for idx, domain in domains.items():\n",
            "    print(f\"Domain {idx} ({domain.category}):\")\n",
            "    outer_str = f\"{domain.a_outer:.2f}\" if domain.a_outer != np.inf else \"inf\"\n",
            "    print(f\"  Radii: {domain.a_inner:.2f} m to {outer_str} m\")\n",
            "    print(f\"  Lower Depth: {domain.d_lower:.2f} m\")\n",
            "    print(f\"  Harmonics: {domain.number_harmonics}\")\n"
        ]
    })

    # --- Cell 12-13: Results Analysis (THE FIX) ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Storing and Exporting Results\n\nThe `Results` class (based on `xarray`) provides a structured way to store simulation data. Since we ran two separate simulations (one per mode), we will manually aggregate the results into a single dataset containing the full 2x2 added mass and damping matrices."
        ]
    })
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Initialize Results object for the FULL system\n",
            "# We explicitly pass the modes [0, 1] so the container can hold the full matrix,\n",
            "# even though problem2 only had mode 1 active.\n",
            "all_modes = np.arange(len(a_list)) # [0, 1]\n",
            "results = Results(problem2, modes=all_modes)\n",
            "\n",
            "# 2. Construct Full Matrices from Separate Mode Results\n",
            "# Shape: (n_freqs, n_modes, n_modes)\n",
            "n_freqs = len(problem2.frequencies)\n",
            "n_modes = len(all_modes) \n",
            "\n",
            "added_mass = np.zeros((n_freqs, n_modes, n_modes))\n",
            "damping = np.zeros((n_freqs, n_modes, n_modes))\n",
            "\n",
            "# Helper to fill matrix columns\n",
            "# coeffs_list contains entries for row_idx (force on body i)\n",
            "# col_idx (motion of body j) is determined by which case we ran\n",
            "def fill_matrix_col(coeffs_list, col_idx):\n",
            "    for c in coeffs_list:\n",
            "        row_idx = c['mode'] # Force on body i\n",
            "        added_mass[0, row_idx, col_idx] = c['real']\n",
            "        damping[0, row_idx, col_idx] = c['imag']\n",
            "\n",
            "# Fill Column 0 (Inner Body Heaving results -> Mode 0)\n",
            "if coeffs1:\n",
            "    fill_matrix_col(coeffs1, 0)\n",
            "\n",
            "# Fill Column 1 (Outer Body Heaving results -> Mode 1)\n",
            "if coeffs2:\n",
            "    fill_matrix_col(coeffs2, 1)\n",
            "\n",
            "# 3. Store in Results Object\n",
            "results.store_hydrodynamic_coefficients(\n",
            "    frequencies=problem2.frequencies,\n",
            "    added_mass_matrix=added_mass,\n",
            "    damping_matrix=damping\n",
            ")\n",
            "\n",
            "# 4. View Dataset\n",
            "print(\"\\n--- Results Dataset (xarray) ---\")\n",
            "print(results.dataset)\n",
            "\n",
            "# 5. Export (Optional)\n",
            "# results.export_to_netcdf(\"tutorial_results.nc\")\n"
        ]
    })

    # Update and Save
    nb['cells'] = new_cells
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=4)
    print(f"Successfully updated {notebook_path}")

if __name__ == "__main__":
    update_notebook()