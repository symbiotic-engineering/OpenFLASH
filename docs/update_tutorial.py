import json
import os

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
    # We assume Cell 0 is the Markdown Title and Cell 1 is Imports
    new_cells = nb['cells'][:2]

    # --- Cell 2: Problem Parameters (Common) ---
    cell_params = {
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
    }
    new_cells.append(cell_params)

    # --- Cell 3: Helper Function for Simulation ---
    cell_helper = {
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
    }
    new_cells.append(cell_helper)

    # --- Cell 4: Case 1 Markdown ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Case 1: Inner Body Heaving (Mode 0)\n",
            "\n",
            "We simulate the case where only the inner body is heaving (`heaving_list = [True, False]`)."
        ]
    })

    # --- Cell 5: Case 1 Code ---
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

    # --- Cell 6: Case 2 Markdown ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Case 2: Outer Body Heaving (Mode 1)\n",
            "\n",
            "Now we simulate the case where only the outer body is heaving (`heaving_list = [False, True]`).\n",
            "\n",
            "**Note:** Currently, the package calculates the diagonal terms of the hydrodynamic coefficient matrix. Off-diagonal (coupling) terms are not yet implemented, so we run separate problems for each mode."
        ]
    })

    # --- Cell 7: Case 2 Code ---
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

    # --- Cell 8: Visualization Markdown ---
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Potential Field Visualization\n",
            "\n",
            "We can visualize the potential field for one of the cases (e.g., Case 2: Outer Body Heaving)."
        ]
    })

    # --- Cell 9: Visualization Code ---
    # Attempt to locate the original visualization code (containing 'plot')
    orig_viz_source = None
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'plot' in str(cell['source']).lower():
            orig_viz_source = cell['source']
            break
    
    if orig_viz_source is None:
        # Fallback if not found
        orig_viz_source = ["# Visualization code missing from original file\n"]

    # Prepend variable binding so the viz code uses the results from Case 2
    viz_source = [
        "# Setup variables for visualization using Case 2 results\n",
        "problem = problem2\n",
        "X = X2\n",
        "\n"
    ]
    if isinstance(orig_viz_source, list):
        viz_source.extend(orig_viz_source)
    else:
        viz_source.append(orig_viz_source)

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_source
    })

    # Update and Save
    nb['cells'] = new_cells
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=4)
    print(f"Successfully updated {notebook_path}")

if __name__ == "__main__":
    update_notebook()