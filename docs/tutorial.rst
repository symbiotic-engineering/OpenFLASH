.. _meem_tutorial:

Tutorial: Solving and Visualizing Potentials in MEEM
====================================================

Contents
--------
.. contents::
   :local:

Introduction
------------
This tutorial walks through using the MEEM package to compute and visualize potentials across multiple regions. By the end, you will have assembled matrices, validated them against expected values, and visualized the homogeneous and particular potentials in defined geometrical regions.

Setup
-----
Ensure you have installed the required modules:

.. code-block:: bash

    pip install numpy scipy pandas matplotlib

Now, import the necessary modules:

.. code-block:: python

    import sys, os
    import numpy as np
    import pandas as pd
    from scipy import linalg
    import matplotlib.pyplot as plt

    sys.path.append("path_to_meem_module")  # Add the path to the MEEM module if necessary

Geometry and Problem Setup
--------------------------
Define the geometry for the problem, including radial and axial coordinates, as well as boundary conditions:

.. code-block:: python

    from meem_engine import MEEMEngine
    from meem_problem import MEEMProblem
    from geometry import Geometry

    r_coordinates = {'a1': 0.5, 'a2': 1.0}  # Example radial coordinates
    z_coordinates = {'h': 1.001}  # Example axial coordinates
    domain_params = [
        {'number_harmonics': N, 'height': 1, 'radial_width': 0.5, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': 0.5},
        {'number_harmonics': M, 'height': 1, 'radial_width': 1.0, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': 0.25},
        {'number_harmonics': K, 'height': 1, 'radial_width': 1.5, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
    ]

    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)
    engine = MEEMEngine([problem])

Matrix Assembly and Visualization
---------------------------------
To verify the assembled matrix `A`, visualize its structure, which is helpful for spotting unexpected values or structure. The following function `visualize_A_matrix` plots the positions of non-zero entries and the block structure of the matrix.

.. code-block:: python

    def visualize_A_matrix(A, title="Matrix Visualization"):
        rows, cols = np.nonzero(A)
        plt.figure(figsize=(6, 6))
        plt.scatter(cols, rows, color='blue', marker='o', s=100)
        plt.gca().invert_yaxis()
        plt.xticks(range(A.shape[1]))
        plt.yticks(range(A.shape[0]))

        # Draw dividing lines to visualize block structure in A matrix
        N, M = 4, 4
        block_dividers = [N, N + M, N + 2 * M]
        for val in block_dividers:
            plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
            plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

        plt.grid(True)
        plt.title(title)
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.show()

    # Generate and verify A matrix
    generated_A = engine.assemble_A(problem)
    visualize_A_matrix(generated_A, title="Generated A Matrix")
        
# Assuming 'A' is the assembled matrix from `MEEMEngine`

Matrix and Vector Validation
----------------------------
We compare our generated matrix and vector to predefined expected values, allowing us to confirm the accuracy of our results. Threshold levels are set to `1e-3` for matrices and vectors to ensure only minor deviations.

.. code-block:: python

    tolerance = 1e-3
    generated_b = engine.assemble_b(problem)
    expected_b = np.array([
        0.0069, 0.0120, -0.0030, 0.0013, 
        0.1560, 0.0808, -0.0202, 0.0090, 
        0, -0.1460, 0.0732, -0.0002, 
        -0.4622, -0.2837, 0.1539, -0.0673
    ], dtype=np.complex128)  # Load expected values

    # Validate the matrix and vector
    try:
        np.testing.assert_allclose(generated_b, expected_b, atol=tolerance, err_msg="b vector does not match expected values")
        print("b vector matches successfully.")
    except AssertionError as e:
        print("b vector does not match expected values. Details:")
        print(e)

Solving the System and Extracting Results
-----------------------------------------
After generating `A` and `b`, we solve the equation `Ax = b`. The solution vector `x` contains coefficients for the homogeneous potential functions, which we use for region-specific potential calculations.

.. code-block:: python

    X = linalg.solve(generated_A, generated_b)

Defining Potential Functions
----------------------------
The functions below calculate the homogeneous potential in different regions. For example, `phi_h_n_i1_func` computes the potential in region 1 (or domain i1) based on coefficients from `x`.

.. code-block:: python

    def phi_h_n_i1_func(n, r, z):
        #  Define the formula to compute potential for region 1
        return (C_1n_1s[n] * R_1n_1(n, r) + C_2n_1s[n] * R_2n_1(n)) * Z_n_i1(n, z)

    def phi_h_m_i2_func(m, r, z):
        return (C_1n_2s[m] * R_1n_2(m, r) + C_2n_2s[m] * R_2n_2(m, r)) * Z_n_i2(m, z)

    def phi_e_k_func(k, r, z):
        return B_ks[k] * Lambda_k_r(k, r) * Z_n_e(k, z)

Visualization of Potentials
---------------------------
Using `plot_potential` and `plot_matching` functions, visualize the homogeneous and particular potentials, as well as the potential continuity at interfaces.

.. code-block:: python

    def plot_potential(phi, R, Z, title):
        plt.figure(figsize=(12, 6))

        # Real part
        plt.subplot(1, 2, 1)
        contour_real = plt.contourf(R, Z, np.real(phi), levels=50, cmap='viridis')
        plt.colorbar(contour_real)
        plt.title(f'{title} - Real Part')
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')

        # Imaginary part
        plt.subplot(1, 2, 2)
        contour_imag = plt.contourf(R, Z, np.imag(phi), levels=50, cmap='viridis')
        plt.colorbar(contour_imag)
        plt.title(f'{title} - Imaginary Part')
        plt.xlabel('Radial Distance (R)')
        plt.ylabel('Axial Distance (Z)')

        plt.tight_layout()
        plt.show()

    def plot_matching(phi1, phi2, phie, a1, a2, R, Z, title):
        idx_a1 = np.argmin(np.abs(R[0, :] - a1))
        idx_a2 = np.argmin(np.abs(R[0, :] - a2))

        Z_line = Z[:, 0]

        # Potential at r = a1
        phi1_a1 = phi1[:, idx_a1]
        phi2_a1 = phi2[:, idx_a1]

        # Potential at r = a2
        phi2_a2 = phi2[:, idx_a2]
        phie_a2 = phie[:, idx_a2]

        plt.figure(figsize=(8, 6))
        plt.plot(Z_line, np.abs(phi1_a1), 'r--', label=f'|{title}_1| at r = a1')
        plt.plot(Z_line, np.abs(phi2_a1), 'b-', label=f'|{title}_2| at r = a1')
        plt.plot(Z_line, np.abs(phi2_a2), 'g--', label=f'|{title}_2| at r = a2')
        plt.plot(Z_line, np.abs(phie_a2), 'k-', label=f'|{title}_e| at r = a2')
        plt.legend()
        plt.xlabel('Z')
        plt.ylabel(f'|{title}|')
        plt.title(f'{title} Matching at Interfaces')
        plt.show()

    # Example of plotting potentials
    plot_potential(phiH, R, Z, 'Homogeneous Potential')
    plot_potential(phiP, R, Z, 'Particular Potential')
    plot_potential(phi, R, Z, 'Total Potential')

    # Extract potentials in different regions for matching
    phi1 = np.where(region1, phi, np.nan)
    phi2 = np.where(region2, phi, np.nan)
    phie = np.where(regione, phi, np.nan)

    # Plot matching at interfaces
    plot_matching(phi1, phi2, phie, a1, a2, R, Z, 'Potential')

Verification of Continuity
--------------------------
To ensure continuity of potentials across interfaces, we calculate and plot the difference in potential across `a1` and `a2`.

.. code-block:: python

    def verify_continuity_at_interfaces():
        # Sample code to compute continuity difference at interfaces
        # Potential at r = a1 from region1 and region2
        idx_a1 = np.argmin(np.abs(R[0, :] - a1))
        phi_a1_region1 = phi1[:, idx_a1]
        phi_a1_region2 = phi2[:, idx_a1]

        # Compute the difference
        difference_a1 = np.abs(phi_a1_region1 - phi_a1_region2)

        # Potential at r = a2 from region2 and exterior region
        idx_a2 = np.argmin(np.abs(R[0, :] - a2))
        phi_a2_region2 = phi2[:, idx_a2]
        phi_a2_exterior = phie[:, idx_a2]

        # Compute the difference
        difference_a2 = np.abs(phi_a2_region2 - phi_a2_exterior)

        # Plot the differences
        plt.figure(figsize=(8, 6))
        plt.plot(Z[:, 0], difference_a1, label='Difference at r = a1')
        plt.plot(Z[:, 0], difference_a2, label='Difference at r = a2')
        plt.legend()
        plt.xlabel('Z')
        plt.ylabel('Potential Difference')
        plt.title('Continuity of Potential at Interfaces')
        plt.show()

Conclusion
----------
In this tutorial, we assembled and validated matrices for potential calculations, visualized potential fields, and verified continuity across regions. This workflow provides a robust approach for calculating and assessing potential fields using the MEEM Python package.
