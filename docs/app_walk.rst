.. _app-module:

============
App Module
============

.. automodule:: app
   :members:
   :undoc-members:
   :show-inheritance:

.. _app-overview:

Application Overview
====================

This Streamlit application (`app.py`) provides an interactive interface for simulating the hydrodynamic behavior of multiple cylinders using MEEM. Users can adjust various parameters to visualize potential and velocity fields, and compute hydrodynamic coefficients (added mass and damping).

 **Run the Streamlit App:**
 Ensure that you have Streamlit installed. You can install it using:

 .. code-block:: bash

   pip install streamlit

 To launch the app, run the following command in the docs folder:

 .. code-block:: bash

   streamlit run app.py

**Key Features:**

* **Interactive Parameters:** Adjust water depth, cylinder radii, heaving states, number of harmonics, wave number, angular frequency, and spatial resolution.
* **Dynamic Domain Configuration:** Supports multi-region calculations.
* **Slant Support:** Allows specification of slanted cylinder walls.
* **Hydrodynamic Coefficients:** Calculates and displays added mass and damping for each heaving mode.
* **Potential and Velocity Field Visualization:** Plots homogeneous, particular, and total potential fields, as well as radial and vertical velocity components.

.. _app-functions:

Functions
=========

.. autofunction:: plot_field
   :noindex:

   Plots the real and imaginary parts of a field using Matplotlib.
   The function returns a Matplotlib figure object.

   :param field: A 2D NumPy array representing the complex field data.
   :type field: numpy.ndarray
   :param R: A 2D NumPy array of radial coordinates.
   :type R: numpy.ndarray
   :param Z: A 2D NumPy array of axial coordinates.
   :type Z: numpy.ndarray
   :param title: The title for the plot.
   :type title: str
   :param a_list: A list of cylinder radii used for plotting vertical boundary lines.
   :type a_list: list[float]
   :param d_list: A list of cylinder depths used for plotting horizontal boundary lines.
   :type d_list: list[float]
   :param plot_imaginary: If True, plots both real and imaginary parts. If False, only plots the real part. Defaults to True.
   :type plot_imaginary: bool
   :returns: A Matplotlib Figure object.
   :rtype: matplotlib.figure.Figure

.. autofunction:: main
   :noindex:

   The main function that sets up and runs the Streamlit application.

   It configures the sidebar for user inputs, parses and validates these inputs,
   sets up the `MEEMProblem` and `MEEMEngine`, solves the hydrodynamic problem,
   computes coefficients, calculates potential and velocity fields, and
   visualizes the results using Matplotlib plots within the Streamlit interface.

   **User Inputs (Sidebar):**

   * **Water Depth (h):** Overall depth of the water. (Slider)
   * **Cylinder Depths (d):** Comma-separated list of depths for each cylinder. (Text Input)
   * **Cylinder Radii (a):** Comma-separated list of radii for each cylinder. (Text Input)
   * **Heaving States:** Comma-separated binary (1=True, 0=False) list indicating if each domain is heaving. (Text Input)
   * **Slanted States:** Comma-separated binary (1=True, 0=False) list indicating if each domain's outer wall is slanted. (Text Input)
   * **Number of Harmonics (NMK):** Comma-separated list specifying the number of terms in the approximation for each domain (inner, middle, exterior). (Text Input)
   * **Wave Number (m0):** The radial wave number. (Number Input)
   * **Angular Frequency (omega):** The angular frequency of the wave system. (Number Input)
   * **Spatial Resolution:** Controls the density of the grid for plotting potential/velocity fields. (Slider)

   **Plot Options (Sidebar Checkboxes):**

   * Show Homogeneous Potential Plots
   * Show Particular Potential Plots
   * Show Total Potential Plots
   * Show Radial Velocity Plots
   * Show Vertical Velocity Plots

   **Simulation Steps:**

   1.  **Input Parsing & Validation:** Converts string inputs to lists of numbers and performs basic validation on their lengths.
   2.  **Domain Parameter Construction:** Builds a list of dictionaries, each describing a domain's properties based on user inputs.
   3.  **Geometry and Problem Setup:** Initializes `Geometry` and `MEEMProblem` objects with the defined parameters, frequencies, and modes.
   4.  **MEEM Engine Initialization:** Creates an `MEEMEngine` instance.
   5.  **Linear System Solving:** Solves the core linear system to find the unknown coefficients (`X`).
   6.  **Hydrodynamic Coefficients Calculation:** Computes and displays added mass and damping coefficients for each heaving mode.
   7.  **Coefficient Reformatting:** Reformats the solution vector `X` into a more usable structure (`Cs`) per region.
   8.  **Cache Access:** Retrieves precomputed arrays (`m_k_arr`, `N_k_arr`) from the engine's cache.
   9.  **Potential Field Calculation:** Iterates through spatial regions to calculate homogeneous and particular potentials based on the computed coefficients.
   10. **Potential Field Plotting:** Generates and displays Matplotlib plots for the selected potential fields.
   11. **Velocity Field Calculation:** Computes radial and vertical velocity components across the domain.
   12. **Velocity Field Plotting:** Generates and displays Matplotlib plots for the velocity fields.
   13. **Completion Message:** Displays a success message upon simulation completion.
