.. _app-module:

===================
Interactive Web App
===================

.. automodule:: app
   :members:
   :undoc-members:
   :show-inheritance:

.. _app-overview:

Application Overview
====================

This Streamlit application (`app.py`) provides an interactive interface for simulating the hydrodynamic behavior of multiple bodies using the OpenFLASH engine. Users can adjust various parameters to visualize potential fields for a single frequency or compute and plot hydrodynamic coefficients over a range of frequencies.

Running the App
---------------

There are two ways to run the Streamlit application:

**Option 1: Use the Web-Based App (Recommended)**

The app is deployed and can be run directly in your web browser, with no installation required. This is the easiest way to get started.

* **Interactive Streamlit App:** `Launch App <app_streamlit.html>`_

**Option 2: Run the App Locally**

Ensure that you have Streamlit and the `openflash` package installed.

.. code-block:: bash

   pip install streamlit

To launch the app, run the following command from your project's root directory:

.. code-block:: bash

   streamlit run docs/app.py

**Key Features:**

* **Interactive Parameters:** Adjust water depth, body radii, step depths, and the number of harmonics.
* **Object-Oriented Setup:** Defines the geometry by creating `SteppedBody` objects, reflecting the modern API.
* **Single Frequency Analysis:** Solves the system for a single wave frequency and visualizes the total potential field in real time.
* **Frequency Sweep Analysis:** Efficiently runs the simulation across a range of frequencies to compute and plot how added mass and damping coefficients change.

.. _app-functions:

Functions
=========
.. autofunction:: main
   :noindex:

The main function that sets up and runs the Streamlit application.

It configures the sidebar for user inputs, parses them, and sets up the problem geometry using the object-oriented API (`SteppedBody`, `ConcentricBodyGroup`, `BasicRegionGeometry`). The interface is split into two main actions: a single frequency test and a frequency sweep.

**User Inputs (Sidebar):**

* **Water Depth (h):** Overall depth of the water.
* **Body Step Depths (d):** Comma-separated list of submerged depths, one for each body.
* **Body Radii (a):** Comma-separated list of radii, one for each body.
* **Heaving Bodies (1/0):** Comma-separated binary list (1=True, 0=False) indicating if each body is heaving.
* **Harmonics (NMK):** Comma-separated list specifying the number of series approximation terms for each fluid domain (number of bodies + 1).
* **Single Frequency Test:**
   * **Angular Frequency (omega):** The specific frequency for the potential field visualization.
   * **Plot Spatial Resolution:** Controls the grid density for the potential plots.
* **Frequency Sweep for Coefficients:**
   * **Start Omega:** The beginning of the frequency range.
   * **End Omega:** The end of the frequency range.
   * **Number of Steps:** The number of frequencies to simulate within the range.

**Simulation Workflows:**

The application logic is divided into two distinct workflows, triggered by buttons in the main interface.

1.  **Run Single Test & Plot Potentials:**

   * Calculates the non-dimensional wavenumber (`m0`) from the user-provided `omega`.
   * Configures the `MEEMProblem` with the single frequency and the active modes of motion.
   * Initializes the `MEEMEngine` and calls `solve_linear_system_multi` to get the solution vector `X`.
   * Computes and displays the hydrodynamic coefficient matrices for that single frequency.
   * Calls `calculate_potentials` to compute the total potential field.
   * Visualizes the real and imaginary parts of the potential field using Matplotlib.

2.  **Run Frequency Sweep & Plot Coefficients:**

   * Creates an array of frequencies based on the user's start, end, and step inputs.
   * Configures the `MEEMProblem` with the full array of frequencies and active modes.
   * Initializes the `MEEMEngine` and calls the highly efficient `run_and_store_results` method. This single method handles the entire simulation loop internally.
   * Extracts the computed `added_mass` and `damping` coefficients from the resulting `xarray.Dataset`.
   * Generates and displays Matplotlib plots showing how the added mass and damping coefficients vary across the simulated frequency range.