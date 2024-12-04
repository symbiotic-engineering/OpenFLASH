MEEM Simulation Streamlit App
=============================

Introduction
------------
The MEEM Simulation app allows users to perform simulations for solving a multi-region problem with custom parameters. The app uses the MEEM (Multi-Region Eigenfunction Expansion Method) engine to compute hydrodynamic coefficients and visualize the results.

Features
--------
1. **Hydrodynamic Coefficients Calculation:** The app calculates the coefficients based on user-defined domain parameters.
2. **Visualization:** The app provides visualizations of the radial, vertical, and total velocity potentials.
3. **Interactive Sidebar:** Users can interactively adjust simulation parameters like height, slant vectors, spatial resolution, and harmonic values.

How to Use
-----------
1. **Run the Streamlit App:**
   Ensure that you have Streamlit installed. You can install it using:
   
   .. code-block:: bash
      pip install streamlit

   To launch the app, run the following command in the docs folder:
   
   .. code-block:: bash
      streamlit run app.py

2. **Adjust Simulation Parameters:**
   The app features a sidebar where you can input different parameters for your simulation:

   - **Water Height (h):** Adjust the height parameter. The default is set to 1.001 meters.
   - **Body Height (d):** Enter a list of body heights separated by commas (e.g., "0.5,0.25").
   - **Diameter (a):** Enter a list of diameters separated by commas (e.g., "0.5,1").
   - **Heaving States:** Define whether each region is heaving (1 for True, 0 for False) by entering a comma-separated list (e.g., "1,1").
   - **Slant Vectors:** Enter slant vectors for each region (e.g., "0,0,1;0,0,1;0,0,1").
   - **Number of Harmonics (NMK):** Define the number of harmonics for each region, separated by commas (e.g., "30,30,30").
   - **Spatial Resolution:** Choose the spatial resolution for the mesh grid. The range is between 10 and 100.
   - **Checkboxes:** Select whether to display different potential plots, such as homogeneous, particular, total, radial velocity, and vertical velocity.

3. **View Domain Parameters:**
   After configuring the parameters, the app displays the domain parameters, including the number of harmonics, radial widths, and slant vectors, which are used in the simulation.

4. **Run the Simulation:**
   Once you've configured the parameters, the app automatically runs the simulation and computes the hydrodynamic coefficients.

5. **Visualization:**
   The app provides visualizations of different potential fields. These include:
   - **Homogeneous Potential Plots**
   - **Particular Potential Plots**
   - **Total Potential Plots**
   - **Radial Velocity Potential Plots**
   - **Vertical Velocity Potential Plots**

   You can interact with the plots and explore the results. The plots are generated using the Matplotlib library, with contour plots showing the variation of potential across the radial and axial distances.

6. **Understanding the Results:**
   The hydrodynamic coefficients and the potential fields are computed based on the user-defined inputs. These results are displayed in the app in a format that helps you understand the spatial variations in the velocity potentials.

Code Explanation
----------------
The main components of the code include:

1. **User Input:**
   The app uses `st.sidebar` for user inputs, where users can adjust various parameters for the simulation.

2. **Geometry and Engine Setup:**
   The `Geometry` and `MEEMEngine` classes are used to configure the simulation domain and run the solver. The problem is set up using these objects, and the linear system is solved using SciPy's `linalg.solve()` function.

3. **Potential and Velocity Calculations:**
   The code calculates different potentials (e.g., homogeneous, particular, total) and velocities (e.g., radial, vertical) using predefined functions like `phi_h_n_inner_func`, `phi_h_m_i_func`, and others.

4. **Visualization:**
   The `plot_potential()` function creates visual representations of the computed potential fields using Matplotlib and Streamlit's `st.pyplot()`.

Troubleshooting
---------------
- Ensure that all necessary Python packages are installed, including `streamlit`, `numpy`, `pandas`, `scipy`, and `matplotlib`.
- If the app does not load, check the browser console for any error messages.
- Ensure that the correct versions of packages are being used to avoid compatibility issues.
- If slant vectors or other inputs are misconfigured, the app will show an error and prompt you to adjust the inputs.

Conclusion
----------
This Streamlit app provides an easy-to-use interface for running complex MEEM simulations. It allows users to customize the parameters, run simulations, and visualize the results interactively. By adjusting parameters in the sidebar, users can explore different configurations and see how they affect the hydrodynamic coefficients and velocity potentials.

For more information, refer to the documentation of the `MEEMEngine`, `Geometry`, and other classes used in the app.
