MEEM Engine Simulation
======================

This is a Streamlit application for simulating the MEEM (Modified Eigenfunction Expansion Method) engine. The application allows users to input parameters and visualize the generated matrix \(A\) resulting from the problem setup.

Overview
--------
The application imports necessary libraries, sets up the problem geometry, collects user inputs, and visualizes the generated matrix. 

### Dependencies

- `streamlit`

- `numpy`

- `pandas`

- `scipy`

- `matplotlib`

### Modules

- `equations`: Contains functions for various equations used in the simulation.  

- `meem_engine`: Contains the `MEEMEngine` class for simulating the MEEM process.  

- `meem_problem`: Contains the `MEEMProblem` class that represents the problem setup.  

- `geometry`: Contains the `Geometry` class for managing the geometric configuration of the problem.  

Usage
-----
1. **Run the Streamlit application**:
   To start the application (cd into docs), run the following command in your terminal:

   streamlit run app.py

2. **Input Parameters**:

- `a1`: Value for parameter \(a_1\).

- `a2`: Value for parameter \(a_2\).

- `h`: Value for height.

3. **Matrix Visualization**:

After entering the parameters, the application will generate the matrix \(A\) and visualize it.

Functions
---------
### visualize_A_matrix(A, title="Matrix Visualization")

Visualizes the matrix \(A\) using a scatter plot.

**Parameters**:

- **A** (ndarray): The matrix to visualize.

- **title** (str): Title for the visualization (default: "Matrix Visualization").

### main()

The main function that runs the Streamlit application.

**Functionality**:

- Sets up the Streamlit interface.

- Collects user input for parameters.

- Configures the geometry and problem instance.

- Assembles the matrix \(A\) and calls the visualization function.

Example
-------
An example of how to configure the geometry and visualize the matrix can be seen in the `main()` function, which initializes the geometry based on user input and calls the visualization function.

