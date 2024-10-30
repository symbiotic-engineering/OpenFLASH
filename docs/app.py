import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt


print(sys.path)
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../package/src')))
from equations import (
    phi_p_i1, phi_p_i2, diff_phi_i1, diff_phi_i2, Z_n_i1, Z_n_i2, Z_n_e,
    m_k, Lambda_k_r, diff_Lambda_k_a2, R_1n_1, R_1n_2, R_2n_2,
    diff_R_1n_1, diff_R_1n_2, diff_R_2n_2, R_2n_1
)
from meem_engine import MEEMEngine
from meem_problem import MEEMProblem
from geometry import Geometry

# Function to visualize A matrix
def visualize_A_matrix(A, title="Matrix Visualization"):
    rows, cols = np.nonzero(A)
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, color='blue', marker='o', s=100)
    plt.gca().invert_yaxis()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))

    N, M = 4, 4  # Replace with actual block sizes if they are dynamic
    block_dividers = [N, N + M, N + 2 * M]
    for val in block_dividers:
        plt.axvline(val - 0.5, color='black', linestyle='-', linewidth=1)
        plt.axhline(val - 0.5, color='black', linestyle='-', linewidth=1)

    plt.grid(True)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    st.pyplot(plt)  # Display the plot in Streamlit
    plt.close()  # Close the figure to avoid display issues

# Main function for the Streamlit app
def main():
    st.title("MEEM Engine Simulation")

    # Geometry configuration and problem instance
    N = 4
    M = 4
    K = 4

    # User input for parameters
    a1 = st.number_input("Enter value for a1:", value=0.5, step=0.01)
    a2 = st.number_input("Enter value for a2:", value=1.0, step=0.01)
    h = st.number_input("Enter value for h:", value=1.001, step=0.01)

    # Setup domain parameters
    domain_params = [
        {'number_harmonics': N, 'height': 1, 'radial_width': a1, 'top_BC': None, 'bottom_BC': None, 'category': 'inner', 'di': 0.5},
        {'number_harmonics': M, 'height': 1, 'radial_width': 1.0, 'top_BC': None, 'bottom_BC': None, 'category': 'outer', 'di': 0.25},
        {'number_harmonics': K, 'height': 1, 'radial_width': 1.5, 'top_BC': None, 'bottom_BC': None, 'category': 'exterior'}
    ]
    
    r_coordinates = {'a1': a1, 'a2': a2}
    z_coordinates = {'h': h}
    geometry = Geometry(r_coordinates, z_coordinates, domain_params)
    problem = MEEMProblem(geometry)
    engine = MEEMEngine([problem])

    # Generate and visualize A matrix
    generated_A = engine.assemble_A(problem)
    visualize_A_matrix(generated_A, title="Generated A Matrix")

    # Further processing and results display...
    # Include additional visualizations and results as needed

if __name__ == "__main__":
    main()
