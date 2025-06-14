import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest
from multi_test_functions import MultiEvaluator

def interpret_capytaine_file(filename, omega):
    file_path = "test/data/" + filename + "-imag.csv"
    df = pd.read_csv(file_path, header=None)
    real_array = (df.to_numpy()) * (-1/omega)
        
    file_path = "test/data/" + filename + "-real.csv"
    df = pd.read_csv(file_path, header=None)
    imag_array = (df.to_numpy()) * (1/omega)

    return(real_array, imag_array)

def interpret_matlab_file(filename):
    file_path = "test/data/" + filename + "-imag - matlab.csv"
    df = (pd.read_csv(file_path, header=None))
    imag_array = df.to_numpy()
        
    file_path = "test/data/" + filename + "-real - matlab.csv"
    df = (pd.read_csv(file_path, header=None))
    real_array = df.to_numpy()
    return(real_array, imag_array)

def plot_difference(title, R, Z, arr):
    plt.figure(figsize=(8, 6))
    plt.contourf(R, Z, arr, levels=1, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Radial Distance (R)')
    plt.ylabel('Axial Distance (Z)')
    plt.show()

def fractional_diff(a,b):
    return abs((a-b)/a)

# arguments: the name of the csv to compare with, an appropriately formatted potential array
# the threshold of closeness, appropriate R and Z, and omega to help with conversion
# tailored for 50x50 points (including nans), evenly spaced, twice the widest radius, and given height

def potential_comparison(filename, filetype, arr, rtol, atol, R, Z, omega, nan_mask):

    if filetype == "matlab":
        real_arr, imag_arr = interpret_matlab_file(filename)
    else:
        real_arr, imag_arr = interpret_capytaine_file(filename, omega)

    real_calc_arr = np.real(arr)
    imag_calc_arr = np.imag(arr)

    is_within_threshold_r = np.isclose(real_arr, np.real(arr), rtol = rtol, atol = atol)
    is_within_threshold_i = np.isclose(imag_arr, np.imag(arr), rtol = rtol, atol = atol)

    for i in range(len(nan_mask)):
        is_within_threshold_r[nan_mask[i]] = np.nan
        is_within_threshold_i[nan_mask[i]] = np.nan

    match_r = np.sum(np.isnan(is_within_threshold_r)) + np.sum(is_within_threshold_r == 1)
    match_i = np.sum(np.isnan(is_within_threshold_i)) + np.sum(is_within_threshold_i == 1)

    return (match_r, match_i, is_within_threshold_r, is_within_threshold_i)

def test_config1():
    # Requires: h, d, a, heaving, m0, g, rho, NMK
    config = MultiEvaluator(h = 1.001, d = [0.5, 0.25], a = [0.5, 1], heaving = [1, 1], m0 = 1, g = 9.81, rho = 1023, NMK = [50, 50, 50])
    A = config.A_matrix()
    b = config.b_vector()
    coeffs = config.Ab_coefficients(A,b)
    hydrocs = config.hydro_coefficients(coeffs)
    R, Z, phi, nanregions = config.potential_matrix()

    
    assert fractional_diff(hydrocs[2], 0.4995) <= 0.001
    assert fractional_diff(hydrocs[3], 0.3679) <= 0.001
    
    sum1, sum2, thres1, thres2 = potential_comparison("Total Potential", "matlab", phi, 0.01, 0.01, R, Z, config.omega, nanregions)
    assert sum1 == 2500
    assert sum2 == 2500

test_config1()