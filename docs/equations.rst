.. _equations.py:

Equation Module
=====================

This module defines a set of functions for solving various equations used in physics and engineering. It includes equations related to cylindrical Bessel functions, specific to the problem at hand, and integrates key mathematical concepts such as Hankel, Bessel, and modified Bessel functions.

Dependencies
------------

- numpy
- scipy (special, integrate, linalg, optimize)
- matplotlib
- constants (external module)

Functions
---------

**m_k(k)**
   Computes the value of `m_k` using Newton's method for root finding.

   :param k: Integer value for the index.
   :return: Computed value of `m_k`.

**m_k_newton(h)**
   Solves for `m_k` using Newton's method with a given height `h`.

   :param h: Height value.
   :return: Result from Newton's method.

**lambda_n1(n)**
   Computes the first eigenvalue `lambda_n1` for the first layer.

   :param n: Integer value for the index.
   :return: Computed value of `lambda_n1`.

**lambda_n2(n)**
   Computes the second eigenvalue `lambda_n2` for the second layer.

   :param n: Integer value for the index.
   :return: Computed value of `lambda_n2`.

**phi_p_a1(z)**
   Computes the function `phi_p` for the first particular solution.

   :param z: Z-coordinate.
   :return: Computed value of `phi_p_a1`.

**phi_p_a2(z)**
   Computes the function `phi_p` for the second particular solution.

   :param z: Z-coordinate.
   :return: Computed value of `phi_p_a2`.

**diff_phi_p_i2_a2(h)**
   Differentiates the second particular solution with respect to height `h`.

   :param h: Height value.
   :return: Differentiated result.

**R_1n_1(n, r)**
   Computes the first radial function for the first layer `R_1n_1`.

   :param n: Integer value for the index.
   :param r: Radial distance.
   :return: Computed value of `R_1n_1`.

**R_1n_2(n, r)**
   Computes the first radial function for the second layer `R_1n_2`.

   :param n: Integer value for the index.
   :param r: Radial distance.
   :return: Computed value of `R_1n_2`.

**diff_R_1n_1(n, r)**
   Differentiates the radial function `R_1n_1` with respect to radius `r`.

   :param n: Integer value for the index.
   :param r: Radial distance.
   :return: Differentiated result.

**diff_R_1n_2(n, r)**
   Differentiates the radial function `R_1n_2` with respect to radius `r`.

   :param n: Integer value for the index.
   :param r: Radial distance.
   :return: Differentiated result.

**R_2n_1(n)**
   Returns 0 as per the given equation for the second radial function of the first layer.

   :param n: Integer value for the index.
   :return: 0.

**R_2n_2(n, r)**
   Computes the second radial function for the second layer `R_2n_2`.

   :param n: Integer value for the index.
   :param r: Radial distance.
   :return: Computed value of `R_2n_2`.

**diff_R_2n_2(n, r)**
   Differentiates the radial function `R_2n_2` with respect to radius `r`.

   :param n: Integer value for the index.
   :param r: Radial distance.
   :return: Differentiated result.

**Z_n_i1(n, z)**
   Computes the `Z_n_i1` function for the first layer.

   :param n: Integer value for the index.
   :param z: Z-coordinate.
   :return: Computed value of `Z_n_i1`.

**Z_n_i2(n, z)**
   Computes the `Z_n_i2` function for the second layer.

   :param n: Integer value for the index.
   :param z: Z-coordinate.
   :return: Computed value of `Z_n_i2`.

**Lambda_k_r(k, r)**
   Computes the function `Lambda_k_r` using the Bessel functions.

   :param k: Integer value for the index.
   :param r: Radial distance.
   :return: Computed value of `Lambda_k_r`.

**diff_Lambda_k_a2(n)**
   Differentiates `Lambda_k_r` with respect to the radius `a2`.

   :param n: Integer value for the index.
   :return: Differentiated result.

**N_k(k)**
   Computes the function `N_k` based on the value of `k`.

   :param k: Integer value for the index.
   :return: Computed value of `N_k`.

**Z_n_e(k, z)**
   Computes the function `Z_n_e` for a given `k` and `z`.

   :param k: Integer value for the index.
   :param z: Z-coordinate.
   :return: Computed value of `Z_n_e`.

**diff_phi_p_i1_dz(z)**
   Differentiates `phi_p_i1` with respect to `z`.

   :param z: Z-coordinate.
   :return: Differentiated result.

**diff_phi_p_i2_dz(z)**
   Differentiates `phi_p_i2` with respect to `z`.

   :param z: Z-coordinate.
   :return: Differentiated result.

**int_R_1n_1(n)**
   Computes the integral of `R_1n_1` for a given index `n`.

   :param n: Integer value for the index.
   :return: Computed integral.

**int_R_1n_2(n)**
   Computes the integral of `R_1n_2` for a given index `n`.

   :param n: Integer value for the index.
   :return: Computed integral.

**int_R_2n_2(n)**
   Computes the integral of `R_2n_2` for a given index `n`.

   :param n: Integer value for the index.
   :return: Computed integral.

**int_phi_p_i1_no_coef()**
   Computes the integral of `phi_p_i1` without coefficients.

   :return: Computed integral.

**int_phi_p_i2_no_coef()**
   Computes the integral of `phi_p_i2` without coefficients.

   :return: Computed integral.

**z_n_d1_d2(n, d)**
   Computes the value of `z_n_d1_d2` for the given index `n` and parameter `d`.

   :param n: Integer value for the index.
   :param d: Parameter for the layer.
   :return: Computed value of `z_n_d1_d2`.
