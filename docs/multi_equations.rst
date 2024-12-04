Multi Equations 
================

This module, `multi_equations.py`, contains a variety of functions for performing computations related to multi-region eigenfunctions, vertical eigenvector coupling, Bessel functions, and other related mathematical operations. It leverages libraries such as `numpy`, `scipy`, and `matplotlib` for efficient scientific computing.

Imports
-------

The following libraries are imported:

- `numpy` (as `np`): For numerical computations, including arrays and mathematical operations.
- `scipy.special`: For special functions such as Bessel functions (`hankel1`, `iv`, `kv`).
- `scipy.integrate`: For numerical integration.
- `scipy.linalg`: For linear algebra functions.
- `matplotlib.pyplot`: For plotting graphs.
- `scipy.optimize`: For optimization routines.
- `multi_constants`: Custom constants, such as `a`, `h`, and `m0`, used across the equations.

Functions
---------

### wavenumber(omega)
Calculates the wavenumber for a given frequency (`omega`), using a root-finding method to solve for `m0` using the equation provided. This function is vital for determining the spatial frequency characteristics of waves.

#### Parameters:
- `omega` (float): The frequency for which the wavenumber is to be calculated.

#### Returns:
- `float`: The calculated wavenumber corresponding to the given frequency.

---

### eigenfunction(x, m0)
Computes the eigenfunction for a given point `x` using the specified `m0` parameter. This function is essential for generating solutions to the multi-region problem.

#### Parameters:
- `x` (float): The spatial coordinate at which to evaluate the eigenfunction.
- `m0` (float): A parameter related to the system's eigenvalue.

#### Returns:
- `float`: The value of the eigenfunction evaluated at `x`.

---

### vertical_eigenvector_coupling(m0, m1, z)
Calculates the vertical coupling between two eigenvectors, indexed by `m0` and `m1`, at a given height `z`. This coupling plays a role in multi-region solutions where eigenvectors interact at different vertical levels.

#### Parameters:
- `m0` (float): The first eigenvector index.
- `m1` (float): The second eigenvector index.
- `z` (float): The vertical height where the coupling is evaluated.

#### Returns:
- `float`: The vertical coupling value at the specified height.

---

### bessel_jv(n, x)
Computes the Bessel function of the first kind of order `n` at point `x`. This is used in problems involving cylindrical symmetry, such as wave propagation in cylindrical coordinates.

#### Parameters:
- `n` (int): The order of the Bessel function.
- `x` (float): The point at which to evaluate the Bessel function.

#### Returns:
- `float`: The value of the Bessel function at the specified point.

---

### bessel_hankel1(n, x)
Computes the Hankel function of the first kind of order `n` at point `x`. This function is used for modeling wave propagation in open space, particularly for radiation problems.

#### Parameters:
- `n` (int): The order of the Hankel function.
- `x` (float): The point at which to evaluate the Hankel function.

#### Returns:
- `complex`: The value of the Hankel function at the specified point.

---

### bessel_iv(n, x)
Computes the modified Bessel function of the first kind of order `n` at point `x`. This function is often used in problems involving heat conduction or diffusion in cylindrical geometries.

#### Parameters:
- `n` (int): The order of the Bessel function.
- `x` (float): The point at which to evaluate the Bessel function.

#### Returns:
- `float`: The value of the modified Bessel function at the specified point.

---

### bessel_kv(n, x)
Computes the modified Bessel function of the second kind of order `n` at point `x`. This function is often used in problems involving wave propagation or thermal conduction in cylindrical coordinates.

#### Parameters:
- `n` (int): The order of the Bessel function.
- `x` (float): The point at which to evaluate the Bessel function.

#### Returns:
- `float`: The value of the modified Bessel function at the specified point.

---

### solve_integral(func, a, b)
Numerically integrates a function `func` over the interval `[a, b]` using scipy's `quad` method. This is used for computing integrals that arise in eigenvalue problems or coupling coefficients.

#### Parameters:
- `func` (function): The function to be integrated.
- `a` (float): The lower bound of the integration interval.
- `b` (float): The upper bound of the integration interval.

#### Returns:
- `float`: The result of the integration.

---

### optimize_function(func, x0)
Optimizes a function `func` starting from an initial guess `x0` using scipy's optimization routines. This is useful for solving for parameters like eigenvalues or minimizing error functions.

#### Parameters:
- `func` (function): The function to be optimized.
- `x0` (float): The initial guess for the optimization.

#### Returns:
- `float`: The optimal value that minimizes the given function.

---

### plot_eigenfunction(x_vals, eigenfunction_vals)
Plots the eigenfunction values against the corresponding spatial coordinates using `matplotlib`. This function is useful for visualizing the solution to the eigenfunction problem.

#### Parameters:
- `x_vals` (array-like): A list or array of spatial coordinates.
- `eigenfunction_vals` (array-like): A list or array of eigenfunction values at the corresponding coordinates.

#### Returns:
- `None`: Displays the plot.

---

### constants_setup()
Initializes and sets up the constants required for solving the equations, such as `a`, `h`, and `m0`. These constants are used across various functions in the module for consistency and efficiency.

#### Returns:
- `None`: Initializes constants in the module's namespace.

---

Conclusion
----------

This module is designed to handle a variety of operations related to multi-region eigenfunction problems, providing a solid foundation for solving complex mathematical models in physics, engineering, and related fields. By utilizing standard scientific computing libraries like `numpy`, `scipy`, and `matplotlib`, this module provides efficient and accurate tools for solving problems involving waves, eigenvectors, and special functions.
