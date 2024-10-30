Coupling
========

Overview
--------

The `coupling` module provides functions for calculating coupling integrals used in the **Matched Eigenfunctions Method (MEEM)**. The primary functions, such as `A_nm`, `A_nm2`, `A_nj`, and others, compute various integral transformations based on the indices provided. This module is essential for performing precise mathematical calculations required for coupled systems.

Function Definitions
--------------------

.. _A_nm:

A_nm
----

.. function:: A_nm(n, m)
   
   Computes the integral `A_nm`, representing coupling integrals between indices `n` and `m`.

   **Parameters:**

   - **n** (*int*): First index in the coupling integral calculation.
   - **m** (*int*): Second index in the coupling integral calculation.

   **Returns:**

   - (*float*): Result of the integral for the given indices `n` and `m`.
   
   **Raises:**

   - **ValueError**: Raised if `n` or `m` are out of the expected range.

.. code-block:: python

   # Example usage
   result = A_nm(2, 3)

   # Result will be a float representing the calculated integral


.. _A_nm2:

A_nm2
-----

.. function:: A_nm2(n, m)
   
   Similar to `A_nm`, this function computes the integral `A_nm2`, with specific transformations applied to indices `n` and `m`.

   **Parameters:**

   - **n** (*int*): First index.
   - **m** (*int*): Second index.

   **Returns:**

   - (*float*): Result of the `A_nm2` integral calculation.


.. _A_nj:

A_nj
----

.. function:: A_nj(n, j)
   
   Calculates the integral `A_nj`, which is essential in the coupling calculations.

   **Parameters:**

   - **n** (*int*): First index in the coupling integral.
   - **j** (*int*): Second index in the coupling integral.

   **Returns:**

   - (*float*): Calculated integral for the given indices.


Helper Functions
----------------

.. _sq:

sq
--

.. function:: sq(x)
   
   Squares the input number `x`.

   **Parameters:**

   - **x** (*float*): Number to be squared.

   **Returns:**

   - (*float*): Result of squaring `x`.

.. code-block:: python

   # Example usage
   result = sq(5)
   # result = 25


.. _nk_sigma_helper:

nk_sigma_helper
---------------

.. function:: nk_sigma_helper(n, k)
   
   A helper function that aids in `A_nm` calculations by handling specific transformations for values of `n` and `k`.

   **Parameters:**

   - **n** (*int*): First index.
   - **k** (*int*): Second index.

   **Returns:**

   - (*float*): Transformed result based on `n` and `k` inputs.


Examples
--------

The following example demonstrates a simple workflow using functions from the `coupling` module to perform a coupling integral calculation:

.. code-block:: python

   # Importing functions from the coupling module
   from coupling import A_nm, A_nm2, A_nj

   # Calculating A_nm and A_nm2
   result_nm = A_nm(2, 3)
   result_nm2 = A_nm2(4, 5)

   # Using A_nj in further calculations
   result_nj = A_nj(3, 6)
   
   # Example results would be of type float based on input indices

Dependencies
------------

The following external libraries are required:

- **NumPy**: Provides numerical operations essential for the calculations.
- **SciPy**: Used for integral transformations and other mathematical operations.

Notes
-----

Ensure that inputs `n` and `m` fall within acceptable ranges to avoid `ValueError`. Some functions in this module assume integer indices for accurate calculations.

