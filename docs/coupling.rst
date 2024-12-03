Coupling
========

Overview
--------

The `coupling` module provides functions for calculating coupling integrals used in the **Matched Eigenfunctions Method (MEEM)**. The primary functions, such as `A_nm`, `A_nm2`, `A_nj`, and `A_nj2`, compute various integral transformations. These calculations are vital for accurately modeling coupled systems in the MEEM framework.

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

   - **ValueError**: Raised if invalid indices are provided.

.. code-block:: python

   # Example usage
   result = A_nm(2, 3)


.. _A_nm2:

A_nm2
-----

.. function:: A_nm2(j, n)
   
   Computes the integral `A_nm2`, applying specific transformations to indices `j` and `n`.

   **Parameters:**

   - **j** (*int*): First index.
   - **n** (*int*): Second index.

   **Returns:**

   - (*float*): Result of the `A_nm2` integral calculation.

.. _A_nj:

A_nj
----

.. function:: A_nj(n, j)
   
   Calculates the integral `A_nj`, essential for coupling calculations.

   **Parameters:**

   - **n** (*int*): First index in the coupling integral.
   - **j** (*int*): Second index in the coupling integral.

   **Returns:**

   - (*float*): Calculated integral for the given indices.

.. _A_nj2:

A_nj2
-----

.. function:: A_nj2(n, j)
   
   Computes the integral `A_nj2`, with additional transformations for `n` and `j`.

   **Parameters:**

   - **n** (*int*): First index.
   - **j** (*int*): Second index.

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


.. _nk_sigma_helper:

nk_sigma_helper
---------------

.. function:: nk_sigma_helper(mk, k, m)
   
   A helper function for `A_nm` and related calculations, handling specific transformations for values of `mk`, `k`, and `m`.

   **Parameters:**

   - **mk** (*float*): Coupled variable derived from `m_k`.
   - **k** (*int*): Index in the transformation.
   - **m** (*int*): Index in the transformation.

   **Returns:**

   - (*tuple*): Transformed components used in further calculations.

Examples
--------

The following example demonstrates a workflow using functions from the `coupling` module:

.. code-block:: python

   from coupling import A_nm, A_nm2, A_nj, sq

   # Calculating A_nm and A_nm2
   result_nm = A_nm(2, 3)
   result_nm2 = A_nm2(4, 5)

   # Using A_nj in further calculations
   result_nj = A_nj(3, 6)

Dependencies
------------

The following external libraries are required:

- **NumPy**: For numerical operations.
- **SciPy**: For integral transformations and mathematical operations.

Notes
-----

Ensure all inputs fall within valid ranges to avoid `ValueError`. Index values are expected to be integers.
