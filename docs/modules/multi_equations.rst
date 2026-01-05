.. _multi_equations-module:

======================
Mathematical Equations
======================

.. automodule:: openflash.multi_equations

.. _multi_equations-overview:

Conceptual Overview
===================

The ``multi_equations`` module is the mathematical heart of the OpenFLASH package. It contains the Python implementations of the core analytical functions required for the Matched Eigenfunction Expansion Method (MEEM), including radial and vertical eigenfunctions, their derivatives, coupling integrals, and terms for constructing the final linear system.

.. warning::
   Most functions in this module are low-level mathematical components used internally by the :class:`~openflash.meem_engine.MEEMEngine`. The average user will typically only need to interact with the **User-Facing Utility Functions** listed below. The other sections are provided for developers and researchers interested in the underlying mathematical theory.

.. _multi_equations-user-api:

User-Facing Utility Functions
=============================

These are high-level helper functions that you may need to use when setting up a simulation.

.. autofunction:: openflash.multi_equations.omega
.. autofunction:: openflash.multi_equations.wavenumber

.. _multi_equations-core-api:

Core Mathematical Components
============================

This section details the core mathematical building blocks of the MEEM formulation. These functions are primarily called by the ``MEEMEngine`` during the matrix assembly process.

Wavenumber Computations
-----------------------
These functions determine the wavenumbers for the exterior fluid domain.

.. autofunction:: openflash.multi_equations.m_k_entry
.. autofunction:: openflash.multi_equations.lambda_ni

Coupling Integrals
------------------
These functions compute the integrals that couple the vertical eigenfunctions at the boundaries between adjacent fluid regions.

.. autofunction:: openflash.multi_equations.I_nm
.. autofunction:: openflash.multi_equations.I_mk

Radial Eigenfunctions
---------------------
These functions define the radial variation of the potential in each type of fluid domain. They include the functions themselves, their derivatives, and optimized vectorized versions used for post-processing.

.. rubric:: Interior Regions (Bessel I)

.. autofunction:: openflash.multi_equations.R_1n
.. autofunction:: openflash.multi_equations.diff_R_1n
.. autofunction:: openflash.multi_equations.R_1n_vectorized
.. autofunction:: openflash.multi_equations.diff_R_1n_vectorized

.. rubric:: Intermediate Regions (Bessel K)

.. autofunction:: openflash.multi_equations.R_2n
.. autofunction:: openflash.multi_equations.diff_R_2n
.. autofunction:: openflash.multi_equations.R_2n_vectorized
.. autofunction:: openflash.multi_equations.diff_R_2n_vectorized

.. rubric:: Exterior Region (Hankel & Bessel K)

.. autofunction:: openflash.multi_equations.Lambda_k
.. autofunction:: openflash.multi_equations.diff_Lambda_k
.. autofunction:: openflash.multi_equations.Lambda_k_vectorized
.. autofunction:: openflash.multi_equations.diff_Lambda_k_vectorized


Vertical Eigenfunctions
-----------------------
These functions define the vertical variation of the potential in each type of fluid domain.

.. rubric:: Interior & Intermediate Regions

.. autofunction:: openflash.multi_equations.Z_n_i
.. autofunction:: openflash.multi_equations.diff_Z_n_i
.. autofunction:: openflash.multi_equations.Z_n_i_vectorized
.. autofunction:: openflash.multi_equations.diff_Z_n_i_vectorized

.. rubric:: Exterior Region

.. autofunction:: openflash.multi_equations.N_k_multi
.. autofunction:: openflash.multi_equations.Z_k_e
.. autofunction:: openflash.multi_equations.diff_Z_k_e
.. autofunction:: openflash.multi_equations.Z_k_e_vectorized
.. autofunction:: openflash.multi_equations.diff_Z_k_e_vectorized


Particular Solution & Hydrodynamic Terms
----------------------------------------
These functions are related to the non-homogeneous parts of the solution and the final calculation of physical coefficients.

.. autofunction:: openflash.multi_equations.phi_p_i
.. autofunction:: openflash.multi_equations.diff_r_phi_p_i
.. autofunction:: openflash.multi_equations.diff_z_phi_p_i
.. autofunction:: openflash.multi_equations.int_R_1n
.. autofunction:: openflash.multi_equations.int_R_2n
.. autofunction:: openflash.multi_equations.excitation_force