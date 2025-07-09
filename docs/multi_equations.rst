.. _multi_equations:

======================
Multi-Region Equations
======================

The `multi_equations` module contains a comprehensive set of mathematical functions
and expressions central to MEEM Engine.
These functions define the fundamental components of the problem, including
eigenfunctions, coupling integrals, and terms for matrix assembly, particularly
adapted for multi-region configurations.

.. automodule:: multi_equations
   :members:

.. raw:: html

   <hr>

Common Computations
-------------------

This section includes general utility functions and fundamental computations
used across various parts of the MEEM formulation.

.. autofunction:: omega
.. autofunction:: scale
.. autofunction:: lambda_ni

.. raw:: html

   <hr>

m_k Wavenumber Computations
---------------------------

Functions related to the calculation of the wavenumber :math:`m_k`, which
are crucial for the exterior region's eigenfunctions.

.. autofunction:: m_k_entry
.. autofunction:: m_k
.. autofunction:: m_k_newton

.. raw:: html

   <hr>

Vertical Eigenvector Coupling Integrals
---------------------------------------

Functions that compute the coupling integrals between vertical eigenfunctions
in adjacent fluid regions.

.. autofunction:: I_nm
.. autofunction:: I_mk
.. autofunction:: I_mk_og

.. raw:: html

   <hr>

Right-Hand Side (b-vector) Terms
--------------------------------

Functions that contribute to the assembly of the right-hand side vector
(:math:`\mathbf{b}`), representing external excitations or boundary conditions.

.. autofunction:: b_potential_entry
.. autofunction:: b_potential_end_entry
.. autofunction:: b_velocity_entry
.. autofunction:: b_velocity_end_entry
.. autofunction:: b_velocity_end_entry_og

.. raw:: html

   <hr>

Particular Solution and Derivatives
-----------------------------------

Functions for the particular solution of the Laplace equation and its
derivatives, often related to incident wave potential.

.. autofunction:: phi_p_i
.. autofunction:: diff_r_phi_p_i
.. autofunction:: diff_z_phi_p_i

.. raw:: html

   <hr>

Radial Eigenfunctions (R1n, R2n, Lambda_k)
------------------------------------------

Functions defining the radial eigenfunctions for the interior (Bessel I type),
intermediate (Bessel K type), and exterior (Hankel and Bessel K type) regions.

.. autofunction:: R_1n
.. autofunction:: R_1n_vectorized
.. autofunction:: diff_R_1n
.. autofunction:: R_2n
.. autofunction:: R_2n_vectorized
.. autofunction:: diff_R_2n
.. autofunction:: Lambda_k
.. autofunction:: Lambda_k_vectoized
.. autofunction:: Lambda_k_og
.. autofunction:: diff_Lambda_k
.. autofunction:: diff_Lambda_k_og

.. raw:: html

   <hr>

Vertical Eigenfunctions (Zn, Zk)
--------------------------------

Functions defining the vertical eigenfunctions for the interior/intermediate
regions (:math:`Z_n^i`) and the exterior region (:math:`Z_k^e`).

.. autofunction:: N_k_multi
.. autofunction:: N_k_og
.. autofunction:: Z_n_i
.. autofunction:: Z_n_i_vectorized
.. autofunction:: diff_Z_n_i
.. autofunction:: Z_k_e
.. autofunction:: Z_k_e_vectorized
.. autofunction:: diff_Z_k_e

.. raw:: html

   <hr>

Hydrodynamic Coefficient Integrals
----------------------------------

Functions used in the calculation of hydrodynamic coefficients, involving
integrals of radial eigenfunctions and other terms.

.. autofunction:: int_R_1n
.. autofunction:: int_R_2n
.. autofunction:: int_phi_p_i_no_coef
.. autofunction:: z_n_d

.. raw:: html

   <hr>

Excitation Phase
----------------

Function to calculate the excitation phase.

.. autofunction:: excitation_phase