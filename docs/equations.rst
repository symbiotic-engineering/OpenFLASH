.. _equations-module:

===================
Equations Module
===================

.. automodule:: equations
   :members:
   :undoc-members: 
   :show-inheritance: 

.. _equations-overview:

Overview
========

This module (`equations.py`) contains the core mathematical functions and formulations derived from the theoretical basis of MEEM. It defines various potential functions, radial and vertical eigenfunctions, and helper functions necessary for constructing and solving problems. The functions are grouped by the equations they correspond to in the underlying theory (e.g., specific textbooks or paper equations).

.. _fundamental-definitions:

Fundamental Definitions
=======================

.. autofunction:: m_k
   :noindex:

   Calculates the wave numbers :math:`m_k` for the vertical eigenfunctions based on the dispersion relation.
   This function solves a transcendental equation to find the appropriate wave numbers.

   :param k: The mode index (integer, usually starting from 0 or 1).
   :type k: int
   :param m0: The incident wave number.
   :type m0: float
   :param h: The water depth.
   :type h: float
   :returns: The calculated wave number :math:`m_k`.
   :rtype: float
   :raises AssertionError: If internal checks for `m0 * m_k_val / np.pi - 0.5` fail, indicating a potential issue with the root finding.

.. _equation-4:

Equation 4: Eigenvalues :math:`\lambda_n`
=========================================

These functions define the eigenvalues for the inner (Region 1) and middle (Region 2) domains.

.. autofunction:: lambda_n1
   :noindex:

   Calculates the eigenvalue :math:`\lambda_n` for the inner fluid domain (Region 1).

   :param n: The mode index.
   :type n: int
   :param h: The total water depth.
   :type h: float
   :param d1: The depth of the inner cylinder.
   :type d1: float
   :returns: The eigenvalue :math:`\lambda_n`.
   :rtype: float

.. autofunction:: lambda_n2
   :noindex:

   Calculates the eigenvalue :math:`\lambda_n` for the middle fluid domain (Region 2).

   :param n: The mode index.
   :type n: int
   :param h: The total water depth.
   :type h: float
   :param d2: The depth of the outer cylinder.
   :type d2: float
   :returns: The eigenvalue :math:`\lambda_n`.
   :rtype: float

.. _equation-5:

Equation 5: Particular Potentials :math:`\phi_p`
=================================================

These functions define the particular solution potentials and their derivatives within different fluid domains. The particular solution accounts for the non-homogeneous boundary condition on the bottom of the heaving cylinders.

.. autofunction:: phi_p_a1
   :noindex:

   Calculates the particular potential at radius :math:`a_1` (inner cylinder radius).
   This is a convenience wrapper for :func:`phi_p_i1`.

   :param z: Vertical coordinate.
   :type z: float
   :param a1: Radius of the inner cylinder.
   :type a1: float
   :returns: The particular potential value.
   :rtype: float

.. autofunction:: phi_p_a2
   :noindex:

   Calculates the particular potential at radius :math:`a_2` (outer cylinder radius).
   This is a convenience wrapper for :func:`phi_p_i2`.

   :param z: Vertical coordinate.
   :type z: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The particular potential value.
   :rtype: float

.. autofunction:: phi_p_i1
   :noindex:

   Calculates the particular potential :math:`\phi_p` for the inner fluid domain (Region 1).

   :param r: Radial coordinate.
   :type r: float
   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :returns: The particular potential value in Region 1.
   :rtype: float

.. autofunction:: phi_p_i2
   :noindex:

   Calculates the particular potential :math:`\phi_p` for the middle fluid domain (Region 2).

   :param r: Radial coordinate.
   :type r: float
   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The particular potential value in Region 2.
   :rtype: float

.. autofunction:: phi_p_i1_i2_a1
   :noindex:

   Calculates the difference of particular solutions :math:`\phi_{p,i1} - \phi_{p,i2}` at radius :math:`a_1`.
   This is often used for matching conditions between domains.

   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The difference in particular potentials.
   :rtype: float

.. autofunction:: diff_phi_p_i2_a2
   :noindex:

   Calculates the radial derivative of the particular potential for Region 2 at radius :math:`a_2`.
   This represents the radial flux/velocity at the outer cylinder wall.

   :param h: Total water depth.
   :type h: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The radial derivative of the particular potential.
   :rtype: float

.. autofunction:: diff_phi_p_i1_i2_a1
   :noindex:

   Calculates the differentiation of the difference of particular solutions at radius :math:`a_1`.
   This is likely the radial derivative of :math:`\phi_{p,i1} - \phi_{p,i2}` at :math:`a_1`.

   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The radial derivative of the particular potential difference.
   :rtype: float

.. autofunction:: diff_phi_helper
   :noindex:

   A helper function for calculating radial derivatives of particular potentials.

   :param r: Radial coordinate.
   :type r: float
   :param di: depth for the specific domain (d1 or d2).
   :type di: float
   :param h: Total water depth.
   :type h: float
   :returns: The calculated derivative.
   :rtype: float

.. autofunction:: diff_phi_i1
   :noindex:

   Calculates the radial derivative of the particular potential for Region 1.
   A specialized call to :func:`diff_phi_helper`.

   :param r: Radial coordinate.
   :type r: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :param h: Total water depth.
   :type h: float
   :returns: The radial derivative.
   :rtype: float

.. autofunction:: diff_phi_i2
   :noindex:

   Calculates the radial derivative of the particular potential for Region 2.
   A specialized call to :func:`diff_phi_helper`.

   :param r: Radial coordinate.
   :type r: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :param h: Total water depth.
   :type h: float
   :returns: The radial derivative.
   :rtype: float

.. _equation-7:

Equation 7: Radial Functions :math:`R_{1n}` (Inner Domain, Region 1)
====================================================================

These functions define the radial eigenfunctions in the inner fluid domain (Region 1).

.. autofunction:: R_1n_1
   :noindex:

   Calculates the radial eigenfunction :math:`R_{1n}^{(1)}` for Region 1 (inner cylinder).
   Uses modified Bessel function of the first kind (:math:`I_0`).

   :param n: The mode index.
   :type n: int
   :param r: Radial coordinate.
   :type r: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :raises ValueError: If `n` is invalid (e.g., negative).
   :returns: The radial eigenfunction value.
   :rtype: float

.. autofunction:: R_1n_2
   :noindex:

   Calculates the radial eigenfunction :math:`R_{1n}^{(2)}` for Region 2 (middle cylinder).
   Uses modified Bessel function of the first kind (:math:`I_0`).

   :param n: The mode index.
   :type n: int
   :param r: Radial coordinate.
   :type r: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :raises ValueError: If `n` is invalid (e.g., negative).
   :returns: The radial eigenfunction value.
   :rtype: float

.. autofunction:: diff_R_1n_1
   :noindex:

   Calculates the radial derivative of the eigenfunction :math:`R_{1n}^{(1)}` for Region 1.
   Uses modified Bessel function of the first kind and its derivative (:math:`I_1`).

   :param n: The mode index.
   :type n: int
   :param r: Radial coordinate.
   :type r: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :param h: Total water depth.
   :type h: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :returns: The radial derivative of the eigenfunction.
   :rtype: float

.. autofunction:: diff_R_1n_2
   :noindex:

   Calculates the radial derivative of the eigenfunction :math:`R_{1n}^{(2)}` for Region 2.
   Uses modified Bessel function of the first kind and its derivative (:math:`I_1`).

   :param n: The mode index.
   :type n: int
   :param r: Radial coordinate.
   :type r: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :param h: Total water depth.
   :type h: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :returns: The radial derivative of the eigenfunction.
   :rtype: float

.. _equation-8:

Equation 8: Radial Functions :math:`R_{2n}` (Middle Domain, Region 2)
=====================================================================

These functions define the radial eigenfunctions in the middle fluid domain (Region 2), often involving Bessel functions of the second kind.

.. autofunction:: R_2n_2
   :noindex:

   Calculates the radial eigenfunction :math:`R_{2n}^{(2)}` for Region 2 (middle cylinder).
   Uses modified Bessel function of the second kind (:math:`K_0`) for :math:`n \ge 1`, and a logarithmic term for :math:`n=0`.

   :param n: The mode index.
   :type n: int
   :param r: Radial coordinate.
   :type r: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The radial eigenfunction value.
   :rtype: float

.. autofunction:: diff_R_2n_2
   :noindex:

   Calculates the radial derivative of the eigenfunction :math:`R_{2n}^{(2)}` for Region 2.
   Uses modified Bessel function of the second kind and its derivative (:math:`K_1`).

   :param n: The mode index.
   :type n: int
   :param r: Radial coordinate.
   :type r: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :param h: Total water depth.
   :type h: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :returns: The radial derivative of the eigenfunction.
   :rtype: float

.. _equation-9:

Equation 9: Vertical Functions :math:`Z_n` (Inner & Middle Domains)
===================================================================

These functions define the vertical eigenfunctions for the inner and middle fluid domains (Region 1 and 2).

.. autofunction:: Z_n_i1
   :noindex:

   Calculates the vertical eigenfunction :math:`Z_n^{(1)}` for the inner fluid domain (Region 1).

   :param n: The mode index.
   :type n: int
   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :returns: The vertical eigenfunction value.
   :rtype: float

.. autofunction:: Z_n_i2
   :noindex:

   Calculates the vertical eigenfunction :math:`Z_n^{(2)}` for the middle fluid domain (Region 2).

   :param n: The mode index.
   :type n: int
   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The vertical eigenfunction value.
   :rtype: float

.. _equation-13:

Equation 13: Radial Functions :math:`\Lambda_k` (Exterior Domain)
=================================================================

These functions define the radial eigenfunctions in the exterior fluid domain.

.. autofunction:: Lambda_k_r
   :noindex:

   Calculates the radial eigenfunction :math:`\Lambda_k` for the exterior fluid domain.
   Uses Hankel function of the first kind (:math:`H_0^{(1)}`) for :math:`k=0` and modified Bessel function of the second kind (:math:`K_0`) for :math:`k \ge 1`.

   :param k: The mode index.
   :type k: int
   :param r: Radial coordinate.
   :type r: float
   :param m0: The incident wave number.
   :type m0: float
   :param a2: Radius of the outer cylinder (reference radius for normalization).
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :returns: The radial eigenfunction value.
   :rtype: float or complex

.. autofunction:: diff_Lambda_k_a2
   :noindex:

   Calculates the radial derivative of the eigenfunction :math:`\Lambda_k` at radius :math:`a_2`.
   This is used for boundary conditions matching.

   :param n: The mode index (often `k` in this context, consider renaming for clarity).
   :type n: int
   :param m0: The incident wave number.
   :type m0: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :returns: The radial derivative value.
   :rtype: float or complex

.. _equation-16:

Equation 16 (or 2.34): Normalization Factor :math:`N_k`
=======================================================

This function defines the normalization factor used for the vertical eigenfunctions.

.. autofunction:: N_k
   :noindex:

   Calculates the normalization factor :math:`N_k` for the vertical eigenfunctions.

   :param k: The mode index.
   :type k: int
   :param m0: The incident wave number.
   :type m0: float
   :param h: Total water depth.
   :type h: float
   :returns: The normalization factor value.
   :rtype: float

.. _equation-14:

Equation 14: Vertical Functions :math:`Z_n^e` (Exterior Domain)
===============================================================

This function defines the vertical eigenfunctions for the exterior fluid domain.

.. autofunction:: Z_n_e
   :noindex:

   Calculates the vertical eigenfunction :math:`Z_n^e` for the exterior fluid domain.

   :param k: The mode index (often `n` in vertical eigenfunction context).
   :type k: int
   :param z: Vertical coordinate.
   :type z: float
   :param m0: The incident wave number.
   :type m0: float
   :param h: Total water depth.
   :type h: float
   :returns: The vertical eigenfunction value.
   :rtype: float

.. _hydrocoefficient-helpers:

Hydrodynamic Coefficient Calculation Helpers
============================================

These functions are used in the calculation of hydrodynamic coefficients (added mass and damping). They often involve derivatives or integrals of other potential and eigenfunction components.

.. autofunction:: diff_phi_p_i1_dz
   :noindex:

   Calculates the vertical derivative of the particular potential for Region 1.

   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :returns: The vertical derivative.
   :rtype: float

.. autofunction:: diff_phi_p_i2_dz
   :noindex:

   Calculates the vertical derivative of the particular potential for Region 2.

   :param z: Vertical coordinate.
   :type z: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The vertical derivative.
   :rtype: float

.. autofunction:: int_R_1n_1
   :noindex:

   Calculates the integral of the radial eigenfunction :math:`R_{1n}^{(1)}` (or a related term) for Region 1.
   Often involves integrating :math:`r R_{1n}^{(1)}` or similar.

   :param n: The mode index.
   :type n: int
   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :returns: The integral value.
   :rtype: float

.. autofunction:: int_R_1n_2
   :noindex:

   Calculates the integral of the radial eigenfunction :math:`R_{1n}^{(2)}` (or a related term) for Region 2.
   Often involves integrating :math:`r R_{1n}^{(2)}` or similar over an annular region.

   :param n: The mode index.
   :type n: int
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The integral value.
   :rtype: float

.. autofunction:: int_R_2n_2
   :noindex:

   Calculates the integral of the radial eigenfunction :math:`R_{2n}^{(2)}` (or a related term) for Region 2.

   :param n: The mode index.
   :type n: int
   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The integral value.
   :rtype: float

.. autofunction:: int_phi_p_i1_no_coef
   :noindex:

   Calculates an integral involving the particular potential :math:`\phi_{p,i1}` and its vertical derivative.
   This specific integral is part of the hydrodynamic coefficient calculation for the inner cylinder at its bottom.

   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param h: Total water depth.
   :type h: float
   :param d1: depth of the inner cylinder.
   :type d1: float
   :returns: The integral value (without the coefficient).
   :rtype: float

.. autofunction:: int_phi_p_i2_no_coef
   :noindex:

   Calculates an integral involving the particular potential :math:`\phi_{p,i2}` and its vertical derivative.
   This specific integral is part of the hydrodynamic coefficient calculation for the outer cylinder at its bottom.

   :param a1: Radius of the inner cylinder.
   :type a1: float
   :param a2: Radius of the outer cylinder.
   :type a2: float
   :param h: Total water depth.
   :type h: float
   :param d2: depth of the outer cylinder.
   :type d2: float
   :returns: The integral value (without the coefficient).
   :rtype: float

.. autofunction:: z_n_d1_d2
   :noindex:

   Calculates a specific vertical term related to mode :math:`n` at the interface between `d1` and `d2`.
   Often represents a value of `Z_n` at a specific depth for matching.

   :param n: The mode index.
   :type n: int
   :returns: The calculated term.
   :rtype: float