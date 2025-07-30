.. _coupling-module:

==================
Coupling Module
==================

.. automodule:: coupling
   :members:
   :undoc-members:
   :show-inheritance:

.. _coupling-overview:

Overview
========

This module (`coupling.py`) defines various coupling integrals and helper functions critical for solving the multi-domain problem using the MEEM (Multi-Domain Eigenfunction Expansion Method). These functions represent the integrals over the fluid boundaries, connecting different regions of the solution domain. They are fundamental in constructing the system of linear equations that determines the unknown coefficients of the potential solutions.

.. _coupling-functions:

Functions
=========

.. autofunction:: sq
   :noindex:

   Calculates the square of an input number.
   
   :param n: The number to be squared.
   :type n: float or int
   :returns: The square of `n`.
   :rtype: float or int

.. autofunction:: A_nm
   :noindex:

   Calculates the coupling integral :math:`A_{nm}`. This integral relates the inner and middle domains
   across a common boundary. The specific formula depends on the values of `n` and `m`.

   :param n: Integer index for one series term.
   :type n: int
   :param m: Integer index for another series term.
   :type m: int
   :raises ValueError: If `n` or `m` are invalid (e.g., negative).
   :returns: The calculated coupling integral value.
   :rtype: float or complex

.. autofunction:: A_nm2
   :noindex:

   Calculates the coupling integral :math:`A_{nm}` (alternative formulation).
   The specific formula depends on the values of `j` and `n`.

   :param j: Integer index for one series term.
   :type j: int
   :param n: Integer index for another series term.
   :type n: int
   :raises ValueError: If `j` or `n` are invalid.
   :returns: The calculated coupling integral value.
   :rtype: float or complex

.. autofunction:: A_nj
   :noindex:

   Calculates the coupling integral :math:`A_{nj}`. This integral often relates different domains or
   series expansions at an interface.

   :param n: Integer index for one series term.
   :type n: int
   :param j: Integer index for another series term.
   :type j: int
   :raises ValueError: If `n` or `j` are invalid.
   :returns: The calculated coupling integral value.
   :rtype: float or complex

.. autofunction:: A_nj2
   :noindex:

   Calculates the coupling integral :math:`A_{nj}` (alternative formulation).
   The specific formula depends on the values of `n` and `j`.

   :param n: Integer index for one series term.
   :type n: int
   :param j: Integer index for another series term.
   :type j: int
   :raises ValueError: If `n` or `j` are invalid.
   :returns: The calculated coupling integral value.
   :rtype: float or complex

.. autofunction:: nk_sigma_helper
   :noindex:

   A helper function that computes several intermediate sigma values used in the `A_mk` integral
   calculations.

   :param mk: A derived wave number, often from `m_k`.
   :type mk: float
   :param k: Integer index for a series term.
   :type k: int
   :param m: Integer index for another series term.
   :type m: int
   :returns: A tuple containing various precomputed sigma values (sigma1, sigma2, sigma3, sigma4, sigma5).
   :rtype: tuple[float, float, float, float, float]

.. autofunction:: A_mk
   :noindex:

   Calculates the coupling integral :math:`A_{mk}`. This integral is likely used to couple the
   outermost domain (exterior region) to an inner domain.

   :param m: Integer index for a series term.
   :type m: int
   :param k: Integer index for another series term.
   :type k: int
   :raises ValueError: If `m` or `k` are invalid.
   :returns: The calculated coupling integral value.
   :rtype: float or complex

.. autofunction:: nk2_sigma_helper
   :noindex:

   A helper function that computes several intermediate sigma values used in the `A_km2` integral
   calculations.

   :param mk: A derived wave number, often from `m_k`.
   :type mk: float
   :returns: A tuple containing various precomputed sigma values (sigma1, sigma2, sigma3, sigma4, sigma5).
   :rtype: tuple[float, float, float, float, float]

.. autofunction:: A_km2
   :noindex:

   Calculates the coupling integral :math:`A_{km}` (alternative formulation, possibly from exterior to interior).
   The specific formula depends on the values of `n` and `k`.

   :param n: Integer index for a series term.
   :type n: int
   :param k: Integer index for another series term.
   :type k: int
   :raises ValueError: If `n` or `k` are invalid.
   :returns: The calculated coupling integral value.
   :rtype: float or complex