.. _meem_engine-module:

===========
MEEM Engine
===========

.. automodule:: openflash.meem_engine
   :no-members:

.. _meem_engine-overview:

Conceptual Overview
===================

The ``MEEMEngine`` is the central processing unit of the OpenFLASH package. It takes one or more :class:`~openflash.meem_problem.MEEMProblem` objects and orchestrates the entire simulation process, from assembling the mathematical system to calculating the final physical results.

The engine is designed around an internal caching system (``ProblemCache``) that significantly optimizes performance, especially when running simulations over multiple frequencies. It pre-calculates parts of the system that are frequency-independent and stores functions to efficiently compute the frequency-dependent parts.

Primary Workflows
-----------------

There are two primary ways to use the ``MEEMEngine``:

1.  **Single Frequency Analysis**: This workflow is ideal for detailed inspection of the system at a single wave frequency. The user typically calls :meth:`~solve_linear_system_multi` to get the solution vector, then uses that vector with post-processing methods like :meth:`~calculate_potentials` or :meth:`~calculate_velocities` to generate spatial field data for visualization.

2.  **Frequency Sweep Analysis**: This is the most common and powerful workflow. The user configures a ``MEEMProblem`` with a range of frequencies and then makes a single call to the :meth:`~run_and_store_results` method. The engine handles the entire loop internally, solving the system for each frequency and packaging all the hydrodynamic coefficients into a convenient :class:`~openflash.results.Results` object.

.. _meem_engine-api:

API Reference
=============

.. autoclass:: openflash.meem_engine.MEEMEngine

   Core Solver Methods
   -------------------
   These are the main methods for running simulations.

   .. automethod:: solve_linear_system_multi

   .. automethod:: run_and_store_results


   Post-Processing & Analysis Methods
   ----------------------------------
   These methods are used after solving the system to compute physical quantities.

   .. automethod:: compute_hydrodynamic_coefficients

   .. automethod:: calculate_potentials

   .. automethod:: calculate_velocities


   Utility & Visualization Methods
   -------------------------------

   .. automethod:: reformat_coeffs

   .. automethod:: visualize_potential