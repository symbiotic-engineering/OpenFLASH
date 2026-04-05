.. _problem_cache-module:

====================
Problem Cache Module
====================

.. automodule:: openflash.problem_cache

.. _problem_cache-overview:

Conceptual Overview
===================

The ``ProblemCache`` class is a crucial **internal component** designed to optimize the performance of the :class:`~openflash.meem_engine.MEEMEngine`. Its purpose is to store pre-calculated components of the mathematical system to avoid redundant computations, especially when solving a problem over a range of frequencies.

.. note::
   As an end-user of the OpenFLASH package, you will **not** need to interact with or create ``ProblemCache`` objects directly. The ``MEEMEngine`` automatically creates and manages a cache for each ``MEEMProblem`` instance it handles.

How it Works
------------

When the ``MEEMEngine`` is initialized with a problem, it builds a ``ProblemCache`` that:

1.  **Analyzes the System**: It identifies which parts of the governing matrices (**A**) and vectors (**b**) are constant (frequency-independent) and which parts change with the wave frequency.
2.  **Pre-computes Templates**: It calculates the frequency-independent parts once and stores them in "template" matrices.
3.  **Stores Calculation Logic**: For the frequency-dependent parts, it stores lightweight functions (closures) that can be quickly executed to calculate the values for any given frequency.

When a user requests a solution at a new frequency, the engine simply copies the pre-computed templates and runs the stored functions to fill in the missing pieces, rather than re-building the entire system from scratch. This caching strategy is the key to the engine's efficiency during frequency sweeps.