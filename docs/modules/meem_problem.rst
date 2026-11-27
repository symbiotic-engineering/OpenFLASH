.. _meem_problem-module:

===================
MEEM Problem Module
===================

.. automodule:: openflash.meem_problem

.. _meem_problem-overview:

Conceptual Overview
===================

The ``MEEMProblem`` class is a fundamental container in the OpenFLASH workflow. Its primary role is to bundle a fully defined **geometry** with the specific **simulation parameters** you want to investigate.

Think of it as the complete "job description" for a simulation run. It holds two key pieces of information:

1.  **What to Simulate:** A :class:`~openflash.geometry.Geometry` object that describes the physical layout of all the bodies and the surrounding fluid.
2.  **How to Simulate It:** The specific wave **frequencies** and **modes of motion** (e.g., which bodies are heaving) that the :class:`~openflash.meem_engine.MEEMEngine` should solve for.

You create a `MEEMProblem` instance and then pass it to the `MEEMEngine` to perform the calculations.

.. _meem_problem-usage:

Example Usage
=============

Creating and configuring a ``MEEMProblem`` is a straightforward process.

.. code-block:: python

   from openflash import MEEMProblem, BasicRegionGeometry
   import numpy as np

   # --- Assume 'geometry' is an already created BasicRegionGeometry object ---
   # geometry = BasicRegionGeometry(...)

   # 1. Create the problem instance with the geometry
   problem = MEEMProblem(geometry)

   # 2. Define the simulation parameters
   frequencies_to_run = np.array([1.5, 2.0, 2.5]) # Frequencies in rad/s

   # 3. Configure the problem with these parameters
   problem.set_frequencies(frequencies_to_run)

   # The 'problem' object is now ready to be passed to the MEEMEngine.


.. _meem_problem-api:

API Reference
=============

.. autoclass:: openflash.meem_problem.MEEMProblem
   :members: set_frequencies
   :undoc-members:
   :show-inheritance: