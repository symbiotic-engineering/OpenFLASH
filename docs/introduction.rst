.. _introduction:

==============
Introduction
==============

Welcome to **OpenFLASH**, a modern Python package for solving wave-body interaction problems for concentric cylindrical structures.

OpenFLASH implements the **Matched Eigenfunction Expansion Method (MEEM)**, a powerful semi-analytical technique that offers significant performance advantages over traditional numerical methods like the Boundary Element Method (BEM). The package is designed to be user-friendly, efficient, and easily extensible for researchers, engineers, and students in marine hydrodynamics.

---

Why OpenFLASH?
--------------

* **High Performance**: Leveraging the semi-analytical nature of MEEM, OpenFLASH is exceptionally fast. Our benchmarks show that it can compute hydrodynamic coefficients up to **10 times faster** than leading open-source BEM packages like Capytaine, especially for frequency sweep analyses. This speed is achieved by an intelligent caching system that minimizes redundant calculations.

* **Intuitive, Object-Oriented API**: Define your physical problem intuitively by creating ``SteppedBody`` objects. The library's object-oriented structure handles the complex task of translating this physical geometry into the mathematical fluid domains required for the solver, making your code cleaner and more readable.

* **Structured Data Output**: Simulation results are not just raw numbers. OpenFLASH packages all outputs into an ``xarray.Dataset``, a powerful data structure that provides labeled dimensions (like 'frequencies', 'modes') and coordinates. This makes your data self-describing, easy to analyze with tools like Pandas, and simple to export to standard scientific formats like NetCDF.

---

Quick Example
-------------

Here is a minimal example of setting up and solving a two-body problem:

.. code-block:: python

   import numpy as np
   from openflash import (
       SteppedBody, ConcentricBodyGroup, BasicRegionGeometry,
       MEEMProblem, MEEMEngine, omega, g
   )

   # 1. Define the physical bodies
   body1 = SteppedBody(a=np.array([5.0]), d=np.array([20.0]), heaving=True)
   body2 = SteppedBody(a=np.array([10.0]), d=np.array([10.0]), heaving=False)

   # 2. Create the geometry from the bodies
   arrangement = ConcentricBodyGroup(bodies=[body1, body2])
   geometry = BasicRegionGeometry(
       body_arrangement=arrangement,
       h=100.0,
       NMK=[30, 30, 30]
   )

   # 3. Set up the problem with simulation parameters
   problem = MEEMProblem(geometry)
   problem.set_frequencies(
       frequencies=np.array([omega(m0=1.0, h=100.0, g=g)]),
   )

   # 4. Run the engine and get results
   engine = MEEMEngine(problem_list=[problem])
   results = engine.run_and_store_results(problem_index=0)

   # 5. Analyze the output
   print(results.get_results())


---

Target Audience
---------------
This documentation is intended for:

* Researchers and students in ocean engineering and marine hydrodynamics.

* Engineers working on wave energy converters or offshore platforms.

* Developers interested in contributing to or extending the codebase.

---

Getting Started
---------------
To begin using OpenFLASH, we recommend the following steps:

1.  **Installation**: Follow the instructions in the :doc:`installation` guide to set up the package in a virtual environment.
2.  **Run the Tutorial**: Walk through the `Jupyter Notebook Tutorial <tutorial.html>`_ for a hands-on, interactive example of a full simulation.
3.  **Explore the Web App**: Try the interactive Streamlit application by following the :doc:`app_walk` guide.
4.  **API Reference**: For detailed information on specific classes and functions, refer to the individual module documentation listed in the sidebar.