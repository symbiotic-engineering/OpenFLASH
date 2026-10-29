.. _results-module:

==============
Results Module
==============

.. automodule:: openflash.results

.. _results-overview:

Conceptual Overview
===================

The ``Results`` class is the primary container for storing, managing, and exporting all outputs from an OpenFLASH simulation. It is built on top of the powerful `xarray` library, which provides labeled, multi-dimensional arrays, making the data self-describing and easy to work with.

When you run a simulation, especially a frequency sweep using the :meth:`~openflash.meem_engine.MEEMEngine.run_and_store_results` method, the engine will return a fully populated ``Results`` object.

Key Features
------------

* **Structured Data:** All data is stored in an ``xarray.Dataset`` with named dimensions (like 'frequencies', 'modes', 'r', 'z') and coordinates, eliminating ambiguity.
* **Comprehensive Storage:** Capable of storing key outputs, including hydrodynamic coefficients (added mass, damping) and detailed spatial field data (potentials, velocities).
* **NetCDF Export:** Provides a simple method to export the entire dataset to a NetCDF (`.nc`) file, a standard format for scientific data that preserves the data's structure and labels.

.. _results-usage:

Example Usage
=============

The most common workflow involves receiving a ``Results`` object from the ``MEEMEngine`` and then accessing or exporting its data.

.. code-block:: python

   from openflash import MEEMEngine, MEEMProblem
   import numpy as np

   # --- Assume 'engine' and 'problem' are already configured ---
   # problem.set_frequencies(np.linspace(0.5, 4.0, 50))

   # 1. Run the simulation to get a populated Results object
   results = engine.run_and_store_results(problem_index=0)

   # 2. Access the underlying xarray.Dataset
   dataset = results.get_results()
   print("--- Accessing Added Mass Data ---")
   print(dataset['added_mass'])

   # 3. Display a summary of the dataset
   print("\n--- Dataset Summary ---")
   results.display_results()

   # 4. Export all results to a file
   results.export_to_netcdf("my_simulation_output.nc")
   print("\nResults saved to my_simulation_output.nc")


.. _results-api:

API Reference
=============

.. autoclass:: openflash.results.Results

   Data Storage Methods
   --------------------
   These methods are used by the ``MEEMEngine`` to populate the dataset. Users typically do not need to call these directly.

   .. automethod:: store_hydrodynamic_coefficients
   .. automethod:: store_all_potentials

   Data Access and Export
   ------------------------
   These methods are the primary public interface for interacting with a populated ``Results`` object.

   .. automethod:: get_results
   .. automethod:: display_results
   .. automethod:: export_to_netcdf