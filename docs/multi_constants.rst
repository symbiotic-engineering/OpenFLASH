.. _multi_constants:

=================
Multi-Constants
=================

This module defines fundamental physical constants that are utilized throughout OpenFLASH.
These constants are set to common default values, but can be overridden by the user
if different physical conditions are required.

.. automodule:: multi_constants
   :members:

Constants Reference
-------------------

Here is a detailed reference for the global constants defined in this module:

.. py:data:: g
   :type: float
   :value: 9.81

   The acceleration due to gravity, in meters per second squared.
   Default value: :math:`g = 9.81` m/s².

.. py:data:: rho
   :type: float
   :value: 1023.0

   The density of water (typically seawater), in kilograms per cubic meter.
   Default value: :math:`\rho = 1023` kg/m³.

.. note::
   Parameters such as water depth, geometric dimensions, wave properties (e.g., wavenumber), and motion types are specific to a given hydrodynamic problem. These should be defined when constructing or configuring your OpenFLASH problem instance.