.. _multi_constants:

=================
Multi-Constants
=================

This module defines various global constants that are utilized throughout OpenFLASH. These include
physical parameters, geometric dimensions, and settings that
can be adjusted for different problem configurations.

.. automodule:: multi_constants
   :members:

Constants Reference
-------------------

Here is a detailed reference for the global constants defined in this module:

.. py:data:: h
   :type: float
   :value: 1.001

   The total water depth in meters. :math:`h = 1.001` m.

.. py:data:: d
   :type: list[float]
   :value: [0.5, 0.25, 0.25]

   A list of depths for each cylindrical domain's structure, in meters.
   The order corresponds to the domains from innermost to outermost.
   For example, `d = [0.5, 0.25, 0.25]` indicates the inner domain's depth is 0.5m,
   and subsequent domains have a depth of 0.25m.

.. py:data:: a
   :type: list[float]
   :value: [0.5, 1, 1]

   A list of radii for each cylindrical domain's structure, in meters.
   The order corresponds to the domains from innermost to outermost.
   For example, `a = [0.5, 1, 1]` indicates the inner domain's radius is 0.5m,
   and subsequent domains have a radius of 1m.

.. py:data:: heaving
   :type: list[int]
   :value: [1, 1, 1]

   A list indicating whether each domain's structure is experiencing a heaving motion.
   A value of `1` (true) indicates heaving, while `0` (false) indicates no heaving.
   `heaving = [1, 1, 1]` indicates all structures are heaving.

.. py:data:: m0
   :type: float
   :value: 1

   Wavenumber in radians per meter.
   It is a key parameter. :math:`m_0 = 1` rad/m.

.. py:data:: g
   :type: float
   :value: 9.81

   The acceleration due to gravity, in meters per second squared.
   :math:`g = 9.81` m/s².

.. py:data:: rho
   :type: float
   :value: 1023

   The density of water (typically seawater), in kilograms per cubic meter.
   :math:`\rho = 1023` kg/m³.

.. py:data:: n
   :type: int
   :value: 3

   An integer parameter.

.. py:data:: z
   :type: int
   :value: 6

   An integer parameter.

.. py:data:: omega
   :type: float
   :value: 2.734109632312753

   The angular frequency, in radians per second.