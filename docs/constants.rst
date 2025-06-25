.. _constants-module:

===================
Constants Module
===================

.. automodule:: constants
   :members:
   :undoc-members:
   :show-inheritance:

.. _constants-overview:

Overview
========

This module (`constants.py`) defines various physical and mathematical constants used throughout the MEEM (Multiple Expansion Eigenfunction Method) simulation project. These values serve as default or fundamental parameters for the calculations.

.. _constants-details:

Defined Constants
=================

Below are the constants defined in this module:

.. autodata:: g
   :annotation: = 9.81

   Acceleration due to gravity in meters per second squared ($m/s^2$).

.. autodata:: pi
   :annotation: = numpy.pi

   The mathematical constant pi, sourced from NumPy for high precision.

.. autodata:: h
   :annotation: = 1.001

   Total water depth in meters.

.. autodata:: a1
   :annotation: = 0.5

   Radius of the first (innermost) cylinder in meters.

.. autodata:: a2
   :annotation: = 1.0

   Radius of the second cylinder in meters.

.. autodata:: d1
   :annotation: = 0.5

   Submerged depth of the first (innermost) cylinder in meters.

.. autodata:: d2
   :annotation: = 0.25

   Submerged depth of the second cylinder in meters.


.. autodata:: m0
   :annotation: = 1

   Radial wave number.

.. autodata:: n
   :annotation: = 3

   Related to a mode or term index.

.. autodata:: z
   :annotation: = 6

   Related to a vertical coordinate or an integer value.

.. autodata:: omega
   :annotation: = 2

   Angular frequency of the incident wave in radians per second.