.. _multi_constants-module:

=================
Constants Module
=================

.. automodule:: openflash.multi_constants

.. _multi_constants-overview:

Conceptual Overview
===================

This module defines fundamental physical constants that are utilized throughout the OpenFLASH package. These constants are set to common default values for hydrodynamic simulations in seawater.

.. _multi_constants-api:

API Reference
=============

.. py:data:: g
   :type: float
   :value: 9.81

   The acceleration due to gravity, in meters per second squared ($m/s^2$).

.. py:data:: rho
   :type: float
   :value: 1023.0

   The density of water, in kilograms per cubic meter ($kg/m^3$). The default value is typical for seawater.