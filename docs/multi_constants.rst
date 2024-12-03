Multi Constants Module
=======================

This module contains the constants used in the simulation for hydrodynamic calculations.

Constants
---------

- **h**: `1.001`  
  The value of h used in the simulation.

- **d**: `[0.5, 0.25]`  
  A list of depths.

- **a**: `[0.5, 1]`  
  A list of amplitude values.

- **heaving**: `[1, 1]`  
  A list indicating whether each component is heaving.  
  `0` or `false` indicates not heaving, `1` or `true` indicates heaving.

- **m0**: `1`  
  The mass value.

- **g**: `9.81`  
  The acceleration due to gravity in m/s².

- **rho**: `1023`  
  The density of the fluid in kg/m³.

- **n**: `3`  
  The number of components in the model.

- **z**: `6`  
  A constant used in the model.

- **omega**: `2.734109632312753`  
  The angular frequency, calculated from `m0` and `g`.  
  Formula:  
  `omega = sqrt(m0 * g)`
