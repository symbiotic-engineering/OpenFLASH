.. currentmodule:: package.constants

Constants
==========

This module defines several physical and mathematical constants used throughout the project.

Constants
----------

- **g**: Acceleration due to gravity (m/s²) — \(9.81\)
- **pi**: The mathematical constant π — \( \pi \)
- **h**: A project-specific constant related to the problem’s domain. 
- **a1**: A project-specific constant that parameterizes aspects of the geometry or conditions in the problem.
- **a2**: A project-specific constant that parameterizes aspects of the geometry or conditions in the problem.
- **d1**: Additional constant used in defining boundary conditions or specific points in the domain.
- **d2**: Additional constants used in defining boundary conditions or specific points in the domain.
- **m0**: Represents a base or initial parameter, often used as a reference value in the calculations. 
- **n**: A constant representing a fixed integer value, often used for indexing or defining specific scenarios in the model. 
- **z**: A placeholder constant, often representing a vertical or depth-related dimension within the domain.
- **omega**: Represents a frequency or angular velocity, typically associated with wave or rotational phenomena.

Usage Example
-------------

You can import and use these constants in your calculations as follows:

```python
from constants import g, pi
```

# Example of using the gravitational constant

mass = 5.0  # Example mass in kg

force = mass * g

print("Force:", force)
