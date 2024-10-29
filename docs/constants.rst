.. currentmodule:: package.constants

Constants
==========

This module defines several physical and mathematical constants used throughout the project.

Constants
----------

- **g**: Acceleration due to gravity (m/s²) — \(9.81\)
- **pi**: The mathematical constant π — \( \pi \)
- **h**: A project-specific constant — \(1.001\)
- **a1**: A project-specific constant — \(0.5\)
- **a2**: A project-specific constant — \(1\)
- **d1**: A project-specific constant — \(0.5\)
- **d2**: A project-specific constant — \(0.25\)
- **m0**: A project-specific constant — \(1\)
- **n**: A project-specific constant — \(3\)
- **z**: A project-specific constant — \(6\)
- **omega**: A project-specific constant — \(2\)

Usage Example
-------------

You can import and use these constants in your calculations as follows:

```python
from package.constants import g, pi  # Ensure to use the correct import path

# Example of using the gravitational constant
mass = 5.0  # Example mass in kg
force = mass * g
print("Force:", force)  # Output the force
