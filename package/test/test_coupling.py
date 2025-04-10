import unittest
import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import scipy.integrate as integrate
import scipy.linalg as linalg
from math import sqrt, cosh, cos, sinh, sin, pi
from scipy.optimize import newton, minimize_scalar
import scipy as sp  # Import scipy correctly
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))  # Path to /package/src
sys.path.append(src_path)

from constants import *  
from equations import *  
import coupling

class TestCouplingIntegrals(unittest.TestCase):

    def test_A_nm(self):
        # Test cases for A_nm
        self.assertAlmostEqual(coupling.A_nm(0, 0), h - d1)  # Example: Test for n=0, m=0
        self.assertAlmostEqual(coupling.A_nm(1, 0), 0)  # Example: Test for n=1, m=0
        self.assertAlmostEqual(coupling.A_nm(0, 1), (-sqrt(2) * sin((pi * 1 * (d1 - h)) / (d2 - h)) * (d2 - h)) / (1 * pi)) #Example: Test for n=0, m=1
        self.assertAlmostEqual(coupling.A_nm(1, 1), -2 * ((-1) ** 1) * 1 * sin((pi * 1 * (d1 - h)) / (d2 - h)) * (d1 - h)**2 * (d2 - h) / (pi * ((d1**2 * 1**2) - (2 * d1 * h * 1**2) - (d2**2 * 1**2) + (2 * d2 * h * 1**2) + (h**2 * 1**2) - (h**2 * 1**2)))) #Example: Test for n=1, m=1
        # Add more test cases, especially edge cases and values that might cause issues

    def test_A_nm2(self):
        # Test cases for A_nm2
        self.assertAlmostEqual(coupling.A_nm2(0, 0), h - d2)
        self.assertAlmostEqual(coupling.A_nm2(1, 0), (-sqrt(2) * sin(pi * 1) * (d2 - h)) / (1 * pi))
        self.assertAlmostEqual(coupling.A_nm2(0, 1), (-sqrt(2) * sin((pi * 1 * (d2 - h)) / (d1 - h)) * (d1 - h)) / (1 * pi))
        # Add more test cases


    # ... (Add tests for nk2_sigma_helper )


if __name__ == '__main__':
    unittest.main()