import numpy as np
from scipy.special import hankel1 as besselh
from scipy.special import iv as besseli
from scipy.special import kv as besselk
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize_scalar
import scipy as sp
from constants import *
from equations import *

#############################################
# Coupling integrals: (m_k is a function)
def sq(n):
    return np.power(n, 2)

def A_nm(n, m):
    """Vectorized version of A_nm."""
    n = np.asarray(n)
    m = np.asarray(m)
    result = np.zeros_like(n, dtype=float)

    # Case 1: n == 0 and m == 0
    mask1 = (n == 0) & (m == 0)
    result[mask1] = h - d1

    # Case 2: m == 0 and 1 <= n
    mask2 = (m == 0) & (n >= 1)
    result[mask2] = 0

    # Case 3: 1 <= m and n == 0
    mask3 = (m >= 1) & (n == 0)
    sigma3 = np.sin((np.pi * m[mask3] * (d1 - h)) / (d2 - h))
    result[mask3] = (-np.sqrt(2) * sigma3 * (d2 - h)) / (m[mask3] * np.pi)

    # Case 4: 1 <= m and 1 <= n
    mask4 = (m >= 1) & (n >= 1)
    sigma4 = np.sin((np.pi * m[mask4] * (d1 - h)) / (d2 - h))
    top4 = -2 * ((-1) ** n[mask4]) * m[mask4] * sigma4 * sq(d1 - h) * (d2 - h)
    bottom4 = np.pi * ((sq(d1) * sq(m[mask4])) - (2 * d1 * h * sq(m[mask4])) - (sq(d2) * sq(n[mask4])) + (2 * d2 * h * sq(n[mask4])) + (sq(h) * sq(m[mask4])) - (sq(h) * sq(n[mask4])))
    result[mask4] = top4 / bottom4

    return result

def A_nm2(j, n):
    """Vectorized version of A_nm2."""
    j = np.asarray(j)
    n = np.asarray(n)
    result = np.zeros_like(j, dtype=float)

    # Case 1: j == 0 and n == 0
    mask1 = (j == 0) & (n == 0)
    result[mask1] = h - d2

    # Case 2: j == 0 and 1 <= n
    mask2 = (j == 0) & (n >= 1)
    sigma2 = (np.pi * n[mask2] * (d2 - h)) / (d1 - h)
    result[mask2] = (-np.sqrt(2) * np.sin(sigma2) * (d1 - h)) / (n[mask2] * np.pi)

    # Case 3: 1 <= j and n == 0
    mask3 = (j >= 1) & (n == 0)
    result[mask3] = (-np.sqrt(2) * np.sin(np.pi * j[mask3]) * (d2 - h)) / (j[mask3] * np.pi)

    # Case 4: 1 <= j and 1 <= n
    mask4 = (j >= 1) & (n >= 1)
    sigma4 = (np.pi * n[mask4] * (d2 - h)) / (d1 - h)
    top4 = -2 * (j[mask4] * (d1 - h) * (d2 - h) * (d1 * np.sin(np.pi * j[mask4]) * np.cos(sigma4) - h * np.sin(np.pi * j[mask4]) * np.cos(sigma4)) -
                  n[mask4] * (d1 - h) * (d2 - h) * (d2 * np.sin(sigma4) * np.cos(np.pi * j[mask4]) - h * np.sin(sigma4) * np.cos(np.pi * j[mask4])))
    bottom4 = np.pi * ((sq(d1) * sq(j[mask4])) - (2 * d1 * h * sq(j[mask4])) - (sq(d2) * sq(n[mask4])) + (2 * d2 * h * sq(n[mask4])) + (sq(h) * sq(j[mask4])) - (sq(h) * sq(n[mask4])))
    result[mask4] = top4 / bottom4

    return result

def A_nj(n, j):
    """Vectorized version of A_nj."""
    n = np.asarray(n)
    j = np.asarray(j)
    result = np.zeros_like(n, dtype=float)

    # Case 1: j == 0 and n == 0
    mask1 = (j == 0) & (n == 0)
    result[mask1] = h - d1

    # Case 2: 1 <= j and n == 0
    mask2 = (j >= 1) & (n == 0)
    result[mask2] = (-np.sqrt(2) * np.sin(np.pi * j[mask2]) * (d1 - h)) / (j[mask2] * np.pi)

    # Case 3: j == 0 and 1 <= n
    mask3 = (j == 0) & (n >= 1)
    sigma3 = (np.pi * n[mask3] * (d1 - h)) / (d2 - h)
    result[mask3] = (-np.sqrt(2) * np.sin(sigma3) * (d2 - h)) / (n[mask3] * np.pi)

    # Case 4: 1 <= j and 1 <= n
    mask4 = (j >= 1) & (n >= 1)
    sigma4 = (np.pi * n[mask4] * (d1 - h)) / (d2 - h)
    top4 = (2 * (j[mask4] * (d1 - h) * (d2 - h) * (d2 * np.sin(np.pi * j[mask4]) * np.cos(sigma4) - h * np.sin(np.pi * j[mask4]) * np.cos(sigma4)) -
                  n[mask4] * (d1 - h) * (d2 - h) * (d2 * np.sin(sigma4) * np.cos(np.pi * j[mask4]) - h * np.sin(sigma4) * np.cos(np.pi * j[mask4]))))
    bottom4 = np.pi * ((-sq(d1) * sq(n[mask4])) + (2 * d1 * h * sq(n[mask4])) + (sq(d2) * sq(j[mask4])) - (2 * d2 * h * sq(j[mask4])) + (sq(h) * sq(j[mask4])) - (sq(h) * sq(n[mask4])))
    result[mask4] = -(top4 / bottom4)

    return result

def A_nj2(n, j):
    """Vectorized version of A_nj2."""
    n = np.asarray(n)
    j = np.asarray(j)
    result = np.zeros_like(n, dtype=float)

    # Case 1: j == 0 and n == 0
    mask1 = (j == 0) & (n == 0)
    result[mask1] = h - d2

    # Case 2: 1 <= j and n == 0
    mask2 = (j >= 1) & (n == 0)
    sigma2 = (np.pi * j[mask2] * (d2 - h)) / (d1 - h)
    result[mask2] = -(np.sqrt(2) * np.sin(sigma2) * (d1 - h)) / (j[mask2] * np.pi)

    # Case 3: j == 0 and 1 <= n
    mask3 = (j == 0) & (n >= 1)
    result[mask3] = -(np.sqrt(2) * np.sin(np.pi * n[mask3]) * (d2 - h)) / (n[mask3] * np.pi)

    # Case 4: 1 <= j and 1 <= n
    mask4 = (j >= 1) & (n >= 1)
    sigma4 = (np.pi * j[mask4] * (d2 - h)) / (d1 - h)
    top4 = -(2 * (j[mask4] * (d1 - h) * (d2 - h) * (d2 * np.sin(sigma4) * np.cos(np.pi * n[mask4]) - h * np.sin(sigma4) * np.cos(np.pi * n[mask4])) -
                   n[mask4] * (d1 - h) * (d2 - h) * (d1 * np.sin(np.pi * n[mask4]) * np.cos(sigma4) - h * np.sin(np.pi * n[mask4]) * np.cos(sigma4))))
    bottom4 = np.pi * ((-sq(d1) * sq(n[mask4])) + (2 * d1 * h * sq(n[mask4])) + (sq(d2) * sq(j[mask4])) - (2 * d2 * h * sq(j[mask4])) + (sq(h) * sq(j[mask4])) - (sq(h) * sq(n[mask4])))
    result[mask4] = top4 / bottom4

    return result

def nk_sigma_helper(mk, k, m):
    sigma1 = np.sqrt(np.sinh(2 * h * m0) + 2 * h * m0 / h)
    sigma2 = np.sin(mk * (d2 - h))
    sigma3 = np.pi ** 2 * m ** 2
    sigma4 = np.sinh(m0 * (d2 - h))
    sigma6 = 2 * h * mk
    sigma5 = np.sqrt(np.sin(sigma6) / sigma6 + 1)
    return sigma1, sigma2, sigma3, sigma4, sigma5

def A_mk(m, k):
    """Vectorized version of A_mk."""
    m = np.asarray(m)
    k = np.asarray(k)
    result = np.zeros_like(m, dtype=float)
    mk_val = np.array([m_k(ki) for ki in k])  # Calculate m_k for each k

    sigma1, sigma2, sigma3, sigma4, sigma5 = nk_sigma_helper(mk_val, k, m)

    # Case 1: k == 0 and m == 0
    mask1 = (k == 0) & (m == 0)
    result[mask1] = -2 * sigma4[mask1] / (np.sqrt(m0) * sigma1[mask1])

    # Case 2: 1 <= k and m == 0
    mask2 = (k >= 1) & (m == 0)
    result[mask2] = -np.sqrt(2) * sigma2[mask2] / (mk_val[mask2] * sigma5[mask2])

    # Case 3: k == 0 and 1 <= m
    mask3 = (k == 0) & (m >= 1)
    result[mask3] = -(2 * ((-1) ** m[mask3]) * m0**(3/2) * sigma4[mask3] * sq(d2 - h)) / \
                   (sigma1[mask3] * (sq(d2) * sq(m0) - 2 * d2 * h * sq(m0) + sq(h) * sq(m0) + sigma3[mask3]))

    # Case 4: 1 <= k and 1 <= m
    mask4 = (k >= 1) & (m >= 1)
    result[mask4] = -(2 * ((-1) ** m[mask4]) * sigma2[mask4] * mk_val[mask4] * sq(d2 - h)) / \
                   (sigma5[mask4] * (sq(d2) * sq(mk_val[mask4]) - 2 * d2 * h * sq(mk_val[mask4]) + sq(h) * sq(mk_val[mask4]) - sigma3[mask4]))

    return result

def nk2_sigma_helper(mk, n):
    top = np.sin(2 * h * mk)
    bottom = 4 * h * mk
    sigma1 = np.sqrt(top / bottom + 0.5)
    sigma2 = np.sin(h * mk)
    sigma3 = (np.pi * h * n) / (d2 - h)
    sigma4 = sq(np.pi) * sq(n)
    sigma5 = np.sinh(2 * h * m0)
    return sigma1, sigma2, sigma3, sigma4, sigma5

def A_km2(n, k):
    """Vectorized version of A_km2."""
    n = np.asarray(n)
    k = np.asarray(k)
    result = np.zeros_like(n, dtype=float)
    mk_val = np.array([m_k(ki) for ki in k])  # Calculate m_k for each k

    sigma1, sigma2, sigma3, sigma4, sigma5 = nk2_sigma_helper(mk_val, n)

    # Case 1: k == 0 and n == 0
    mask1 = (k == 0) & (n == 0)
    result[mask1] = (-2 * np.sqrt(h) * np.sinh(h * m0)) / (np.sqrt(m0) * np.sqrt(sigma5[mask1] + 2 * h * m0))

    # Case 2: 1 <= k and n == 0
    mask2 = (k >= 1) & (n == 0)
    result[mask2] = sigma2[mask2] / (mk_val[mask2] * sigma1[mask2])

    # Case 3: k == 0 and 1 <= n
    mask3 = (k == 0) & (n >= 1)
    top3 = np.sqrt(2) * (m0 * (d2 - h) * (d2 * np.sinh(h * m0) * np.cos(sigma3[mask3]) - h * np.sinh(h * m0) * np.cos(sigma3[mask3])) +
                       np.pi * n[mask3] * np.cosh(h * m0) * np.sin(sigma3[mask3]) * (d2 - h))
    bottom3 = np.sqrt((sigma5[mask3] / (4 * h * m0)) + 0.5) * (sq(d2) * sq(m0) - 2 * d2 * h * sq(m0) + sq(h) * sq(m0) + sigma4[mask3])
    result[mask3] = top3 / bottom3

    # Case 4: 1 <= k and 1 <= n
    mask4 = (k >= 1) & (n >= 1)
    top4 = np.sqrt(2) * (mk_val[mask4] * (d2 - h) * (d2 * sigma2[mask4] * np.cos(sigma3[mask4]) - h * sigma2[mask4] * np.cos(sigma3[mask4])) -
                       np.pi * n[mask4] * np.cos(h * mk_val[mask4]) * np.sin(sigma3[mask4]) * (d2 - h))
    bottom4 = sigma1[mask4] * (sq(d2) * sq(mk_val[mask4]) - 2 * d2 * h * sq(mk_val[mask4]) + sq(h) * sq(mk_val[mask4]) - sigma4[mask4])
    result[mask4] = top4 / bottom4

    return result