import sys
import os
sys.path.append(os.path.relpath('../'))
from multi_condensed import Problem
import numpy as np
import pickle

######################################
class SProblem(Problem):
  def __init__(self, h, d, a, heaving, NMK, m0, rho, slopes, scale = None):
    self.h = h
    self.d = d
    self.a = a
    self.heaving = heaving
    self.NMK = NMK
    self.m0 = m0
    self.rho = rho
    self.scale = a if scale is None else scale
    self.size = NMK[0] + NMK[-1] + 2 * sum(NMK[1:len(NMK) - 1])
    self.boundary_count = len(NMK) - 1
    self.m_k = self.m_k_array()
    self.N_k = self.N_k_array()
    self.slopes = slopes
    self.thetas = self.get_angles()
  
  def get_angles(self):
    def arccot(x):
      if x == 0: return np.pi/2
      else: return np.atan(1/x)
    return [arccot(slope) for slope in self.slopes]

  def det_region(self, r):
    region = 0
    for i in range(self.boundary_count):
      if r <= self.a[i]:
        return region
      else: region += 1
    return region

######################################
# staircase with outline on exterior corners
def make_slant_region1(d1, d2, a1, a2, res):
  a_prime = []
  d_prime = []
  delta_d = (d2 - d1)/res
  delta_a = (a2 - a1)/res
  offset = (delta_d < 0)
  for i in range(res):
     a_prime.append(a1 + (1 + i) * delta_a)
     d_prime.append(d1 + (offset + i) * delta_d)
  return a_prime, d_prime

# staircase with outlines through centers, starting horizontal, end vertical
def make_slant_region2(d1, d2, a1, a2, res):
  a_prime = []
  d_prime = []
  delta_d = (d2 - d1)/res
  delta_a = (a2 - a1)/res
  # offset = (delta_d < 0)
  for i in range(res):
     a_prime.append(a1 + (0.5 + i) * delta_a)
     d_prime.append(d1 + (i) * delta_d)
  a_prime.append(a2)
  d_prime.append(d2)
  return a_prime, d_prime

# staircase with outlines through centers, starting vertical, end horizontal
def make_slant_region3(d1, d2, a1, a2, res):
  a_prime = []
  d_prime = []
  delta_d = (d2 - d1)/res
  delta_a = (a2 - a1)/res
  # offset = (delta_d < 0)
  for i in range(res):
     a_prime.append(a1 + (1 + i) * delta_a)
     d_prime.append(d1 + (0.5 + i) * delta_d)
  return a_prime, d_prime

# Get d and a to make a staircase
def slant_approx_vars(a, d_in, d_out, heaving, NMK, res, version):
  if version == 1:
     make_slant_region = make_slant_region1
  elif version == 2:
     make_slant_region = make_slant_region2
  elif version == 3:
     make_slant_region = make_slant_region3
  else:
     raise ValueError
  
  a_prime = []
  d_prime = []
  heaving_prime = []
  NMK_prime = []
  slopes = []
  for i in range(len(a)):
    if d_in[i] == d_out[i]: # horizontal region
        a_prime.append(a[i])
        d_prime.append(d_in[i])
        heaving_prime.append(heaving[i])
        NMK_prime.append(NMK[i])
        slopes.append(0)
    else: # slanted region
       heaving_prime += ([heaving[i]] * res)
       NMK_prime += ([NMK[i]] * res)
       a_inner = 0 if i == 0 else a[i - 1]
       a_add, d_add = make_slant_region(d_in[i], d_out[i], a_inner, a[i], res)
       a_prime += a_add
       d_prime += d_add
       slope = (d_in[i]-d_out[i])/(a[i] - a_inner)
       slopes += ([slope] * res)
  NMK_prime.append(NMK[-1])
  return d_prime, a_prime, heaving_prime, NMK_prime, slopes

######################################
def solve_problem(prob_style, h, a, d_in, d_out, heaving, m0, rho, res, version, nmk = 150, NMK = None):
  if NMK is None: NMK = [nmk for _ in range(len(a) + 1)]
  d_prime, a_prime, heaving_prime, NMK_prime, slopes = slant_approx_vars(a, d_in, d_out, heaving, NMK, res, version)
  if prob_style is Problem: prob = prob_style(h, d_prime, a_prime, heaving_prime, NMK_prime, m0, rho)
  else: prob = prob_style(h, d_prime, a_prime, heaving_prime, NMK_prime, m0, rho, slopes)
  a_matrix = prob.a_matrix()
  b_vector = prob.b_vector()
  x = prob.get_unknown_coeffs(a_matrix, b_vector)
  cs = prob.reformat_coeffs(x)
  return x, cs, prob

def plot_both(prob, cs):
  prob.plot_potentials(cs)
  prob.plot_velocities(cs)

######################################
def update_data_file(data, name):
  with open(name, "wb") as f:
    pickle.dump(data, f)
