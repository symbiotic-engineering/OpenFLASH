import numpy as np

import sys
import os
sys.path.append(os.path.relpath('../../'))
from multi_condensed import Problem

import h5py
from itertools import product

import warnings
warnings.filterwarnings("ignore") # Inelegant, but gets rid of LinAlgWarnings

class ConvergenceProblem(Problem):
  # class should be instantiated with the same NMK/region, which will be the maximum allowed amount.
  # NMK will vary up to that, or until appropriate convergence is determined to have been reached.

  def convergence_study(self, f, group_path):
    nmk_max = self.NMK[0]
    full_a_matrix = self.a_matrix()
    all_a_matrices = self.get_sub_matrices(full_a_matrix)
    omega = self.angular_freq(self.m0) * self.rho
    grp = f.require_group(group_path)
    for i in range(self.boundary_count):
      am_lst = []
      dp_lst = []
      self.heaving = [1 if index == i else 0 for index in range(self.boundary_count)]
      full_b_vector = self.b_vector()
      all_b_vectors = self.get_sub_vectors(full_b_vector, True)
      full_c_vector = self.c_vector()
      all_c_vectors = self.get_sub_vectors(full_c_vector, False)
      particular_contribution = self.int_phi_p_i(i) # only region i is heaving
      for nmk in range(1, nmk_max + 1):
        x = self.get_unknown_coeffs(all_a_matrices[nmk - 1], all_b_vectors[nmk - 1])
        raw_hydro = 2 * np.pi * (np.dot(all_c_vectors[nmk - 1], x[:-nmk]) + particular_contribution)
        # follow the capytaine convention
        am_lst.append(raw_hydro.real * self.rho) # added mass
        dp_lst.append(raw_hydro.imag * omega * self.rho) # damping
        try:
            if self.stopping_point(am_lst, dp_lst):
                break
        except Exception as e:
            print(f"Error in stopping_point: {e}, nmk = {nmk}")
            break
      grp.create_dataset("ams" + str(i), data=am_lst)
      grp.create_dataset("dps" + str(i), data=dp_lst)
    

  def get_sub_matrices(self, full_a_matrix):
    nmk_max = self.NMK[0]
    block_dimension = self.boundary_count * 2
    all_a_matrices = [np.zeros((block_dimension * nmk, block_dimension * nmk), dtype=complex) for nmk in range(1, nmk_max + 1)]
    for i in range(block_dimension):
      for j in range(block_dimension):
        block = full_a_matrix[i * nmk_max : (i+1) * nmk_max, j * nmk_max : (j+1) * nmk_max]
        for nmk in range(1, nmk_max + 1):
          sub_block = block[:nmk, :nmk]
          all_a_matrices[nmk - 1][i * nmk : (i+1) * nmk, j * nmk : (j+1) * nmk] = sub_block
    return all_a_matrices

  def get_sub_vectors(self, full_vector, exterior):
    # exterior: whether or not the vector contains information about the exterior region
    # True: b-vector style. False: c-vector style.
    nmk_max = self.NMK[0]
    block_dimension = self.boundary_count * 2 - (not exterior)
    all_vectors = [np.zeros(block_dimension * nmk, dtype=complex) for nmk in range(1, nmk_max + 1)]
    for i in range(block_dimension):
      block = full_vector[i * nmk_max : (i+1) * nmk_max]
      for nmk in range(1, nmk_max + 1):
        sub_block = block[:nmk]
        all_vectors[nmk - 1][i * nmk : (i+1) * nmk] = sub_block
    return all_vectors

  def c_vector(self):
    heaving, NMK, boundary_count = self.heaving, self.NMK, self.boundary_count
    c = np.zeros((self.size - NMK[-1]), dtype=complex)
    col = 0
    for n in range(NMK[0]):
        c[n] = heaving[0] * self.int_R_1n(0, n)* self.z_n_d(n)
    col += NMK[0]
    for i in range(1, boundary_count):
        M = NMK[i]
        for m in range(M):
            c[col + m] = heaving[i] * self.int_R_1n(i, m)* self.z_n_d(m)
            c[col + M + m] = heaving[i] * self.int_R_2n(i, m)* self.z_n_d(m)
        col += 2 * M
    return c

  def stopping_point(self, am_lst, dp_lst):
    if len(am_lst) < 6: return False
    for i in range(1, 4): # Apply to the most recent 3 values
      if abs((am_lst[-i] - am_lst[-i-3])/am_lst[-i-3]) > 0.001:
        return False # added mass value differs from one 3 before it by less than .1%
      if abs((dp_lst[-i] - dp_lst[-i-3])/am_lst[-i-3]) > 0.001:
        return False
    else: return True # suitably converged

# Actual Runtime Code
heights = [1, 10, 50, 100]
depth_fractions = [0.01, 0.1, 0.5]
radial_widths = [0.1, 1, 5]
m0 = 1
rho = 1023

domains = [heights] + [depth_fractions] * 4 + [radial_widths] * 4
all_configs = product(*domains)

def format_val(val):
    if isinstance(val, float):
        # Round to 1 significant figure
        if val == 0:
            return "0"
        else:
            return f"{val:.1g}"
    else:
        return str(val)

with h5py.File("data/dg_out.h5", "a") as f:
  for idx, config in enumerate(product(*domains)):
    h = config[0]
    d = [config[i] * h for i in range(1, 5)]
    nmk = 100
    a = [config[5]]
    for i in range(6, 9):
      high_val = int(210 * (h - d[i - 6]) / config[i - 1])
      nmk = min(nmk, high_val)
      a.append(a[-1] + config[i])
    high_val = int(210 * (h - d[3]) / config[8])
    nmk = min(nmk, high_val)
    NMK = [nmk] * 5
    group_path = "/".join(f"v{i}_{format_val(val)}" for i, val in enumerate(config))
    prob = ConvergenceProblem(h, d, a, [1, 1, 1, 1], NMK, m0, rho)
    prob.convergence_study(f, group_path)
    if idx % 1000 == 0:
      print(f"Progress: {idx}")