import numpy as np

import sys
import os
sys.path.append(os.path.relpath('../'))
from multi_condensed import Problem

import warnings
warnings.filterwarnings("ignore") # Inelegant, but gets rid of LinAlgWarnings

class ConvergenceProblem(Problem):
  # class should be instantiated with the same NMK/region, which will be used for "true value".
  # NMK will vary up to the amount determined by nmk_max, which should be less than that.

  def convergence_study(self, nmk_max):
    full_a_matrix = self.a_matrix()
    all_a_matrices = self.get_sub_matrices(full_a_matrix)
    omega = self.angular_freq(self.m0)
    output = {}
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
      output["ams" + str(i)] = am_lst
      output["dps" + str(i)] = dp_lst
      x = self.get_unknown_coeffs(full_a_matrix, full_b_vector)
      am, dp = self.hydro_coeffs(x, "capytaine")
      output["am" + str(i)] = am
      output["dp" + str(i)] = dp
    return output
  
  def convergence_study_over_m0s(self, nmk_max, m0s): # First m0 should be the one given in Problem instantiation
    full_a_matrix = self.a_matrix()
    output = {}
    b_vector_lst = []
    c_vector_lst = []
    all_c_vectors_lst = []
    for i in range(self.boundary_count):
      self.heaving = [1 if index == i else 0 for index in range(self.boundary_count)]
      b_vector_lst.append(self.b_vector())
      c_vector_lst.append(self.c_vector())
      all_c_vectors_lst.append(self.get_sub_vectors(c_vector_lst[i], False))
    
    for m0 in m0s:
      if m0 != self.m0:
          self.change_m0(m0)
          full_a_matrix = self.a_matrix_from_old(full_a_matrix)
          for i in range(self.boundary_count):
            self.heaving = [1 if index == i else 0 for index in range(self.boundary_count)]
            b_vector_lst = [self.b_vector_from_old(b_vector) for b_vector in b_vector_lst]
      all_a_matrices = self.get_sub_matrices(full_a_matrix)
      omega = self.angular_freq(self.m0)

      for i in range(self.boundary_count):
        out_for_m0 = {}
        self.heaving = [1 if index == i else 0 for index in range(self.boundary_count)]
        full_b_vector = b_vector_lst[i]
        full_c_vector = c_vector_lst[i]
        all_b_vectors = self.get_sub_vectors(full_b_vector, True)
        particular_contribution = self.int_phi_p_i(i) # only region i is heaving
        am_lst = []
        dp_lst = []
        for nmk in range(1, nmk_max + 1):
          x = self.get_unknown_coeffs(all_a_matrices[nmk - 1], all_b_vectors[nmk - 1])
          raw_hydro = 2 * np.pi * (np.dot(full_c_vector[nmk - 1], x[:-nmk]) + particular_contribution)
          # follow the capytaine convention
          am_lst.append(raw_hydro.real * self.rho) # added mass
          dp_lst.append(raw_hydro.imag * omega * self.rho) # damping
        out_for_m0["ams" + str(i)] = am_lst
        out_for_m0["dps" + str(i)] = dp_lst
        x = self.get_unknown_coeffs(full_a_matrix, full_b_vector)
        am, dp = self.hydro_coeffs(x, "capytaine")
        out_for_m0["am" + str(i)] = am
        out_for_m0["dp" + str(i)] = dp
      output[m0] = out_for_m0
    return output
    

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