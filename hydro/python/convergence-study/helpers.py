# Common functions used by all the notebooks in studying i-region convergence

import math
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from scipy import stats


import sys
import os
sys.path.append(os.path.relpath('../../'))
from multi_condensed import Problem

import warnings
warnings.filterwarnings("ignore") # removing condition number warnings

# The relevant ConvergenceProblemI class
class ConvergenceProblemI(Problem):
  # Input has a single heaving region.
  # Varies NMK only over that region.

  def change_m0_mk(self, m0, mk):
      self.m0 = m0
      self.m_k = mk

  def convergence_study_over_m0s(self, nmk_max, m0s, mks):
    region = self.heaving.index(1)
    full_a_matrix = self.a_matrix()
    full_b_vector = self.b_vector()
    full_c_vector = self.c_vector()
    all_c_vectors = self.get_c_vectors(full_c_vector, nmk_max, region)
    output = {}
    
    for idx, m0 in enumerate(m0s):
      out_for_m0 = {}
      if m0 != self.m0:
          self.change_m0_mk(m0, mks[idx])
          full_a_matrix = self.a_matrix_from_old(full_a_matrix)
          full_b_vector = self.b_vector_from_old(full_b_vector)
      all_a_matrices = self.get_sub_matrices(full_a_matrix, nmk_max, region)
      all_b_vectors = self.get_b_vectors(full_b_vector, nmk_max, region)
      particular_contribution = self.int_phi_p_i(region) # only region i is heaving
      omega = self.angular_freq(self.m0)
      am_lst = []
      dp_lst = []
      for nmk in range(1, nmk_max + 1):
        x = self.get_unknown_coeffs(all_a_matrices[nmk - 1], all_b_vectors[nmk - 1])
        raw_hydro = 2 * np.pi * (np.dot(all_c_vectors[nmk - 1], x[:-self.NMK[-1]]) + particular_contribution)
        # follow the capytaine convention
        am_lst.append(raw_hydro.real * self.rho) # added mass
        dp_lst.append(raw_hydro.imag * omega * self.rho) # damping
      out_for_m0["ams"] = am_lst
      out_for_m0["dps"] = dp_lst
      x = self.get_unknown_coeffs(full_a_matrix, full_b_vector)
      am, dp = self.hydro_coeffs(x, "capytaine")
      out_for_m0["am"] = am
      out_for_m0["dp"] = dp
      output[m0] = out_for_m0
    return output
  
  def cut_rows(self, full_arr, region, big_nmk, nmk):
      arr = np.asarray(full_arr)
      left_potential_changed = False
      # Left boundary
      if region == 0: # innermost region
        pass # No changes
      else: # Has a left boundary
        if self.d[region] < self.d[region - 1]: # Taller fluid, change velocity
          before = big_nmk * ((region - 1) + self.boundary_count)
        else: # Shorter fluid, change potentials
          before = big_nmk * (region - 1)
          left_potential_changed = True
        arr = np.delete(arr, np.s_[before + nmk : before + big_nmk], axis = 0)

      # Right boundary
      if region == (self.boundary_count - 1): # outermost body region
        # no comparison necessary, alter potential
        if left_potential_changed:
          before = big_nmk * (region - 1) + nmk
        else:
          before = big_nmk * region
        pass
      else:
        if self.d[region] < self.d[region + 1]: # Taller fluid, change velocity
          if region == 0: # vector not previously changed
            before = big_nmk * (region + self.boundary_count)
          else:
            before = big_nmk * (region + self.boundary_count - 1) + nmk
        else: # Shorter fluid, change potentials
          if left_potential_changed:
            before = big_nmk * (region - 1) + nmk
          else:
            before = big_nmk * region
      arr = np.delete(arr, np.s_[before + nmk : before + big_nmk], axis = 0)
      return arr
  
  def cut_cols(self, full_arr, region, big_nmk, nmk):
    arr = np.asarray(full_arr)
    if arr.ndim == 1:
      ax = 0
    else:
      ax = 1
    if region == 0: # no before, and only 1 set of terms
      arr = np.delete(arr, np.s_[nmk:big_nmk], axis=ax)
    else: # 2 sets of terms, and some before terms
      before = big_nmk * (2 * region - 1)
      arr = np.delete(arr, np.s_[before + big_nmk + nmk:before + 2 * big_nmk], axis=ax)
      arr = np.delete(arr, np.s_[before + nmk:before + big_nmk], axis=ax)
    return arr


  def get_sub_matrices(self, full_a_matrix, nmk_max, region):
    big_nmk = self.NMK[0]
    all_a_matrices = []

    for nmk in range(1, nmk_max + 1):
      # Cut out the columns
      arr = self.cut_cols(full_a_matrix, region, big_nmk, nmk)
      # Cut out the rows
      arr = self.cut_rows(arr, region, big_nmk, nmk)
      all_a_matrices.append(arr)
    return all_a_matrices

  def get_b_vectors(self, full_vector, nmk_max, region):
    big_nmk = self.NMK[0]
    all_vectors = []

    for nmk in range(1, nmk_max + 1):
      vec = self.cut_rows(full_vector, region, big_nmk, nmk)
      all_vectors.append(vec)

    return all_vectors
  
  def get_c_vectors(self, full_vector, nmk_max, region):
    big_nmk = self.NMK[0]
    all_vectors = []

    for nmk in range(1, nmk_max + 1):
      vec = self.cut_cols(full_vector, region, big_nmk, nmk)
      all_vectors.append(vec)

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
  
###################################
# Data processing functions

def merge_dicts(dict1, dict2): # Creates a copy, favoring the values of dict2 for shared keys.
  dict3 = dict1.copy()
  for key in dict2.keys():
    dict3[key] = dict2[key]
  return dict3

def update_data_file(data, name):
  with open(name, "wb") as f:
    pickle.dump(data, f)

def open_pkl_file(name):
  with open(name, "rb") as f:
    return pickle.load(f)

def convergence_point(data_dict, hydro, error):
  hydros = data_dict[hydro + "s"]
  true_value = data_dict[hydro]
  incumbent = len(hydros)
  for nmk in range(len(hydros), 0, -1):
    if abs((hydros[nmk - 1] - true_value)/true_value) <= error:
      incumbent = nmk
    else: # after this point, consistently <= error away.
      break
  return incumbent

def generate_convergence_data(data, error_lst):
  # Modifies the given list of dictionaries in place.
  hydro_keys = ["am", "dp"]
  for data_dict in data:
    for m0 in data_dict["m0s"]:
      for hydro in hydro_keys:
          for error in error_lst:
            data_dict[m0][f"convergence point {error:.2g} " + hydro] = convergence_point(data_dict[m0], hydro, error)

def scale_by(lst, val):
  return [entry/val for entry in lst]

def generate_log_data(all_prob_dicts):
  hydro_keys = ["am", "dp"]
  log_data = []
  for data_dict in all_prob_dicts:
    output = data_dict.copy()
    for m0 in data_dict["m0s"]:
      m0_output = {}
      for hydro in hydro_keys:
          if data_dict[m0]["convergence point 0.01 " + hydro] < 150:
            true_y = data_dict[m0][hydro]
            ys = [np.log(abs(entry - 1)) for entry in scale_by(data_dict[m0][hydro + "s"], true_y)]
            slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(1, 151)), ys)
            m0_output[hydro] = {"ys" : ys,
                                "slope" : slope,
                                "intercept" : intercept,
                                "r2_value" : r_value ** 2,
                                "p_value" : p_value,
                                "std_err" : std_err}
          else: m0_output[hydro] = None # Did not converge
      output[m0] = m0_output
    log_data.append(output)
  return log_data

def wrap_m0(f):
  # wraps a function f that only takes data_dict to fit situations where it should intake m0
  return lambda data_dict, m0 : f(data_dict)

###################################
# Data plotting functions
def func_applier(f, data):
  output = []
  for data_dict in data:
    for m0 in data_dict["m0s"]:
      output.append(f(data_dict, m0))
  return output

def plot_relation_grid(data, x_funcs, y_funcs):
  cols_x = x_funcs.keys()
  cols_y = y_funcs.keys()

  # compute the points to plot into a data frame
  transforms = {}
  for key in cols_x:
    transforms[key] = func_applier(x_funcs[key], data)
  for key in cols_y:
    transforms[key] = func_applier(y_funcs[key], data)
  transform_df = pd.DataFrame(transforms)

  # actually plot the points
  g = sns.PairGrid(transform_df, x_vars=cols_x, y_vars=cols_y, height=3, aspect=1)
  g.map(sns.scatterplot, s=10, alpha=0.7)
  plt.tight_layout()
  plt.show()

# Plotting functions for convergence against 2 and 3 variables.
def plot_3tuples_2d(data, cmap='plasma', xlabel = "X", ylabel = "Y", clabel = "color", title = "Untitled"):
    
    xs = [point[0] for point in data]
    ys = [point[1] for point in data]
    values = [point[2] for point in data]

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(xs, ys, c=values, cmap=cmap, s = 5)
    cbar = plt.colorbar(sc)
    cbar.set_label(clabel)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_hydros_against_2(data, hydro, error, xfunc, yfunc, xlab, ylab):
  xs = []
  ys = []
  zs = []
  for data_dict in data:
    for m0 in data_dict["m0s"]:
      z = data_dict[m0][f"convergence point {error:.2g} " + hydro]
      zs.append(z)
      xs.append(xfunc(data_dict, m0))
      ys.append(yfunc(data_dict, m0))
  data = np.column_stack((xs, ys, zs))
  plot_3tuples_2d(data, xlabel = xlab, ylabel = ylab,
                  clabel = f"convergence point {error:.2g}", title = hydro)

def plot_4tuples_3d(data, cmap = "plasma", xlab = "X", ylab = "Y", zlab = "Z", title = "Untitled", clabel = "color"):
    # Extract coordinates and values
    xs = [point[0] for point in data]
    ys = [point[1] for point in data]
    zs = [point[2] for point in data]
    values = [point[3] for point in data]

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs, ys, zs, c=values, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(clabel)
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    ax.set_title(title)
    plt.show()

def plot_hydros_against_3(data, hydro, error, xfunc, yfunc, zfunc, xlab, ylab, zlab):
  xs = []
  ys = []
  zs = []
  vals = []
  for data_dict in data:
    for m0 in data_dict["m0s"]:
      val = data_dict[m0][f"convergence point {error:.2g} " + hydro]
      vals.append(val)
      xs.append(xfunc(data_dict, m0))
      ys.append(yfunc(data_dict, m0))
      zs.append(zfunc(data_dict, m0))
  data = np.column_stack((xs, ys, zs, vals))
  plot_4tuples_3d(data, cmap='plasma', xlab = xlab, ylab = ylab, zlab = zlab,
                        clabel = f"convergence point {error:.2g}", title = hydro)
  
# Plots the shapes of the configurations, visually with color corresponding to convergence point.
def plot_regions_grid(data_dicts, m0s, color_func, plots_per_row=10, figsize_per_plot=(2, 2)):
    """
    Plot a grid of region plots for a list of data_dicts.
    Each row contains up to plots_per_row subplots.
    """
    total = len(data_dicts)
    rows = math.ceil(total / plots_per_row)
    cols = plots_per_row

    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_plot[0] * cols,
                                      figsize_per_plot[1] * rows),
                             squeeze=False)

    for idx, data_dict in enumerate(data_dicts):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        a = data_dict["a"]
        d = data_dict["d"]
        h = data_dict["h"]
        m0 = m0s[idx]

        # Set limits
        ax.set_xlim(0, 2 * a[-1])
        ax.set_ylim(-h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')

        # Determine color
        body_color = color_func(data_dict, m0)

        # Draw regions
        x_edges = [0, a[0], a[1], a[2]]
        for i in range(3):
            if data_dict["region"] == i:
              region_color = (0, 0.7, 0.125) # Green, not in the plasma colormap
            else: region_color = body_color
            x0 = x_edges[i]
            width = x_edges[i+1] - x0
            height = d[i]
            rect = Rectangle((x0, -height), width, height,
                             facecolor=region_color, edgecolor=None)
            ax.add_patch(rect)

        # Label with m0 value
        ax.set_title(f"m0 = {m0 : .4g}", fontsize=8)

    # Hide any unused axes
    for idx in range(total, rows * cols):
        fig.delaxes(axes.flatten()[idx])

    plt.tight_layout()
    plt.show()

def get_plasma_color(value, min_value = 1, max_value=150):
    # # Clamp to [1, max_value]
    # v = min(max(value, 1), max_value)
    # Normalize into [0,1]
    norm = (value - min_value) / (max_value - min_value)
    return cm.get_cmap('plasma')(norm)

def filter_and_plot_shapes(all_prob_dicts, restriction, color_func, ppr = 10, figsize_per_plot = (2, 2)):
  plotting_dicts, plotting_m0s = [], []
  for data_dict in all_prob_dicts:
    for m0 in data_dict["m0s"]:
      if restriction(data_dict, m0):
        plotting_dicts.append(data_dict)
        plotting_m0s.append(m0)
  plot_regions_grid(plotting_dicts, plotting_m0s, color_func, plots_per_row=ppr, figsize_per_plot=figsize_per_plot)

def first_m0_filter_and_plot_shapes(all_prob_dicts, restriction, color_func, ppr = 10, figsize_per_plot = (2, 2)):
  plotting_dicts, plotting_m0s = [], []
  for data_dict in all_prob_dicts:
    m0 = data_dict["m0s"][0]
    if restriction(data_dict, m0):
      plotting_dicts.append(data_dict)
      plotting_m0s.append(m0)
  plot_regions_grid(plotting_dicts, plotting_m0s, color_func, plots_per_row=ppr, figsize_per_plot=figsize_per_plot)
  
def plot_hypothesis(data, hydro, error, prediction, xlab, linerange = 150):
  x_line = np.linspace(0, linerange, linerange)
  plt.plot(x_line, x_line, label='y = x', color = "red")
  xs = []
  ys = []
  for data_dict in data:
      for m0 in data_dict["m0s"]:
        x = prediction(data_dict, m0)
        y = data_dict[m0][f"convergence point {error:.2g} " + hydro]
        xs.append(x)
        ys.append(y)
  plt.scatter(xs, ys, s = 1)
  plt.xlabel(xlab)
  plt.ylabel("convergence point")
  plt.title(hydro + " " + str(error))
  plt.grid()
  plt.show()