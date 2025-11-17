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
from scipy.optimize import curve_fit


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

# Converging over all regions & heaves
class ConvergenceProblemNeighbors(ConvergenceProblemI):
  def get_all_heaves(self):
    heaves = []
    for i in range(self.boundary_count):
      heave = [1 if i == j else 0 for j in range(self.boundary_count)]
      heaves.append(heave)
    return heaves
  
  def e_varied_matrices(self, full_a_matrix, nmk_max):
    big_nmk = self.NMK[-1]
    all_a_matrices = []
    for i in range(1, nmk_max + 1):
      a_matrix = full_a_matrix[: (self.size - big_nmk + i), : (self.size - big_nmk + i)]
      all_a_matrices.append(a_matrix)
    return all_a_matrices

  def e_varied_b_vectors(self, full_b_vector, nmk_max):
    big_nmk = self.NMK[-1]
    all_b_vectors = []
    for i in range(1, nmk_max + 1):
      b_vector = full_b_vector[: (self.size - big_nmk + i)]
      all_b_vectors.append(b_vector)
    return all_b_vectors
    
  def full_convergence_study(self, nmk_max, m0s, mks):
    # Do study convergence across all regions, for all distinct heaves
    heaves = self.get_all_heaves()
    full_a_matrix = self.a_matrix()
    full_bs_across_heaves = []
    full_cs_across_heaves = []
    all_cs_across_heaves = []
    for heave_vector in heaves:
      self.heaving = heave_vector
      full_bs_across_heaves.append(self.b_vector())
      full_c_vector = self.c_vector()
      full_cs_across_heaves.append(full_c_vector)
      all_cs_across_regions = []
      for region in range(self.boundary_count):
        all_cs_across_regions.append(self.get_c_vectors(full_c_vector, nmk_max, region))
      all_cs_across_regions.append([full_c_vector for i in range(nmk_max)]) # for the e region
      all_cs_across_heaves.append(all_cs_across_regions)
    output = {}

    for idx, m0 in enumerate(m0s):
      out_for_m0 = {}
      self.change_m0_mk(m0, mks[idx])
      omega = self.angular_freq(self.m0)

      full_a_matrix = self.a_matrix_from_old(full_a_matrix) # match the m0
      all_a_matrices_across_regions = [self.get_sub_matrices(full_a_matrix, nmk_max, region) for region in range(self.boundary_count)]
      all_a_matrices_across_regions.append(self.e_varied_matrices(full_a_matrix, nmk_max))

      for heaving_region, heave_vector in enumerate(heaves):
        self.heaving = heave_vector
        full_b_vector = self.b_vector_from_old(full_bs_across_heaves[heaving_region]) # match the m0
        full_c_vector = full_cs_across_heaves[heaving_region]
        particular_contribution = self.int_phi_p_i(heaving_region)
        out_for_heave = {}
        for region in range(self.boundary_count + 1):
          if region == self.boundary_count:
            all_b_vectors = self.e_varied_b_vectors(full_b_vector, nmk_max)
          else:
            all_b_vectors = self.get_b_vectors(full_b_vector, nmk_max, region)
          all_c_vectors = all_cs_across_heaves[heaving_region][region]
          am_lst, dp_lst = [], []
          for nmk in range(1, nmk_max + 1):
            x = self.get_unknown_coeffs(all_a_matrices_across_regions[region][nmk - 1], all_b_vectors[nmk - 1])
            sub_x = x[:-nmk] if region == self.boundary_count else x[:-self.NMK[-1]]
            raw_hydro = 2 * np.pi * (np.dot(all_c_vectors[nmk - 1], sub_x) + particular_contribution)
            # follow the capytaine convention
            am_lst.append(raw_hydro.real * self.rho) # added mass
            dp_lst.append(raw_hydro.imag * omega * self.rho) # damping
          out_for_region = {"ams" : am_lst,
                            "dps" : dp_lst}
          out_for_heave[region] = out_for_region
        x = self.get_unknown_coeffs(full_a_matrix, full_b_vector)
        am, dp = self.hydro_coeffs(x, "capytaine")
        out_for_heave["am"], out_for_heave["dp"] = am, dp
        out_for_m0[heaving_region] = out_for_heave
      output[m0] = out_for_m0
    return output

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
  for data_dict in all_prob_dicts:
    for m0 in data_dict["m0s"]:
      for hydro in hydro_keys:
          true_y = data_dict[m0][hydro]
          ys = [np.log(abs(entry - 1)) for entry in scale_by(data_dict[m0][hydro + "s"], true_y)]
            # slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(1, 151)), ys)
          data_dict[m0]["log errors " + hydro] = ys
  return all_prob_dicts

def wrap_m0(f):
  # wraps a function f that only takes data_dict to fit situations where it should intake m0
  return lambda data_dict, m0 : f(data_dict)

def subdivide_lst(data_lst, f, rtol = 0.01):
  group_dict = {}
  for data_dict in data_lst:
    for m0 in data_dict["m0s"]:
      val = f(data_dict, m0)
      stored = False
      keys = ["h", "d", "a", "region", m0]
      out_dict = {key: data_dict[key] for key in keys}
      out_dict["m0s"] = [m0]
      for key in group_dict.keys():
        if abs((val - key)/key) < rtol:
          group_dict[key].append(out_dict)
          stored = True
          break
      if not stored:
        group_dict[val] = [out_dict]
  return [group_dict[key] for key in group_dict.keys()] # turn dict into lst of lsts

def subdivide_by_constants(data, xlab, variable_funcs):
  other_funcs = variable_funcs.copy()
  del other_funcs[xlab]
  subdivision = [data]
  for key in other_funcs.keys():
    subdivision = [entry for sublist in subdivision for entry in subdivide_lst(sublist, other_funcs[key])]
  return subdivision

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

def plot_hydros_against_1(data, hydro, error, xfunc, xlabel = "X", ylabel = None, title = None):
  xs = [xfunc(config, m0) for config in data for m0 in config["m0s"]]
  ys = [config[m0][f"convergence point {error:2g} " + hydro] for config in data for m0 in config["m0s"]]
  plt.scatter(xs, ys)

  if ylabel is None:
    ylabel = f"convergence point {error:.2g}"
  if title is None:
    title = hydro
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()

######## GEOMETRIES, with color corresponding to convergence point.
def plot_regions_grid(data_dicts, m0s, color_func, plots_per_row=10, figsize_per_plot=(2, 2)):
    """
    Plot a grid of region plots for a list of data_dicts.
    Each row contains up to plots_per_row subplots.
    """
    total = len(data_dicts)
    rows = math.ceil(total / plots_per_row)
    cols = plots_per_row

    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows), squeeze=False)

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
        x_edges = [0] + [a[i] for i in range(len(d))]
        for i in range(len(d)):
            if data_dict["region"] == i:
              region_color = (0, 0.7, 0.125) # Green, not in the plasma colormap
            else: region_color = body_color
            x0 = x_edges[i]
            width = x_edges[i+1] - x0
            height = d[i]
            rect = Rectangle((x0, -height), width, height, facecolor=region_color, edgecolor=None)
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

######## HISTOGRAM
def histogram(data, hydro, error, ylab = "count", title = "convergence"):
  vals = [data_dict[m0][f"convergence point {error:.2g} " + hydro]
          for data_dict in data for m0 in data_dict["m0s"]]
  if min(vals) == max(vals):
    bins = [vals[0] - 0.5, vals[0] + 0.5]
  else:
    bins = np.arange(min(vals)-0.5, max(vals)+1.5, 1)
  plt.hist(vals, bins = bins, edgecolor = "black")
  plt.xlabel(f"convergence point {error:.2g} " + hydro)
  plt.ylabel(ylab)
  plt.title(hydro + " " + title)
  plt.show()
  return vals

######## LOG PLOTS
def plot_one_convergence_and_log(data_dict, m0, hydro, ax1, ax2, color = "Red", scale = False, alpha = 1,
                             label = None, error = 0.01, nmk_max = 150, trunc = 0, smooth = False):
  xs = list(range(1, nmk_max + 1))[trunc:]
  ys1 = data_dict[m0][hydro + "s"][trunc:]
  ys2 = data_dict[m0]["log errors " + hydro][trunc:]
  true_val = data_dict[m0][hydro]
  if scale:
    ys1 = scale_by(ys1, true_val)
    true_val = 1
  ax1.plot(xs, ys1, color = color, alpha=alpha, label = label)
  ax1.axhline(true_val, color = color, linestyle='--')
  if smooth:
    xs, ys2 = filter_local_maxima(xs, ys2)
  # if smooth:
  #   new_xs = []
  #   new_ys = []
  #   incumbent = min(ys2)
  #   for i in range(len(xs) - 1, -1, -1):
  #     if ys2[i] > incumbent:
  #       new_ys.append(ys2[i])
  #       new_xs.append(xs[i])
  #       incumbent = ys2[i]
  #   xs, ys2 = new_xs, new_ys
  ax2.plot(xs, ys2, color = color, alpha=alpha, label = label)

def plot_set_convergence_and_log(data, hydro, colors = None, scale = False, alpha = 1,
                                 label_func = (lambda data_dict, m0 : None), error = 0.01,
                                 nmk_max = 150, show_error = False, trunc = 0, smooth = False):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
  for idx, pair in enumerate(data):
    data_dict, m0 = pair
    color = "Red" if colors is None else colors[idx]
    plot_one_convergence_and_log(data_dict, m0, hydro, ax1, ax2, color = color, scale = scale, alpha = alpha,
                             label = label_func(data_dict, m0), error = error, nmk_max = nmk_max, trunc = trunc, smooth = smooth)
  if show_error:
    ax2.axhline(np.log(error), color = "Black", linestyle = "--")
  ax1.set_xlabel('NMK')
  if scale:
    ax1.set_ylabel("Ratio of computed " + hydro + " to true value")
  ax1.set_ylabel("Computed " + hydro )
  ax2.set_title("Raw convergence of " + hydro)
  ax2.set_xlabel('NMK')
  ax2.set_ylabel('Log error to true value')
  ax2.set_title("Log error convergence of " + hydro)
  ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
  plt.grid()
  plt.tight_layout()
  plt.show()

def selective_m0(f, data_dict, m0 = None, m0s = None, all_m0s = False):
  if all_m0s:
    return [f(data_dict, m0) for m0 in data_dict["m0s"]]
  if m0 is None:
    if m0s is None:
      return [f(data_dict, data_dict["m0s"][0])]
    else:
      return [f(data_dict, m0) for m0 in m0s]
  else:
    return [f(data_dict, m0)]

def data_dict_to_convergence_plot_data(data_dict, m0 = None, m0s = None, all_m0s = False):
  return selective_m0((lambda data_dict, m0 : (data_dict, m0)), data_dict, m0 = m0, m0s = m0s, all_m0s = all_m0s)
  
def many_data_dicts_to_convergence_plot_data(all_dicts, m0 = None, m0s = None, all_m0s = False):
  lst = []
  for data_dict in all_dicts:
    lst = lst + data_dict_to_convergence_plot_data(data_dict, m0 = m0, m0s = m0s, all_m0s = all_m0s)
  return lst

# generates a [colors] input
def color_by_f_value(f, all_prob_dicts, m0 = None, m0s = None, all_m0s = False, cmap = "viridis"):
  f_vals = []
  for prob in all_prob_dicts:
    f_vals = f_vals + selective_m0(f, prob, m0 = m0, m0s = m0s, all_m0s = all_m0s)
  max_val = max(f_vals)
  min_val = min(f_vals)
  return [cm.get_cmap(cmap)((f_val - min_val)/(max_val - min_val)) for f_val in f_vals]

####### ERROR FITTING
def filter_local_maxima(xs, ys):
  new_xs = []
  new_ys = []
  for x in xs:
    index = x - 1
    add_entry = False
    if x == 1:
      add_entry = ys[index] > ys[index + 1]
    elif x == xs[-1]:
      add_entry = ys[index] > ys[index - 1]
    else:
      add_entry = ((ys[index] > ys[index + 1]) and (ys[index] > ys[index - 1]))
    if add_entry:
      new_xs.append(x)
      new_ys.append(ys[index])
  return new_xs, new_ys

def r2(f, xs, ys, a, b):
    yhat = [f(x, a, b) for x in xs]
    ys, yhat = np.array(ys), np.array(yhat)
    ss_res = np.sum((ys - yhat)**2)
    ss_tot = np.sum((ys - np.mean(ys))**2)
    return 1 - ss_res / ss_tot

def r2_underestimates(f, xs, ys, a, b):
    yhat = [f(x, a, b) for x in xs]
    ys, yhat = np.array(ys), np.array(yhat)
    ybar = np.mean([ys[i] for i in range(len(xs)) if ys[i] < yhat[i]])
    ss_res, ss_tot = 0, 0
    for i in range(len(xs)):
      if ys[i] < yhat[i]:
        ss_res += (ys[i] - yhat[i])**2
        ss_tot += (ys[i] - ybar)**2
    return 1 - ss_res / ss_tot

def fit_parameters(cf, m0, hydro, local_maxima = False, plot_comparison = False, print_params = True,
                   nmk_max = 150, linear_model = False, r2_lin = False):
  full_xs = list(range(1, nmk_max + 1))
  full_ys = [cf[m0]["log errors " + hydro][i] for i in range(nmk_max)]
  if local_maxima:
    xs, ys = filter_local_maxima(full_xs, full_ys)
  else:
    xs, ys = full_xs, full_ys

  if not linear_model:
    f = lambda x, a1, a2 : (- a1 * np.log(x/a2))
    popt, pcov = curve_fit(f, xs, ys, p0=(1, 1))
  else:
    f = lambda x, a1, a2 : (- a1 * x + a2)
    popt, pcov = curve_fit(f, np.log(xs), ys, p0=(1, 1))
    popt[1] = np.exp(popt[1]/popt[0])

  if r2_lin:
    f = lambda x, a1, a2 : (- a1 * x + a2)
    alpha, beta, xs = popt[0], popt[0] * np.log(popt[1]), np.log(full_xs)
  else:
    f = lambda x, a1, a2 : (- a1 * np.log(x/a2))
    alpha, beta, xs = popt[0], popt[1], full_xs
  r2_val = r2(f, xs, full_ys, alpha, beta)
  r2_under = r2_underestimates(f, xs, full_ys, alpha, beta)
  
  if print_params: print("Best-fit parameters:", popt)
  if plot_comparison:
    ys_calc = [f(x, *popt) for x in full_xs]
    plt.scatter(full_xs, full_ys, label="Data")
    plt.plot(full_xs, ys_calc, color="red", label="Fit")
    plt.legend()
    plt.show()

  return popt, pcov, r2_val, r2_under

def reorder(lst, order):
    return [lst[i] for i in order]

def multi_fit_parameters(cfs, sort_func, hydro, sort_label = "No Label", local_maxima = False, plot_comparison = False,
                         print_params = True, plot_multi_log_comparison = True, plot_multi_params = True, nmk_max = 150,
                         linear_model = False, r2_lin = False):
  meta_xs, meta_ys1, meta_ys2, covs, r2s, r2_unders = [], [], [], [], [], []
  for cf in cfs:
    for m0 in cf["m0s"]:
      popt, pcov, r2, r2_under = fit_parameters(cf, cf["m0s"][0], hydro, local_maxima=local_maxima,
                                                plot_comparison=plot_comparison, print_params = print_params,
                                                nmk_max = nmk_max, linear_model=linear_model, r2_lin = r2_lin)
      meta_xs.append(sort_func(cf, m0))
      meta_ys1.append(popt[0])
      meta_ys2.append(popt[1])
      covs.append(pcov)
      r2s.append(r2)
      r2_unders.append(r2_under)

  if plot_multi_log_comparison:
    colors = color_by_f_value(sort_func, cfs, all_m0s = True)
    f = lambda x, a1, a2 : (- a1 * np.log(x/a2))
    nmks = list(range(1, nmk_max + 1))
    for i in range(len(meta_xs)):
      ys_calc = [f(x, meta_ys1[i], meta_ys2[i]) for x in nmks]
      plt.plot(nmks, ys_calc, color = colors[i])
    plt.title("Fitted curves for different " + sort_label)
    plt.xlabel("NMK")
    plt.ylabel("Predicted Log Error")
    plt.show()
  
  order = sorted(range(len(meta_xs)), key=lambda i: meta_xs[i])
  meta_xs, meta_ys1, meta_ys2 = reorder(meta_xs, order), reorder(meta_ys1, order), reorder(meta_ys2, order)

  if plot_multi_params:
    plt.plot(meta_xs, meta_ys1)
    plt.xlabel(sort_label)
    plt.ylabel("alpha")
    plt.show()
    result = stats.linregress(meta_xs, meta_ys1)
    print(f"fitted slope: {result.slope:.3g}, slope/avg: {result.slope/np.mean(meta_ys1):.3g}")

    plt.plot(meta_xs, meta_ys2)
    plt.xlabel(sort_label)
    plt.ylabel("beta")
    plt.show()
    result = stats.linregress(meta_xs, meta_ys2)
    print(f"fitted slope: {result.slope:.3g}, slope/avg: {result.slope/np.mean(meta_ys2):.3g}")

  return meta_xs, meta_ys1, meta_ys2, covs, r2s, r2_unders
      