import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import sys
from pathlib import Path
HERE = Path.cwd().resolve()
up1_path = str((HERE / ".." ).resolve())
up2_path = str((HERE / ".." / ".." ).resolve())
for pathstr in [up1_path, up2_path]:
  if pathstr not in sys.path:
      sys.path.insert(0, pathstr)

from helpers import convergence_point, ConvergenceProblemI


def innermost_vars(h, d, a, m0): # predicts terms needed in innermost region
  return {"(h-d)/radwidth" : (h-d[0])/a[0],
          "(h-d_out)/(h-d)" : (h-d[1])/(h-d[0]),
          "radwidth/dist_to_e" : a[0]/(a[-1]-a[0]),
          "m0 * dist_to_e" : m0 * (a[-1]-a[0])}

def middle_vars(h, d, a, m0, i): # predicts terms needed in middle region i
  return {"(h-d)/radwidth" : (h-d[i])/(a[i] - a[i-1]),
          "(h-d_out)/(h-d)" : (h-d[i+1])/(h-d[i]),
          "(h-d_in)/(h-d)" : (h-d[i-1])/(h-d[i]),
          "radwidth/dist_to_e" : (a[i] - a[i-1])/(a[-1]-a[i]),
          "radwidth/dist_to_c" : (a[i] - a[i-1])/a[i-1],
          "m0 * dist_to_e" : m0 * (a[-1]-a[i])}

def outermost_vars(h, d, a, m0): # predicts terms needed in outermost region
  return {"(h-d)/radwidth" : (h-d[-1])/(a[-1] - a[-2]),
          "h/(h-d)" : h/(h-d[-1]),
          "(h-d_in)/(h-d)" : (h-d[-2])/(h-d[-1]),
          "radwidth/dist_to_c" : (a[-1] - a[-2])/a[-2],
          "m0h" : m0 * h}

def exterior_vars(h, d, a, m0):
  return {"m0h" : m0 * h,
          "(h-d)/h" : (h-d[-1])/h}

def filter_one_convergence_plus(cf, plus):
  for m0 in cf["m0s"]:
    cutoff_am = min(cf[m0]["convergence point 0.01 am"] + plus, len(cf[m0]["ams"]))
    cutoff_dp = min(cf[m0]["convergence point 0.01 dp"] + plus, len(cf[m0]["dps"]))
    cf[m0]["ams"] = cf[m0]["ams"][:cutoff_am]
    cf[m0]["dps"] = cf[m0]["dps"][:cutoff_dp]
  return cf

# NEED EXTERIOR
innermost_vars_cf = lambda cf, m0 : innermost_vars(cf["h"], cf["d"], cf["a"], m0)
middle_vars_cf = lambda cf, m0, region = 1 : middle_vars(cf["h"], cf["d"], cf["a"], m0, region)
outermost_vars_cf = lambda cf, m0 : outermost_vars(cf["h"], cf["d"], cf["a"], m0)
exterior_vars_cf = lambda cf, m0 : exterior_vars(cf["h"], cf["d"], cf["a"], m0)
# NEED EXTERIOR

def variables_guess_variants(model, region_type, *params):
  inner_model, variables_used, guess = model(*params)
  if region_type == "innermost": vars_cf = innermost_vars_cf
  elif region_type == "middle": vars_cf = middle_vars_cf
  elif region_type == "outermost": vars_cf = outermost_vars_cf
  else: vars_cf = exterior_vars_cf
  fit_model, cf_params_to_alpha_beta, err_from_nmk_model, nmk_from_err_model = get_model_variants(vars_cf, variables_used, inner_model)
  return variables_used, guess, fit_model, cf_params_to_alpha_beta, err_from_nmk_model, nmk_from_err_model

def fit_model_weighted(cfs, hydro, model, guess, get_vars, variables_used, underweight = 1):
  xs = [[] for _ in range(len(variables_used) + 1)]
  ys = []
  for cf in cfs:
    for m0 in cf["m0s"]:
      vals = get_vars(cf, m0)
      for nmk in range(len(cf[m0][hydro + "s"])): # up to nmk_max
        xs[0].append(nmk + 1)
        for i in range(len(variables_used)):
          xs[i + 1].append(vals[variables_used[i]])
        ys.append(cf[m0]["log errors " + hydro][nmk])
  
  xs = [np.asarray(x, dtype=float) for x in xs]
  ys = np.asarray(ys, dtype=float)
  
  def residuals(params):
    yhat = model(xs, *params)
    r = yhat - ys
    w = np.ones_like(r)
    w[r < 0] = underweight
    return np.sqrt(w) * r
  
  res = least_squares(residuals, x0=guess)
  return res.x, res

fit_inner_model_weighted = lambda cfs, hydro, model, guess, variables_used, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, innermost_vars_cf, variables_used, underweight = underweight)
fit_middle_model_weighted = lambda cfs, hydro, model, guess, variables_used, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, middle_vars_cf, variables_used, underweight = underweight)
fit_outer_model_weighted = lambda cfs, hydro, model, guess, variables_used, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, outermost_vars_cf, variables_used, underweight = underweight)
fit_exterior_model_weighted = lambda cfs, hydro, model, guess, variables_used, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, exterior_vars_cf, variables_used, underweight = underweight)

def get_model_variants(get_vars, variables_used, var_params_to_alpha_beta): # format: list of nondim variables used in order, params
  def fit_model(xs, *params):
    nmk = xs[0]
    vars_ = xs[1:]
    alpha, beta = var_params_to_alpha_beta(vars_, *params)
    return - alpha * np.log(nmk/beta)
  def cf_params_to_alpha_beta(cf, m0, *params):
    all_vars = get_vars(cf, m0)
    vars = [all_vars[key] for key in variables_used]
    return var_params_to_alpha_beta(vars, *params) # alpha, beta
  def err_from_nmk_model(nmk, cf, m0, *params):
    alpha, beta = cf_params_to_alpha_beta(cf, m0, *params)
    return (beta/nmk)**alpha
  def nmk_from_err_model(err, cf, m0, *params):
    alpha, beta = cf_params_to_alpha_beta(cf, m0, *params)
    return beta / (err ** (1/alpha))
  return fit_model, cf_params_to_alpha_beta, err_from_nmk_model, nmk_from_err_model

def predict_nmk(cf, m0, model, fraction_converged, use_m0 = False):
  alpha, beta = model(cf, m0) if use_m0 else model(cf)
  nmk = beta / (fraction_converged ** (1/alpha))
  return math.ceil(nmk)

def evaluate_one_geom(model_am, model_dp, cf, m0, fraction_converged, use_m0 = False, true_vals = None, correct_nmk = None, contained_in_cf = False):
    am_nmk = predict_nmk(cf, m0, model_am, fraction_converged, use_m0 = use_m0)
    dp_nmk = predict_nmk(cf, m0, model_dp, fraction_converged, use_m0 = use_m0)

    if true_vals is None: pass
    else: true_am, true_dp = true_vals[0], true_vals[1]
    if correct_nmk is None: pass
    if not contained_in_cf: pass
    else:
      predicted_am, predicted_dp = cf[m0]["ams"][am_nmk - 1], cf[m0]["dps"][dp_nmk - 1]

    am_fraction_off = abs((predicted_am - true_am)/true_am)
    dp_fraction_off = abs((predicted_dp - true_dp)/true_dp)

    return {"am error" : am_fraction_off,
            "dp error" : dp_fraction_off,
            "am nmk predicted" : am_nmk,
            "dp nmk predicted" : dp_nmk,
            "am nmk actual" : correct_nmk[0],
            "dp nmk actual" : correct_nmk[1]}

def evaluate_one_geom_precomputed_geom(model_am, model_dp, cf, m0, fraction_converged, use_m0 = False):
  true_vals = (cf[m0]["am"], cf[m0]["dp"])
  correct_nmk = (convergence_point(cf[m0], "am", fraction_converged), convergence_point(cf[m0], "dp", fraction_converged))
  return evaluate_one_geom(model_am, model_dp, cf, m0, fraction_converged, use_m0 = use_m0, true_vals = true_vals, correct_nmk = correct_nmk, contained_in_cf = True)

def evaluate_many_precomputed_geoms(cfs, model_am, model_dp, fraction_converged):
  data = [evaluate_one_geom_precomputed_geom(model_am, model_dp, cf, m0, fraction_converged, use_m0 = use_m0) for cf in cfs for m0 in cf["m0s"]]
  return data

def extractf(cf, m0, hydro):
   true_val = cf[m0][hydro]
   computed_vals = np.array(cf[m0][hydro + "s"])
   errs = abs((computed_vals - true_val)/true_val)
   return np.log10(errs)

extractf_am = lambda cf, m0: extractf(cf, m0, "am")
extractf_dp = lambda cf, m0: extractf(cf, m0, "dp")

def model_helper(model, cf, m0, nmks = range(1, 151)):
  alpha, beta = model(cf, m0)
  return np.log10(np.array([(beta/nmk)** alpha for nmk in nmks]))

def model_wrapper(model):
  return lambda cf, m0 : model_helper(model, cf, m0, nmks = range(1, 151))

def compare_model_curves(cfs, extractf, modelf, summary_fs, nmks = list(range(1, 151)), ncols = 10, figsize_per_ax = (2.2, 2.2),):
    cf_count = len(cfs)
    nrows = math.ceil(cf_count / ncols)

    fig, axs = plt.subplots(nrows, ncols, sharex=True,
                            figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows), squeeze=False)

    flat_axs = axs.ravel()

    for i, cf in enumerate(cfs):
      for m0 in cf["m0s"]:
        ax = flat_axs[i]
        ys1 = np.asarray(extractf(cf, m0), dtype=float)
        ys2 = np.asarray(modelf(cf, m0), dtype=float)

        ax.scatter(nmks, ys1, s=18, color = "cornflowerblue")
        ax.plot(nmks, ys2, linewidth=1.2, color = "red")

        summary_lines = [f"{key}: {float(summary_fs[key](cf)):.3g}" for key in summary_fs.keys()]

        ax.text(0.98, 0.98, "\n".join(summary_lines),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))

    for j in range(cf_count, len(flat_axs)): # Hide unused axes
        flat_axs[j].set_visible(False)

    for r in range(nrows):
      for c in range(ncols):
        ax = axs[r, c]
        if ax.get_visible():
          ax.tick_params(axis="x", bottom=True, labelbottom=(r == nrows - 1)) # Only numerical labels for bottom row
          ax.tick_params(axis="y", left=True, labelleft=True)
          ax.set_xlabel("NMK" if r == nrows - 1 else "") # Axis labels only on outer edges
          ax.set_ylabel("log(error)" if c == 0 else "")

    fig.tight_layout()
    plt.show()

def plot_on_ax_nmks(ax1, cfs, hydro, models_dict, err, ax2 = None, plot_type = "hist"):
  cf_key = f"convergence point {err:.2g} " + hydro
  true_nmks = [cf[m0][cf_key] for cf in cfs for m0 in cf["m0s"]]
  sample_size = len(true_nmks)
  all_nmk_diffs, all_nmk_fracs = [], []
  diff_mus, diff_stds = [], []
  frac_mus, frac_stds = [], []
  labels_hist, labels_box = [], []
  for key in models_dict.keys():
    nmk_from_err_model, params = models_dict[key]
    calculated_nmks = [math.ceil(nmk_from_err_model(err, cf, m0, *params)) for cf in cfs for m0 in cf["m0s"]]
    nmk_diffs = [calculated_nmks[i] - true_nmks[i] for i in range(sample_size)]
    mu1, std1 = np.mean(nmk_diffs), np.std(nmk_diffs)
    diff_mus.append(mu1)
    diff_stds.append(std1)
    all_nmk_diffs.append(nmk_diffs)
    nmk_fracs = [nmk_diffs[i]/true_nmks[i] for i in range(sample_size)] 
    mu2, std2 = np.mean(nmk_fracs), np.std(nmk_fracs)
    frac_mus.append(mu2)
    frac_stds.append(std2)
    all_nmk_fracs.append(nmk_fracs)
    labels_hist.append(key + rf": $\mu_{{diff}}={mu1:.3g}, \sigma_{{diff}}={std1:.3g}, \mu_{{frac}}={mu2:.3g}, \sigma_{{frac}}={std2:.3g}$")
    labels_box.append(key)

  if plot_type == "hist":
    weights = np.ones(sample_size) / sample_size
    for i in range(len(models_dict.keys())):
      nmk_diffs, nmk_fracs = all_nmk_diffs[i], all_nmk_fracs[i]
      bins = np.arange(min(nmk_diffs)-0.5, max(nmk_diffs)+1.5, 1)
      ax1.hist(nmk_diffs, bins=bins, weights = weights, histtype='step', linewidth=0.5)
      bin_width = 0.04
      bins = np.arange(min(nmk_fracs)-bin_width/2,
                      max(nmk_fracs)+bin_width*1.5,
                      bin_width)
      ax1.set_xlabel("Predicted NMK - True NMK")
      if ax2 is not None:
        ax2.hist(nmk_fracs, bins=bins, weights = weights, histtype='step', linewidth=0.5, label = labels_hist[i])
        ax2.set_xlabel("(Predicted NMK - True NMK)/(True NMK)")
  else:
    ax1.boxplot(all_nmk_diffs, whis=(5, 95), tick_labels = labels_box, vert=False)
    ax1.set_xlabel("Predicted NMK - True NMK")
    ax1.grid(True, axis='x', alpha = 0.5)
    if ax2 is not None:
      ax2.boxplot(all_nmk_fracs, whis=(5, 95), tick_labels = labels_box, vert=False)
      ax2.set_xlabel("(Predicted NMK - True NMK)/(True NMK)")
      ax2.grid(True, axis='x', alpha = 0.5)

  return sample_size, all_nmk_diffs, all_nmk_fracs, diff_mus, diff_stds, frac_mus, frac_stds

def compare_nmk_distributions(cfs, hydro, models_dict, err, title = None, plot_type = "hist"):
  fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
  statpack = plot_on_ax_nmks(axs[0], cfs, hydro, models_dict, err, axs[1], plot_type = plot_type)
  
  if plot_type == "hist":
    axs[0].set_ylabel("Frequency")
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  else:
    axs[0].set_ylabel("Fit Version")

  fig.suptitle(title)
  plt.show()
  print("Sample size: " + str(statpack[0]))
  return statpack[1:]

def get_error_single_vary(cf, m0, nmk, nmk_big, true_am, true_dp):
  if nmk <= len(cf[m0]["ams"]): # has already been computed
    am, dp = cf[m0]["ams"][nmk - 1], cf[m0]["dps"][nmk - 1]
  else:
    am, dp = cf[m0]["am"] * 1.0001, cf[m0]["dp"] * 1.0001 # placeholder to reduce excessive computation
    # NMK = [nmk if i == cf["region"] else nmk_big for i in range(len(cf["a"]))]
    # heaving = [1 if i == cf["region"] else 0 for i in range(len(cf["a"]))]
    # prob = ConvergenceProblemI(cf["h"], cf["d"], cf["a"], heaving, NMK, m0, 1023)
    # x = prob.get_unknown_coeffs(prob.a_matrix(), prob.b_vector())
    # am, dp = prob.hydro_coeffs(x, "capytaine")
  return (am - true_am)/true_am, (dp - true_dp)/true_dp

def box_plot_on_ax_errs(ax, cfs, hydro, models_dict, err, nmk_big = 200):
  true_vals = [cf[m0][hydro] for cf in cfs for m0 in cf["m0s"]]
  sample_size = len(true_vals)
  all_calculated_errs = []
  err_mus, err_stds = [], []
  labels_box = []
  for key in models_dict.keys():
    nmk_from_err_model, params = models_dict[key]
    errs = [get_error_single_vary(cf, m0, math.ceil(nmk_from_err_model(err, cf, m0, *params)), nmk_big, cf[m0]["am"], cf[m0]["dp"])
            for cf in cfs for m0 in cf["m0s"]]
    idx = 0 if hydro == "am" else 1
    errs = [abs(err[idx]) for err in errs]
    all_calculated_errs.append(errs)
    mu1, std1 = np.mean(errs), np.std(errs)
    err_mus.append(mu1)
    err_stds.append(std1)
    labels_box.append(key)
  ax.boxplot(all_calculated_errs, whis=(5, 95), tick_labels = labels_box, vert=False)
  ax.set_xlabel("|Relative Error to True|")
  ax.grid(True, axis='x', alpha = 0.5)
  return sample_size, all_calculated_errs, err_mus, err_stds

def compare_err_distributions(cfs, hydro, models_dict, err, nmk_big = 200, title = None):
  fig, ax = plt.subplots(1, 1, figsize=(5, 5))
  sample_size, all_calculated_errs, err_mus, err_stds = box_plot_on_ax_errs(ax, cfs, hydro, models_dict, err, nmk_big = nmk_big)
  ax.set_ylabel("Fit Version")
  fig.suptitle(title)
  plt.show()
  print("Sample size: " + str(sample_size))
  return all_calculated_errs, err_mus, err_stds

def compare_abs_nmk_rel_err(cfs, hydro, models_dict, err, nmk_big = 200, title = None):
  fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
  err_statpack = box_plot_on_ax_errs(axs[0], cfs, hydro, models_dict, err, nmk_big = nmk_big)
  nmk_statpack = plot_on_ax_nmks(axs[1], cfs, hydro, models_dict, err, plot_type = "box")
  axs[0].set_ylabel("Fit Version")
  fig.suptitle(title)
  plt.show()
  print("Sample size: " + str(err_statpack[0]))
  return err_statpack[1:], nmk_statpack[1:]

def filter_converged(cfs):
  return [cf for cf in cfs if
          (max([cf[m0]["convergence point 0.01 am"] for m0 in cf["m0s"]] + [cf[m0]["convergence point 0.01 dp"] for m0 in cf["m0s"]]) < 150)]

def twenty_minimum(nmk_from_err_model):
  return lambda err, cf, m0, *params : max(nmk_from_err_model(err, cf, m0, *params), 20)

def print_err_nmk_abs_from_statpacks(statpacks):
  print("Average Errors:", ",".join(f"{x:.3g}" for x in statpacks[0][1]))
  print("STDEV Errors:", ",".join(f"{x:.3g}" for x in statpacks[0][2]))
  print("Average NMK off:", ",".join(f"{x:.3g}" for x in statpacks[1][2]))
  print("STDEV NMK off:", ",".join(f"{x:.3g}" for x in statpacks[1][3]))