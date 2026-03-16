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

from helpers import convergence_point


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

# NEED EXTERIOR
innermost_vars_cf = lambda cf, m0 : innermost_vars(cf["h"], cf["d"], cf["a"], m0)
middle_vars_cf = lambda cf, m0, region : middle_vars(cf["h"], cf["d"], cf["a"], m0, region)
outermost_vars_cf = lambda cf, m0 : outermost_vars(cf["h"], cf["d"], cf["a"], m0)
# NEED EXTERIOR

def fit_model_weighted(cfs, hydro, model, guess, get_vars, variables_used, nmk_max = 150, underweight = 1):
  xs = [[] for _ in range(len(variables_used) + 1)]
  ys = []
  for cf in cfs:
    for m0 in cf["m0s"]:
      vals = get_vars(cf, m0)
      for nmk in range(nmk_max):
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

fit_inner_model_weighted = lambda cfs, hydro, model, guess, variables_used, nmk_max = 150, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, innermost_vars_cf, variables_used, nmk_max = nmk_max, underweight = underweight)
fit_middle_model_weighted = lambda cfs, hydro, model, guess, variables_used, nmk_max = 150, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, middle_vars_cf, variables_used, nmk_max = nmk_max, underweight = underweight)
fit_outer_model_weighted = lambda cfs, hydro, model, guess, variables_used, nmk_max = 150, underweight = 1 : fit_model_weighted(cfs, hydro, model, guess, outermost_vars_cf, variables_used, nmk_max = nmk_max, underweight = underweight)

def get_model_variants(get_vars, variables_used, var_params_to_alpha_beta, m0_arg = False): # format: list of nondim variables used in order, params
  def fit_model(xs, *params):
    nmk = xs[0]
    vars_ = xs[1:]
    alpha, beta = var_params_to_alpha_beta(vars_, *params)
    return - alpha * np.log(nmk/beta)
  if m0_arg:
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
  else:
    def cf_params_to_alpha_beta(cf, *params):
      all_vars = get_vars(cf, cf["m0s"][0])
      vars = [all_vars[key] for key in variables_used]
      return var_params_to_alpha_beta(vars, *params) # alpha, beta
    def err_from_nmk_model(nmk, cf, *params):
      alpha, beta = cf_params_to_alpha_beta(cf, *params)
      return (beta/nmk)**alpha
    def nmk_from_err_model(err, cf, *params):
      alpha, beta = cf_params_to_alpha_beta(cf, *params)
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

def evaluate_many_precomputed_geoms(cfs, model_am, model_dp, fraction_converged, use_m0 = False):
  data = [evaluate_one_geom_precomputed_geom(model_am, model_dp, cf, cf["m0s"][0], fraction_converged, use_m0 = use_m0) for cf in cfs]
  return data

def extractf(cf, m0, hydro):
   true_val = cf[m0][hydro]
   computed_vals = np.array(cf[m0][hydro + "s"])
   errs = abs((computed_vals - true_val)/true_val)
   return np.log10(errs)

extractf_am = lambda cf: extractf(cf, cf["m0s"][0], "am")
extractf_dp = lambda cf: extractf(cf, cf["m0s"][0], "dp")

def model_helper(model, cf, nmks = range(1, 151)):
  alpha, beta = model(cf)
  return np.log10(np.array([(beta/nmk)** alpha for nmk in nmks]))

def model_wrapper(model):
  return lambda cf : model_helper(model, cf, nmks = range(1, 151))

def compare_model_curves(cfs, extractf, modelf, summary_fs, nmks = list(range(1, 151)), ncols = 10, figsize_per_ax = (2.2, 2.2)):
    cf_count = len(cfs)
    nrows = math.ceil(cf_count / ncols)

    fig, axs = plt.subplots(nrows, ncols, sharex=True,
                            figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows), squeeze=False)

    flat_axs = axs.ravel()

    for i, cf in enumerate(cfs):
      ax = flat_axs[i]
      ys1 = np.asarray(extractf(cf), dtype=float)
      ys2 = np.asarray(modelf(cf), dtype=float)

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

def compare_nmk_distributions(cfs, hydro, models_dict, err, title = None):
  cf_key = f"convergence point {err:.2g} " + hydro
  true_nmks = [cf[cf["m0s"][0]][cf_key] for cf in cfs]
  sample_size = len(true_nmks)
  weights = np.ones(sample_size) / sample_size
  diff_mus, diff_stds = [], []
  frac_mus, frac_stds = [], []
  fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
  for key in models_dict.keys():
    nmk_from_err_model, params = models_dict[key]
    calculated_nmks = [nmk_from_err_model(err, cf, *params) for cf in cfs]
    nmk_diffs = [calculated_nmks[i] - true_nmks[i] for i in range(sample_size)]
    mu1, std1 = np.mean(nmk_diffs), np.std(nmk_diffs)
    diff_mus.append(mu1)
    diff_stds.append(std1)
    bins = np.arange(min(nmk_diffs)-0.5, max(nmk_diffs)+1.5, 1)
    axs[0].hist(nmk_diffs, bins=bins, weights = weights, histtype='step', linewidth=0.5)
    nmk_fracs = [nmk_diffs[i]/true_nmks[i] for i in range(sample_size)] 
    mu2, std2 = np.mean(nmk_fracs), np.std(nmk_fracs)
    frac_mus.append(mu2)
    frac_stds.append(std2)
    bin_width = 0.04
    bins = np.arange(min(nmk_fracs)-bin_width/2,
                    max(nmk_fracs)+bin_width*1.5,
                    bin_width)
    axs[1].hist(nmk_fracs, bins=bins, weights = weights, histtype='step', linewidth=0.5,
                label = (key + rf": $\mu_{{diff}}={mu1:.3g}, \sigma_{{diff}}={std1:.3g}, \mu_{{frac}}={mu2:.3g}, \sigma_{{frac}}={std2:.3g}$"))
  axs[0].set_xlabel("Predicted NMK - True NMK")
  axs[1].set_xlabel("(Predicted NMK - True NMK)/(True NMK)")
  axs[0].set_ylabel("Frequency")
  fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  fig.suptitle(title)
  plt.show()
  print("Sample size: " + str(sample_size))
  return diff_mus, diff_stds, frac_mus, frac_stds
