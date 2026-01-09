import numpy as np

import sys
import os
sys.path.append(os.path.relpath('../../'))
sys.path.append(os.path.relpath('../'))
from multi_condensed import Problem
from helpers import *

import pickle

# def subdictionary(dict0, key_lst):
#   return {k: dict0[k] for k in key_lst if k in dict0}


target_file = "data/ps5.pkl" #"data/ps4-all-configs-v3.pkl"

with open("data/ps5.pkl", "rb") as f:
  configurations = pickle.load(f)

# eliminate unneeded data
# key_lst = ['name', 'h', 'a', 'd_in', 'd_out', 'heaving', 'm0', 'rho',
#            'MEEM box AM', 'MEEM box DP', 'CPT box AM', 'CPT box DP', 'CPT slant AM', 'CPT slant DP']
# configurations = [subdictionary(config, key_lst) for config in configurations]

def solve_prob(config, res, version, nmk): # Specifically for our 2-region case
  NMK = [nmk, nmk, nmk]
  a, d_in, d_out, heaving = config["a"], config["d_in"], config["d_out"], config["heaving"]
  d_prime, a_prime, heaving_prime, NMK_prime, _ = slant_approx_vars(a, d_in, d_out, heaving, NMK, res, version)
  prob = Problem(config["h"], d_prime, a_prime, heaving_prime, NMK_prime, config["m0"], config["rho"])
  a_matrix = prob.a_matrix()
  b_vector = prob.b_vector()
  x = prob.get_unknown_coeffs(a_matrix, b_vector)
  am, dp = prob.hydro_coeffs(x, "capytaine")
  return am, dp

# I will run this with version 3 first.
# After getting data to interpret in the meanwhile, I'll set this to run on version 2, then 1.
# Then splice it all together.

for config in (configurations[19:38] + configurations[38:57]):
   a = config["a"]
   a[1] = a[0] + 4
   config["a"] = a

# Fix NMK = 200, vary res from 10 to 60 by intervals of 10
resolutions = [10, 20, 30, 40, 50, 60]
for config in (configurations[20:38] + configurations[39:57]):
   config["AMs by res"] = []
   config["DPs by res"] = []
   for res in resolutions:
      am, dp = solve_prob(config, res, 2, 200)
      config["AMs by res"].append(am)
      config["DPs by res"].append(dp)
      update_data_file(configurations, target_file)
   print("Finished res variation for " + config["name"])

# Fix res = 50, vary NMK = [50, 100, 150, 200, 250, 300]
nmks = [50, 100, 150, 200, 250, 300]
for config in (configurations[20:38] + configurations[39:57]):
   config["AMs by nmk"] = []
   config["DPs by nmk"] = []
   for nmk in nmks:
      am, dp = solve_prob(config, 50, 2, nmk)
      config["AMs by nmk"].append(am)
      config["DPs by nmk"].append(dp)
      update_data_file(configurations, target_file)
   print("Finished nmk variation for " + config["name"])