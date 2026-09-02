import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np


###########################
# Used in ps4, ps5 notebooks
def plot_multiple_fade(x, ys, colors = ["Reds", "Greens", "Blues"], labs = None, last_k = None, title = "plot", xlab = "x", ylab = "y", hline = None, hlab = None):
    if hline is not None:
      plt.axhline(y = hline, color='orange', label = hlab)

    if type(ys[0][0]) != list:
       ys = [ys]

    if labs is None or type(labs[0]) != list:
       labs = [labs]

    for i in range(len(ys)):
      plot_fade(x, ys[i], labs[i], last_k, colors[i])

    if xlab == "theta":
      ax = plt.gca()
      ax.xaxis.set_major_locator(MultipleLocator(np.pi / 8))
      ax.xaxis.set_major_formatter(FuncFormatter(angle_format_func))

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def angle_format_func(value, tick_number):
    frac = 1 / 16
    multiple = round(value / (np.pi * frac))
    num = multiple
    denom = int(1 / frac)

    if multiple == 0:
        return "0"

    # Reduce the fraction
    common = np.gcd(abs(num), denom)
    num //= common
    denom //= common

    if denom == 1:
        coeff = f"{num}" if abs(num) != 1 else ("-" if num == -1 else "")
        return rf"${coeff}\pi$"
    else:
        coeff = "-" if num < 0 else ""
        num_string = "" if abs(num) == 1 else abs(num)
        return rf"${coeff}\frac{{{num_string}\pi}}{{{denom}}}$"

def plot_fade(x, ys, labs = None, last_k = None, cname = "Blue"):
  if labs is None:
      labs = [None] * len(ys)

  if last_k is not None: # get rid of early extreme values for scaling
      x = x[-last_k:]
      ys = [y[-last_k:] for y in ys]

  cmap = plt.get_cmap(cname)
  color_vals = np.linspace(0.8, 0.2, len(ys))

  for y, label, color_val in zip(ys, labs, color_vals):
        plt.plot(x, y, label = label, color=cmap(color_val))

def make_label(num_lst, key):
  return [key + " = " + str(num) for num in num_lst]

def ratio_conversion(arr):
  return [[value / row[-1] for value in row] for row in arr]

def percent_off(arr):
    return [
        [((row[j] / row[j - 1]) - 1) for j in range(1, len(row))]
        for row in arr
    ]

def extract(configurations, key, length):
  all_series = []
  for i in range(length):
    single_series = [config[key][i] for config in configurations]
    all_series.append(single_series)
  return all_series

def list_ratios(lst_of_lsts, base_lst):
  # Element i of each list  is expressed as a ratio to the the ith entry of base_lst
  output = []
  for lst in lst_of_lsts:
    output.append([lst[i]/base_lst[i] for i in range(len(base_lst))])
  return output