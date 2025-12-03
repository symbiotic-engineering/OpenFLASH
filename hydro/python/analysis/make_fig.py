import matplotlib.pyplot as plt
import os

plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Sample Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.savefig("pubs/JFM/figs/sample_plot.png")