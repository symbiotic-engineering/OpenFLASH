import numpy as np
import matplotlib.pyplot as plt
import colors   # import the colors.py functions
import fonts # import the fonts.py functions
# Generate some data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create the plot
colors.get_colors()     # create the color variables
colors.what_colors()    # remind yourself of the color names if needed
print(colors.purple)    # access the color variables using . notation
fonts.get_fonts()
plt.plot(x, y, label='sin(x)', color=colors.purple, linestyle='-', linewidth=2)

# Show plot
plt.show()
