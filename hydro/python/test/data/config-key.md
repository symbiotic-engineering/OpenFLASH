Parenthetical values are matching points (real, imag), with rtol = 0.03, real atol = 0.01, imag atol = 0.0001. <br> 
Other values are what was inputted into capytaine for the corresponding configurations, and what the hydro coefficient outputs were. <br>
Note: Capytaine convention does not multiply hydro coefficients by a factor of h^3.<br>
The hydro coefficients/point matching in double parenthesis was data given by the code at the bottom of test_potential.py,
which matches better sometimes.

config0:<br>
h = 1.001 <br>
d = [0.5, 0.25] <br>
a = [0.5, 1] <br>
heaving = [1, 1] <br>
m0 = 1 <br>
g = 9.81 <br>
rho = 1023 <br>
zdensities = [10, 10] <br>
rdensities = [20, 20] <br>
tdensities = [50, 100] <br>
added_mass = 1620.53, radiation_damping = 3221.55, (2500, 2500) <br>
((added_mass = 1586.99, radiation_damping = 3192.55, (2500, 2500)))

config1: <br>
h = 1.5 <br>
d = [1.1, 0.85, 0.75, 0.4, 0.15] <br>
a = [0.3, 0.5, 1, 1.2, 1.6] <br>
heaving = [1, 1, 1, 1, 1] <br>
m0 = 1 <br>
g = 9.81 <br>
rho = 1023 <br>
zdensities = [20, 10, 30, 20, 15]<br>
rdensities = [10, 10, 20, 10, 15]<br>
tdensities = [40, 50, 70, 80, 100]<br>
added_mass = 4760.37, radiation_damping = 11539.05, (2500,2500)<br>
((added_mass = 4684.70, radiation_damping = 11442.87, (2499, 1825)))

config2:<br>
h = 100<br>
d = [29, 7, 4]<br>
a = [3, 5, 10]<br>
heaving = [1, 1, 1]<br> 
m0 = 1<br>
g = 9.81<br>
rho = 1023<br>
zdensities = [40, 10, 10]<br>
rdensities = [15, 10, 25]<br>
tdensities = [50, 80, 200]<br>
added_mass = 1386873.78, radiation_damping = 283.92, (2450, 2439)<br>
((added_mass = 1355730.94, radiation_damping = 184.13, (2464, 2455)))

config3:<br>
h = 1.9<br>
d = [0.5, 0.7, 0.8, 0.2, 0.5]<br>
a = [0.3, 0.5, 1, 1.2, 1.6]<br>
heaving = [1, 1, 1, 1, 1]<br>
m0 = 1<br>
g = 9.81<br>
rho = 1023<br>
zdensities = [15, 10, 30, 15, 25]<br>
rdensities = [10, 10, 20, 10, 15]<br>
tdensities = [40, 50, 70, 80, 100]<br>
added_mass = 3470.42, radiation_damping = 6124.83, (621, 408) <br>
((added_mass = 5863.80, radiation_damping = 6745.27, (2478, 698)))

config4:<br>
h = 1.001 <br>
d = [0.5, 0.25] <br>
a = [0.5, 1] <br>
heaving = [0, 1] <br>
m0 = 1<br>
g = 9.81<br>
rho = 1023<br>
zdensities = [10, 10] <br>
rdensities = [20, 20] <br>
tdensities = [50, 100] <br>
added_mass = 943.31, radiation_damping = 1847.53, (2500, 2500)

config5:<br>
h = 1.001 <br>
d = [0.5, 0.25] <br>
a = [0.5, 1] <br>
heaving = [1, 0] <br>
m0 = 1<br>
g = 9.81<br>
rho = 1023<br>
zdensities = [10, 10] <br>
rdensities = [20, 20] <br>
tdensities = [50, 100] <br>
added_mass = 291.39, radiation_damping = 189.86, (2500, 2500)

config6:<br>
h = 100<br>
d = [29, 7, 4]<br>
a = [3, 5, 10]<br>
heaving = [0, 1, 1]<br> 
m0 = 1<br>
g = 9.81<br>
rho = 1023<br>
zdensities = [40, 10, 10]<br>
rdensities = [15, 10, 25]<br>
tdensities = [50, 80, 200]<br>
added_mass = 1306746.66, radiation_damping = 279.84, (2479,2439)
