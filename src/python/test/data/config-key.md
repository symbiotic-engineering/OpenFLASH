Match is the number of matching potential points for (real, imag), with rtol = 0.01, real/imag atol = 0.01 (2500 max).<br> 
t_densities and face_units determined the number of panels in the Capytaine mesh.  <br>
Capytaine added_mass and radiation_damping are listed first.<br>
MEEM added_mass and radiation_damping with 100 coefficients per region, converted to the capytaine convention, are listed second. <br>
(Capytaine convention does not multiply hydro coefficients by a factor of h^3).<br>

config0:<br>
h = 1.001 <br>
d = [0.5, 0.25] <br>
a = [0.5, 1] <br>
heaving = [1, 1] <br>
t_densities = [50, 100] <br>
face_units = 90 <br>
panels = 6750 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 1586.35, radiation_damping = 3188.92 <br>
MEEM: added_mass = 1600.79, radiation_damping = 3222.35 <br>
Match: (2500, 2500)

config1: <br>
h = 1.5 <br>
d = [1.1, 0.85, 0.75, 0.4, 0.15] <br>
a = [0.3, 0.5, 1, 1.2, 1.6] <br>
heaving = [1, 1, 1, 1, 1] <br>
t_densities = [30, 50, 100, 120, 160] <br>
face_units = 93 <br>
panels = 8930 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 4746.05, radiation_damping = 11688.44 <br>
MEEM: added_mass = 4792.01, radiation_damping = 11683.90 <br>
Match: (2500, 2500)

config2:<br>
h = 100<br>
d = [29, 7, 4]<br>
a = [3, 5, 10]<br>
heaving = [1, 1, 1]<br> 
t_densities = [30, 50, 100]<br> 
face_units = 110 <br> 
panels = 5330 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 1340420.77, radiation_damping = 207.74 <br>
MEEM: added_mass = 1367585.23, radiation_damping = 210.62 <br>
Match: (2378, 2500)

config3:<br>
h = 1.9<br>
d = [0.5, 0.7, 0.8, 0.2, 0.5]<br>
a = [0.3, 0.5, 1, 1.2, 1.6]<br>
heaving = [1, 1, 1, 1, 1]<br>
t_densities = [30, 50, 100, 120, 160] <br>
face_units = 105 <br>
panels = 11660 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 5869.05, radiation_damping = 6875.46 <br>
MEEM: added_mass = 6057.48, radiation_damping = 6886.69 <br>
Match: (2341, 2429)

config4:<br>
h = 1.001 <br>
d = [0.5, 0.25] <br>
a = [0.5, 1] <br>
heaving = [0, 1] <br>
t_densities = [50, 100] <br>
face_units = 90 <br>
panels = 6750 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 924.69, radiation_damping = 1836.75 <br>
MEEM: added_mass = 931.66, radiation_damping = 1852.36 <br>
Match: (2500, 2500)

config5:<br>
h = 1.001 <br>
d = [0.5, 0.25] <br>
a = [0.5, 1] <br>
heaving = [1, 0] <br>
t_densities = [50, 100] <br>
face_units = 90 <br>
panels = 6750 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 286.31, radiation_damping = 185.42 <br>
MEEM: added_mass = 288.36, radiation_damping = 188.42 <br>
Match: (2500, 2500)

config6:<br>
h = 100<br>
d = [29, 7, 4]<br>
a = [3, 5, 10]<br>
heaving = [0, 1, 1]<br> 
t_densities = [30, 50, 100]<br> 
face_units = 110 <br> 
panels = 5330 <br>
m0 = 1 <br>
rho = 1023 <br>
g = 9.81 <br>
CPT: added_mass = 1264061.71, radiation_damping = 205.57 <br>
MEEM: added_mass = 1290352.23, radiation_damping = 208.55 <br>
Match: (2417, 2500)
