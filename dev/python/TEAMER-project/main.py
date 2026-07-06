import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openflash as of
from scipy.special import hankel1e 
import capytaine as cpt
import xarray as xr
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
import os
input_data_dir = os.path.dirname(__file__)
output_dir = os.path.join(input_data_dir, 'outputs')

######################## INPUTS ##########################
# Geometry
r1 = 2.5/2 # bottom radius [m]
r2 = 8.4/2 # top radius [m]
d1 = 14.45 # draft [m]
d2 = 14.45-7.32 # vertical distance to bottom of slant [m]
d3 = 14.45-7.32-5.08 # vertical distance to top of slant [m]

# Settings
rho=1000 # water density [kg/m**3]
g=9.81 # acceleration due to gravity [m/s**2]
omegas = np.linspace(0.4,1.5,10) # frequencies [rad/s]
h = 100 # water depth (m)


############################ FUNCTIONS ###########################
def CorPower_geom(r1,r2,d1,d2,d3,num_subdivs):
    d_list = np.linspace(d2,d3,num_subdivs)
    a_list = np.linspace(r1,r2,num_subdivs)
    d_list[0] = d1
    return d_list, a_list

def compute_hydrostatics(a_list, d_list, rho, g):
    hydrostatic_stiffness = 0
    mass = 0
    for i in range(len(a_list)):    
        if i==0:
            area_i = np.pi*a_list[i]**2
        else:
            area_i = np.pi*(a_list[i]**2-a_list[i-1]**2)
        volume_i = area_i * d_list[i]
        hydrostatic_stiffness += rho * g * area_i
        mass += rho * volume_i 
    return hydrostatic_stiffness, mass 


def compute_CorPower_data(r1,r2,d1,d2,d3,omegas,rho,g,h,show_geom=False,save_data=True):

    heaving_list = True 
    num_subdivs=10
    truncastion_order = 50

    # Generate correct input
    d_list, a_list = CorPower_geom(r1,r2,d1,d2,d3,num_subdivs)
    num_regions = len(d_list)
    NMK = [truncastion_order] * (num_regions+1)

    # Show plot of geom
    if show_geom:
        fig, ax = plt.subplots()
        for i, (a, d) in enumerate(zip(a_list,d_list)):
            if i==0:
                a_in=0 
                ax.fill([a_in,a,a,a_in],[-d,-d,0,0], color=[0.7,0.7,0.7], edgecolor='black', label='Approximated geometry')       
            else:
                a_in=a_list[i-1]
                ax.fill([a_in,a,a,a_in],[-d,-d,0,0], color=[0.7,0.7,0.7], edgecolor='black')
        ax.plot([0,r1,r1,r2,r2],[-d1,-d1,-d2,-d3,0],color="red",label="True outline")
        ax.set_aspect('equal')
        ax.legend(loc='best', frameon=False)
        ax.set_xlim([0, -np.min(-d_list)])
        ax.set_ylim([np.min(-d_list),0])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title("CorPower WEC profile")
        # fig.savefig("CorPower_profile", format='pdf', dpi=300)

    # Single Body
    bodies_sweep = []
    body = of.SteppedBody(
        a=np.array(a_list),
        d=np.array(d_list),
        slant_angle= np.zeros_like(a_list),
        heaving=heaving_list
    )
    bodies_sweep.append(body)

    # Create arrangement
    arrangement_sweep = of.ConcentricBodyGroup(bodies_sweep)

    # Create geometry
    geometry_sweep = of.BasicRegionGeometry(
        body_arrangement=arrangement_sweep,
        h=h,
        NMK=NMK
    )

    # Create the MEEMProblem instance
    problem = of.MEEMProblem(geometry_sweep)

    # Set the frequencies for the sweep
    problem.set_frequencies(omegas)

    # Initialize a new MEEM Engine for this problem
    engine = of.MEEMEngine(problem_list=[problem])

    # Solve
    results = engine.run_and_store_results(0)
    OpenFLASH_results = results.get_results()

    # Reorganize to match Capytaine 
    added_mass_data = OpenFLASH_results.added_mass.values
    radiation_damping_data = OpenFLASH_results.damping.values
    excitation_force_mag_data = OpenFLASH_results.excitation_force.values
    excitation_force_phase_data = OpenFLASH_results.excitation_phase.values
    excitation_force_data = excitation_force_mag_data[:,None] * np.exp(1j * excitation_force_phase_data[:,None])

    wavenumbers = np.array([of.wavenumber(omega,h) for omega in omegas])

    influenced_dof_cat = pd.Categorical(["Heave"])
    radiating_dof_cat = pd.Categorical(["Heave"])

    # Create the Dataset
    hydrodynamic_dataset = xr.Dataset(
        data_vars={
            "added_mass": (["omega", "radiating_dof", "influenced_dof"], added_mass_data),
            "radiation_damping": (["omega", "radiating_dof", "influenced_dof"], radiation_damping_data),
            "excitation_force": (["omega", "wave_direction", "influenced_dof"], excitation_force_data)
        },
        coords={
            "g":g,
            "rho":rho,
            "water_depth":h,
            "forward_speed":0.0,
            "wave_direction": np.array([0.0]),
            "omega": omegas,
            "radiating_dof": radiating_dof_cat,
            "influenced_dof": influenced_dof_cat,
            "period": ("omega", (2*np.pi)/omegas), # seconds
            "wavenumber": ("omega", wavenumbers), # rad/m
            "wavelength": ("omega", (2*np.pi)/wavenumbers)                
        }
    )

    # Export
    # if save_data:
    #     results.export_to_netcdf("hydrodynamic_data.nc")

    # Hydrostatics
    hydrostatic_stiffness, mass = compute_hydrostatics(a_list, d_list, rho, g)
    hydrostatic_dataset = {}
    hydrostatic_dataset["center_of_buoyancy"] = np.array([0,0,0])
    hydrostatic_dataset["center_of_mass"] = np.array([0,0,0])
    hydrostatic_dataset["disp_mass"] = mass
    hydrostatic_dataset["hydrostatic_stiffness"] = xr.DataArray([[hydrostatic_stiffness]],
                                dims=['influenced_dof', 'radiating_dof'],
                                coords={'influenced_dof': ["Heave"],
                                'radiating_dof': ["Heave"]},
                                name="hydrostatic_stiffness"
                                )
    

    # Save   
    if save_data: 
        cpt.export_dataset(os.path.join(output_dir, 'CorPower_hydrodynamics.nc'), hydrodynamic_dataset)


    return hydrodynamic_dataset, hydrostatic_dataset

hydrodynamic_dataset, hydrostatic_dataset = compute_CorPower_data(r1,r2,d1,d2,d3,omegas,rho,g,h,show_geom=False,save_data=True)


# Practice
# mesh = cpt.mesh_vertical_cylinder(length=10.0, radius=1.0, center=(0, 0, 0),faces_max_radius=0.3)
# fb = cpt.FloatingBody(mesh=mesh,
#                       center_of_mass=(0,0,0))
# fb.rotation_center = np.array([0,0,0])
# fb.add_translation_dof(name="Heave")
# input_dict = {"omega":np.linspace(2,3,10),
#                 "wave_direction":[0.0],
#                 "radiating_dof":["Heave"],
#                 "water_depth":1000,
#                 "rho":1000,
#                 "g":9.81}
# test_matrix = xr.Dataset(coords=input_dict)
# dataset = cpt.BEMSolver().fill_dataset(test_matrix, fb)



# cpt_hydrostatics = fb.compute_hydrostatics()


# from capytaine.io.legacy import export_hydrostatics_from_dataset
# export_hydrostatics_from_dataset(output_dir, cpt_hydrostatics)


# for hydrostatics, may need to use cpt since the function for generating the .dat file requires bodies.
