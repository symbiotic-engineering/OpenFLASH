# These functions are used in many files, 

import capytaine as cpt
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys, os

# removes capytaine warnings from clogging outputs
import logging

class CapytaineSlantSolver:
    
    # arguments for whether or not to display mesh, panel count, hydro coefficients, computation time, or excitation phase
    def __init__(self, mesh, panel_count, hydros, times, phase):
      self.show_mesh = mesh
      self.show_pc = panel_count
      self.show_hydros = hydros
      self.show_times = times
      self.show_phase = phase
      self.solver = cpt.BEMSolver()
      logging.getLogger("capytaine").setLevel(logging.ERROR)

    # use to get rid of prints
    def __deafen(self, function, *args, **kwargs):
        real_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, "w")
            output = function(*args, **kwargs)
        finally:
            sys.stdout = real_stdout
        return output

    def __timed_solve(self, problem, reps):
        t_lst = []
        for i in range(reps):
            t0 = time.perf_counter()
            result = self.solver.solve(problem, keep_details = True)
            t1 = time.perf_counter()
            t_lst.append(t1 - t0)
        tdiff = sum(t_lst)/reps
        return result, tdiff

    def __get_points(self, a, d_in, d_out): # These points define the outline of the body
        pt_lst = [(0, - d_in[0])]
        for i in range(len(a)):
            pt_lst.append((a[i], - d_out[i]))
            if i < (len(a) - 1): # not last body region
                if d_out[i] != d_in[i + 1]: # vertical face exists
                    pt_lst.append((a[i], - d_in[i + 1]))
            else: # need vertical face to water surface
                pt_lst.append((a[i], 0))
        return pt_lst

    # compute number of panels along each surface given total number along the outline
    def __get_f_densities(self, pt_lst, total_units):
        face_lengths = np.array([])
        for i in range(len(pt_lst) - 1):
            p1, p2 = pt_lst[i], pt_lst[i + 1]
            face_length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) # one of these two values will be zero
            face_lengths = np.append(face_lengths, face_length)
        total_length = sum(face_lengths)
        each_face_densities = np.vectorize(lambda x: max(1, x/total_length * total_units))(face_lengths) # each face needs at least one panel
        remainders = each_face_densities % 1
        each_face_densities = each_face_densities.astype(int)
        remaining_units = total_units - sum(each_face_densities)
        if remaining_units < 0: # high proportion of small faces
            for u in range(remaining_units * -1):
                i = np.argmax(each_face_densities) # cut density from the largest faces
                each_face_densities[i] = (each_face_densities[i]) - 1
        else:
            for u in range(remaining_units): # distribute remaining units where most needed
                i = np.argmax(remainders)
                each_face_densities[i] = (each_face_densities[i]) + 1
                remainders[i] = 0
        assert sum(each_face_densities) == total_units
        return each_face_densities

    def __make_face(self, p1, p2, f_density, t_density):
        zarr = np.linspace(p1[1], p2[1], f_density + 1)
        rarr = np.linspace(p1[0], p2[0], f_density + 1)
        xyz = np.array([np.array([x/np.sqrt(2),y/np.sqrt(2),z]) for x,y,z in zip(rarr,rarr,zarr)])
        return cpt.AxialSymmetricMesh.from_profile(xyz, nphi = t_density)

    def __faces_and_heaves(self, heave_status, p1, p2, f_density, t_density, meshes, mask, panel_ct):
        mesh = self.__make_face(p1, p2, f_density, t_density)
        meshes += mesh
        new_panels = f_density * t_density
        if heave_status:
            direction = [0, 0, 1]
        else:
            direction = [0, 0, 0]
        for i in range(new_panels):
            mask.append(direction)
        return meshes, mask, (panel_ct + new_panels)

    def get_excitation_phase(self, result):
        return np.angle((cpt.assemble_dataset([result]))["excitation_force"][0][0][0])

    def __make_body(self, pts, t_densities, f_densities, heaving):
        meshes = cpt.meshes.meshes.Mesh()
        panel_ct = 0
        mask = []
        heave_region = -1
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            if p1[0] != p2[0]: # face spans some horizontal distance
                heave_region += 1 # advance to next region
                # make a horizontal face
                meshes, mask, panel_ct = self.__faces_and_heaves(heaving[heave_region], p1, p2, f_densities[i], t_densities[heave_region], meshes, mask, panel_ct)
            else: # make a vertical face
                if p1[1] <= p2[1]: # body on inside
                  j = heave_region # defer to variables of inner region
                else: # body on outside
                  j = heave_region + 1 # defer to variables of outer region
                meshes, mask, panel_ct = self.__faces_and_heaves(heaving[j], p1, p2, f_densities[i], t_densities[j], meshes, mask, panel_ct)
        body = self.__deafen(cpt.FloatingBody, mesh = meshes) # unclosed boundary warnings
        # , lid_mesh = meshes.generate_lid() # consider adding lid mesh to above function
        return body, panel_ct, mask

    def construct_and_solve(self, a, d_in, d_out, heaving, t_densities, face_units, h, m0, rho, reps, f_densities = None):
        pt_lst = self.__get_points(a, d_in, d_out)
        if f_densities is None:
            f_densities = self.__get_f_densities(pt_lst, face_units)
        
        body, panel_count, mask = self.__make_body(pt_lst, t_densities, f_densities, heaving)
        body.dofs["Heave"] = mask  
        if self.show_mesh: body.show_matplotlib()
        
        rad_problem = cpt.RadiationProblem(body = body, wavenumber = m0, water_depth = h, rho = rho)
        result, t_diff = self.__timed_solve(rad_problem, reps)

        diff_problem = cpt.DiffractionProblem(body = body, wavenumber = m0, water_depth = h, rho = rho)
        result_d, t_diff_d = self.__timed_solve(diff_problem, reps)

        if self.show_pc: print("Panel Count: ", panel_count)
        if self.show_hydros:
          print(result.added_mass)
          print(result.radiation_damping)
        if self.show_times:
          print("Solve Time (Radiation): ", t_diff)
          print("Solve Time (Diffraction): ", t_diff_d)
        if self.show_phase: print("Excitation Phase: ", self.get_excitation_phase(result_d))
        return result, t_diff, result_d, t_diff_d, panel_count
    
    def __get_region(self, a, r):
        # assumes r >= 0
        # returns -1 if in outermost, extends-to-infinity region
        region = 0
        for rad in a:
            if r <= rad:
                return region
            else:
                region += 1
        return -1
    
    def __above_line(self, p1, p2, x, y):
        x1, y1 = p1
        x2, y2 = p2

        if x2 == x1:
            raise ValueError(f"The line defined by points {p1} and {p2} is vertical.")

        slope = (y2 - y1) / (x2 - x1)
        y_on_line = y1 + slope * (x - x1)

        return y > y_on_line
    
    
    # given a body definable by a, d_in, d_out, returns a function that says whether or not a given point is in the body.
    # that function assumes points are at/below the surface.
    def get_body_bounds_from_regions(self, a, d_in, d_out):
        pt_lst = self.__get_points(a, d_in, d_out)

        def is_inside(r, z):
            region = self. __get_region(a, r)
            if region == -1: return False
            inner_rad = 0 if region == 0 else a[region - 1]
            return self.__above_line((inner_rad, - d_in[region]), (a[region], -d_out[region]), r, z)
        
        return is_inside
        
    def __get_points_and_mask(self, h, a, d_in, d_out, res):

        is_inside = self.get_body_bounds_from_regions(a, d_in, d_out)

        R_range = np.linspace(0.0, 2 * a[-1], num = res)
        Z_range = np.linspace(0, -h, num = res) 
        R, Z = np.meshgrid(R_range, Z_range)

        # Flatten R and Z into 1D arrays
        r_flat = R.ravel()
        z_flat = Z.ravel()

        # Create a validity mask by applying not is_inside(r, z) to each point
        valid_mask = np.array([(not is_inside(r, z)) for r, z in zip(r_flat, z_flat)])

        # Create full 3D points for valid locations (assuming y = y_value)
        valid_points = np.column_stack((r_flat[valid_mask],
                                        np.full(np.sum(valid_mask), 0),
                                        z_flat[valid_mask]))
        
        return R, Z, valid_mask, valid_points
    
    def get_potential_array(self, h, a, d_in, d_out, res, rad_result):
        R, Z, valid_mask, valid_points = self.__get_points_and_mask(h, a, d_in, d_out, res)

        valid_results = self.solver.compute_potential(valid_points, rad_result)

        # Build output array filled with NaNs, alter valid points to finite.
        result = np.full(R.size, np.nan + np.nan*1j)
        result[valid_mask] = valid_results

        return R, Z, result.reshape(R.shape)

    def get_velocity_arrays(self, h, a, d_in, d_out, res, rad_result):
        R, Z, valid_mask, valid_points = self.__get_points_and_mask(h, a, d_in, d_out, res)

        valid_results = self.solver.compute_velocity(valid_points, rad_result)

        # Extract columns: r (or x) = computed[:, 0], z = computed[:, 2]
        r_vel_flat = np.full(R.size, np.nan + np.nan*1j)
        z_vel_flat = np.full(R.size, np.nan + np.nan*1j)

        r_vel_flat[valid_mask] = valid_results[:, 0]
        z_vel_flat[valid_mask] = valid_results[:, 2]

        # Reshape to original grid shape
        r_vel = r_vel_flat.reshape(R.shape)
        z_vel = z_vel_flat.reshape(R.shape)

        return R, Z, r_vel, z_vel
    
    def __plot_contour(self, R, Z, data, color_label, title):
        plt.contourf(R, Z, data, cmap='viridis', levels = 50)
        plt.colorbar(label = color_label)
        plt.contour(R, Z, data, colors='black', linestyles='solid', linewidths=0.05, levels=50)
        plt.xlabel('R')
        plt.ylabel('Z')
        plt.title(title)
        plt.show()

    def plot_potential(self, h, a, d_in, d_out, res, rad_result, MEEM_convention = False):
        # Also returns arrays, no recomputation necessary
        R, Z, potential_array = self.get_potential_array(h, a, d_in, d_out, res, rad_result)

        if MEEM_convention:
            omega = rad_result.omega
            potential_array = potential_array * 1j / omega

        real_phi = np.real(potential_array)
        imag_phi = np.imag(potential_array)

        self.__plot_contour(R, Z, real_phi, "Potential", "Real Potential with Capytaine")
        self.__plot_contour(R, Z, imag_phi, "Potential", "Imaginary Potential with Capytaine")

        return real_phi, imag_phi

    def plot_velocities(self, h, a, d_in, d_out, res, rad_result, MEEM_convention = False):
        R, Z, vr, vz = self.get_velocity_arrays(h, a, d_in, d_out, res, rad_result)

        if MEEM_convention:
            omega = rad_result.omega
            vr = vr * 1j / omega
            vz = vz * 1j / omega

        real_vr = np.real(vr)
        imag_vr = np.imag(vr)
        real_vz = np.real(vz)
        imag_vz = np.imag(vz)

        self.__plot_contour(R, Z, real_vr, "Radial Velocity", "Real radial velocity with Capytaine")
        self.__plot_contour(R, Z, imag_vr, "Radial Velocity", "Imag radial velocity with Capytaine")
        self.__plot_contour(R, Z, real_vz, "Vertical Velocity", "Real vertical velocity with Capytaine")
        self.__plot_contour(R, Z, imag_vz, "Vetical Velocity", "Imag vertical velocity with Capytaine")

        return real_vr, imag_vr, real_vz, imag_vz
    
    def plot_from_array(self, h, a, data, color_lab = None, title = None):
        res = len(data)
        R_range = np.linspace(0.0, 2 * a[-1], num = res)
        Z_range = np.linspace(0, -h, num = res) 
        R, Z = np.meshgrid(R_range, Z_range)
        self.__plot_contour(R, Z, data, color_lab, title)

    

    
    