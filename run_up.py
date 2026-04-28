
import numpy as np
import time as time
import matplotlib.pyplot as plt
import multiscale_up_formulation
import up_formulation
'''
Compression of a Poroelastic cylinder

Confined = True : Confined Compression
Confined = False : Unconfined Compression

Draw Results = True : Draws results in WebGUI

'''
Confined = True
DrawResults = False
R = 3.6 * 10**(-3) # [m] Radius of the cylindrical sample
H = 1e-3 # [m] Height of the cylindrical sample

# Material Properties

nu = 0.49 # Poisson's ratio
G = 2e3
alpha = 0.9 # Biot coefficient
k = 1e-18 # [m^2] Intrinsic Permeability
phi = 0.8 # Porosity
viscosity = 1e-3 # [N s] Fluid viscosity
chi = 1e-10 # [Pa^-1] Fluid Compressibility
rho = 1000 # [kg/m^3] Fluid density
g = -9.81 # [m/s^2] Gravitational acceleration

# Simulation Parameters

dt = 1.25e-3 # [s] Starting Time step size
dt_max = 1000 # [s] Maximum time step size for dynamic time stepping
t_end = 300 # [s] Total simulation time
dt_growth = 1.045 # Growth factor for dynamic time stepping

u_max = H*0.1 # [m] Maximum displacement of the compression
compression_time = 92.2 # [s] Time duration for which the compression is applied

# Mesh Parameters
nx = 10 # Number of elements in X direction (radial)
ny = 20 # Number of elements in Y direction (axial)
order = 3 # Polynomial order for the finite element space
grading = 0.95 # Mesh grading parameter, controls bias of elements near the surface
 
# Solver settings
tol = 1e-16 # GMRes solver tolerance
maxsteps = 500 # GMRes iterations per timestep
restart = 150 # GMRes iterations before restart 

# Timing code
t0 = time.time()
time_vals, F_solid, F_fluid = multiscale_up_formulation.ConfinedCompression(G, nu, viscosity, alpha, phi, k,
     chi, rho, g, dt, dt_max, dt_growth, R, H, u_max , compression_time, t_end, nx,
     ny, grading, order, tol, maxsteps, restart, Confined, DrawResults)
t1 = time.time()
print("Simulation time: ", t1 - t0, " seconds", flush=True)
plt.plot(time_vals, F_solid,'--', color='green',lw=2,  label='Solid Force')
plt.plot(time_vals, F_fluid,'--', color='blue',lw=2,  label='Fluid Force')
plt.plot(np.array(time_vals), np.array(F_solid) + np.array(F_fluid), label='Total Force', linestyle='-', color='black', lw=3)
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Confined Compression: Force vs Time')
plt.legend()
plt.grid()
plt.xlim(0, 300)
print('max force = ' + str(np.max(F_solid + F_fluid)))
plt.show()





##### TO DO: Change alpha to be a tensor #####


##### TO DO: Extract p, grad p, strain from each timestep and give it to a function to update parameters #####