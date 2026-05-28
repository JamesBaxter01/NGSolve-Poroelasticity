import importlib
import multiscale_up_formulation_tensor as upC
importlib.reload(upC)
import numpy as np 
import matplotlib.pyplot as plt

C_cauchy_6x6 = np.array(
[[8460, 4120, 4120,    0,    0,    0],
 [4120, 8460, 4120,    0,    0,    0],
 [4120, 4120, 8460,    0,    0,    0],
 [   0,    0,    0, 2970,    0,    0],
 [   0,    0,    0,    0, 2970,   -0],
 [   0,    0,    0,    0,   -0, 2970]])

K_perm = np.array([
    [4.585, 6.623e-4, 4.644e-4],
    [6.623e-4, 4.587, 8.201e-4],
    [4.644e-4, 8.201e-4, 4.586],
]) * 1e-19


Confined = True
DrawResults = False
R = 3.6 * 10**(-3) # [m] Radius of the cylindrical sample
H = 0.922e-3 # [m] Height of the cylindrical sample

# Material Properties


alpha = 0.85 # Biot coefficient
#k = 1.8e-18 # [m^2] Intrinsic Permeability
phi = 0.76 # Porosity
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

print(1.8e-18 / 4.585e-19)

perm_scaling_factor = 1.8e-18 / 4.585e-19

C_scaling_factor = 9

C = C_cauchy_6x6 * C_scaling_factor
K_perm_scaled = K_perm * perm_scaling_factor

time_vals, F_solid, F_fluid = upC.ConfinedCompression(
    C,
    viscosity,
    alpha,
    phi,
    K_perm_scaled,
    chi,
    rho,
    g,
    dt,
    dt_max,
    dt_growth,
    R,
    H,
    u_max,
    compression_time,
    t_end,
    nx,
    ny,
    grading,
    order,
    tol,
    maxsteps,
    restart,
    DrawResults=False
)