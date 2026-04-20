from ngsolve import *
from netgen.meshing import Mesh as NetMesh, MeshPoint, Element2D, Element1D, FaceDescriptor
import netgen.meshing as meshing

def MakeStructuredMesh(nx, ny, Lx=3.6, Ly=1.0):
    m = NetMesh()
    m.dim = 2
    r = 0.95  # grading ratio (smaller = more aggressive)
    # Add points row by row
    pids = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            x = Lx * i / nx
            y = Ly * (1 - r**j) / (1 - r**ny)
            row.append(m.Add(MeshPoint(meshing.Pnt(x, y, 0))))
        pids.append(row)
    
    # Add face descriptor
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    
    # Add quad elements
    for j in range(ny):
        for i in range(nx):
            m.Add(Element2D(1, [pids[j][i], pids[j][i+1],
                                pids[j+1][i+1], pids[j+1][i]]))
    
    # Boundary edges
    # Bottom (bc=1), Right (bc=2), Top (bc=3), Left (bc=4)
    for i in range(nx):
        m.Add(Element1D([pids[0][i], pids[0][i+1]], index=1))      # bottom
    for j in range(ny):
        m.Add(Element1D([pids[j][nx], pids[j+1][nx]], index=2))    # right
    for i in range(nx):
        m.Add(Element1D([pids[ny][i+1], pids[ny][i]], index=3))    # top
    for j in range(ny):
        m.Add(Element1D([pids[j+1][0], pids[j][0]], index=4))      # left

    m.SetBCName(0, "bottom")
    m.SetBCName(1, "sides")
    m.SetBCName(2, "top")
    m.SetBCName(3, "sides")

    return Mesh(m)

from netgen.occ import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import time as time
import matplotlib.pyplot as plt
from ngsolve.krylovspace import GMRes
from ngsolve.webgui import Draw 

def BulkModulus(G, nu):
    return 2*G*(1 + nu) / (3 - 6*nu)

def YoungsModulus(G, nu):
    return 2*G*(1 + nu)

def LameParameter(G, nu):
    return 2*G*nu / (1 - 2*nu)


def ConfinedCompression(G, nu, viscosity, alpha, phi, k, chi, rho, g, dt, dt_max, dt_growth, R, H, u_max , compression_time, t_end, nx, ny, order, DrawResults=False):
    L = H
    growth_factor = dt_growth


    lame_star = 2*nu / (1 - 2*nu)  # Dimensionless Lamé parameter

    C11, C12, C13, C14 = lame_star + 2, lame_star,     lame_star,     0
    C21, C22, C23, C24 = lame_star,     lame_star + 2, lame_star,     0
    C31, C32, C33, C34 = lame_star,     lame_star,     lame_star + 2, 0
    C41, C42, C43, C44 = 0,             0,             0,             1
    

    cauchy_values = (C11, C12, C13, C14,
                     C21, C22, C23, C24,
                     C31, C32, C33, C34,
                     C41, C42, C43, C44)
    
    Cauchy_tensor_star = CoefficientFunction(cauchy_values, dims=(4, 4)).Compile()

    def div_ax(u):
    # div(u) = du_r/dr + du_z/dz + u_r/r
        return (grad(u)[0,0] + u[0]/r + grad(u)[1,1])

    def eps_ax(u):
        # Returns the 4 non-zero strain components in a vector
        # [eps_rr, eps_zz, eps_th, 2*eps_rz]
        gu = grad(u)
        return CF((gu[0,0], gu[1,1], u[0]/r, gu[0,1] + gu[1,0]))
    
    
    def Stress_ax_Anisotropic(u_vec):
        e_ax = eps_ax(u_vec)
        sigma_v = Cauchy_tensor_star * e_ax
        
        # Mapping to a 3x3 for Axissymmetric (r, z, theta)
        # sigma_v[0] = rr, [1] = zz, [2] = theta-theta, [3] = rz
        return CF( (sigma_v[0], sigma_v[3], 0,
                    sigma_v[3], sigma_v[1], 0,
                    0,          0,          sigma_v[2]), dims=(3,3) )
    
    def AxialGrad(v):
        # This creates a 3x3 matrix from a 2D vector v
        # [dv_r/dr, dv_r/dz, 0]
        # [dv_z/dr, dv_z/dz, 0]
        # [0,       0,       v[0]/r]
        gv = Grad(v)
        return CF((gv[0,0], gv[0,1], 0,
                gv[1,0], gv[1,1], 0,
                0,       0,       v[0]/r), dims=(3,3))




    twopi = 2*np.pi
    k_ref = k
    
    # Parameter defining
    K = BulkModulus(G, nu) # [Pa] Bulk modulus
    S = phi * chi + ((1-alpha)*(alpha-phi)) / K    
    
    tau = L**2 * viscosity / (k * G)

    dt_star = dt / tau # Non-dimensional time step size
    S_star = S * G
 
    k11, k12 = k_ref, 0
    k21, k22 = 0, k_ref

    k_values = (k11/k_ref, k12/k_ref,
                k21/k_ref, k22/k_ref)
    
    k_star = CoefficientFunction(k_values, dims=(2,2)).Compile()

    R_Star = R / H
    H_Star = 1.0
    mesh = MakeStructuredMesh(nx, ny, Lx=R_Star, Ly=H_Star)


    V = VectorH1(mesh, order=order, dirichlet="top|bottom", dirichletx="sides")
    Q = H1(mesh,order=order-1, dirichlet="top")
    
    (u,v) = V.TnT()
    (p,q) = Q.TnT()

    gfu_star = GridFunction(V)
    gfp_star = GridFunction(Q)
    gfp_star.Set(0)

    u_old_star = GridFunction(V)
    u_old_star.Set((0,0))

    p_old_star = GridFunction(Q)
    p_old_star.Set(0)

    n = specialcf.normal(mesh.dim)
    # Weak forms
    traction = 0 # Traction BC

    tbar = CoefficientFunction((0, traction))
    b = CoefficientFunction((0, rho*g)) # Body force

    qbar = CoefficientFunction(0) # Flux BC

    r = x
    y = z
    
    if DrawResults == True:
        from ngsolve.webgui import Draw
        sceneu = Draw(gfu_star, mesh, deformation=True)
        scenep = Draw(G * gfp_star, mesh)

        # K* (Stiffness)
    # Note: Ensure Cauchy_tensor_star is a 4x4 matrix for the [rr, zz, th, rz] vector
    a_K = BilinearForm(V)
    a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
    #pre_a_K = Preconditioner(a_K, type="direct")
    a_K.Assemble()
    pre_a_K = a_K.mat.Inverse(freedofs = V.FreeDofs(), inverse="sparsecholesky")

    # Q* (Coupling) - Use the updated div_ax
    a_Q12 = BilinearForm(trialspace=Q, testspace=V)
    a_Q12 += p * div_ax(v) * twopi * r * dx
    a_Q12.Assemble()

    a_Q21 = BilinearForm(trialspace=V, testspace=Q)
    a_Q21 += q * div_ax(u) * twopi * r * dx
    a_Q21.Assemble()

    # S* (Storage)
    a_S = BilinearForm(Q)
    a_S += p * q * twopi * r * dx
    a_S.Assemble()

    # H* (Conductivity)
    a_H = BilinearForm(Q)
    a_H += (k_star * grad(p) * grad(q)) * twopi * r * dx
    a_H.Assemble()

    # Preconditioner for Pressure (H1-like norm)



    # Normalize traction and body forces
    tbar_star = tbar / G
    b_star = (b * L) / G

    # b_star is (f_r, f_z). Usually (0, -rho*g)
    b_f = LinearForm(V)
    b_f += (v * tbar_star) * twopi * r * ds("top")
    b_f += (v * b_star) * twopi * r * dx
    b_f.Assemble()

    qbar_star = (qbar * L) / (G * viscosity) 
    k_n = InnerProduct(n, k_star * n)
    b_q = LinearForm(Q)

    
    b_q += qbar_star * k_n * q * twopi * r * ds("top")
    b_q.Assemble()


    time_vals_nondim, F_solid, F_fluid = [], [], []

    t = 0
    t_end_star = t_end / tau


    
    u_max_star = u_max / L
    t_ramp = compression_time # [s] Time to reach full load
    t_ramp_star = t_ramp / tau


    # 1. Define the normal vector and vertical direction


    dt_star_max = dt_max / tau # Set a reasonable upper limit for your physical scaled time step size


    A = BlockMatrix([
        [a_K.mat,        -alpha * a_Q12.mat],
        [(alpha/dt_star) * a_Q21.mat,   a_H.mat + S_star/dt_star * a_S.mat]
        ])

    F = BlockVector([
        b_f.vec, 
        (alpha/dt_star) * a_Q21.mat * u_old_star.vec + 
                    (S_star/dt_star) * a_S.mat * p_old_star.vec + 
                    b_q.vec])

    # Create the Block solution vector (linking to your GridFunctions)
    sol = BlockVector([gfu_star.vec, gfp_star.vec])

    rhs_resid = F.CreateVector()
    correction = F.CreateVector()

    # This is for solid force
    v_test = GridFunction(V)
    v_test.Set(CF((0, 1)), definedon=mesh.Boundaries("bottom"))


    while t < t_end_star:
        t += dt_star
        #print(f"Time: {t*tau:.4f} s / {t_end:.4f} s", end='\r')
        uy_star = min(t / t_ramp_star, 1.0) * u_max_star
        disp_cf_star = CF((0, -uy_star))
        gfu_star.Set(disp_cf_star, definedon=mesh.Boundaries("top"))
        # 1. Update the RHS (F) FIRST using the state from the END of the previous step
        # This ensures the 'source' for pressure is based on the previous equilibrium

        A = BlockMatrix([
        [a_K.mat,        -alpha * a_Q12.mat],
        [(alpha/dt_star) * a_Q21.mat,   a_H.mat + S_star/dt_star * a_S.mat]
        ])
        F[0].data = b_f.vec
        F[1].data = ((alpha/dt_star) * a_Q21.mat * u_old_star.vec + 
                    (S_star/dt_star) * a_S.mat * p_old_star.vec + 
                    b_q.vec)

        a_PreP = BilinearForm(Q)
        # Scale both the Gradient and the Mass term by r
        a_PreP += (grad(p) * grad(q) + ((S_star / dt_star) * p * q)) * twopi * r * dx

        # BDDC or AMG works great here
        #pre_a_P = Preconditioner(a_PreP, type="direct")
        a_PreP.Assemble()
        pre_a_P = a_PreP.mat.Inverse(freedofs = Q.FreeDofs(), inverse="sparsecholesky")

        pre_C = BlockMatrix([
        [pre_a_K, None],
        [None, dt_star * pre_a_P]
        ])
        

        # 3. Synchronize the 'sol' BlockVector
        sol[0].data = gfu_star.vec
        sol[1].data = gfp_star.vec

        # 4. Calculate the residual
        # Since sol[0] has the new BC and F[1] has the old state, 
        # the solver correctly identifies the 'change' over dt.
        rhs_resid.data = F - A * sol

        ################################
        # Assuming r = x and twopi = 2 * pi
        fluid_force_star = Integrate(alpha * gfp_star * twopi * r, mesh, definedon=mesh.Boundaries("bottom"))

        # Then your scaling remains the same
        fluid_force_phys = fluid_force_star * (G * L**2)

        F_fluid.append(fluid_force_phys)
        ################################
        
        rhs_resid[0].data[~V.FreeDofs()] = 0.0
        rhs_resid[1].data[~Q.FreeDofs()] = 0.0



        #print("Relative pressure residual:", Rp / bp if bp > 0 else 0)
        # 5. Solve and update
        correction[:] = 0.0
        GMRes(A=A, b=rhs_resid, pre=pre_C, x=correction, tol=1e-8, printrates=False, maxsteps=500)
        sol.data += correction


        sigma_star = Stress_ax_Anisotropic(gfu_star)  
        
        # Use the Virtual Work method to get the dimensionless force
        # Note: v_test only needs to be defined once outside the loop
        solid_force_star = Integrate(InnerProduct(sigma_star, AxialGrad(v_test)) * twopi * r, mesh)
        
        # Unscale and store
        solid_force_phys = solid_force_star * (G * L**2)
        F_solid.append(solid_force_phys)


        #print(f"Solid: {solid_force_phys:.6f}, Fluid: {fluid_force_phys:.6f}, Total: {solid_force_phys + fluid_force_phys:.6f}, \
        #    Time: {t*tau:.4f} s / {t_end:.4f} s, dt: {dt_star * tau:.6f} s")

        #
        # 6. Handover (Update history for next step)
        u_old_star.vec.data = gfu_star.vec
        p_old_star.vec.data = gfp_star.vec

        time_vals_nondim.append(t)
        if DrawResults == True: 
            sceneu.Redraw()
            scenep.Redraw()

        dt_star = min(dt_star * growth_factor, dt_star_max)

    
    time_vals = np.array(time_vals_nondim)*tau
    F_solid = np.array(F_solid)
    F_fluid = np.array(F_fluid)

    time_vals = np.insert(time_vals, 0, 0.0)
    F_solid =  np.insert(F_solid, 0, 0.0)
    F_fluid =  np.insert(F_fluid, 0, 0.0)

    
    return time_vals, F_solid, F_fluid


