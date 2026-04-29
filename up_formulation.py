from ngsolve import *
from netgen.meshing import Mesh as NetMesh, MeshPoint, Element2D, Element1D, FaceDescriptor
import netgen.meshing as meshing
import scipy
import numpy as np
def MakeStructuredMesh(nx, ny, Lx, Ly, grading):
    m = NetMesh()
    m.dim = 2
    r = grading
    pids = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            x = Lx * i / nx
            y = Ly * (1 - r**j) / (1 - r**ny)
            row.append(m.Add(MeshPoint(meshing.Pnt(x, y, 0))))
        pids.append(row)
    
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    
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
import numpy as np
import time as time
import matplotlib.pyplot as plt
from ngsolve.krylovspace import GMRes
import time as time

def BulkModulus(G, nu):
    return 2*G*(1 + nu) / (3 - 6*nu)

def YoungsModulus(G, nu):
    return 2*G*(1 + nu)

def LameParameter(G, nu):
    return 2*G*nu / (1 - 2*nu)



def ConfinedCompression(G, nu, viscosity, alpha, phi, k, chi, rho, g, dt, dt_max, dt_growth, R, H, u_max , compression_time, t_end, nx, ny, grading, order, tol, maxsteps, restart, Confined=True, DrawResults=False):
    
    def div_ax(u):
        return (grad(u)[0,0] + u[0]/r + grad(u)[1,1])

    def eps_ax(u):
        gu = grad(u)
        return CF((gu[0,0], gu[1,1], u[0]/r, gu[0,1] + gu[1,0]))
        
        
    def Stress_ax_Anisotropic(u_vec):
        e_ax = eps_ax(u_vec)
        sigma_v = Cauchy_tensor_star * e_ax
        return CF( (sigma_v[0], sigma_v[3], 0,
                        sigma_v[3], sigma_v[1], 0,
                        0,          0,          sigma_v[2]), dims=(3,3) )
        
    def AxialGrad(v):
        gv = Grad(v)
        return CF((gv[0,0], gv[0,1], 0,
                    gv[1,0], gv[1,1], 0,
                    0,       0,       v[0]/r), dims=(3,3))
        
    start_time = time.time()
    L = H
    growth_factor = dt_growth

    lame_star = 2*nu / (1 - 2*nu)  

    C11, C12, C13, C14 = lame_star + 2, lame_star,     lame_star,     0
    C21, C22, C23, C24 = lame_star,     lame_star + 2, lame_star,     0
    C31, C32, C33, C34 = lame_star,     lame_star,     lame_star + 2, 0
    C41, C42, C43, C44 = 0,             0,             0,             1
    

    cauchy_values = (C11, C12, C13, C14,
                     C21, C22, C23, C24,
                     C31, C32, C33, C34,
                     C41, C42, C43, C44)
    
    Cauchy_tensor_star = CoefficientFunction(cauchy_values, dims=(4, 4)).Compile()
    twopi = 2*np.pi
    k_ref = k

    K = BulkModulus(G, nu) 
    S = phi * chi + ((1-alpha)*(alpha-phi)) / K    
    
    tau = L**2 * viscosity / (k * G)

    dt_star = dt / tau 
    S_star = S * G
    
    k11, k12 = k_ref, 0
    k21, k22 = 0, k_ref
    k_values = (k11/k_ref, k12/k_ref,
                k21/k_ref, k22/k_ref)
    
    k_star = CoefficientFunction(k_values, dims=(2,2)).Compile()

    R_Star = R / H
    H_Star = 1.0
    mesh = MakeStructuredMesh(nx, ny, R_Star, H_Star, grading)
    if Confined == True:
        V = VectorH1(mesh, order=order, dirichlet="top|bottom", dirichletx="sides")
        Q = H1(mesh,order=order-1, dirichlet="top")
    if Confined == False:
        V = VectorH1(mesh, order=order, dirichlet="top|bottom")
        Q = H1(mesh,order=order-1, dirichlet="top|sides")
    
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

    traction = 0 
    tbar = CoefficientFunction((0, traction))
    b = CoefficientFunction((0, rho*g))

    qbar = CoefficientFunction(0) 
    r = x
   
    if DrawResults == True:
        from ngsolve.webgui import Draw
        sceneu = Draw(gfu_star, mesh, deformation=True)
        scenep = Draw(G * gfp_star, mesh)
        scene_darcy = Draw(-(k/viscosity) * (G/L) * grad(gfp_star), mesh, name="darcy_flux [m/s]")
        scene_vel = Draw((L/tau) * (gfu_star - u_old_star) / dt_star, mesh, name="solid_velocity [m/s]")

    a_K = BilinearForm(V)
    a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
    a_K.Assemble()
    pre_a_K = a_K.mat.Inverse(freedofs = V.FreeDofs(), inverse="sparsecholesky")

    a_Q = BilinearForm(trialspace=Q, testspace=V)
    a_Q += alpha * p * div_ax(v) * twopi * r * dx
    a_Q.Assemble()

    a_S = BilinearForm(Q)
    a_S += p * q * twopi * r * dx
    a_S.Assemble()

    a_H = BilinearForm(Q)
    a_H += (k_star * grad(p) * grad(q)) * twopi * r * dx
    a_H.Assemble()

    tbar_star = tbar / G
    b_star = (b * L) / G

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
    t_ramp = compression_time 
    t_ramp_star = t_ramp / tau

    dt_star_max = dt_max / tau 


    A = BlockMatrix([
        [a_K.mat,        -a_Q.mat],
        [(1/dt_star) * a_Q.mat.T,   a_H.mat + S_star/dt_star * a_S.mat]
        ])

    F = BlockVector([
        b_f.vec, 
        (1/dt_star) * a_Q.mat.T * u_old_star.vec + 
                    (S_star/dt_star) * a_S.mat * p_old_star.vec + 
                    b_q.vec])

    sol = BlockVector([gfu_star.vec, gfp_star.vec])

    rhs_resid = F.CreateVector()
    correction = F.CreateVector()

    v_test = GridFunction(V)
    v_test.Set(CF((0, 1)), definedon=mesh.Boundaries("bottom"))


    a_Schur = BilinearForm(Q)
    a_Schur += (k_star * grad(p) * grad(q)) * twopi * r * dx
    a_Schur += (S_star / dt_star) * p * q * twopi * r * dx
    a_Schur += alpha**2 * p * q * twopi * r * dx
    a_Schur.Assemble()
    pre_Schur = a_Schur.mat.Inverse(freedofs=Q.FreeDofs(), inverse="sparsecholesky")
        
 
    
    while t < t_end_star:
        t += dt_star
        uy_star = min(t / t_ramp_star, 1.0) * u_max_star
        disp_cf_star = CF((0, -uy_star))
        gfu_star.Set(disp_cf_star, definedon=mesh.Boundaries("top"))


        A = BlockMatrix([
        [a_K.mat,        - a_Q.mat],
        [(1/dt_star) * a_Q.mat.T,   a_H.mat + S_star/dt_star * a_S.mat]
        ])

        F[0].data = b_f.vec
        F[1].data = ((1/dt_star) * a_Q.mat.T * u_old_star.vec + 
                    (S_star/dt_star) * a_S.mat * p_old_star.vec + 
                    b_q.vec)
        """
        a_Schur = BilinearForm(Q)
        a_Schur += (k_star * grad(p) * grad(q)) * twopi * r * dx          # H block
        a_Schur += ((S_star)/dt_star) * p * q * twopi * r * dx  # augmented S
        a_Schur += alpha**2 * p * q * twopi * r * dx
        a_Schur.Assemble()
        pre_Schur = a_Schur.mat.Inverse(freedofs=Q.FreeDofs(), inverse="sparsecholesky")
   

        pre_C = BlockMatrix([
            [dt_star * pre_a_K, None],
            [None,   dt_star * pre_Schur]
        ])
        """


        pre_C = BlockMatrix([
        [dt_star * pre_a_K, None],
        [None,   dt_star * pre_Schur]
        ])

        sol[0].data = gfu_star.vec
        sol[1].data = gfp_star.vec

        rhs_resid.data = F - A * sol

        fluid_force_star = Integrate(alpha * gfp_star * twopi * r, mesh, definedon=mesh.Boundaries("bottom"))

        fluid_force_phys = fluid_force_star * (G * L**2)

        F_fluid.append(fluid_force_phys)
        
        rhs_resid[0].data[~V.FreeDofs()] = 0.0
        rhs_resid[1].data[~Q.FreeDofs()] = 0.0

        correction[:] = 0.0

    
        GMRes(A=A, b=rhs_resid, pre=pre_C, x=correction, tol=tol, printrates="\r", maxsteps=maxsteps, restart=restart)
        sol.data += correction
        sigma_star = Stress_ax_Anisotropic(gfu_star)  
        solid_force_star = Integrate(InnerProduct(sigma_star, AxialGrad(v_test)) * twopi * r, mesh)
        solid_force_phys = solid_force_star * (G * L**2)
        F_solid.append(solid_force_phys)


        elapsed = time.time() - start_time
        #print(f" Total Force: {solid_force_phys + fluid_force_phys:.6f}, "
        #    f"Simulation time: {t*tau:.4f} s / {t_end:.4f} s, "
        #    f"dt: {dt_star * tau:.6f} s, "
        #    f"Elapsed time: {elapsed:.2f} s")
        if DrawResults == True: 
            sceneu.Redraw()
            scenep.Redraw()
            scene_darcy.Redraw()
            scene_vel.Redraw()

        u_old_star.vec.data = gfu_star.vec
        p_old_star.vec.data = gfp_star.vec

        time_vals_nondim.append(t)


        dt_star = min(dt_star * growth_factor, dt_star_max)

    
    time_vals = np.array(time_vals_nondim)*tau
    F_solid = np.array(F_solid)
    F_fluid = np.array(F_fluid)

    time_vals = np.insert(time_vals, 0, 0.0)
    F_solid =  np.insert(F_solid, 0, 0.0)
    F_fluid =  np.insert(F_fluid, 0, 0.0)

    
    return time_vals, F_solid, F_fluid



def Consolidation(G, nu, viscosity, alpha, n, k, chi, rho, g, dt, dt_max, dt_growth, R, H, traction, t_end, nx, ny, grading, order, tol, Confined=True, DrawResults=False):
    def div_ax(u):
        return (grad(u)[0,0] + u[0]/r + grad(u)[1,1])

    def eps_ax(u):
        gu = grad(u)
        return CF((gu[0,0], gu[1,1], u[0]/r, gu[0,1] + gu[1,0]))
        
        
    def Stress_ax_Anisotropic(u_vec):
        e_ax = eps_ax(u_vec)
        sigma_v = Cauchy_tensor_star * e_ax
        return CF( (sigma_v[0], sigma_v[3], 0,
                        sigma_v[3], sigma_v[1], 0,
                        0,          0,          sigma_v[2]), dims=(3,3) )
        
    def AxialGrad(v):
        gv = Grad(v)
        return CF((gv[0,0], gv[0,1], 0,
                    gv[1,0], gv[1,1], 0,
                    0,       0,       v[0]/r), dims=(3,3))

    L = H
    growth_factor = dt_growth

    lame_star = LameParameter(G, nu) / G 

    C11, C12, C13, C14 = lame_star + 2, lame_star,     lame_star,     0
    C21, C22, C23, C24 = lame_star,     lame_star + 2, lame_star,     0
    C31, C32, C33, C34 = lame_star,     lame_star,     lame_star + 2, 0
    C41, C42, C43, C44 = 0,             0,             0,             1

    cauchy_values = (C11, C12, C13, C14,
                     C21, C22, C23, C24,
                     C31, C32, C33, C34,
                     C41, C42, C43, C44)
    
    Cauchy_tensor_star = CoefficientFunction(cauchy_values, dims=(4, 4)).Compile()

    twopi = 2*np.pi
    k_ref = k
    
    K = BulkModulus(G, nu)
    S = n * chi + ((1-alpha)*(alpha-n)) / K    
    
    tau = L**2 * viscosity / (k * G)

    dt_star = dt / tau
    S_star = S * G
 
    k11, k12 = k_ref, 0
    k21, k22 = 0, k_ref

    k_values = (k11/k_ref, k12/k_ref,
                k21/k_ref, k22/k_ref)
    
    k_star = CoefficientFunction(k_values, dims=(2,2)).Compile()

    R_Star = R / H
    H_Star = 1.0
    mesh = MakeStructuredMesh(nx, ny, Lx=R_Star, Ly=H_Star, grading=grading)

    if Confined == True:
        V = VectorH1(mesh, order=order, dirichlet="bottom", dirichletx="sides", dirichletz="sides")
    if Confined == False:
        V = VectorH1(mesh, order=order, dirichlet="bottom")

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

    tbar = CoefficientFunction((0, traction))
    b = CoefficientFunction((0, rho*g))

    qbar = CoefficientFunction(0) 
    r = x
   
    if DrawResults == True:
        from ngsolve.webgui import Draw
        sceneu = Draw(gfu_star, mesh, deformation=True)
        scenep = Draw(G * gfp_star, mesh)
        scene_darcy = Draw(-(k/viscosity) * (G/L) * grad(gfp_star), mesh, name="darcy_flux [m/s]")
        scene_vel = Draw((L/tau) * (gfu_star - u_old_star) / dt_star, mesh, name="solid_velocity [m/s]")

    a_K = BilinearForm(V)
    a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
    a_K.Assemble()
    pre_a_K = a_K.mat.Inverse(freedofs = V.FreeDofs(), inverse="sparsecholesky")

    a_Q = BilinearForm(trialspace=Q, testspace=V)
    a_Q += p * div_ax(v) * twopi * r * dx
    a_Q.Assemble()

    a_S = BilinearForm(Q)
    a_S += p * q * twopi * r * dx
    a_S.Assemble()

    a_H = BilinearForm(Q)
    a_H += (k_star * grad(p) * grad(q)) * twopi * r * dx
    a_H.Assemble()
    tbar = CF((0, traction))
    tbar_star = tbar / G
    b_star = (b * L) / G

    b_f = LinearForm(V)
    b_f += (v * tbar_star) * twopi * r * ds("top")
    b_f += (v * b_star) * twopi * r * dx
    b_f.Assemble()

    qbar_star = (qbar * L) / (G * viscosity) 
    k_n = InnerProduct(n, k_star * n)
    b_q = LinearForm(Q)

    b_q += qbar_star * k_n * q * twopi * r * ds("top")
    b_q.Assemble()

    a_PreP = BilinearForm(Q)
    a_PreP += (grad(p) * grad(q) + p * q) * twopi * r * dx
    a_PreP.Assemble()
    pre_a_P = a_PreP.mat.Inverse(freedofs = Q.FreeDofs(), inverse="sparsecholesky")

    A = BlockMatrix([
        [a_K.mat,        -alpha * a_Q.mat],
        [(alpha/dt_star) * a_Q.mat.T,   a_H.mat + S_star/dt_star * a_S.mat]
        ])

    F = BlockVector([
        b_f.vec, 
        (alpha/dt_star) * a_Q.mat.T * u_old_star.vec + 
                    (S_star/dt_star) * a_S.mat * p_old_star.vec + 
                    b_q.vec])

    sol = BlockVector([gfu_star.vec, gfp_star.vec])

    rhs_resid = F.CreateVector()
    correction = F.CreateVector()
    time_vals_nondim = []

    t = 0
    t_end_star = t_end / tau
    dt_star_max = dt_max / tau 
    all_p_values, displacement_max = [], []
    
    while t < t_end_star:

        A = BlockMatrix([
        [a_K.mat,        -alpha * a_Q.mat],
        [(alpha/dt_star) * a_Q.mat.T,   a_H.mat + S_star/dt_star * a_S.mat]
        ])

        F = BlockVector([
        b_f.vec, 
        (alpha/dt_star) * a_Q.mat.T * u_old_star.vec + 
            (S_star/dt_star) * a_S.mat * p_old_star.vec + 
            b_q.vec])


        pre_C = BlockMatrix([
            [dt_star * pre_a_K, None],
            [None,   dt_star * pre_a_P]
        ])
        
        sol[0].data = gfu_star.vec
        sol[1].data = gfp_star.vec

        rhs_resid.data = F - A * sol
        
        rhs_resid[0].data[~V.FreeDofs()] = 0.0
        rhs_resid[1].data[~Q.FreeDofs()] = 0.0

        correction[:] = 0.0
        GMRes(A=A, b=rhs_resid, pre=pre_C, x=correction, tol=tol, printrates=False, maxsteps=500, restart=150)
        sol.data += correction

        #print(f"Solid: {solid_force_phys:.6f}, Fluid: {fluid_force_phys:.6f}, Total: {solid_force_phys + fluid_force_phys:.6f}, \
        #    Time: {t*tau:.4f} s / {t_end:.4f} s, dt: {dt_star * tau:.6f} s")
        if DrawResults == True: 
            sceneu.Redraw()
            scenep.Redraw()
            scene_darcy.Redraw()
            scene_vel.Redraw()
 
        u_old_star.vec.data = gfu_star.vec
        p_old_star.vec.data = gfp_star.vec

        p_values = []
        heights = np.linspace(0, H, 50) 
        for y in heights:
            val = gfp_star(mesh(0,y,0))
            p_values.append(val)

        p_values = np.array(p_values)*G

        all_p_values.append(p_values)

        total_uy = Integrate(gfu_star.components[1] * twopi * r, mesh, definedon=mesh.Boundaries("top"))

        top_area = Integrate(1 * twopi * r, mesh, definedon=mesh.Boundaries("top"))

        top_settlement = total_uy / top_area
        displacement_max.append(np.abs(top_settlement))
        time_vals_nondim.append(t)

        dt_star = min(dt_star * growth_factor, dt_star_max)
        t += dt_star

    
    time_vals = np.array(time_vals_nondim)*tau
    
    return time_vals, all_p_values, displacement_max



