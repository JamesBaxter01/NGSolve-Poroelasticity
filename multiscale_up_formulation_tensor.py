from ngsolve import *
from netgen.meshing import Mesh as NetMesh, MeshPoint, Element2D, Element1D, FaceDescriptor
import netgen.meshing as meshing


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
from ngsolve.krylovspace import GMRes
import time as time

def BulkModulus(G, nu):
    return 2*G*(1 + nu) / (3 - 6*nu)

def YoungsModulus(G, nu):
    return 2*G*(1 + nu)

def LameParameter(G, nu):
    return 2*G*nu / (1 - 2*nu)

def axisymmetric_stiffness_from_C6(C):
    """
    Extract the axisymmetric stiffness block from a 6x6 Voigt tensor.

    Expected 6x6 ordering is (xx, yy, zz, yz, xz, xy). The axisymmetric
    strain vector used here is (rr, zz, theta-theta, rz), mapped as
    (x, y, z, xy).
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Expected C to be a 6x6 stiffness matrix")
    axis_indices = [0, 1, 2, 5]
    return C[np.ix_(axis_indices, axis_indices)]

def bulk_modulus_from_C6(C):
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Expected C to be a 6x6 stiffness matrix")
    return (
        C[0, 0] + C[1, 1] + C[2, 2]
        + 2 * (C[0, 1] + C[0, 2] + C[1, 2])
    ) / 9

def axisymmetric_permeability_from_k(k):
    """
    Convert either a scalar permeability or a 3x3 tensor into the 2x2
    axisymmetric permeability block used by the (r,z) model.

    Expected 3x3 ordering is (x, y, z), with x mapped to radial r and y mapped
    to axial z in this axisymmetric formulation.
    """
    k_array = np.asarray(k, dtype=float)
    if k_array.shape == ():
        return float(k_array) * np.eye(2)
    if k_array.shape == (2, 2):
        return k_array
    if k_array.shape == (3, 3):
        axis_indices = [0, 1]
        return k_array[np.ix_(axis_indices, axis_indices)]
    raise ValueError("Expected k to be a scalar, a 2x2 matrix, or a 3x3 matrix")

def default_stiffness_scaling(comp_err, comp_ezz, comp_eth, comp_erz,
                               C_axis_star, gf_err, gf_ezz, gf_eth, gf_erz,
                               gf_vel_mag, gf_vr, gf_vz,
                               gf_p, gf_p_old, dp_dt, **kwargs):

    def strain_stiffening_scale(comp, q, b):
        return 1 + q * (exp(b * comp) - 1)

    # Your fitted normalized parameters
    q11, b11 = 0.384899,   1.58449674
    q55, b55 = 0.20827433, 3.30233778
    q99, b99 = 0.39786299, 1.53177068

    scale_rr = strain_stiffening_scale(comp_err, q11, b11)
    scale_zz = strain_stiffening_scale(comp_ezz, q55, b55)
    scale_th = strain_stiffening_scale(comp_eth, q99, b99)

    # For shear/coupling strain, use the A55 law unless you have a separate fit
    scale_rz = strain_stiffening_scale(comp_erz, q55, b55)

    scales = (scale_rr, scale_zz, scale_th, scale_rz)
    cauchy_values = []
    for i in range(4):
        for j in range(4):
            cauchy_values.append(sqrt(scales[i] * scales[j]) * C_axis_star[i, j])

    return CoefficientFunction(tuple(cauchy_values), dims=(4, 4))


def default_permeability_scaling(comp_err, comp_ezz, comp_eth, comp_erz,
                                  C_axis_star, gf_err, gf_ezz, gf_eth, gf_erz,
                                  gf_vel_mag, gf_vr, gf_vz,
                                  gf_p, gf_p_old, dp_dt, k_ref, k_axis_star,
                                  **kwargs):

    bxx = 2.03435832
    byy = 2.49276041
    bzz = 2.03149295

    scale_rr = exp(-bxx * comp_err)
    scale_zz = exp(-byy * comp_ezz)

    scales = (scale_rr, scale_zz)
    k_values = []
    for i in range(2):
        for j in range(2):
            k_values.append(sqrt(scales[i] * scales[j]) * k_axis_star[i, j])

    return CoefficientFunction(tuple(k_values), dims=(2, 2))


def ConfinedCompression(C, viscosity, alpha, phi, k, chi, rho, g, dt, dt_max, dt_growth, R, H, u_max , compression_time, t_end, nx, ny, grading, order, tol, maxsteps, restart, Confined=True, DrawResults=False):
    start_time = time.time()
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

    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("ConfinedCompression expects C to be a 6x6 stiffness matrix")
    G_ref = float(C[0, 0])
    if G_ref <= 0:
        raise ValueError("C[0,0] must be positive because it is used as G_ref")
    C_axis = axisymmetric_stiffness_from_C6(C)
    C_axis_star = C_axis / G_ref

    twopi = 2*np.pi
    k_axis = axisymmetric_permeability_from_k(k)
    k_ref = float(k_axis[0, 0])
    if k_ref <= 0:
        raise ValueError("The radial permeability k[0,0] must be positive because it is used as k_ref")
    k_axis_star = k_axis / k_ref

    K = bulk_modulus_from_C6(C) 
    S = phi * chi + ((1-alpha)*(alpha-phi)) / K    
    
    tau = L**2 * viscosity / (k_ref * G_ref)

    dt_star = dt / tau 
    S_star = S * G_ref
    


    R_Star = R / H
    H_Star = 1.0
    mesh = MakeStructuredMesh(nx, ny, R_Star, H_Star, grading)
    if Confined == True:
        V = VectorH1(mesh, order=order, dirichlet="top|bottom", dirichletx="sides")
    if Confined == False:
        V = VectorH1(mesh, order=order, dirichlet="top|bottom")
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

    traction = 0 
    tbar = CoefficientFunction((0, traction))
    b = CoefficientFunction((0, rho*g))

    qbar = CoefficientFunction(0) 
    r = x
    W = H1(mesh, order=order-1)
    gf_Czz_star = GridFunction(W, name="Czz_star")
    gf_kzz_star = GridFunction(W, name="kzz_star")


    def update_material(gfu_star, u_old_star, gfp_star, p_old_star, dt_star, W, k_ref, k_axis_star, C_axis_star, L, tau):
        """
        Compute all strain-dependent material properties from current field solutions.
                
        Inputs:
            gfu_star        : current displacement GridFunction
            u_old_star      : previous displacement GridFunction  
            gfp_star        : current pressure GridFunction
            p_old_star      : previous pressure GridFunction
            dt_star         : current dimensionless time step
            W               : scalar H1 space for projections
            k_ref           : reference permeability
            C_axis_star     : dimensionless 4x4 axisymmetric stiffness
            L, tau          : characteristic length and time scales
            stiffness_scaling_fn  : optional custom stiffness function
            permeability_scaling_fn : optional custom permeability function

        Outputs:
            Cauchy_tensor_star : 4x4 CoefficientFunction
            k_star             : 2x2 CoefficientFunction
        """
        # strain
        strain = eps_ax(gfu_star)
        gf_err = GridFunction(W)
        gf_ezz = GridFunction(W)
        gf_eth = GridFunction(W)
        gf_erz = GridFunction(W)

        gf_err.Set(strain[0])
        gf_ezz.Set(strain[1])
        gf_eth.Set(strain[2])
        gf_erz.Set(strain[3])

        comp_err = IfPos(-gf_err - 1e-7, -gf_err, 0.0)
        comp_ezz = IfPos(-gf_ezz - 1e-7, -gf_ezz, 0.0)
        comp_eth = IfPos(-gf_eth - 1e-7, -gf_eth, 0.0)
        comp_erz = IfPos(-gf_erz - 1e-7, -gf_erz, 0.0)

            # --- Velocity ---
        vel = (L / tau) * (gfu_star - u_old_star) / dt_star
        gf_vr, gf_vz = GridFunction(W), GridFunction(W)
        gf_vr.Set(vel[0])
        gf_vz.Set(vel[1])
        gf_vel_mag = GridFunction(W)
        gf_vel_mag.Set(sqrt(gf_vr**2 + gf_vz**2))

        # --- Pressure rate ---
        gf_p     = gfp_star
        gf_p_old = p_old_star
        dp_dt    = (gf_p - gf_p_old) / dt_star

        # --- Bundle everything for the scaling functions ---
        fields = dict(
            comp_err=comp_err, comp_ezz=comp_ezz, comp_eth=comp_eth, comp_erz=comp_erz,
            C_axis_star=C_axis_star,
            gf_err=gf_err, gf_ezz=gf_ezz, gf_eth=gf_eth, gf_erz=gf_erz,
            gf_vel_mag=gf_vel_mag, gf_vr=gf_vr, gf_vz=gf_vz,
            gf_p=gf_p, gf_p_old=gf_p_old, dp_dt=dp_dt,
            k_ref=k_ref, k_axis_star=k_axis_star,
        )

        Cauchy_tensor_star = default_stiffness_scaling(**fields)
        k_star             = default_permeability_scaling(**fields)

        return Cauchy_tensor_star, k_star
        
    Cauchy_tensor_star, k_star = update_material(gfu_star, u_old_star, gfp_star, p_old_star, dt_star, W, k_ref, k_axis_star, C_axis_star, L, tau)

    if DrawResults == True:
        from ngsolve.webgui import Draw
        sceneu = Draw(gfu_star, mesh, "displacement", deformation=True)
        scenep = Draw(G_ref * gfp_star, mesh, "pressure [Pa]")
        scene_darcy = Draw(
            -(k_ref/viscosity) * (G_ref/L) * (k_star * grad(gfp_star)),
            mesh,
            name="darcy_flux [m/s]",
        )
        scene_vel = Draw((L/tau) * (gfu_star - u_old_star) / dt_star, mesh, name="solid_velocity [m/s]")
        scene_Czz = Draw(G_ref * gf_Czz_star, mesh, "Czz [Pa]")
        scene_kzz = Draw(k_ref * gf_kzz_star, mesh, "kzz [m^2]")


    a_K = BilinearForm(V)
    a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
    pre_a_K = Preconditioner(a_K, "bddc")
    a_K.Assemble()
    #pre_a_K = a_K.mat.Inverse(freedofs = V.FreeDofs(), inverse="sparsecholesky")

    a_Q = BilinearForm(trialspace=Q, testspace=V)
    a_Q += alpha * p * div_ax(v) * twopi * r * dx
    a_Q.Assemble()

    a_S = BilinearForm(Q)
    a_S += p * q * twopi * r * dx
    a_S.Assemble()

    a_H = BilinearForm(Q)
    a_H += (k_star * grad(p) * grad(q)) * twopi * r * dx
    a_H.Assemble()

    tbar_star = tbar / G_ref
    b_star = (b * L) / G_ref

    b_f = LinearForm(V)
    b_f += (v * tbar_star) * twopi * r * ds("top")
    b_f += (v * b_star) * twopi * r * dx
    b_f.Assemble()

    qbar_star = (qbar * L) / (G_ref * viscosity) 
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
 

    while t < t_end_star:
        #print(f"Time: {t*tau:.4f} s / {t_end:.4f} s, dt: {dt_star * tau:.6f} s", flush=True)
        t += dt_star
        max_iter = 20
        print(
            f"wall={time.time()-start_time:.2f}s, "
            f"model t={t*tau:.4f}/{t_end:.1f}, "
            f"dt={dt_star*tau:.6f}",
            flush=True,
        )
        uy_star = min(t / t_ramp_star, 1.0) * u_max_star
        disp_cf_star = CF((0, -uy_star))
        gfu_star.Set(disp_cf_star, definedon=mesh.Boundaries("top"))
        tol_picard = 1e-6  

        # once per timestep
        a_K = BilinearForm(V)
        a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
        pre_a_K = Preconditioner(a_K, "direct")
        a_K.Assemble()

        a_Schur = BilinearForm(Q)
        a_Schur += (k_star * grad(p) * grad(q)) * twopi * r * dx          # H block
        a_Schur += ((S_star)/dt_star) * p * q * twopi * r * dx  # augmented M
        pre_Schur = Preconditioner(a_Schur, "direct")
        a_Schur.Assemble()
        
        for picard_iter in range(max_iter):

            gfu_star.vec.data = sol[0]
            gfp_star.vec.data = sol[1]
            Cauchy_tensor_star, k_star = update_material(gfu_star, u_old_star, gfp_star, p_old_star, dt_star, W, k_ref, k_axis_star, C_axis_star, L, tau)
            a_K = BilinearForm(V)
            a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
            a_K.Assemble()


            a_H = BilinearForm(Q)
            a_H += (k_star * grad(p) * grad(q)) * twopi * r * dx
            a_H.Assemble()

            A = BlockMatrix([
            [a_K.mat,        -alpha * a_Q.mat],
            [(1/dt_star) * a_Q.mat.T,   a_H.mat + S_star/dt_star * a_S.mat]
            ])

            F = BlockVector([
            b_f.vec,
            (1/dt_star) * a_Q.mat.T * u_old_star.vec + 
                (S_star/dt_star) * a_S.mat * p_old_star.vec + 
                b_q.vec])
            
            pre_C = BlockMatrix([
                [dt_star * pre_a_K, None],
                [None,   dt_star * pre_Schur]
            ])

            rhs_resid.data = F - A * sol

            fluid_force_star = Integrate(alpha * gfp_star * twopi * r, mesh, definedon=mesh.Boundaries("bottom"))

            fluid_force_phys = fluid_force_star * (G_ref * L**2)
            
            rhs_resid[0].data[~V.FreeDofs()] = 0.0
            rhs_resid[1].data[~Q.FreeDofs()] = 0.0

            correction[:] = 0.0

            GMRes(A=A, b=rhs_resid, pre=pre_C, x=correction, tol=tol, printrates=False, maxsteps=maxsteps, restart=restart)

            correction_norm = sqrt(InnerProduct(correction, correction))
            sol.data += correction
            sigma_star = Stress_ax_Anisotropic(gfu_star)  
            solid_force_star = Integrate(InnerProduct(sigma_star, AxialGrad(v_test)) * twopi * r, mesh)
            solid_force_phys = solid_force_star * (G_ref * L**2)


            if correction_norm < tol_picard:
                break


        F_fluid.append(fluid_force_phys)
        F_solid.append(solid_force_phys)
        gf_Czz_star.Set(Cauchy_tensor_star[1,1])
        gf_kzz_star.Set(k_star[1,1])
        if DrawResults == True: 

            sceneu.Redraw()
            scenep.Redraw()
            scene_darcy.Redraw()
            scene_vel.Redraw()
            scene_Czz.Redraw()
            scene_kzz.Redraw()
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


    def velocity(gfu_star, u_old_star, dt_star):
        vel = (L/tau) * (gfu_star - u_old_star) / dt_star
        return vel
    
    L = H
    growth_factor = dt_growth

    lame_star = LameParameter(G, nu) / G 


    twopi = 2*np.pi
    k_axis = axisymmetric_permeability_from_k(k)
    k_ref = float(k_axis[0, 0])
    if k_ref <= 0:
        raise ValueError("The radial permeability k[0,0] must be positive because it is used as k_ref")
    k_axis_star = k_axis / k_ref
    
    K = BulkModulus(G, nu)
    S = n * chi + ((1-alpha)*(alpha-n)) / K    
    
    tau = L**2 * viscosity / (k_ref * G)

    dt_star = dt / tau
    S_star = S * G
 
    k_star = CoefficientFunction(tuple(k_axis_star.reshape(-1)), dims=(2,2)).Compile()

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
        sceneu = Draw(gfu_star, mesh, deformation=gfu_star*1000)
        scenep = Draw(G * gfp_star, mesh, max=1000)
        scene_darcy = Draw(
            -(k_ref/viscosity) * (G/L) * (k_star * grad(gfp_star)),
            mesh,
            name="darcy_flux [m/s]",
        )
        scene_vel = Draw((L/tau) * (gfu_star - u_old_star) / dt_star, mesh, name="solid_velocity [m/s]")
    from ngsolve.webgui import Draw
    # Strain components
    W = H1(mesh, order=order-1)
    gf_C22 = GridFunction(W)
    scene_C22 = Draw(gf_C22, mesh, name="Effective G (C22)")
    
    def update_strain_scaling(gfu_star, W):
        strain = eps_ax(gfu_star)

        gf_err = GridFunction(W)
        gf_ezz = GridFunction(W)
        gf_eth = GridFunction(W)
        gf_erz = GridFunction(W)

        gf_err.Set(strain[0])
        gf_ezz.Set(strain[1])
        gf_eth.Set(strain[2])
        gf_erz.Set(strain[3])

        comp_err = IfPos(-gf_err - 1e-7, -gf_err, 0.0)
        comp_ezz = IfPos(-gf_ezz - 1e-7, -gf_ezz, 0.0)
        comp_eth = IfPos(-gf_eth - 1e-7, -gf_eth, 0.0)
        comp_erz = IfPos(-gf_erz - 1e-7, -gf_erz, 0.0)

        scale_rr = 10**(comp_err*100)
        scale_zz = 10**(comp_ezz*100)
        scale_th = 10**(comp_eth*100)
        scale_rz = 10**(comp_erz*100)

        C11 = scale_rr * (lame_star + 2)
        C22 = scale_zz * (lame_star + 2)
        C33 = scale_th * (lame_star + 2)
        C44 = scale_rz

        C12 = ((scale_rr + scale_zz) / 2) * lame_star
        C13 = ((scale_rr + scale_th) / 2) * lame_star
        C23 = ((scale_zz + scale_th) / 2) * lame_star

        cauchy_values = (C11, C12, C13, 0,
                        C12, C22, C23, 0,
                        C13, C23, C33, 0,
                        0,   0,   0,   C44)
        

        return CoefficientFunction(cauchy_values, dims=(4, 4))
    
    Cauchy_tensor_star = update_strain_scaling(gfu_star, W)
    a_K = BilinearForm(V)
    a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
    a_K.Assemble()
    pre_a_K = a_K.mat.Inverse(freedofs = V.FreeDofs(), inverse="bddc")

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
        t += dt_star
        max_iter = 5
        tol_picard = 1e-6

        for picard_iter in range(max_iter):
            Cauchy_tensor_star = update_strain_scaling(gfu_star, W)

            a_K = BilinearForm(V)
            a_K += InnerProduct(Cauchy_tensor_star * eps_ax(u), eps_ax(v)) * twopi * r * dx
            a_K.Assemble()
            pre_a_K = a_K.mat.Inverse(freedofs=V.FreeDofs(), inverse="sparsecholesky")


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
            correction_norm = sqrt(InnerProduct(correction, correction))
            sol.data += correction
            #print(f"  Picard iter {picard_iter+1}, correction norm = {correction_norm:.2e}")
            #print(f"Solid: {solid_force_phys:.6f}, Fluid: {fluid_force_phys:.6f}, Total: {solid_force_phys + fluid_force_phys:.6f}, \
            #    Time: {t*tau:.4f} s / {t_end:.4f} s, dt: {dt_star * tau:.6f} s")
            if correction_norm < tol_picard:
                break
        
        if DrawResults == True: 
            sceneu.Redraw()
            scenep.Redraw()
            scene_darcy.Redraw()
            scene_vel.Redraw()

        strain = eps_ax(gfu_star)
        gf_ezz = GridFunction(W)
        gf_ezz.Set(strain[1])
        comp_ezz = IfPos(-gf_ezz - 1e-7, -gf_ezz, 0.0)

        scale_zz = 10**(comp_ezz*100)


        gf_C22.Set(scale_zz)  # multiply by G to recover dimensional value
        scene_C22.Redraw()
        
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

    
    time_vals = np.array(time_vals_nondim)*tau
    
    return time_vals, all_p_values, displacement_max
