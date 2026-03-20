import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from contextlib import redirect_stdout
sys.path.append('../..')
sys.path.append('./data')

from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.jax.nlforces.elastic_dry_fric_3d import ElasticDryFriction3D
from scipy import io as sio
import tmdsimpy.utils.harmonic as hutils
import tmdsimpy.nlutils as nlutils

import time


from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.continuation as cont_utils
from tmdsimpy.jax.solvers import NonlinearSolverOMP
from datetime import datetime

save_timestamp = datetime.now().isoformat()


large_system = True

if not large_system:
    # Load matrices (relative path from current script)
    M = sio.loadmat('data/TMD_smallbrb_M.mat')['M']
    K = sio.loadmat('data/TMD_smallbrb_K.mat')['K']
    #Q = sio.loadmat('data/TMD_smallbrb_Q.mat')['Qxyn']
    L = sio.loadmat('data/TMD_smallbrb_L.mat')['L']
    Rmat = sio.loadmat('data/TMD_smallbrb_R.mat')['R']
    LamT = sio.loadmat('data/TMD_smallbrb_LamT.mat')['LamT']
    Fv = sio.loadmat('data/TMD_smallbrb_Fv.mat')['Fv'].squeeze()
    Kpre = sio.loadmat('data/TMD_smallbrb_Kpre.mat')['dRstat']
    patch_areas = sio.loadmat('data/TMD_smallbrb_patch_areas.mat')['PatchAreas'].squeeze()
    upxyn = sio.loadmat('data/TMD_smallbrb_upxyn.mat')['upxyn']
    #uxyn = sio.loadmat('data/TMD_smallbrb_uxyn.mat')['uxyn'] #unused
    
    # Compute Qxyn and Txyn
    Npatches = patch_areas.size   # or however you know it
    select = np.reshape(((np.arange(Npatches) * 6)[:, None] + np.arange(3)), -1, order='F')

    Qxyn = L[select, :]
    Txyn = LamT[:, select]
else:
    default_sys_fname = './data/BRB_ROM_U_122ELS4py.mat'
    system_matrices = sio.loadmat(default_sys_fname)
    M, K = system_matrices['M'], system_matrices['K']
    L  = system_matrices['L']
    Fv = system_matrices['Fv'][:, 0] #fine
    Rmat = system_matrices['R']
    
    Qm = np.array(system_matrices['Qm'].todense()) 
    Tm = np.array(system_matrices['Tm'].todense())
    
    patch_areas = Tm.sum(axis = 0)
    
    Qxyn = np.kron(Qm, np.eye(3)) @ L[:3*Qm.shape[1], :]
    
    Nnl = Qxyn.shape[0]
    Txyn = L[:3*Qm.shape[1], :].T @ np.kron(Tm, np.eye(3))
    
    
    print("Large system goes here")

Nt = 128



params = np.array([0.6618, 10**14.4598, 10**12.2353])

# Compute per-patch stiffnesses
KT_patches0 = patch_areas * params[1]
KN_patches0 = patch_areas * params[2]
MU_patches0 = params[0] * np.ones(patch_areas.shape)

prestress = 12845

vib_sys = VibrationSystem(M, K)

for patch in range(len(patch_areas)):
    dryfric = ElasticDryFriction3D(Qxyn[3*patch:3*patch+3, :], 
                                   Qxyn[3*patch:3*patch+3, :].T, 
                                   KT_patches0[patch], KN_patches0[patch], MU_patches0[patch])
    
    vib_sys.add_nl_force(dryfric)

#ref_nlforce = ElasticDryFriction3D(np.eye(3), np.eye(3), params[1], params[2], params[0])

vib_sys.set_prestress_mu()

#t, dtduxyn = ref_nlforce.force(upxyn[0])





# Assemble full Kst
Kst = Txyn @ np.diag(np.r_[KT_patches0, KT_patches0, KN_patches0]) @ Qxyn
K0 = K + Kst;
X0 = np.linalg.solve(K0,(Fv * prestress)).squeeze()

# function to solve
pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)

R0, dR0dX = pre_fun(X0)

static_config={'max_steps' : 30,
                'reform_freq' : 1,
                'verbose' : False, 
                'xtol'    : 1e-12, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
static_solver = NonlinearSolverOMP(config=static_config) 

Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=False, xtol=1e-11)

epmc_config={'max_steps' : 12, # balance with reform_freq
            'reform_freq' : 2, #>1 corresponds to BFGS 
            'xtol'    : None, # Just use the one passed from continuation
            'rtol'    : 1e-9,
            'etol'    : None,
            'xtol_rel' : 1e-6, 
            'rtol_rel' : None,
            'etol_rel' : None,
            'stopping_tol' : ['xtol'], # stop on xtol 
            'accepting_tol' : ['xtol_rel', 'rtol'], # accept solution on these
            'armijo_iters': 5
            }

epmc_solver = NonlinearSolverOMP(config=epmc_config)

Astart = -10
Aend = -3.5
dsmin = 0.005
dsmax = 0.05
ds = 0.1
FracLam = 0.9

vib_sys.update_force_history(Xpre)

# Use the prestress solution as the intial slider positions for AFT as well
# This influences residual tractions and may slightly change the results of
# the simulation.
vib_sys.set_aft_initialize(Xpre)

# Reset to real friction coefficient after updating frictionless slider
# positions
# This is needed so that the friction coefficient is used in EPMC 
# (rather than 0 tangential forces)
vib_sys.reset_real_mu()

# Recalculate stiffness with real mu (including stiffness from friction)
Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)

Kpre = (dRpredX + dRpredX.T) / 2.0


eigvals, eigvecs = static_solver.eigs(Kpre, M, 
                                      subset_by_index=[0, 9])


h_max = 1

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Ndof = vib_sys.M.shape[0]

Fl = np.zeros(Nhc*Ndof)

# Static Forces
Fl[:Ndof] = prestress*Fv # EPMC static force

# EPMC phase constraint - No cosine component at accel
Fl[Ndof:2*Ndof] = Rmat[2, :] 

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements (prediction for 0th harmonic)
Uwxa0[:Ndof] = Xpre

# Mode Shape (from linearized system for prediction)
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

print(f"Initial eigenvalue sqrted: {Uwxa0[-3]}")

# Initial Damping (low amplitude as prescribed)
desired_zeta = np.array([0.087e-2, 0.034e-2]) 
zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta
# Continuation really wants this to be zero, which kinda makes sense if you
#think about it
# Amplitude (Desired starting amplitude)
Uwxa0[-1] = Astart

t0 = time.time()

R = vib_sys.epmc_res(Uwxa0, Fl, h, Nt=Nt, calc_grad=True)[0]

str(R[0]) # This forces JAX operations to block

print(f"Norm of the initial residual: {np.linalg.norm(R)}")

t1 = time.time()

print('EPMC Residual Run Time (with gradient): {: 7.3f} s'.format(t1 - t0))

t0 = time.time()

R = vib_sys.epmc_res(Uwxa0, Fl, h, Nt=Nt, calc_grad=False)[0]

str(R[0]) # This forces JAX operations to block

t1 = time.time()

print('EPMC Residual Run Time (without gradient): {: 7.3f} s'.format(t1 - t0))

###############################################################################
####### 15. EPMC Continuation                                           #######
###############################################################################

# This block actually executes the full continuation for the EPMC solution.

epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                          calc_grad=calc_grad)

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 4,
                   'MaxSteps'   : 40, # May need more depending on ds and dsmin
                   'dsmin'      : dsmin,
                   'dsmax'      : dsmax,
                   'verbose'    : 1,
                   'xtol'       : 1e-6*np.sqrt(Uwxa0.shape[0]), 
                   'corrector'  : 'Ortho', # Ortho, Pseudo
                   'nsolve_verbose' : True,
                   'FracLam' : FracLam,
                   'FracLamList' : [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop' : 0.05 # stop if backtracks to before lam0
                   }

# The conditioning of the static displacements should be small since these
# displacements are small, but very important
CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)

# Increasing delta means that these cofficients will be smaller in conditioned 
# space. - reduces importance of higher harmonics when calculating arc length
CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3) 

# Allow different CtoP for static displacements than harmonics.
CtoP[:Ndof] = CtoPstatic[:Ndof] 

# Exactly take damping and frequency regardless of delta for conditioning
CtoP[-3:-1] = np.abs(Uwxa0[-3:-1]) 

# scale so step size is similar order as fraction of total distance from 
# start to end
CtoP[-1] = np.abs(Aend-Astart)

cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, 
                           config=continue_config)
    
t0 = time.time()

Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))

#%% Helper Functions 

def findDryParameters(E_patches, KT_patches0, KN_patches0, MU_patches0, ti_kt, tau_kt, ti_mu, tau_mu):
    '''
    Estimate dry contact stiffness parameters for each patch based on the
    cumulative dissipated energy and initial stiffness values.

    Parameters
    ----------
    E_patches : np.ndarray of shape (Npatches,)
        Cumulative dissipated energy for each patch.
    KT_patches0 : np.ndarray of shape (Npatches,)
        Initial tangential stiffness values for each patch.
    KN_patches0 : np.ndarray of shape (Npatches,)
        Initial normal stiffness values for each patch.

    Returns
    -------
    KT_patches : np.ndarray of shape (Npatches,)
        Estimated tangential stiffness for each patch under dry conditions.
    KN_patches : np.ndarray of shape (Npatches,)
        Estimated normal stiffness for each patch under dry conditions.
    '''

    KT_mod = lambda E: nlutils.first_order_exponential(1.0, ti_kt, tau_kt, E)
    KN_mod = lambda E: nlutils.first_order_exponential(1.0, ti_kt, tau_kt, E)
    MU_mod = lambda E: nlutils.first_order_exponential(1.0, ti_mu, tau_mu, E)
    
    Npatches = E_patches.shape[0]
    
    KT_patches = np.zeros(Npatches)
    KN_patches = np.zeros(Npatches)
    MU_patches = np.zeros(Npatches)
    
    print(f"KT_mod sample: {np.mean(KT_mod(E_patches[0]))}")
    
    for patch in range(Npatches):
        KT_patches[patch] = KT_patches0[patch] * KT_mod(E_patches[patch])
        KN_patches[patch] = KN_patches0[patch] * KN_mod(E_patches[patch])
        MU_patches[patch] = MU_patches0[patch] * MU_mod(E_patches[patch])
    
    return KT_patches, KN_patches, MU_patches

def calculate_E_dot(Uwxa, h, Nt, w, nl_force, la):
    omega = Uwxa[-3]
    xi = Uwxa[-2]
    la = la #Calculated Hysteretic Energy needs to reflect target amplitude

    X = Uwxa[:-3]
    Nhc = hutils.Nhc(h)

    X_mat = X.reshape(Nhc, Ndof).T
    # Harmonic order indices: 0 → h0, others → h1c, h1s, ...
    scale_vec = np.ones(Nhc)
    scale_vec[1:] *= 10 ** la  # apply scaling to higher harmonics

    X_mat = X_mat * scale_vec[np.newaxis, :]
    Unl = nl_force.Q @ X_mat
    # print(Unl)
    # # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl.T, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl.T, 1) # Nt x Ndnl
   
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
   
    unlth0 = Unl[0]
    
    fnl = nl_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    fnl = fnl[0]
    
    period = 2*np.pi/w
    dt = period / Nt #Quadrature dt for trapezoidal rule.
    
    inst_power = np.sum(fnl * unltdot, axis=1) #Shape (Nt,)
    
    E_cycle = np.trapz(inst_power, dx=dt)
    
    Edot = E_cycle / period #J/s
    return Edot

def slip_distance_per_cycle(Uwxa, h, Nt, w, nl_force, la, tol=1e-6):
    """
    Compute TRUE slip distance (not total shear motion) for a single dry-friction patch.
    Slip occurs only when |F_t| is plateaued (unchanging & nonzero).
    DOF ordering: [tangential_x, tangential_y, normal].
    """
    # --- unpack Uwxa like your Edot routine ---
    omega = Uwxa[-3]
    xi    = Uwxa[-2]

    X   = Uwxa[:-3]
    Nhc = hutils.Nhc(h)

    # --- rebuild harmonic coefficients and scale ---
    X_mat = X.reshape(Nhc, 3).T
    scale = np.ones(Nhc)
    scale[1:] *= 10 ** la
    X_mat = X_mat * scale[np.newaxis, :]

    # --- get nonlinear DOF harmonics ---
    Unl = nl_force.Q @ X_mat

    # --- time-domain history ---
    unlt    = hutils.time_series_deriv(Nt, h, Unl.T, 0)        # (Nt x 3)
    unltdot = w * hutils.time_series_deriv(Nt, h, Unl.T, 1)    # (Nt x 3)

    # --- nonlinear force history (Nt x 3) ---
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    unlth0 = Unl[0]
    fnl = nl_force.local_force_history(unlt, unltdot, h, cst, unlth0)[0]

    # ---- extract tangential components ----
    tx, ty = unlt[:,0], unlt[:,1]
    Fx, Fy = fnl[:,0], fnl[:,1]

    Ft = np.sqrt(Fx**2 + Fy**2)

    # numerical derivative to detect plateau
    dFt = np.abs(np.diff(Ft))

    # boolean mask for slip
    # (Ft > 0 ensures it's not zero-force)
    slip_mask = np.zeros(Nt-1, dtype=bool)
    slip_mask[:] = (Ft[:-1] > tol) & (dFt < tol)

    # displacement increments
    dx = np.diff(tx)
    dy = np.diff(ty)
    ds = np.sqrt(dx**2 + dy**2)

    # slip increments only when plateaued
    slip_dist = np.sum(ds[slip_mask])

    period = 2*np.pi/w
    slip_rate = slip_dist / period

    return slip_dist, slip_rate




#%% Fit experimental data
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# -------------------------
# Load data
# -------------------------
data = loadmat('./data/hysteresis_data.mat')

cumE = data['cumE'].flatten()
diss = data['DISS'].flatten()
Kt_norm = data['Kt_norm'].flatten()
mu_norm = data['mu_norm'].flatten()


# -------------------------
# Model with t0 fixed = 1
# y(E) = t_inf + (1 - t_inf) * exp(-E/tau)
# -------------------------
def fixed_t0_model(E, t_inf, tau):
    return t_inf + (1.0 - t_inf) * np.exp(-E / tau)


# -------------------------
# Fit helper for fixed t0
# -------------------------
def fit_fixed_t0(E, y, label='y', plot=True, ax=None):
    # Clean
    mask = (~np.isnan(E)) & (~np.isnan(y))
    Ef = E[mask]
    yf = y[mask]

    # Sort
    idx = np.argsort(Ef)
    Ef = Ef[idx]
    yf = yf[idx]

    # Initial guesses
    t_inf_guess = yf[-1]      # last point
    tau_guess = (Ef.max() - Ef.min())/3 if Ef.max() > 0 else 1.0

    p0 = [t_inf_guess, tau_guess]

    # Bounds: tau>0
    lower = [-np.inf, 1e-12]
    upper = [ np.inf, np.inf]

    popt, pcov = curve_fit(
        fixed_t0_model,
        Ef,
        yf,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000
    )

    perr = np.sqrt(np.diag(pcov))

    # Plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        Efit = np.linspace(Ef.min(), Ef.max(), 300)
        yfit = fixed_t0_model(Efit, *popt)

        ax.plot(Ef, yf, 'o', label=f'{label} data')
        ax.plot(Efit, yfit, '-', label=f'{label} fit')
        ax.set_xlabel("Cumulative Energy (J)")
        ax.set_ylabel(label)
        ax.set_title(f'{label} fit (fixed t0=1)\nt_inf={popt[0]:.4f}±{perr[0]:.2g}, tau={popt[1]:.4g}±{perr[1]:.2g}')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

    return popt, perr


# -------------------------
# Fit Kt_norm and mu_norm
# -------------------------
p_Kt, e_Kt = fit_fixed_t0(cumE, Kt_norm, 'Kt_norm')
p_mu, e_mu = fit_fixed_t0(cumE, mu_norm, 'mu_norm')

plt.show()

# -------------------------
# Print numeric results
# -------------------------
print("\n--- Kt_norm Fit (t0 fixed = 1) ---")
print(f"t_inf = {p_Kt[0]:.6g} ± {e_Kt[0]:.2g}")
print(f"tau   = {p_Kt[1]:.6g} ± {e_Kt[1]:.2g}")

print("\n--- mu_norm Fit (t0 fixed = 1) ---")
print(f"t_inf = {p_mu[0]:.6g} ± {e_mu[0]:.2g}")
print(f"tau   = {p_mu[1]:.6g} ± {e_mu[1]:.2g}")

Npatches = Nnl // 3
tau_Kt_per_patch = p_Kt[1] * patch_areas / np.sum(patch_areas)
tau_mu_per_patch = p_mu[1] * patch_areas / np.sum(patch_areas)

ti_Kt = p_Kt[0]
ti_mu = p_mu[0]



#%% Determine Unworn System Energy Loss
patch_range = range(Nnl // 3)
Edot_unweathered = np.zeros(Uwxa_full.shape[0]) 
for level in range(Uwxa_full.shape[0]):
    Edot_total = 0
    for patch in patch_range:
        Edot = calculate_E_dot(Uwxa_full[level, :], h, Nt, Uwxa_full[level, -3], vib_sys.nonlinear_forces[patch], Uwxa_full[level, -1])
        Edot_total += Edot
    Edot_unweathered[level] = Edot_total

plt.plot(Uwxa_full[:, -1], Edot_unweathered)
plt.show()

# Perform interpolation
A_target = np.interp(diss[0], Edot_unweathered, Uwxa_full[:, -1])

print("A_target =", A_target)

#%% Wear Bulk Code

def return_E_dot_from_system(KT_patches0, KN_patches0, E_patches, unweathered):
    vib_sys = VibrationSystem(M, K)
    KT_patches, KN_patches = findDryParameters(E_patches, KT_patches0, KN_patches0)
    KT_average = np.mean(KT_patches)
    
    for patch in range(len(patch_areas)):
        dryfric = ElasticDryFriction3D(Qxyn[3*patch:3*patch+3, :], 
                                       Qxyn[3*patch:3*patch+3, :].T, 
                                       KT_patches[patch], KN_patches[patch], params[0])

        vib_sys.add_nl_force(dryfric)
        
    ##%% Prestress Analysis
    vib_sys.set_prestress_mu()
        
    #t, dtduxyn = ref_nlforce.force(upxyn[0])
    
    
    # Assemble per-patch stiffnesses
    # Assemble full Kst
    Kst = Txyn @ np.diag(np.r_[KT_patches, KT_patches, KN_patches]) @ Qxyn
    K0 = K + Kst;
    X0 = np.linalg.solve(K0,(Fv * prestress)).squeeze()

    # Calculate an initial guess
    X0 = np.linalg.solve(K0,(Fv * prestress))
    with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
    

        # function to solve
        pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)
    
        R0, dR0dX = pre_fun(X0)
    
        print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))
        
        t0 = time.time()
        Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                                  verbose=False, xtol=1e-11)
        
        t1 = time.time()
        
        print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))
    
        print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))
        
        
        vib_sys.update_force_history(Xpre)
    
        # Use the prestress solution as the intial slider positions for AFT as well
        # This influences residual tractions and may slightly change the results of
        # the simulation.
        vib_sys.set_aft_initialize(Xpre)
    
        # Reset to real friction coefficient after updating frictionless slider
        # positions
        # This is needed so that the friction coefficient is used in EPMC 
        # (rather than 0 tangential forces)
        vib_sys.reset_real_mu()
    
        # Recalculate stiffness with real mu (including stiffness from friction)
        Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)
    
        Kpre = (dRpredX + dRpredX.T) / 2.0
    
    
        eigvals, eigvecs = static_solver.eigs(Kpre, M, 
                                              subset_by_index=[0, 9])
    
    
        h_max = 1
    
        h = np.array(range(h_max+1))
    
        Nhc = hutils.Nhc(h)
    
        Ndof = vib_sys.M.shape[0]
    
        Fl = np.zeros(Nhc*Ndof)
    
        # Static Forces
        Fl[:Ndof] = prestress*Fv # EPMC static force
    
        # EPMC phase constraint - No cosine component at accel
        Fl[Ndof:2*Ndof] = Rmat[2, :] 
    
        Uwxa0 = np.zeros(Nhc*Ndof + 3)
    
        # Static Displacements (prediction for 0th harmonic)
        Uwxa0[:Ndof] = Xpre
    
        # Mode Shape (from linearized system for prediction)
        mode_ind = 0
        Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])
    
        # Linear Frequency (for prediction of low amplitude EPMC)
        Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))
        
        print(f"Wear Study eigenvalue sqrted: {Uwxa0[-3]}")
    
        # Initial Damping (low amplitude as prescribed)
        desired_zeta = np.array([0.087e-2, 0.034e-2]) 
        zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
        Uwxa0[-2] = 2*Uwxa0[-3]*zeta
        # Continuation really wants this to be zero, which kinda makes sense if you
        #think about it
        # Amplitude (Desired starting amplitude)
        Uwxa0[-1] = Astart
        
        #Fl = np.zeros(Nhc*Ndof)
    
        # Static Forces
        #Fl[:Ndof] = prestress*Fv
        
        epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                                  calc_grad=calc_grad)
    
    
        # The conditioning of the static displacements should be small since these
        # displacements are small, but very important
        CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)
    
        # Increasing delta means that these cofficients will be smaller in conditioned 
        # space. - reduces importance of higher harmonics when calculating arc length
        CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3) 
    
        # Allow different CtoP for static displacements than harmonics.
        CtoP[:Ndof] = CtoPstatic[:Ndof] 
    
        # Exactly take damping and frequency regardless of delta for conditioning
        CtoP[-3:-1] = np.abs(Uwxa0[-3:-1]) 
    
        # scale so step size is similar order as fraction of total distance from 
        # start to end
        CtoP[-1] = np.abs(Aend-Astart)
    
        cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, 
                                   config=continue_config)
            
        t0 = time.time()
        
        Uwxa_full_worn = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
        
        t1 = time.time()
    print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))
    
    
    
    if unweathered: #Unworn system energy loss calcaultion for elastic dry friction
        patch_range = range(Nnl // 3)
        Edot_unweathered = np.zeros(Uwxa_full_worn.shape[0]) 
        for level in range(Uwxa_full_worn.shape[0]):
            Edot_total = 0
            for patch in patch_range:
                Edot = calculate_E_dot(Uwxa_full_worn[level, :], h, Nt, Uwxa_full_worn[level, -3], vib_sys.nonlinear_forces[patch], Uwxa_full_worn[level, -1])
                Edot_total += Edot
            Edot_unweathered[level] = Edot_total

        # Perform interpolation
        A_target = np.interp(diss[0], Edot_unweathered, Uwxa_full_worn[:, -1])
        A_end = A_target
    print("A_end =", A_end)
    
    
    Edot_patches = np.zeros(Nnl // 3) #Npatches = Nnl // 3
    for patch in patch_range:
        Edot = calculate_E_dot(Uwxa_full_worn[-1, :], h, Nt, Uwxa_full_worn[-1, -3], vib_sys.nonlinear_forces[patch], Aend)
        Edot_patches[patch] = Edot
        
    return Edot_patches

#%% Wear Study

# for nlforce in vib_sys.nonlinear_forces:
#     print(calculate_E_dot(Uwxa_full[-1, :], h, Nt * 32, Uwxa_full[-1, -3], nlforce))

patch_range = range(KT_patches0.shape[0])
quadrature_method = 'euler' #euler or heun
final_time = 12*3600 + 1 #12 hours
results_Edot = {}      # dictionary: key = dt, value = E_dot_patches_history
results_Epatch = {}    # dictionary: key = dt, value = E_patches_history

slow_dts = np.array([1800, 900, 450, 225])

for dt_idx in range(len(slow_dts)):
    tstart = time.time()
    slow_times = np.array(range(0,final_time,slow_dts[dt_idx]))
    
    Aend = A_target
    
    #Start time loop 
    E_patches = np.zeros(Nnl // 3)
    E_patches_history = np.zeros((len(slow_times), Nnl // 3))
    E_dot_patches_history = np.zeros((Nnl // 3, len(slow_times)))
    E_patches_history[0, :] = E_patches #E_patches is zero obviously
#%% Slow Time Step    
    for i in range(len(slow_times) - 1):
        print('')
        print(f"======== TIME: {slow_times[i]} ==========")
        print('')
        
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):

            vib_sys = VibrationSystem(M, K)
            KT_patches, KN_patches, MU_patches = findDryParameters(E_patches, KT_patches0, KN_patches0, MU_patches0, 
                                                                   ti_Kt, tau_Kt_per_patch, ti_mu, tau_mu_per_patch)
            KT_average = np.mean(KT_patches)
            print(f"KT_AVERAGE: {KT_average:.4e}")
            
            for patch in range(len(patch_areas)):
                dryfric = ElasticDryFriction3D(Qxyn[3*patch:3*patch+3, :], 
                                               Qxyn[3*patch:3*patch+3, :].T, 
                                               KT_patches[patch], KN_patches[patch], MU_patches[patch])
        
                vib_sys.add_nl_force(dryfric)
                
            ##%% Prestress Analysis
            vib_sys.set_prestress_mu()
                
            #t, dtduxyn = ref_nlforce.force(upxyn[0])
            
            
            # Assemble per-patch stiffnesses
            # Assemble full Kst
            Kst = Txyn @ np.diag(np.r_[KT_patches, KT_patches, KN_patches]) @ Qxyn
            K0 = K + Kst;
            X0 = np.linalg.solve(K0,(Fv * prestress)).squeeze()
        
            # Calculate an initial guess
            X0 = np.linalg.solve(K0,(Fv * prestress))
            
        
            # function to solve
            pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)
        
            R0, dR0dX = pre_fun(X0)
        
            print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))
            
            t0 = time.time()
            Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                                      verbose=False, xtol=1e-11)
            
            t1 = time.time()
            
            print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))
        
            print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))
            
            
            vib_sys.update_force_history(Xpre)
    
            # Use the prestress solution as the intial slider positions for AFT as well
            # This influences residual tractions and may slightly change the results of
            # the simulation.
            vib_sys.set_aft_initialize(Xpre)
    
            # Reset to real friction coefficient after updating frictionless slider
            # positions
            # This is needed so that the friction coefficient is used in EPMC 
            # (rather than 0 tangential forces)
            vib_sys.reset_real_mu()
    
            # Recalculate stiffness with real mu (including stiffness from friction)
            Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)
    
            Kpre = (dRpredX + dRpredX.T) / 2.0
    
    
            eigvals, eigvecs = static_solver.eigs(Kpre, M, 
                                                  subset_by_index=[0, 9])
    
    
            h_max = 1
    
            h = np.array(range(h_max+1))
    
            Nhc = hutils.Nhc(h)
    
            Ndof = vib_sys.M.shape[0]
    
            Fl = np.zeros(Nhc*Ndof)
    
            # Static Forces
            Fl[:Ndof] = prestress*Fv # EPMC static force
    
            # EPMC phase constraint - No cosine component at accel
            Fl[Ndof:2*Ndof] = Rmat[2, :] 
    
            Uwxa0 = np.zeros(Nhc*Ndof + 3)
    
            # Static Displacements (prediction for 0th harmonic)
            Uwxa0[:Ndof] = Xpre
    
            # Mode Shape (from linearized system for prediction)
            mode_ind = 0
            Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])
    
            # Linear Frequency (for prediction of low amplitude EPMC)
            Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))
            
            print(f"Wear Study eigenvalue sqrted: {Uwxa0[-3]}")
    
            # Initial Damping (low amplitude as prescribed)
            desired_zeta = np.array([0.087e-2, 0.034e-2]) 
            zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
            Uwxa0[-2] = 2*Uwxa0[-3]*zeta
            # Continuation really wants this to be zero, which kinda makes sense if you
            #think about it
            # Amplitude (Desired starting amplitude)
            Uwxa0[-1] = Astart
            
            #Fl = np.zeros(Nhc*Ndof)
    
            # Static Forces
            #Fl[:Ndof] = prestress*Fv
            
            epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                                      calc_grad=calc_grad)
        
        
            # The conditioning of the static displacements should be small since these
            # displacements are small, but very important
            CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)
        
            # Increasing delta means that these cofficients will be smaller in conditioned 
            # space. - reduces importance of higher harmonics when calculating arc length
            CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3) 
        
            # Allow different CtoP for static displacements than harmonics.
            CtoP[:Ndof] = CtoPstatic[:Ndof] 
        
            # Exactly take damping and frequency regardless of delta for conditioning
            CtoP[-3:-1] = np.abs(Uwxa0[-3:-1]) 
        
            # scale so step size is similar order as fraction of total distance from 
            # start to end
            CtoP[-1] = np.abs(Aend-Astart)
        
            cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, 
                                       config=continue_config)
                
            t0 = time.time()
            
            Uwxa_full_worn = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
            
            t1 = time.time()
            print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))
#%% Energy Calculation
            
            if i == 0: #Unworn system energy loss calcaultion for elastic dry friction
                patch_range = range(Nnl // 3)
                Edot_unweathered = np.zeros(Uwxa_full_worn.shape[0]) 
                for level in range(Uwxa_full_worn.shape[0]):
                    Edot_total = 0
                    for patch in patch_range:
                        Edot = calculate_E_dot(Uwxa_full_worn[level, :], h, Nt, Uwxa_full_worn[level, -3], vib_sys.nonlinear_forces[patch], Uwxa_full_worn[level, -1])
                        Edot_total += Edot
                    Edot_unweathered[level] = Edot_total

                # Perform interpolation
                A_target = np.interp(diss[0], Edot_unweathered, Uwxa_full_worn[:, -1])
                A_end = A_target
                print("A_end =", A_end)
            
            
            Edot_patches = np.zeros(Nnl // 3) #Npatches = Nnl // 3
            for patch in patch_range:
                Edot = calculate_E_dot(Uwxa_full_worn[-1, :], h, Nt, Uwxa_full_worn[-1, -3], vib_sys.nonlinear_forces[patch], Aend)
                Edot_patches[patch] = Edot
            
                    
            dT = slow_times[i+1] - slow_times[i]
            E_patches = E_patches + Edot_patches * dT
            E_patches_history[i+1, :] = E_patches
            E_dot_patches_history[:, i] = Edot_patches
            E_patches_sum = np.sum(E_patches_history, axis=1)
    plt.plot(slow_times[:-1], np.sum(E_dot_patches_history, axis=0)[:-1], label = f"dT = {slow_dts[dt_idx]}")
    tend = time.time()
    print(f"That step took {tend - tstart}")
    
    # After finishing the slow_dt loop iteration:

    results_Edot[slow_dts[dt_idx]] = E_dot_patches_history.copy()
    results_Epatch[slow_dts[dt_idx]] = E_patches_history.copy()

plt.title("Wear Dissipation vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Dissipation (W)")
plt.legend()
print('done')
plt.show()

start_time = datetime.now()
timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
filename = f"wear_study_{timestamp_str}.npz"

np.savez(
    filename,
    timestamp=start_time.isoformat(),
    large_system=large_system,
    slow_dts=slow_dts,
    results_Edot=results_Edot,
    results_Epatch=results_Epatch,
    final_time=final_time,
    allow_pickle=True,
)

