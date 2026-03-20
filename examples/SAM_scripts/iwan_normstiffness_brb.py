import sys
import os
from scipy import io as sio
from scipy.stats import gmean
import numpy as np
import time
sys.path.append('../..')
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.nlforces.unilateral_spring import UnilateralSpring
#from tmdsimpy.nlforces.arcstiffness import ArcStiffness
from tmdsimpy.nlforces.normstiffness import NormStiffness
from tmdsimpy.solvers import NonlinearSolver
import tmdsimpy.nlutils as hutils_sam

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.continuation import Continuation

import matplotlib.pyplot as plt
from plot_style import apply_academic_plot_style


USE_LATEX_TEXT = apply_academic_plot_style(prefer_tex=True)
print(f"[plot-style] LaTeX text rendering: {'enabled' if USE_LATEX_TEXT else 'disabled'}")


###############################################################################
####### 1. Load System Matrices                                         #######
###############################################################################
system_fname = './data/brb_iwan4_mesh.mat'
system_matrices = sio.loadmat(system_fname)

M = system_matrices['M']
K = system_matrices['K']
Ndof = M.shape[1]

Q = np.array(system_matrices['Qxyn'])
T = np.array(system_matrices['Txyn'])

Nnl,Nnodes = Q.shape


###############################################################################
####### 2. Establish Vibration System                                   #######
###############################################################################

damp_ab = [0.087e-2*2*(168.622*2*np.pi), 0.0]
#Proportional damping arbitrary value

vib_sys = VibrationSystem(M, K, ab=damp_ab)

lparams =[5.5, 13.8, -0.95, -2, 12.35] #Plot on MATLAB

lparams = [4.957259556938594, 11.737845846995377, -0.023785113541144, -2.953624228947613, 16.933461451698992] #best damping
lparams = [5.530289365884571, 15.859238733093676, -0.999870602783605, -5.746331062210725, 16.888652840667923] #Best Freq
lpsci = [1, 1, 0, 1, 1]

iwan_parameters = [10 ** lparams[i] if lpsci[i] == 1 else lparams[i] for i in range(len(lparams))]



patch_areas = np.array([0.000385483218541205, 0.000715750926938309, 0.000715750512690334, 0.000715750596958764, 0.000385483218541205]) #Iwan parameters


for i in range(Nnl):
    Ls = Q[i:i+1, :]
    Lf = T[:, i:i+1]
    
    Fs = iwan_parameters[0] * patch_areas[i // 3]
    Kt = iwan_parameters[1] * patch_areas[i // 3]
    Chi = iwan_parameters[2]
    Bt = iwan_parameters[3]
    Kn = iwan_parameters[4] * patch_areas[i // 3]
    
    tmp_nl_force = None
    if i % 3 == 0 or i % 3 == 1: #x or y 
        tmp_nl_force = VectorIwan4(Ls, Lf, Kt,
                                   Fs,
                                   Chi,
                                   Bt)
    else:
        tmp_nl_force = UnilateralSpring(Ls, Lf, Kn) #Linear penalty stiffness for the normal dimension
    
    vib_sys.add_nl_force(tmp_nl_force)

iwan_forces = vib_sys.nonlinear_forces

Astart = -10
Aend = -4

# Normal - settings for higher accuracy as used in previous papers
h_max = 3 # harmonics 0, 1, 2, 3
Nt = 1<<7 # 2**7 = 128 AFT steps 

ds = 0.002
dsmax = 0.005
dsmin = 0.0005
# Adjust weighting of amplitude v. other in continuation to hopefully 
# reduce turning around. Higher puts more emphasis on continuation 
# parameter (amplitude)
FracLam = 1
###############################################################################
####### 3. Prestress Analysis                                           #######
###############################################################################
h_max = 3
h = np.array(range(h_max + 1))
Nhc = hutils.Nhc(h)

Fv = system_matrices['Fv'][:, 0]
prestress = 12249.0 #stolen from brb_epmc

KT_patches0 = iwan_parameters[1] * patch_areas
KN_patches0 = iwan_parameters[-1] * patch_areas


Kst = T @ np.diag(np.r_[KT_patches0, KT_patches0, KN_patches0]) @ Q
K0 = K + Kst;
X0 = np.linalg.solve(K0,(Fv * prestress)).squeeze()



pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)
R0, dR0dX = pre_fun(X0)
print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))
                                                      
                                                      
static_config={'max_steps' : 100,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
static_solver = NonlinearSolver() 

t0 = time.time()

Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

t1 = time.time()

print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))

print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))

vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()


###############################################################################
####### 11. Updated Eigenvalue Analysis After Prestress                 #######
###############################################################################

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

#Imported prestress stiffness matrix

K_stat_imported = sio.loadmat('./data/K_static.mat')['dRstat']


eigvals, eigvecs = static_solver.eigs(Kpre, system_matrices['M'], 
                                      subset_by_index=[0, 9])

print(f"Eigenvalues of Iwan Prestress: {eigvals}")


#eigvals = eigvals[1:]
#eigvecs = eigvecs[:, 1:] #BANDAID SOLUTION TO FIX LARGE NEG EIGENVALUE

###############################################################################
####### 12. Updated Damping Matrix After Prestress                      #######
###############################################################################

# This block resets the damping matrix after prestress analysis to 
# achieve the desired levels of viscous linear damping for the first and second
# bending modes

# First and Second Bending Modes damping ratios (taken from experiments on BRB)
desired_zeta = np.array([0.087e-2, 0.034e-2]) 

# 1st and 2nd bending mode = total 1st and 3rd modes
omega_12 = np.array([np.sqrt(eigvals)[0:3:2]]).reshape(2, 1) 

# Matrix problem for proportional damping
prop_mat = np.hstack((1/(2.0*omega_12), omega_12/2.0))

pre_ab = np.linalg.solve(prop_mat, desired_zeta)

vib_sys.set_new_C(C=pre_ab[0]*vib_sys.M + pre_ab[1]*Kpre)

###############################################################################
####### 13. EPMC Initial Guess                                          #######
###############################################################################

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Ndof = vib_sys.M.shape[0]

Fl = np.zeros(Nhc*Ndof)

# Static Forces
Fl[:Ndof] = prestress*Fv # EPMC static force

# EPMC phase constraint - No cosine component at accel
Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] 

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements (prediction for 0th harmonic)
Uwxa0[:Ndof] = Xpre

# Mode Shape (from linearized system for prediction)
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta


Uwxa0[-1] = Astart

###############################################################################
####### 15. EPMC Continuation                                           #######
###############################################################################

# This block actually executes the full continuation for the EPMC solution.

epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                          calc_grad=calc_grad)
epmc_config={'max_steps' : 12, # balance with reform_freq
            'reform_freq' : 2, #>1 corresponds to BFGS 
            'verbose' : True, 
            'xtol'    : None, # Just use the one passed from continuation
            'rtol'    : 1e-9,
            'etol'    : None,
            'xtol_rel' : 1e-6, 
            'rtol_rel' : None,
            'etol_rel' : None,
            'stopping_tol' : ['xtol'], # stop on xtol 
            'accepting_tol' : ['xtol_rel', 'rtol'] # accept solution on these
            }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolver()

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 4,
                   'MaxSteps'   : 2500, # May need more depending on ds and dsmin
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

R0, dRdX0, dRda = epmc_fun(Uwxa0)
print("Norm of initial solution residual: {}".format(np.linalg.norm(R0)))

cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, 
                           config=continue_config)

t0 = time.time()

Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))


freqs = Uwxa_full[:, -3]/2/np.pi
amps = Uwxa_full[:, -1]

iwan_freqs = freqs
iwan_xis = Uwxa_full[:, -2]
iwan_amps = amps

plt.plot(amps, freqs)
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Natural Frequency")
plt.title("Iwan: " + np.array2string(np.round(lparams, 5)))
plt.show()

np.save('data/Uwxa_full_iwan.npy', Uwxa_full)

Uwxa_full_iwan = Uwxa_full

#%% Attempting to Backsolve for kt

Axy = np.diag(np.repeat(patch_areas, 2))
An = np.diag(patch_areas)


# Tangential components (x and y), interleaved by patch: 0,1, 3,4, 6,7, ...
# Keep this ordering aligned with Axy = repeat(patch_areas, 2) and
# patch_idx*2 + xyn so each tangential DOF maps consistently to its local s_i.
xy_rows = np.column_stack((np.arange(0, Q.shape[0], 3), np.arange(1, Q.shape[0], 3))).reshape(-1)
Qxy = Q[xy_rows, :]

# Normal components (n): rows 2, 5, 8, ...
n_rows = np.arange(2, Q.shape[0], 3)
Qn = Q[n_rows, :]

Txy = Qxy.T
Tn = T[:, n_rows]

# Build a normal-only unilateral contact model and extract the prestressed
# tangent matrix for the normal direction contribution.
normal_sys = VibrationSystem(M, K, ab=damp_ab)
for patch_idx, row_idx in enumerate(n_rows):
    Ls = Q[row_idx:row_idx+1, :]
    Lf = T[:, row_idx:row_idx+1]
    Kn_patch = iwan_parameters[-1] * patch_areas[patch_idx]
    normal_sys.add_nl_force(UnilateralSpring(Ls, Lf, Kn_patch))

# Use fully-closed-contact tangent as only an initial guess for the static solve.
Knl_n0 = Tn @ An @ Qn * iwan_parameters[-1]
X0_normal = np.linalg.solve(K + Knl_n0, (Fv * prestress)).squeeze()
normal_solver = NonlinearSolver()
normal_pre_fun = lambda U, calc_grad=True: normal_sys.static_res(U, Fv * prestress)
Xpre_normal, _, _, _ = normal_solver.nsolve(
    normal_pre_fun,
    X0_normal,
    verbose=True,
    xtol=1e-13,
)
_, dRpredX_normal = normal_sys.static_res(Xpre_normal, Fv * prestress)
Kn = (dRpredX_normal + dRpredX_normal.T) / 2.0

row = Uwxa_full[0, :]        # use first converged solution
omega_star = row[-3]

h1c = row[Ndof : 2*Ndof]
h1s = row[2*Ndof : 3*Ndof]

phi_star_complex = h1c + 1j*h1s
# Normalize by first component phase
phi_star = np.real(phi_star_complex * np.exp(-1j*np.angle(phi_star_complex[0])))

# --- Mass-normalize (CRITICAL) ---
phi_star /= np.sqrt(phi_star.T @ M @ phi_star)

# --- Baseline frequency (same mode index you continued) ---
#omega = np.sqrt(eigvals_pre[mode_ind])

# --- Robust stiffness identification ---
kt_pred = (
    omega_star**2
    - phi_star.T @ Kn @ phi_star
) / (
    (Qxy @ phi_star).T @ Axy @ (Qxy @ phi_star)
)

print("Identified tangential stiffness kt =", kt_pred)


#%% Curve fit to find s(A) and b(A)

kt_pred_array = np.zeros(Uwxa_full.shape[0])
phi_levels = np.zeros((Ndof, Uwxa_full.shape[0]))

for level in range(Uwxa_full.shape[0]):
    h1c_level = Uwxa_full[level, Ndof: 2*Ndof]
    h1s_level = Uwxa_full[level, 2*Ndof: 3*Ndof]
    phi_complex = h1c_level + 1j*h1s_level
    phi = np.real(phi_complex * np.exp(-1j*np.angle(phi_complex[0])))
    phi /= np.sqrt(phi.T @ M @ phi)
    phi_levels[:, level] = phi
    
    kt_pred_array[level] = (Uwxa_full[level, -3]**2 - phi.T @ Kn @ phi
    ) / (
         (Qxy @ phi).T @ Axy @ (Qxy @ phi)
         )
    


'''
kt_pred_array = (Uwxa_full[:, -3]**2 - phi_star.T @ Kn @ phi_star
) / (
     (Qxy @ phi_star).T @ Axy @ (Qxy @ phi_star)
     )
'''
plt.plot(Uwxa_full[:, -1], kt_pred_array)
plt.show()


def kt_eff_model_A(A, bA, sA):
    return kt_pred * np.power(np.power(A / sA, bA) + 1, -1 - 1 / bA)

lA_vals = Uwxa_full[:, -1]
A_vals = 10 ** Uwxa_full[:, -1]


# Fit only on the physically meaningful segment:
# positive stiffness and monotonic softening (exclude contaminated tail points).
hardening_tol = 5e-3  # allow small numerical wiggles before declaring hardening
fit_stop = len(A_vals)
for i in range(1, len(A_vals)):
    if (kt_pred_array[i] <= 0) or (kt_pred_array[i] > kt_pred_array[i-1] * (1 + hardening_tol)):
        fit_stop = i
        break

# Optional cutoff: reject high-amplitude points after kt drops below half of
# its maximum (useful when modal coupling corrupts the tail).
# Set False (or delete this block) to disable.
USE_HALF_MAX_KT_CUTOFF = False
halfmax_stop = len(A_vals)
if USE_HALF_MAX_KT_CUTOFF:
    kt_half_threshold = 0.5 * np.max(kt_pred_array)
    below_half = np.where(kt_pred_array < kt_half_threshold)[0]
    if below_half.size > 0:
        halfmax_stop = int(below_half[0])

fit_mask = (
    (np.arange(len(A_vals)) < fit_stop)
    & (np.arange(len(A_vals)) < halfmax_stop)
    & (kt_pred_array > 0)
)
if np.count_nonzero(fit_mask) < 8:
    fit_mask = ((np.arange(len(A_vals)) < halfmax_stop) & (kt_pred_array > 0))
if np.count_nonzero(fit_mask) < 2:
    raise RuntimeError("Insufficient positive stiffness points after fit cutoff(s).")

rejected_idx = min(fit_stop, halfmax_stop)
rejected_start_lA = lA_vals[rejected_idx] if rejected_idx < len(lA_vals) else None

p0 = [2.7, gmean(A_vals[fit_mask])]

from scipy.optimize import curve_fit

popt, _ = curve_fit(
    kt_eff_model_A,
    A_vals[fit_mask],
    kt_pred_array[fit_mask],
    p0=p0,
    maxfev=20000
)

bA, sA = popt



kt_fit_plot = kt_eff_model_A(A_vals, bA, sA)

plt.figure()
if rejected_start_lA is not None:
    plt.axvspan(
        rejected_start_lA,
        lA_vals[-1],
        color='red',
        alpha=0.15,
        label='Rejected backbone curve'
    )
plt.plot(lA_vals, kt_pred_array, 'k.', alpha=0.4, label='Identified $k_t$')
plt.plot(lA_vals, kt_fit_plot, 'r-', lw=2, label='NormStiffness fit')
plt.title('Predicted Tangential Stiffness vs. Log Amplitude')
plt.xlabel(r'Log Amplitude, $\log_{10}(A)$')
plt.ylabel(r'Predicted Tangential Stiffness, $k_t$')
plt.legend()
plt.grid(True)
plt.show()


A_levels = np.power(10, lA_vals)        # convert log amplitude to linear
ui_levels_from_iwan = np.zeros((Qxy.shape[0], Uwxa_full.shape[0]))

# for level in range(Uwxa_full.shape[0]):
#     # extract scaled h1c and h1s
#     h1c_star = Uwxa_full[level, Ndof:2*Ndof]    # stored as h1c*
#     h1s_star = Uwxa_full[level, 2*Ndof:3*Ndof]  # stored as h1s*

#     # reconstruct *true physical* complex amplitudes
#     phi_complex_star = h1c_star + 1j*h1s_star
#     phi_complex = phi_complex_star * A_levels[level]  # scale by 10^lA

#     # optional: normalize phase relative to first DOF
#     phi_complex *= np.exp(-1j*np.angle(phi_complex[0]))

#     # take magnitude of tangential DOFs (projected via Qxy)
#     ui_levels[:, level] = np.abs(Qxy @ phi_complex)
    

"""
Convert EPMC harmonic solution row into local displacement/velocity time series.
"""

Nhc = hutils.Nhc(h)
for level in range(Uwxa_full.shape[0]):
    Uwxa = Uwxa_full[level, :]
    # --- Extract harmonic coefficients ---
    Uhc = Uwxa[:Nhc*Ndof].reshape(Nhc, Ndof)

    
    # --- Amplitude scaling ---
    A = 10**Uwxa[-1]
    Uhc = Uhc * A
    
    # --- Physical time series ---
    ut = hutils.time_series_deriv(1 << 7, h, Uhc, order=0)
    utdot = hutils.time_series_deriv(1 << 7, h, Uhc, order=1)
    
    # --- Convert to local nonlinear coordinates ---
    unlt = ut @ Qxy.T
    
    ui_levels_from_iwan[:, level] = 0.5 * (
        np.max(unlt, axis=0) - np.min(unlt, axis=0)
    )
    
    if level == 0:
        print(f"Initial u levels: {0.5 * (np.max(unlt, axis=0) - np.min(unlt, axis=0))}")



# Map sA to each nonlinear DOF using the full local-amplitude curve u_i(A).
A_sort_idx = np.argsort(A_levels)
A_sorted = A_levels[A_sort_idx]
ui_sorted = ui_levels_from_iwan[:, A_sort_idx]
A_eval = np.clip(sA, A_sorted[0], A_sorted[-1])
s_i = np.array([
    np.interp(A_eval, A_sorted, ui_sorted[row, :])
    for row in range(ui_sorted.shape[0])
])
ui_v_a = s_i / A_eval

b_i = bA * np.ones(Qxy.shape[0])


#%% Arcstiffness Vibration Simulation

###############################################################################
####### 1. Load System Matrices                                         #######
###############################################################################
system_fname = './data/brb_iwan4_mesh.mat'
system_matrices = sio.loadmat(system_fname)

M = system_matrices['M']
K = system_matrices['K']
Ndof = M.shape[1]

Q = np.array(system_matrices['Qxyn'])
T = np.array(system_matrices['Txyn'])

Nnl,Nnodes = Q.shape

vib_sys = VibrationSystem(M, K, ab = damp_ab)

for i in range(Nnl):
    Ls = Q[i:i+1, :]
    Lf = T[:, i:i+1]
    patch_idx = i // 3
    xyn = i % 3

    if i % 3 != 2:  # Tangential DOF
        Kt = kt_pred * patch_areas[patch_idx]  # or use Qxy[patch_idx, tangential_component]
        bi = b_i[patch_idx * 2 + xyn] #0 1 3 4 6 7 etc
        si = s_i[patch_idx * 2 + xyn]
        tmp_nl_force = NormStiffness(Ls, Lf, Kt, bi, si)
    else:  # Normal DOF
        Kn = iwan_parameters[4] * patch_areas[patch_idx]
        tmp_nl_force = UnilateralSpring(Ls, Lf, Kn)
    vib_sys.add_nl_force(tmp_nl_force)

# Normal - settings for higher accuracy as used in previous papers
h_max = 3 # harmonics 0, 1, 2, 3
Nt = 1<<7 # 2**7 = 128 AFT steps 

ds = 0.002
dsmax = 0.005
dsmin = 0.0005
# Adjust weighting of amplitude v. other in continuation to hopefully 
# reduce turning around. Higher puts more emphasis on continuation 
# parameter (amplitude)
FracLam = 1
###############################################################################
####### 3. Prestress Analysis                                           #######
###############################################################################
h_max = 3
h = np.array(range(h_max + 1))
Nhc = hutils.Nhc(h)

Fv = system_matrices['Fv'][:, 0]
prestress = 12249.0 #stolen from brb_epmc


Kstxy = Txy @ Axy @ Qxy * kt_pred
Kstn = Tn @ An @ Qn * iwan_parameters[-1]
K0 = K + Kstxy + Kstn;
X0 = np.linalg.solve(K0,(Fv * prestress)).squeeze()

eigvals_K0, _ = static_solver.eigs(K0, M)

pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)
R0, dR0dX = pre_fun(X0)
print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))
                                                      
                                                      
static_config={'max_steps' : 100,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
static_solver = NonlinearSolver() 

t0 = time.time()

Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

t1 = time.time()

print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))

print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))

vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()


###############################################################################
####### 11. Updated Eigenvalue Analysis After Prestress                 #######
###############################################################################

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

#Imported prestress stiffness matrix


eigvals, eigvecs = static_solver.eigs(Kpre, system_matrices['M'], 
                                      subset_by_index=[0, 9])\

print(f"Eigenvalues of Arcstiffness Prestress: {eigvals}")

#eigvals = eigvals[1:]
#eigvecs = eigvecs[:, 1:] #BANDAID SOLUTION TO FIX LARGE NEG EIGENVALUE

###############################################################################
####### 12. Updated Damping Matrix After Prestress                      #######
###############################################################################

# This block resets the damping matrix after prestress analysis to 
# achieve the desired levels of viscous linear damping for the first and second
# bending modes

# First and Second Bending Modes damping ratios (taken from experiments on BRB)
desired_zeta = np.array([0.087e-2, 0.034e-2]) 

# 1st and 2nd bending mode = total 1st and 3rd modes
omega_12 = np.array([np.sqrt(eigvals)[0:3:2]]).reshape(2, 1) 

# Matrix problem for proportional damping
prop_mat = np.hstack((1/(2.0*omega_12), omega_12/2.0))

pre_ab = np.linalg.solve(prop_mat, desired_zeta)

vib_sys.set_new_C(C=pre_ab[0]*vib_sys.M + pre_ab[1]*Kpre)

###############################################################################
####### 13. EPMC Initial Guess                                          #######
###############################################################################

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Ndof = vib_sys.M.shape[0]

Fl = np.zeros(Nhc*Ndof)

# Static Forces
Fl[:Ndof] = prestress*Fv # EPMC static force

# EPMC phase constraint - No cosine component at accel
Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] 

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements (prediction for 0th harmonic)
Uwxa0[:Ndof] = Xpre

# Mode Shape (from linearized system for prediction)
# Match the physical branch from Iwan by frequency proximity, not raw index.
omega_target = iwan_freqs[0] * 2 * np.pi
mode_ind = int(np.argmin(np.abs(np.sqrt(np.real(eigvals)) - omega_target)))
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta


Uwxa0[-1] = Astart

###############################################################################
####### 15. EPMC Continuation                                           #######
###############################################################################

# This block actually executes the full continuation for the EPMC solution.

epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                          calc_grad=calc_grad)
epmc_config={'max_steps' : 12, # balance with reform_freq
            'reform_freq' : 2, #>1 corresponds to BFGS 
            'verbose' : True, 
            'xtol'    : None, # Just use the one passed from continuation
            'rtol'    : 1e-9,
            'etol'    : None,
            'xtol_rel' : 1e-6, 
            'rtol_rel' : None,
            'etol_rel' : None,
            'stopping_tol' : ['xtol'], # stop on xtol 
            'accepting_tol' : ['xtol_rel', 'rtol'] # accept solution on these
            }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolver()

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 4,
                   'MaxSteps'   : 2500, # May need more depending on ds and dsmin
                   'dsmin'      : dsmin,
                   'dsmax'      : dsmax,
                   'verbose'    : 1,
                   'xtol'       : 1e-5*np.sqrt(Uwxa0.shape[0]), 
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

R0, dRdX0, dRda = epmc_fun(Uwxa0)
print("Norm of initial solution residual: {}".format(np.linalg.norm(R0)))

cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, 
                           config=continue_config)

t0 = time.time()

Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))


freqs = Uwxa_full[:, -3]/2/np.pi
amps = Uwxa_full[:, -1]

plt.plot(amps, freqs, label = 'NormStiffness')
plt.plot(iwan_amps, iwan_freqs, label = 'Iwan')
plt.legend()
plt.xlabel("Logarithm of Modal Amplitude")
plt.ylabel("Natural Frequency (Hz)")
plt.title(f"Natural Frequency with Respect to Ampltiude")
plt.show()

xis = Uwxa_full[:, -2]

plt.plot(amps, xis, label = 'NormStiffness')
plt.plot(iwan_amps, iwan_xis, label = 'Iwan')
plt.legend()
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Self-Excitation Factor (1/s)")
plt.title(f"Self Excitation Factor with Respect to Amplitude")
plt.show()

np.save('data/Uwxa_full_arc.npy', Uwxa_full)

norm_forces = vib_sys.nonlinear_forces

Uwxa_full_norm = Uwxa_full

#%% Diagnose the Hysteretic Forces

def extract_local_time_series(Uwxa_row, Q, Ndof, h, Nt):
    """
    Convert EPMC harmonic solution row into local displacement/velocity time series.
    """

    Nhc = hutils.Nhc(h)

    # --- Extract harmonic coefficients ---
    Uhc = Uwxa_row[:Nhc*Ndof].reshape(Nhc, Ndof)

    # --- Amplitude scaling ---
    A = 10**Uwxa_row[-1]
    Uhc = Uhc * A

    # --- Physical time series ---
    ut = hutils.time_series_deriv(Nt, h, Uhc, order=0)
    utdot = hutils.time_series_deriv(Nt, h, Uhc, order=1)

    # --- Convert to local nonlinear coordinates ---
    unlt = ut @ Q.T
    unltdot = utdot @ Q.T

    return unlt, unltdot, Uhc


def plot_hysteresis_loops(nlforces, Uwxa_full, Q, Ndof, h, Nt, title, line):
    title = title + f" line: {line}" 
    # pick amplitude level (middle of continuation usually informative)
    level = line
    Uwxa_row = Uwxa_full[level]

    unlt, unltdot, Uhc = extract_local_time_series(
        Uwxa_row, Q, Ndof, h, Nt
    )

    # harmonic basis evaluation (required by local_force_history)
    cst = hutils.time_series_deriv(Nt, h, np.eye(hutils.Nhc(h)), order=0)

    # zeroth harmonic
    unlth0 = (Q @ Uhc[0]).squeeze()

    # ---- first pass: count hysteretic DOFs ----
    Nhyst = 0
    for nlforce in nlforces:
        if nlforce.nl_force_type() == 1:
            Nhyst += 1

    if level == 0:
        print(f"Initial u levels: {0.5 * (np.max(unlt, axis=0) - np.min(unlt, axis=0))}")
        
    # ---- subplot layout ----
    ncols = 5
    nrows = int(np.ceil(Nhyst / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(14, 3 * nrows),
        sharex=False,
        sharey=False,
        squeeze=False
    )
    fig.suptitle(title, fontsize=14)

    axes = axes.flatten()
    hysteretic_index = 0

    for nlforce, sl in zip(nlforces, range(Q.shape[0])):

        if nlforce.nl_force_type() != 1:
            continue
        
        unlt_line = unlt[:, [sl]]
        unltdot_line = unltdot[:, [sl]]
    
        ft, _, _ = nlforce.local_force_history(
            unlt_line,
            unltdot_line,
            h,
            cst,
            unlth0[sl],
            max_repeats=2
        )

        ax = axes[hysteretic_index]

        ft_1d = np.ravel(ft)
        # local_force_history returns the converged second pass.
        ax.plot(np.r_[unlt[:, sl], unlt[0, sl]], np.r_[ft_1d, ft_1d[0]])
        if hasattr(nlforce, "s"):
            s_val = getattr(nlforce, "s")
            title_str = f"DOF {hysteretic_index}, s = {s_val:.3e}"
        else:
            title_str = f"DOF {hysteretic_index}"
        
        ax.set_title(title_str)

        ax.set_xlabel("u")
        ax.set_ylabel("f")
        ax.grid(True)

        hysteretic_index += 1

    # ---- hide unused axes ----
    for k in range(hysteretic_index, len(axes)):
        axes[k].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
'''

for line in [0, 10, 20, 30]:

    plot_hysteresis_loops(
        iwan_forces,
        Uwxa_full_iwan,
        Q,
        Ndof,
        h,
        Nt,
        title="Iwan",
        line=line
    )
    
    plot_hysteresis_loops(
        norm_forces,
        Uwxa_full_norm,
        Q,
        Ndof,
        h,
        Nt,
        title="NormStiffness",
        line=line
    )
'''
