'''
Nonlinear Model test with Multiple DOFs in Sample System
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tmdsimpy.nlforces.normstiffness import NormStiffness
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.utils.harmonic import *
import matplotlib.pyplot as plt
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.jax.solvers import NonlinearSolverOMP
from tmdsimpy.solvers import NonlinearSolver

from tmdsimpy.vibration_system import VibrationSystem
from sdof_iwan_epmc import sdof_uwxa_full
import tmdsimpy.nlutils as nlutils
from scipy import io as sio
import time
from scipy.stats import gmean
from plot_style import apply_academic_plot_style


USE_LATEX_TEXT = apply_academic_plot_style(prefer_tex=True)
print(f"[plot-style] LaTeX text rendering: {'enabled' if USE_LATEX_TEXT else 'disabled'}")


os.system('clear')

#%% Iwan Modeling

M = 1 * np.diag([10, 100, 30])
C = M*0.005
c = 0.005

K = 1e7 * np.array([[100, -10, 0], [-10, 15, -5], [0, -5, 10]])


Ndof = M.shape[0]


Q = np.array([[-1, 1, 0],
              [0, -6, 6]])
T = Q.T

kt = 1e6
Fs = 1e0 # N, Match Jenkins
chi = -0.7  # Have a more full hysteresis loop than chi=0.0
beta = 0.01 # Smooth Transition

vib_sys = VibrationSystem(M, K, C = C)

for i in range(Q.shape[0]):
    iwan_force = VectorIwan4(np.atleast_2d(Q[i, :]), np.atleast_2d(T[:, i]).T, kt, Fs, chi, beta)
    vib_sys.add_nl_force(iwan_force)
    
ref_nlforces = vib_sys.nonlinear_forces

Astart = -10
Aend = 3

# Normal - settings for higher accuracy as used in previous papers
h_max = 1 # harmonics 0, 1, 2, 3
Nt = 1 << 7  # 2**7 = 128 AFT steps
h = np.array(range(h_max+1))
Nhc = hutils.Nhc(h)
mode_ind = 0
ds = 0.01
dsmax = 0.02
dsmin = 0.005
# Adjust weighting of amplitude v. other in continuation to hopefully
# reduce turning around. Higher puts more emphasis on continuation
# parameter (amplitude)
FracLam = 1

static_solver = NonlinearSolver()
Fv = np.zeros(Ndof)
# Forcing Vector
Fv[-1] = 1 * 10 ** Astart  # Cosine force vector at arbitrary dof

eigvals_pre, eigvecs_pre = static_solver.eigs(K, M)

X0 = np.real(eigvecs_pre[:, mode_ind])

pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv)
R0, dR0dX = pre_fun(X0)
print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))

static_config={'max_steps' : 30,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
#static_solver = NonlinearSolverOMP(config=static_config)


Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

#vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre_iwan = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre_iwan, M) #Temporary
Fl = np.zeros(Nhc*Ndof)
Fl[:Ndof] = Fv
Fl[2*Ndof] = 1 #Sine Forcing


Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Mode Shape (from linearized system for prediction)

Uwxa0[Ndof:2*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = c  # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta
#Uwxa0[-2] = 0 #This is the modification to damping that is not originally correct -- I think

Uwxa0[-1] = Astart


def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 20,  # balance with reform_freq
               'reform_freq': 1,  # >1 corresponds to BFGS
               'verbose': False,
               'xtol': None,  # Just use the one passed from continuation
               'rtol': 1e-9,
               'etol': None,
               'xtol_rel': 1e0,
               'rtol_rel': None,
               'etol_rel': None,
               'stopping_tol': ['xtol'],  # stop on xtol
               # accept solution on these
               'accepting_tol': ['xtol_rel', 'rtol']
               }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolverOMP(config=epmc_config)

continue_config = {'DynamicCtoP': True,
                   'TargetNfev': 8,
                   'MaxSteps': 5000,  # May need more depending on ds and dsmin
                   'dsmin': dsmin,
                   'dsmax': dsmax,
                   'verbose': 1,
                   'xtol': 1e-6*np.sqrt(Uwxa0.shape[0]),
                   'corrector': 'Ortho',  # Ortho, Pseudo
                   'nsolve_verbose': False,
                   'FracLam': FracLam,
                   'FracLamList': [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop': 0.05,  # stop if backtracks to before lam0,
                   'armijo_iters': 30
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

Uwxa_full_iwan = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
Uwxa_full = Uwxa_full_iwan
t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))

plt.plot(Uwxa_full_iwan[:, -1], Uwxa_full_iwan[:, -3])
plt.show()

plt.plot(Uwxa_full_iwan[:, -1], Uwxa_full_iwan[:, -2])
plt.show()

#breakpoint()
#%% Sigmoid Stiffness Testing

harmonic_norm_iwan = nlutils.nonlinear_harmonic_norm(Uwxa_full_iwan, Q) 
log_hnorm_iwan = np.log10(harmonic_norm_iwan)


'''
Curvefit for NormStiffness, Establish Vibration System
'''

w_star = Uwxa_full_iwan[0, -3]


kt_pred_array = np.zeros(Uwxa_full.shape[0])
phi_levels = np.zeros((Ndof, Uwxa_full.shape[0]))

for level in range(Uwxa_full.shape[0]):
    h1c_level = Uwxa_full[level, Ndof: 2*Ndof]
    h1s_level = Uwxa_full[level, 2*Ndof: 3*Ndof]
    phi_complex = h1c_level + 1j*h1s_level
    phi = np.real(phi_complex * np.exp(-1j*np.angle(phi_complex[0])))
    phi /= np.sqrt(phi.T @ M @ phi)
    phi_levels[:, level] = phi
    
    kt_pred_array[level] = (Uwxa_full[level, -3]**2 - phi.T @ K @ phi
    ) / (
         (Q @ phi).T @ (Q @ phi)
         )

        
plt.plot(Uwxa_full[:, -1], kt_pred_array)
plt.show()

kt_pred = kt_pred_array[0]


def kt_eff_model_A(A, bA, sA):
    return kt_pred * np.power(np.power(A / sA, bA) + 1, -1 - 1 / bA)


A_vals = 10**Uwxa_full[:, -1]

p0 = [2.0, gmean(A_vals)]

from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import curve_fit

popt, _ = curve_fit(
    kt_eff_model_A,
    A_vals,
    kt_pred_array,
    p0=p0,
    maxfev=20000
)

bA, sA = popt



lA_values = np.log10(A_vals)

kt_fit_plot = kt_eff_model_A(A_vals, bA, sA)


plt.figure()
plt.plot(lA_values, kt_pred_array, 'k.', alpha=0.4, label='Identified $k_t$')
plt.plot(lA_values, kt_fit_plot, 'r-', lw=2, label='Norm fit')
plt.xlabel('Log Modal Amplitude')
plt.ylabel('Effective Tangential Stiffness')
plt.legend()
plt.grid(True)
plt.show()

ui_levels = np.zeros((Q.shape[0], Uwxa_full.shape[0]))


for level in range(Uwxa_full.shape[0]):
    # extract scaled h1c and h1s
    h1c_star = Uwxa_full[level, Ndof:2*Ndof]    # stored as h1c*
    h1s_star = Uwxa_full[level, 2*Ndof:3*Ndof]  # stored as h1s*

    # reconstruct *true physical* complex amplitudes
    phi_complex_star = h1c_star + 1j*h1s_star
    phi_complex = phi_complex_star * A_vals[level]  # scale by 10^lA

    # optional: normalize phase relative to first DOF
    phi_complex *= np.exp(-1j*np.angle(phi_complex[0]))

    # take magnitude of tangential DOFs (projected via Qxy)
    ui_levels[:, level] = np.abs(Q @ phi_complex)

print(ui_levels)
print(Uwxa_full[:, -1])
print(ui_levels / 10**Uwxa_full[:, -1])
ui_v_a = ui_levels / 10**Uwxa_full[:, -1]

# global log amplitude at transition
# A_c = 10**sA in linear units
idx_c = np.argmin(np.abs(A_vals - sA))

# displacements at the transition
uphi_c = ui_levels[:, idx_c]

# coordinate-specific log amplitude shifts
s_i = uphi_c

s_i = (sA * ui_v_a)[:, 0] #relation between amplitude and nonlinear amplitude

vib_sys = VibrationSystem(M, K, C)

for nlforce in range(Q.shape[0]):
    normstiff = NormStiffness(Q[[nlforce], :], Q[[nlforce], :].T, kt_pred, bA, s_i[nlforce])
    vib_sys.add_nl_force(normstiff)


#%%


test_nlforces = vib_sys.nonlinear_forces

#static_solver = NonlinearSolverOMP(config=epmc_config)
Fv = np.zeros(Ndof)
# Forcing Vector
Fv[-1] = 1  # Cosine force vector at arbitrary dof

eigvals_pre, eigvecs_pre = static_solver.eigs(K, M)

X0 = np.real(eigvecs_pre[:, mode_ind])

pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv)
R0, dR0dX = pre_fun(X0)
print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))
Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

#vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre, M) #Temporary


def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 4,  # balance with reform_freq
               'reform_freq': 1,  # >1 corresponds to BFGS
               'verbose': False,
               'xtol': None,  # Just use the one passed from continuation
               'rtol': 1e-9,
               'etol': None,
               'xtol_rel': 1e0,
               'rtol_rel': None,
               'etol_rel': None,
               'stopping_tol': ['xtol'],  # stop on xtol
               # accept solution on these
               'accepting_tol': ['xtol_rel', 'rtol']
               }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolverOMP(config=epmc_config)

continue_config = {'DynamicCtoP': True,
                   'TargetNfev': 4,
                   'MaxSteps': 2500,  # May need more depending on ds and dsmin
                   'dsmin': dsmin,
                   'dsmax': dsmax,
                   'verbose': 1,
                   'xtol': 1e-6*np.sqrt(Uwxa0.shape[0]),
                   'corrector': 'Ortho',  # Ortho, Pseudo
                   'nsolve_verbose': False,
                   'FracLam': FracLam,
                   'FracLamList': [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop': 0.05  # stop if backtracks to before lam0
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

Uwxa_full_sigmoid = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))

plt.plot(Uwxa_full_sigmoid[:, -1], Uwxa_full_sigmoid[:, -3])
plt.show()

plt.plot(Uwxa_full_sigmoid[:, -1], Uwxa_full_sigmoid[:, -2])
plt.show()

plt.plot(Uwxa_full_iwan[:, -1], Uwxa_full_iwan[:, -3])
plt.plot(Uwxa_full_sigmoid[:, -1], Uwxa_full_sigmoid[:, -3])
plt.title('Natural Frequency with Respect to Amplitude')
plt.ylabel('Natural Frequency (rad/s)')
plt.xlabel('Logarithm of Amplitude')
plt.legend(('Iwan Model', 'NormStiffness Model'))
plt.show()

plt.plot(Uwxa_full_iwan[:, -1], Uwxa_full_iwan[:, -2])
plt.plot(Uwxa_full_sigmoid[:, -1], Uwxa_full_sigmoid[:, -2])
plt.title('Self-Excitation Factor with Respect to Amplitude')
plt.ylabel('Self-Excitation Factor (1/s)')
plt.xlabel('Logarithm of Amplitude')
plt.legend(('Iwan Model', 'NormStiffness Model'))
plt.show()

#%% Hyst loop

# --- extract line and basic sizes ---
line = Uwxa_full_iwan[35, :]
Ndof = Q.shape[1]

omega, la = line[-3], line[-1]
scale = 10.0**la

# --- reshape harmonic block ---
Hblock = line[:-3].reshape(-1, Ndof)        # (Nhc, Ndof)
Nhc    = Hblock.shape[0]
H      = (Nhc - 1) // 2

# --- project directly into nonlinear coordinates ---
Xnl = np.zeros((Nhc, Q.shape[0]))

Xnl[0]    = Q @ Hblock[0]                   # h0 (unscaled)
Xnl[1::2] = scale * (Q @ Hblock[1::2].T).T  # hkc
Xnl[2::2] = scale * (Q @ Hblock[2::2].T).T  # hks

# --- time history (and derivatives) ---
unlt = time_series_deriv(
    Nt=Nt,
    h=h,
    X0=Xnl,
    order=0,
)

unltdot = time_series_deriv(
    Nt=Nt,
    h=h,
    X0=Xnl,
    order=1,
)

cst = np.ones((Nt, Nhc))
unlh0 = np.ones((1)) * np.mean(unlt)
fnl_arc, _, _ = normstiff.local_force_history(unlt, unltdot, 
                                              h, cst, unlh0)

fnl_iwan, _, _ = iwan_force.local_force_history(unlt, unltdot, 
                            h, np.ones((Nt, Nhc)), np.ones((1))*np.mean(unlt))

plt.plot(unlt, fnl_arc)
plt.plot(unlt, fnl_iwan)
plt.show()


#Timing analysis

import time
import numpy as np

N_runs = 1_000

Nt = 1 << 3
cst = np.ones((Nt, Nhc))

unlt = time_series_deriv(
    Nt=Nt,
    h=h,
    X0=Xnl,
    order=0,
)

unltdot = time_series_deriv(
    Nt=Nt,
    h=h,
    X0=Xnl,
    order=1,
)

# --- Time normstiff ---
t0 = time.perf_counter()
for _ in range(N_runs):
    fnl_arc, _, _ = normstiff.local_force_history(
        unlt, unltdot, h, cst, unlh0
    )
t1 = time.perf_counter()

avg_time_normstiff = (t1 - t0) / N_runs

# --- Time iwan_force ---
t0 = time.perf_counter()
for _ in range(N_runs):
    fnl_iwan, _, _ = iwan_force.local_force_history(
        unlt,
        unltdot,
        h,
        np.ones((Nt, Nhc)),
        np.ones((1)) * np.mean(unlt),
    )
t1 = time.perf_counter()

avg_time_iwan = (t1 - t0) / N_runs

# --- Print results ---
print(f"Average time per call (normstiff): {avg_time_normstiff:.6e} s")
print(f"Average time per call (iwan_force): {avg_time_iwan:.6e} s")


