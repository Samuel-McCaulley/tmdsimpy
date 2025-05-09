
import matplotlib.pyplot as plt
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.jax.solvers import NonlinearSolverOMP
from tmdsimpy.vibration_system import VibrationSystem

import os
from scipy import io as sio
import numpy as np
import time


def sdof_nonlinear_experiment(m, c, k, nlforce, config):
    ###############################################################################
    ####### 1. Load System Matrices                                         #######
    ###############################################################################
    system_fname = './data/brb_iwan4_mesh.mat'
    system_matrices = sio.loadmat(system_fname)
    
    M = np.array([[m]])
    c = 0.01
    K = np.array([[k]])
    Ndof = M.shape[1]
    
    Q = np.array([[1]])
    T = np.array([[1]])
    
    Nnl, Nnodes = Q.shape
    
    
    ###############################################################################
    ####### 2. Establish Vibration System                                   #######
    ###############################################################################
    
    damp_ab = [c, 0.0]
    # Proportional damping arbitrary value
    
    vib_sys = VibrationSystem(M, K, ab=damp_ab)
    
    
    vib_sys.add_nl_force(nlforce)
    
    Astart = -8
    Aend = 1
    
    # Normal - settings for higher accuracy as used in previous papers
    h_max = 3  # harmonics 0, 1, 2, 3
    Nt = 1 << 7  # 2**7 = 128 AFT steps
    
    ds = 0.001
    dsmax = 0.01
    dsmin = 0.01
    # Adjust weighting of amplitude v. other in continuation to hopefully
    # reduce turning around. Higher puts more emphasis on continuation
    # parameter (amplitude)
    FracLam = 0.5
    
    static_solver = NonlinearSolver()
    
    
    eigvals, eigvecs = static_solver.eigs(K, M)
    
    
    ###############################################################################
    ####### 13. EPMC Initial Guess                                          #######
    ###############################################################################
    
    h = np.array(range(h_max+1))
    
    Nhc = hutils.Nhc(h)
    
    Fl = np.zeros(Nhc*Ndof)
    
    # Forcing Vector
    Fl[1] = 1  # Cosine force vector
    
    Uwxa0 = np.zeros(Nhc*Ndof + 3)
    
    # Mode Shape (from linearized system for prediction)
    mode_ind = 0
    Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])
    
    # Linear Frequency (for prediction of low amplitude EPMC)
    Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))
    
    # Initial Damping (low amplitude as prescribed)
    zeta = c  # This is what mass/stiff prop damping should give
    Uwxa0[-2] = 2*Uwxa0[-3]*zeta
    
    
    Uwxa0[-1] = Astart
    
    ###############################################################################
    ####### 15. EPMC Continuation                                           #######
    ###############################################################################
    
    # This block actually executes the full continuation for the EPMC solution.
    
    
    def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                                calc_grad=calc_grad)
    
    
    epmc_config = {'max_steps': 12,  # balance with reform_freq
                   'reform_freq': 2,  # >1 corresponds to BFGS
                   'verbose': True,
                   'xtol': None,  # Just use the one passed from continuation
                   'rtol': 1e-9,
                   'etol': None,
                   'xtol_rel': 1e-6,
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
                       'MaxSteps': 5000,  # May need more depending on ds and dsmin
                       'dsmin': dsmin,
                       'dsmax': dsmax,
                       'verbose': 1,
                       'xtol': 1e-10*np.sqrt(Uwxa0.shape[0]),
                       'corrector': 'Ortho',  # Ortho, Pseudo
                       'nsolve_verbose': True,
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
    
    Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
    
    t1 = time.time()
    
    print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))
    
    freqs = Uwxa_full[:, -3]/2/np.pi
    freq_diffs = freqs - freqs[0]
    amps = Uwxa_full[:, -1]
    
    plt.plot(amps, freqs)
    plt.xlabel("Log Modal Amplitude")
    plt.ylabel("Natural Frequency")
    plt.show()
    
    plt.plot(amps, freq_diffs)
    plt.xlabel("Log Modal Amplitude")
    plt.ylabel("Difference of Natural Frequencies")
    plt.show()
    
    np.save('data/Uwxa_full_iwan.npy', Uwxa_full)
    
    ks = np.zeros((Uwxa_full.shape[0]))
    
    for i in range(Uwxa_full.shape[0]):
        Uwxa = np.atleast_2d(Uwxa_full[i, :]).T
        Ut = hutils.time_series_deriv(Nt, h, Uwxa[:-3], 0)
        Udot = hutils.time_series_deriv(Nt, h, Uwxa[:-3], 1)
        t = np.linspace(0, 1, num=Nt, endpoint=False)
        Ft = np.squeeze(nlforce.local_force_history(
            Ut, Udot, h, np.ones((Nt, Nhc)), Uwxa[0])[0])
        ks[i] = (Ft[np.argmax(Ut)] - Ut[np.argmin(Ut)]) /\
            (Ut[np.argmax(Ut)] - Ut[np.argmin(Ut)])
    
    ks_diffs = ks - ks[0]
    
    
    plt.plot(freq_diffs, ks_diffs)
    plt.xlabel("Difference of Natural Frequency")
    plt.ylabel("Difference of Secant Stiffness")
    plt.title(f"m = {m}, k = {k}")
    plt.show()
    
    return (freq_diffs, ks_diffs, Uwxa_full[-1, -1] >= Aend)

def sdof_uwxa_full(m, c, k, nlforce, config):
    ###############################################################################
    ####### 1. Load System Matrices                                         #######
    ###############################################################################
    
    M = np.array([[m]])
    c = 0.01
    K = np.array([[k]])
    Ndof = M.shape[1]
    
    Q = np.array([[1]])
    T = np.array([[1]])
    
    Nnl, Nnodes = Q.shape
    
    
    ###############################################################################
    ####### 2. Establish Vibration System                                   #######
    ###############################################################################
    
    damp_ab = [c, 0.0]
    # Proportional damping arbitrary value
    
    vib_sys = VibrationSystem(M, K, ab=damp_ab)
    
    
    vib_sys.add_nl_force(nlforce)
    
    try: Astart = config['Astart']
    except: Astart = -8
    try: Aend = config['Aend']
    except: Aend = -2
    
    # Normal - settings for higher accuracy as used in previous papers
    h_max = 3  # harmonics 0, 1, 2, 3
    Nt = 1 << 7  # 2**7 = 128 AFT steps
    
    ds = 0.001
    dsmax = 0.01
    dsmin = 0.01
    # Adjust weighting of amplitude v. other in continuation to hopefully
    # reduce turning around. Higher puts more emphasis on continuation
    # parameter (amplitude)
    FracLam = 0.5
    
    static_solver = NonlinearSolverOMP()
    
    
    eigvals, eigvecs = static_solver.eigs(K, M)
    
    
    ###############################################################################
    ####### 13. EPMC Initial Guess                                          #######
    ###############################################################################
    
    h = np.array(range(h_max+1))
    
    Nhc = hutils.Nhc(h)
    
    Fl = np.zeros(Nhc*Ndof)
    
    # Forcing Vector
    Fl[1] = 1  # Cosine force vector
    
    Uwxa0 = np.zeros(Nhc*Ndof + 3)
    
    # Mode Shape (from linearized system for prediction)
    mode_ind = 0
    Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])
    
    # Linear Frequency (for prediction of low amplitude EPMC)
    Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))
    
    # Initial Damping (low amplitude as prescribed)
    zeta = c  # This is what mass/stiff prop damping should give
    Uwxa0[-2] = 2*Uwxa0[-3]*zeta
    
    
    Uwxa0[-1] = Astart
    
    ###############################################################################
    ####### 15. EPMC Continuation                                           #######
    ###############################################################################
    
    # This block actually executes the full continuation for the EPMC solution.
    
    
    def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                                calc_grad=calc_grad)
    
    
    epmc_config = {'max_steps': 12,  # balance with reform_freq
                   'reform_freq': 1,  # >1 corresponds to BFGS
                   'verbose': True,
                   'xtol': None,  # Just use the one passed from continuation
                   'rtol': 1e-9,
                   'etol': None,
                   'xtol_rel': 1e-6,
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
                       'MaxSteps': 5000,  # May need more depending on ds and dsmin
                       'dsmin': dsmin,
                       'dsmax': dsmax,
                       'verbose': 1,
                       'xtol': 1e-6*np.sqrt(Uwxa0.shape[0]),
                       'corrector': 'Ortho',  # Ortho, Pseudo
                       'nsolve_verbose': True,
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
    
    Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
    
    t1 = time.time()
    
    print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))
    
    return Uwxa_full