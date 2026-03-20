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




#%% Helper Functions 


def run_single_wear_study(slow_dt):
    sys.stdout = sys.__stdout__  # reconnect to parent terminal
    print(f"[{os.getpid()}] Starting run with dt={slow_dt}")
    sys.stdout.flush()
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

    prestress = 12845

    vib_sys = VibrationSystem(M, K)

    for patch in range(len(patch_areas)):
        dryfric = ElasticDryFriction3D(Qxyn[3*patch:3*patch+3, :], 
                                       Qxyn[3*patch:3*patch+3, :].T, 
                                       KT_patches0[patch], KN_patches0[patch], params[0])
        
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
    dsmin = 0.01
    dsmax = 0.20
    ds = 0.1
    FracLam = 0.9


    h_max = 1

    h = np.array(range(h_max+1))

    Nhc = hutils.Nhc(h)

    Ndof = vib_sys.M.shape[0]

    Fl = np.zeros(Nhc*Ndof)

    # Static Forces
    Fl[:Ndof] = prestress*Fv # EPMC static force

    # EPMC phase constraint - No cosine component at accel
    Fl[Ndof:2*Ndof] = Rmat[2, :]

    # for nlforce in vib_sys.nonlinear_forces:
    #     print(calculate_E_dot(Uwxa_full[-1, :], h, Nt * 32, Uwxa_full[-1, -3], nlforce))

    slow_times = np.array(range(0,10800+1,slow_dt))
    
    Aend = -3.5
    
    #Start time loop 
    E_patches = np.zeros(Nnl // 3)
    E_patches_history = np.zeros((len(slow_times), Nnl // 3))
    E_patches_history[0, :] = E_patches #E_patches is zero obviously
    for i in range(len(slow_times) - 1):
        print('')
        print(f"======== TIME: {slow_times[i]} ==========")
        print('')
        sys.stdout.flush()
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):

            vib_sys = VibrationSystem(M, K)
            KT_patches, KN_patches = findDryParameters(E_patches, KT_patches0, KN_patches0)
            KT_average = np.mean(KT_patches)
            print(f"KT_AVERAGE: {KT_average:.4e}")
            
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
            
            Edot_patches = np.zeros(Nnl // 3) #Npatches = Nnl // 3
            for patch in list(range(Nnl // 3)):
                Edot = calculate_E_dot(Uwxa_full_worn[-1, :], h, Nt, Uwxa_full_worn[-1, -3], vib_sys.nonlinear_forces[patch])
                Edot_patches[patch] = Edot
            
                    
            dT = slow_times[i+1] - slow_times[i]
            E_patches = E_patches + Edot_patches * dT
            E_patches_history[i+1, :] = E_patches
            E_patches_average = np.mean(E_patches_history, axis=1)
    return slow_times, E_patches_history

    print('done')
    plt.show()
    


def findDryParameters(E_patches, KT_patches0, KN_patches0):
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
    
    #Placeholder KT, KN
    mod0 = 1 #Represents virgin system
    modIt = 5 #Represents added plasticity
    modIn = 6 #Represents added plasticity 
    tau = 200
    KT_mod = lambda E: nlutils.first_order_exponential(mod0, modIt, tau, E)
    KN_mod = lambda E: nlutils.first_order_exponential(mod0, modIn, tau, E)
    
    Npatches = E_patches.shape[0]
    
    KT_patches = np.zeros(Npatches)
    KN_patches = np.zeros(Npatches)
    
    print(f"KT_mod sample: {np.mean(KT_mod(E_patches[0]))}")
    
    for patch in range(Npatches):
        KT_patches[patch] = KT_patches0[patch] * KT_mod(E_patches[patch])
        KN_patches[patch] = KN_patches0[patch] * KN_mod(E_patches[patch])
    
    return KT_patches, KN_patches

def calculate_E_dot(Uwxa, h, Nt, w, nl_force):
    omega = Uwxa[-3]
    xi = Uwxa[-2]
    la = Uwxa[-1]

    X = Uwxa[:-3]
    Nhc = hutils.Nhc(h)
    Ndof = nl_force.Q.shape[1]

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
    
    # if E_cycle / period < 0: 
    #      print("Negative energy")
    #      print(f'Energy in a Cycle: {E_cycle}')
    #      plt.plot(unlt[:, 0], fnl[:, 0])
    #      plt.show()
    #      plt.plot(unlt[:, 1], fnl[:, 1])
    #      plt.show()
    #      plt.plot(unlt[:, 2], fnl[:, 2])
    #      plt.show()
    # else:
    #     print("Hysteresis Could be Activated!")
    Edot = E_cycle / period #J/s
    # if Edot > 1000:
    #     print("Big energy")
    #     print(f'Energy in a Cycle: {E_cycle}')
    #     plt.plot(unlt[:, 0], fnl[:, 0])
    #     plt.show()
    #     plt.plot(unlt[:, 1], fnl[:, 1])
    #     plt.show()
    #     plt.plot(unlt[:, 2], fnl[:, 2])
    #     plt.show()
    #     breakpoint()
    # elif Edot < -1000:
    #     print('Big negative energy')
    #     print(f'Energy in a Cycle: {E_cycle}')
    #     plt.plot(unlt[:, 0], fnl[:, 0])
    #     plt.show()
    #     plt.plot(unlt[:, 1], fnl[:, 1])
    #     plt.show()
    #     plt.plot(unlt[:, 2], fnl[:, 2])
    #     plt.show()
    #     breakpoint()
    return Edot



#%% Wear Study


from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt

slow_dts = [3600, 1800, 900]

def run_and_average(slow_dt):
    slow_times, E_patches_history = run_single_wear_study(slow_dt)
    E_patches_average = np.mean(E_patches_history, axis=1)
    return slow_dt, slow_times, E_patches_average

if __name__ == "__main__":
    nproc = min(len(slow_dts), cpu_count())
    print(f"Running {len(slow_dts)} wear simulations on {nproc} cores...")

    with Pool(processes=nproc) as pool:
        results = list(
            tqdm(pool.imap_unordered(run_and_average, slow_dts), total=len(slow_dts))
        )

    results.sort(key=lambda x: x[0])

    plt.figure(figsize=(8, 5))
    for slow_dt, slow_times, E_patches_average in results:
        plt.plot(slow_times, E_patches_average, label=f"Δt = {slow_dt} s")

    plt.title("Average Patch Wear Energy vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Patch Wear Energy (J)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




