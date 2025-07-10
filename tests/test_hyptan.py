"""
Test and Verification of the hysteretic Sigmoid Stiffness Model


Items to verify
  1. Correct loading force displacement relationship 
      (against conservative implementation)
  2. The correct hysteretic forces (compare with Masing assumptions)
  2a. Consistent dissipation across discretization / range of N to choose?
  3. Correct derivatives of force at a given time instant with respect to 
      those displacements
  4. Correct harmonic derivaitves
  5. Test for a range of values of parameters

"""


import sys
import numpy as np
import unittest
import matplotlib.pyplot as plt

# Python Utilities
sys.path.append('..')
import verification_utils as vutils

sys.path.append('../..')
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.nlforces.tanh_force import HypTangentForce
from tmdsimpy.jax.solvers import NonlinearSolverOMP
from tmdsimpy.vibration_system import VibrationSystem

###############################################################################
#### Function for testing time series                                      ####
###############################################################################

def time_series_forces(Unl, h, Nt, w, tanh_force):
    
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
    
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
    
    unlth0 = Unl[0]
    
    fnl, dfduh, dfdudh = tanh_force.local_force_history(unlt, unltdot, h, cst, unlth0)
    
    plt.plot(unlt, fnl)
    plt.show()
    
    plt.plot(dfduh[:, 0, 0, :])
    plt.show()
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
    
    return fnl, dfduh


###############################################################################
#### Test Class      ####
###############################################################################

class TestHypTan(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        """
        Define tolerances here for all the tests

        Returns
        -------
        None.

        """
        super(TestHypTan, self).__init__(*args, **kwargs)      
        
        
        analytical_tol = 1e-5 # Comparison to analytical backbones (v. this is discrete)
        
        analytical_tol_slip = 1e-4 # Fully slipped state tolerance
        
        atol_grad = 1e-9 # Absolute gradient tolerance
        
        self.tols = (analytical_tol, analytical_tol_slip, atol_grad)

        ##############################
        # Systems
                
        # Simple Mapping to spring displacements
        Q = np.array([[1.0]])
        T = np.array([[1.0]])
        
        kt = 1000
        b = 10**6.4
        s = 10**-6
        
        self.hysteretic_force  = HypTangentForce(Q, T, kt, b, s)
        
    def test_masing_zero_dc_offset(self):
        """
        Test that a harmonic displacement produces a hysteresis loop 
        with zero DC offset in the force.
        """
        # Test setup
        h = np.array([0, 1, 2, 3])       # Only first harmonic (cosine)
        Nt = 128                # Number of time points per cycle
        w = 1.0                 # Frequency (rad/s)
        Amplitude = 8e-7     # Displacement amplitude
    
        # Harmonic coefficients for cosine input (zeroth harmonic = 0)
        Unl = [0, Amplitude, 0, 0, 0, 0, 0]
        # Compute force time series
        fnl, dfduh = time_series_forces(np.atleast_2d(Unl).T, h, Nt, w, self.hysteretic_force)
    
        dFdU = hutils.get_fourier_coeff(h, dfduh)
        print(dFdU)
        # Check zero DC offset in force (mean should be near zero)
        mean_force = np.mean(fnl)
        print(mean_force)
        self.assertAlmostEqual(mean_force, 0.0, 
                               delta=self.tols[0],
                               msg='Force has non-zero DC offset.')
        
    def test_static_stiffness(self):
        M = np.array([[1]])
        K = np.array([[1]])
        
        static_config = {"nsolve_verbose": True,
                         'reform_requency': 2}
        
        static_solver = NonlinearSolverOMP(static_config)
        
        vib_sys = VibrationSystem(M, K)
        vib_sys.add_nl_force(self.hysteretic_force)
        
        Fv = np.zeros(1)
        # Forcing Vector
        Fv[-1] = 1e-6  # Cosine force vector at arbitrary dof

        eigvals_pre, eigvecs_pre = static_solver.eigs(K, M)

        X0 = np.real(eigvecs_pre[:, 0])*1e-6

        pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv)
        R0, dR0dX = pre_fun(X0)
        
        
        Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                                  verbose=True, xtol=1e-16)
        
        Kpre = dRdX
        
        print(Kpre)
        
        
    
if __name__ == '__main__':
    unittest.main()
    