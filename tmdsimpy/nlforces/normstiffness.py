import numpy as np
from .nonlinear_force import HystereticForce
from ..solvers import NonlinearSolver
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import traceback
from scipy.special import erf
from scipy.special import erfc
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import curve_fit

# Harmonic Functions for AFT
from ..utils import harmonic as hutils
from ..nlutils import *


class NormStiffness(HystereticForce):
    def __init__(self, Q, T, kt, b, s):
       
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.b = b
        self.s = s
       
        assert self.b > 0, 'Bump parameter must be positive.'
        assert self.Q.shape[0] == 1, 'Not tested for simultaneous Iwan elements.'
       
        self.init_history()
        
        
    def nl_force_type(self):
        return 1 #Is indeed hysteretic
        

       
    def set_prestress_mu(self):
        """
        Not implemented for Iwan element.
       
        Returns
        -------
        None
       
        Notes
        -----
        Intention is to
        set friction coefficient to zero while saving initial value in a
        different variable. Useful for prestress analysis.
       
        This is non-trivial for the Iwan implementation, so it is not
        yet implemented. One can simply not include the nonlinear force
        to get the same effect with the Iwan element.
       
        """
       
        assert False, 'Prestress mu is not implemented for Iwan Element.'
       
    def init_history(self):
        """
        Method to initialize history variables for the hysteretic model.
       
        This consists of setting previous displacements and forces
        to be zero.

        Returns
        -------
        None.

        """
       
        self.up = 0
        self.fp = 0
        self.maxup = 0
        self.unlth0 = 0
        return
       
    def init_history_harmonic(self, unlth0, h=np.array([0])):
        """
        Initialize history variables for harmonic (AFT) analysis.

        Parameters
        ----------
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            History displacements are initialized at this value.
        h : numpy.ndarray, sorted
            List of harmonics used in subsequent analysis.
            The default is `numpy.array([0])`.

        Returns
        -------
        None.

        """
               
        self.up = unlth0
        self.fp = 0
        self.dupduh = np.zeros((hutils.Nhc(h)))
       
        self.dupduh[0] = 1 # Base slider position taken as zeroth harmonic
       
        self.dfpduh = np.zeros((1,hutils.Nhc(h)))   
        self.unlth0 = unlth0
        self.maxup = 0 
        #self.maxup is an absolute limit where virtual sliders
        #are set with respect to unlth0
        return
   
    def force(self, X, update_hist=False):
        """
        Calculate global nonlinear forces for some global displacement vector.

        Parameters
        ----------
        X : (N,) numpy.ndarray
            Global displacements
        update_hist : bool, optional
            Flag to save displacement and force from the evaluation as history
            variables for subsequent calls to this function.
            The default is False.

        Returns
        -------
        F : (N,) numpy.ndarray
            Global nonlinear force
        dFdX : (N,N) numpy.ndarray
            Derivative of `F` with respect to `X`.
       
        """
       
        unl = self.Q @ X
       
        fnl, dfnldunl = self.instant_force(unl,
                                                    np.zeros_like(unl),
                                                    update_prev=update_hist)
       
        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)
           
        F = self.T @ fnl
       
        dFdX = self.T @ dfnldunl @ self.Q
       
        return F, dFdX
   
    
    def absolute_force(self, x):
        '''
        Inputs:
            self
            x, scaled by 1 or 1/2 depending on if displacement surpasses 
            self.up
        '''
        return self.kt * x / np.power(1 + np.power(x / self.s, self.b), 1 / self.b)
    
    def absolute_stiffness(self, x):
        '''
        Inputs:
            self
            x, scaled by 1 or 1/2 depending on if displacement surpasses 
            self.up
        '''
        
        return self.kt * np.power(np.power(x / self.s, self.b) + 1, -1 - 1 / self.b)
        
        
        
    def instant_force(self, unl, unldot, update_prev=False):
        """
        Calculates local force based on local nonlinear displacements.

        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (Nnl,) numpy.ndarray
            Local nonlinear velocities to evaluate force at.
        update_prev : bool, optional
            Flag to store the results of the evaluation for the start of the
            subsequent step.
            The default is False.

        Returns
        -------
        fnl : float
            Evaluated local nonlinear forces.
        dfnldunl : float
            Derivative of `fnl` with respect to `unl`.
        dfnlsliders_dunl : (Nsliders+1,) numpy.ndarray
            Derivative of `fnl` at each slider with respect to `unl`.

        Notes
        -----
       
        Implementation only allows for a single nonlinear element, thus
        shapes of first two outputs are reduced to scalar.
        """
        fp = self.fp
        up = self.up
        
        x = unl - up
        
        if np.abs(unl - self.unlth0) > self.maxup: #There will be force scaling relating to initial loading
            fnl = np.sign(x) * 1/2 * self.absolute_force(np.abs(2*(unl - self.unlth0))) #This new loading
            dfnldunl = self.absolute_stiffness(np.abs(2 * (unl - self.unlth0)))
        else: #This is prior loading where minor loop hysteresis occurs
            fnl = np.sign(x) * self.absolute_force(np.abs(x))
            fnl += fp
            dfnldunl = self.absolute_stiffness(np.abs(x))
        
        
        
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
            # Establish new virtual slider limit
            if np.abs(unl - self.unlth0) > self.maxup:
                self.maxup = np.abs(unl - self.unlth0)
       
        return fnl, dfnldunl
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False, initial_loading = False):
        """
        Evaluates the force at instantaneous displacement and velocity
        with harmonic derivatives for the Arctangent Stiffness model.
        """
        # Number of harmonics
        Nhc = len(cst)
    
        # Placeholder for harmonic derivatives
        dfduh = np.zeros((1, 1, Nhc))
        dfdudh = np.zeros((1, 1, Nhc))
        
        # Call the base instant_force function to get force and derivative
        fnl, dfnldunl = self.instant_force(unl, unldot, update_prev=update_prev)
        
        # Reshape fnl for output compatibility
        fnl = np.atleast_1d(fnl)
    
        # Derivative wrt displacement harmonics
        # Assumes a linear combination of cst and dfnldunl
        # dfduh = np.einsum('i,j->ij', dfnldunl, cst-self.dupduh).reshape((1, 1, Nhc))
        dfduh = (dfnldunl * (cst-self.dupduh)).reshape((1, 1, Nhc))  + self.dfpduh

        # Derivative wrt velocity harmonics (assumes zero; modify as needed)
        dfdudh = np.zeros_like(dfduh)
    
        # Store results for continuity between calls
        if update_prev:
            self.dupduh = cst
            self.dfpduh = dfduh
            
        return fnl, dfduh, dfdudh

    def local_force_history_crit(self, unlt, unltdot, h, cst, unlth0,
                                 max_repeats=2, atol=1e-10, rtol=1e-10):
        # Match VectorIwan4 strategy: evaluate exactly two passes and
        # keep only the second-pass values in the returned arrays.
        Nt, Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
    
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
    
        self.init_history_harmonic(unlth0, h)

        for _ in range(2):
            for ti in range(Nt):
                fttmp, dfdutmp, dfdudtmp = self.instant_force_harmonic(
                    unlt[ti, :],
                    unltdot[ti, :],
                    h,
                    cst[ti, :],
                    update_prev=True
                )
    
                ft[ti, :] = fttmp
                dfduh[ti, :, :, :] = dfdutmp
                dfdudh[ti, :, :, :] = dfdudtmp
    
        return ft, dfduh, dfdudh
    
    
    def local_force_history(self, unlt, unltdot, h, cst, unlth0,
                            max_repeats=2, atol=1e-10, rtol=1e-10):
    
        Nt, Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
    
        ft = np.zeros_like(unlt)
        dfduh_hist = np.zeros((Nt, Nhc))

        # Local aliases to reduce repeated attribute lookups in the hot loop.
        kt = self.kt
        b = self.b
        inv_b = 1.0 / b
        inv_s = 1.0 / self.s

        # Match VectorIwan4 critical-point detection:
        # evaluate reversal points from displacement slope changes.
        dup = unlt - np.roll(unlt, 1, axis=0)
        dun = np.roll(unlt, -1, axis=0) - unlt
        vector_set = np.equal(np.sign(dup), np.sign(dun))
        vector_set[0] = False
        crit_mask = np.logical_not(vector_set).reshape(-1)
    
        unlt_crit = unlt[crit_mask, :]
        unltdot_crit = unltdot[crit_mask, :]
        cst_crit = cst[crit_mask, :]
    
        # Match VectorIwan4: two internal passes at critical points.
        ft_crit, dfduh_crit, _ = self.local_force_history_crit(
            unlt_crit,
            unltdot_crit,
            h,
            cst_crit,
            unlth0,
            max_repeats=2,
            atol=1e-10,
            rtol=1e-10
        )
    
        ft[crit_mask] = ft_crit
        dfduh_hist[crit_mask, :] = dfduh_crit[:, 0, 0, :]
    
        crit_inds = np.where(crit_mask)[0]
        crit_inds = np.append(crit_inds, crit_inds[0])
    
        for i in range(len(crit_inds) - 1):
            start = crit_inds[i] + 1
            stop = crit_inds[i + 1]
            stop = stop + Nt if stop == 0 else stop
    
            if stop <= start:
                continue
    
            dupduh = cst[start - 1, :]
            up = unlt[start - 1, :]
            f0 = ft[start - 1, :]
    
            segment_cst = cst[start:stop, :]
            delta_cst = segment_cst - dupduh
    
            dx = unlt[start:stop] - up
            x_scaled_pow = np.power(np.abs(dx) * inv_s, b)
            denom = np.power(1.0 + x_scaled_pow, inv_b)

            ft_segment = f0 + kt * dx / denom

            # dfnldunl = kt * (1 + (|x|/s)^b)^(-1 - 1/b)
            dfnldunl = kt / ((1.0 + x_scaled_pow) * denom)

            dfduh_hist[start:stop, :] = (
                dfnldunl[:, 0, None] * delta_cst
                + dfduh_hist[start - 1, :]
            )
    
            ft[start:stop] = ft_segment
    
        dfduh_hist = dfduh_hist - np.mean(dfduh_hist, axis=0)
        
        ft -= np.mean(ft)

        dfduh = dfduh_hist.reshape((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        
        return ft, dfduh, dfdudh

        
    
    
    
