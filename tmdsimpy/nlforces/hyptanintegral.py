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

# Harmonic Functions for AFT
from ..utils import harmonic as hutils
from ..nlutils import *


class HypTanIntegral(HystereticForce):
    def __init__(self, Q, T, kt, b, s):
       
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.b = b
        self.s = s
       
       
        '''
        Force defined as
        f(x) = multiple * (operation(x) - intercept)
        '''
        self.multiple = self.kt/(1 + np.tanh(self.b*self.s))
       
        self.C = self.multiple / self.b * (-self.b*self.s - np.log(2) + np.log1p(np.exp(2*self.b*self.s)))
       
        assert self.Q.shape[0] == 1, 'Not tested for simultaneous Iwan elements.'
       
        self.init_history()
        
    @staticmethod
    def gather_parameters_from_backbone(Q, XlamP_full, M):
        '''
        Inputs:
            Q: (Nnl, N), nonlinear DOF matrix (each row corresponds to one nonlinear force)
            XlamP_full: (Ncont, N*Nhc + 3), continuation solution array
            M: (N, N), full mass matrix
    
        Outputs:
            B: (Nnl,), estimated rates of nonlinearity for each nonlinear force
            S: (Nnl,), estimated s values (nonlinearity centers for each force)
            k_t: (Nnl,), estimated tangential stiffnesses
        '''
        
        # Step 1: Harmonic norms per nonlinear DOF
        nlharmnorms = nonlinear_harmonic_norm(XlamP_full, Q)  # (Ncont, Nnl)
    
        # Step 2: Extract omega (rad/s) from last column (-3)
        omega = XlamP_full[:, -3]  # (Ncont,)
    
        Nnl = Q.shape[0]
        S = np.zeros(Nnl)
        B = np.zeros(Nnl)
    
        # Step 3: For each nonlinear force, estimate s_i from steepest dω/d||x||
        for i in range(Nnl):
            x_norm = nlharmnorms[:, i]
            domega_dx = np.gradient(omega, x_norm)
            idx_max = np.argmax(np.abs(domega_dx))
            S[i] = x_norm[idx_max]
    
            # Frequency and slope at s_i
            omega_s = omega[idx_max]
            slope = np.abs(domega_dx[idx_max])
    
            # Modal mass for ith nonlinear DOF
            modal_mass = Q[i] @ M @ Q[i].T
    
            # Tangential stiffness k_t_i
            k_t_i = modal_mass * (omega[0]**2 - omega[-1]**2)
    
            # Approximate m_i = 1 + tanh(b*s) -- assume b ~ 1 for first pass
            m_i = 1 + np.tanh(S[i])
    
            # Compute b_i using the formula
            if k_t_i != 0 and slope != 0:
                B[i] = (2 * m_i * modal_mass * omega_s * slope) / k_t_i
            else:
                B[i] = 0.0  # handle division by zero gracefully
    
        # Step 4: Tangential stiffness for all nonlinear DOFs (vectorized)
        modal_masses = np.einsum('ij,jk,ik->i', Q, M, Q)
        delta_omega_sq = omega[0]**2 - omega[-1]**2
        k_t = modal_masses * delta_omega_sq
    
        return B, S, k_t

        
       
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
       
        return
   
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
       
        fnl, dfnldunl, dfnlsliders_dunl = self.instant_force(unl,
                                                    np.zeros_like(unl),
                                                    update_prev=update_hist)
       
        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)
           
        F = self.T @ fnl
       
        dFdX = self.T @ dfnldunl @ self.Q
       
        return F, dFdX
       
   
    
   
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
       
   
    def instant_force(self, unl, unldot, update_prev=False, initial_loading = False):
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
        signx = np.sign(unl - self.up)
        x = np.abs(unl - self.up) * (1 +  initial_loading)
    
        fnl =signx * (self.multiple * self.s - self.multiple/self.b * 
                      (-np.log(2) + np.log1p(np.exp(-2*self.b*(x-self.s)))) + self.C)
        fnl /= (1 + initial_loading)
        
        fnl += self.fp
       
        dfnldunl = self.multiple * (1 - np.tanh(self.b * (x - self.s)))
       
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
       
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
        fnl, dfnldunl = self.instant_force(unl, unldot, update_prev=update_prev, initial_loading=initial_loading)
        
        # Reshape fnl for output compatibility
        fnl = np.atleast_1d(fnl)
    
        # Derivative wrt displacement harmonics
        # Assumes a linear combination of cst and dfnldunl
        # dfduh = np.einsum('i,j->ij', dfnldunl, cst-self.dupduh).reshape((1, 1, Nhc))
        dfduh = np.einsum('i, j->ij', dfnldunl, cst).reshape((1, 1, Nhc))
        
        # Derivative wrt velocity harmonics (assumes zero; modify as needed)
        dfdudh = np.zeros_like(dfduh)
    
        # Store results for continuity between calls
        if update_prev:
            self.dupduh = cst
            self.dfpduh = dfduh
            
        if h[0] == 0:
            dfduh[0, 0, 0] = 0 #Zeroth harmonic should be controlled to zero
    
        return fnl, dfduh, dfdudh

    def local_force_history_crit(self, unlt, unltdot, h, cst, unlth0, \
                                 max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        Modified `local_force_history` to pass out slider states as well
        as other returns.

        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unltdot : (Nt,Nnl) numpy.ndarray
            Local velocities, rows are different time instants and
            columns are different displacement DOFs.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nt,Nhc) numpy.ndarray
            Evaluation of each harmonic component (columns) at a given instant
            in time (row = instant in time). These are without any harmonic
            coefficients, so are just cosine and sine evaluations.
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions to a time series of displacements.
            This is passed to `init_history_harmonic` to initialize model.
        max_repeats : int, optional
            Number of times to repeat the time series to converge the 
            initial state with `local_force_history`. 
            Two is sufficient for slider models. 
            The default is 2.
        atol : float, optional
            Absolute tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.
        rtol : float, optional
            Relative tolerance on force time series convergence to steady-state
            (final state of cycle).
            The default is 1e-10.

        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces. First index is time instants, second index
            is which local nonlinear force DOF.
        dfduh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces with respect to displacement harmonic
            coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        dfdudh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces with respect to velocities harmonic
            coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        fsliders : (Nt, Nsliders+1) numpy.ndarray
            For each instant in time (row), the columns are the force of each
            slider in integrating the Iwan nonlinearity force.
        dfslidersduh : (Nt, Nsliders+1, Nhc) numpy.ndarray
            The derivative of `fsliders` with respect to the harmonic
            coefficients of the displacement `unlt`.

        Notes
        -----

        Function is intended to be called for only a subset of the full times
        of a cycle. These times should just be the velocity reversal points.
        This allows to the calculation of those points more directly
        to improve the efficiency of `local_force_history`.

        Shapes of outputs rely on having `Nnl == 1`.

        """
        its = 0
        
        rcheck = 0
        acheck = 0
        
        # Initialize Memory - Assumption on shape is reasonable for mechanical 
        # systems, but may not be perfect.
        Nt,Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
        
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        
        
        # Only initialize before the loop. History is propogated through 
        # repeated loops over the period
        self.init_history_harmonic(unlth0, h)
        fp = self.fp
        
        while( (its == 0) or (acheck > atol and rcheck > rtol and its < max_repeats) ):
            
            # Time Loop                
            for ti in range(Nt):
                if its == 0 and ti == 0: #Very initial load from unlth0 to first critical point
                    fttmp,dfdutmp,dfdudtmp = \
                        self.instant_force_harmonic(unlt[ti, :], unltdot[ti, :], \
                                                    h, cst[ti, :], update_prev=True, initial_loading = True)
                else:
                    # Update this to immediately save into array without tmps
                    fttmp,dfdutmp,dfdudtmp = \
                        self.instant_force_harmonic(unlt[ti, :], unltdot[ti, :], \
                                                    h, cst[ti, :], update_prev=True)
                
                ft[ti,:] = fttmp
                dfduh[ti,:,:,:] = dfdutmp
                dfdudh[ti,:,:,:] = dfdudtmp


                
            its = its + 1
            
            acheck = np.abs(ft[ti, :] - fp)
            rcheck = np.abs(acheck / (ft[ti, :]+np.finfo(float).eps) )
            
            fp = ft[ti, :]
        
        return ft, dfduh, dfdudh

    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2, 
                            atol=1e-10, rtol=1e-10):
        """
        Evaluate local forces and derivatives for steady-state harmonic motion.
        """
        # Initialize outputs
        Nt, Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
    
        # Identify reversal points (velocity direction changes)
        dup = unlt - np.roll(unlt, 1, axis=0)
        dun = np.roll(unlt, -1, axis=0) - unlt
        vector_set = np.equal(np.sign(dup), np.sign(dun))
        vector_set[0] = False  # First point isn't a reversal
    
        # Extract reversal points and process them
        crit_mask = np.logical_not(vector_set).flatten()
        unlt_crit = unlt[crit_mask, :]
        unltdot_crit = unltdot[crit_mask, :]
        cst_crit = cst[crit_mask, :]
    
        # Process critical points (velocity reversals)
        ft_crit, dfduh_crit, dfdudh_crit = self.local_force_history_crit(
            unlt_crit, unltdot_crit, h, cst_crit, unlth0, 
            max_repeats=max_repeats, atol=atol, rtol=rtol
        )
    
        # Assign critical point results
        ft[crit_mask] = ft_crit
        dfduh[crit_mask] = dfduh_crit
        dfdudh[crit_mask] = dfdudh_crit
    
        # Process segments between reversals
        crit_inds = np.where(crit_mask)[0]
        crit_inds = np.append(crit_inds, crit_inds[0])  # Wrap for periodic boundary
    
        for i in range(len(crit_inds) - 1):
            start = crit_inds[i] + 1
            stop = crit_inds[i + 1]
            stop = stop + Nt if stop == 0 else stop
            
            if stop <= start:
                continue
            
    
            # Get previous state from critical point
            up = unlt[start-1, :]
            f0 = ft[start-1, :]
            
            self.up = up
            self.fp = f0;
    
            # Current segment data
            segment_cst = cst[start:stop, :]
    
            signx = np.sign(unlt[start:stop] - up)
            x = np.abs(unlt[start:stop] - up) # Prevent log(0) in stiffness calc
               
            # Force calculation
            ft[start:stop] = signx * (self.multiple * (x
        - (
            np.abs(self.b*(x-self.s))
            + np.log1p(np.exp(-2*np.abs(self.b*(x-self.s))))
            - np.log(2)
          ) / self.b
      ) + self.C) + f0
            
            # Key Fix: Use raw harmonic basis (cst) without reversal subtraction
            dfnldunl = self.multiple * (1 - np.tanh(self.b * (x - self.s)))

            dfduh_segment = dfnldunl.reshape(-1, 1, 1, 1) * segment_cst.reshape(-1, 1, 1, Nhc)
            dfduh_segment[:, 0, 0, 0] = np.zeros(dfduh_segment.shape[0])
            
            dfduh[start:stop] = dfduh_segment
    
        # Set the DC harmonic to zero because idk 
        #dfduh[:, :, :, 0] = np.zeros((Nt, Ndnl, Ndnl))
        #dfduh -= np.mean(dfduh, axis = 0)
        
        
        # Reset harmonic information i think
        self.init_history_harmonic(unlth0, h)
        return ft, dfduh, dfdudh
        
    
    
    