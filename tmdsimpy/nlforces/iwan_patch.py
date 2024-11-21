import numpy as np
from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from ..utils import harmonic as hutils

class IwanPatch(HystereticForce):
    """
    Implementation of the 4-parameter Iwan model for two isotropic
    tangential dimensions and a linear stiffness model for a normal dimension
    
    Parameters
    ----------
    Q : (3, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system 
        to the `Nnl` local nonlinear DOFs.
    T : (N, 3) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
    kt : float
        Tangential stiffness coefficient.
    Fs : float
        Slip force.
    chi : float
        Controls microslip damping slope. Recommended to have `chi > -1`.
        Smaller values of `chi` may not work.
    beta : float, positive
        Controls discontinuity at beginning of macroslip (zero is smooth).
    kn : float
        Normal stiffness coefficient.
    
    """
    
    def __init__(self, Q, T, kt, Fs, chi, beta, kn):
        
        assert Q.shape[0] == 3, "Not tested for multiple patches"
        
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.Fs = Fs*1.0
        self.chi = chi*1.0
        self.beta = beta*1.0
        self.kn = kn
        
        self.phi_max = Fs * (1+beta)/(kt * (beta + (chi + 1)/(chi + 2)))
        self.coefE = (kt * (kt * (beta + (chi + 1)/(chi + 2))/(Fs * (1 + beta)))**(1 + chi)) / ((1 + beta) * (chi + 2))
        
        
        self.init_history()
        
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
        unlth0 : (3,) numpy.ndarray
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
        self.fpsliders = np.zeros((self.Nsliders+1)) # Slider at the delta is not counted in Nsliders
        self.dupduh = np.zeros((hutils.Nhc(h)))
        
        self.dupduh[0] = 1 # Base slider position taken as zeroth harmonic 
        
        self.dfpduh = np.zeros((1,hutils.Nhc(h)))
        self.dfpslidersduh = np.zeros((self.Nsliders+1,hutils.Nhc(h)))
        
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
    
    def instant_force(self, unl, unldot, update_prev=False):
        """
        Calculates local force based on local nonlinear displacements.

        Parameters
        ----------
        unl : (3,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (3,) numpy.ndarray
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
    
        
        fnl = np.zeros_like(unl)
        
        tmp_unl = unl

        unl = unl - self.up
                
        sux = np.sign(np.real(unl[0]))
        ux = sux * unl[0]
        fnl[0] = ((self.kt * ux - self.coefE * ux**(2 + self.chi)) * (ux < self.phi_max) + self.Fs * (ux >= self.phi_max)) * sux
        
        suy = np.sign(np.real(unl[1]))
        
        uy = suy * unl[1]
        
        fnl[1] = ((self.kt * uy - self.coefE * uy**(2 + self.chi)) * (uy < self.phi_max) + self.Fs * (uy >= self.phi_max)) * suy
        
        
        fnl[2] = self.kn * unl[2]
        
        dfnldunl = np.zeros_like(unl)
        
        dfnldunl[0] = (self.kt-self.coefE*(2+self.chi)*ux**(1+self.chi))*(ux<self.phi_max)
        dfnldunl[1] = (self.kt-self.coefE*(2+self.chi)*uy**(1+self.chi))*(uy<self.phi_max)
        dfnldunl[2] = self.kn
        
        #FIX
        
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
        
        fnl = fnl + self.fp
        
        return fnl, dfnldunl