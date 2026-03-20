# Functions for vectorizing a Iwan (4-par) model AFT calculation
import matplotlib.pyplot as plt
import numpy as np
# from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from ..utils import harmonic as hutils

from .nonlinear_force import HystereticForce 


class ArcStiffnessRewrite(HystereticForce):
    """
    4-Parameter Iwan Element Nonlinearity with vectorized force calculations.
    
    
    Im trying to get this changed to match Iwan4 in the fact that it works

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
    T : (N, Nnl) numpy.ndarray
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
    Nsliders : int, optional
        Number of discrete sliders for the Iwan element.
        Note that this does not include 1 additional slider for the
        delta function at phimax.
        Default is 100 (commonly used in literature).
    alphasliders : float, optional
        Determines the non-uniform discretization (see [1]_).
        For midpoint rule, using anything other than 1.0 has
        significantly higher error.
        The default is 1.0.

    See Also
    --------
    Iwan4Force :
        Standard implementation of the Iwan element, generally a slower
        implementation than the present class.

    Notes
    -----
    This class exploits the fact that only reversal points need to be
    calculated to reach steady-state.
    After that, all intermediate times can be calculated in parallel
    (vectorized here), to be faster.
    This does not change the results.

    This implementation is only tested for `Nnl == 1`.

    `local_force_history` implementation is the only difference relative to
    `Iwan4Force` for standard functions. This also adds a new function
    to help in calculations of `local_force_history_crit`, but that function
    should not be needed for must public calls.

    Iwan nonlinearity is based on [1]_.

    References
    ----------
    .. [1]
       Segalman, D.J., 2005. A Four-Parameter Iwan Model for Lap-Type
       Joints. J. Appl. Mech 72, 752–760.

    """
    
    def __init__(self, Q, T, kt, Fs, chi, beta, Nsliders=100, alphasliders=1.0):
        
        self.Q = Q
        self.T = T
        self.kt = kt*1.0
        self.Fs = Fs*1.0
        self.chi = chi*1.0
        self.beta = beta*1.0
        self.Nsliders = Nsliders
        
        self.phimax = self.Fs * (1 + self.beta) / self.kt \
                        / (self.beta + (self.chi + 1)/(self.chi+2))
        
        self.R = self.Fs*(self.chi + 1) / self.phimax**(self.chi+2)\
                        / (self.beta + (self.chi + 1)/(self.chi+2))
                        
        self.S = self.Fs*self.beta / self.phimax\
                        / (self.beta + (self.chi + 1)/(self.chi+2))
        
        # self.S = self.Fs / self.phimax *(self.beta / (self.beta + (self.chi+1)/(self.chi+2)))
                        
        if(alphasliders > 1):
            deltaphi1 = self.phimax * (alphasliders - 1) / (alphasliders**(Nsliders) - 1)
        else:
            deltaphi1 = self.phimax / Nsliders
        
        delta_phis = deltaphi1 * alphasliders**np.array(range(Nsliders))
        
        self.phisliders = np.concatenate( (np.cumsum(delta_phis) - delta_phis*0.5,\
                                          np.array([self.phimax])) )
        
        # Segalman, 2005, eqn 25
        R = self.Fs*(self.chi + 1) / (self.phimax**(self.chi+2) \
                                     *(self.beta + (self.chi + 1)/(self.chi + 2)))
        
        # Segalman, 2005, eqn 26
        S = self.Fs*self.beta / (self.phimax \
                                   *(self.beta + (self.chi + 1)/(self.chi + 2)))
        
        # These absorb kt
        self.sliderweights = np.concatenate( \
                             (R * self.phisliders[:-1]**(self.chi)*delta_phis, \
                              np.atleast_1d(S)) )
        
        assert self.Q.shape[0] == 1, 'Not tested for simultaneous Iwan elements.'
        
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
        self.fpsliders = np.zeros((self.Nsliders+1)) # Slider at the delta is not counted in Nsliders
        
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
        def _is_jax_array(x):
            return hasattr(x, '__module__') and 'jax' in x.__module__
        
        # Stuck Force
        fnlsliders = unl - self.up + self.fpsliders

        # Mask of stuck sliders == places with unit derivative
        dfnlsliders_dunl = np.less_equal(np.abs(fnlsliders), self.phisliders)
        
        if _is_jax_array(fnlsliders):
            import jax.numpy as jnp
            idxs = jnp.logical_not(dfnlsliders_dunl)
            fnlsliders = fnlsliders.at[idxs].set(
                self.phisliders[idxs] * jnp.sign(fnlsliders[idxs])
            )
        else:
            idxs = np.logical_not(dfnlsliders_dunl)
            fnlsliders[idxs] = self.phisliders[idxs] * np.sign(fnlsliders[idxs])

        # # Additional derivative information does not need to be output:
        # dfnlsliders_dup = -dfnlsliders_dunl
        # dfnlsliders_dfp = dfnlsliders_dunl
        
        # Integration
        fnl = fnlsliders @ self.sliderweights
        dfnldunl = dfnlsliders_dunl @ self.sliderweights
        
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
            self.fpsliders = fnlsliders
        
        return fnl, dfnldunl, dfnlsliders_dunl
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False):
        """
        Evaluates the force at a instantaneous set of displacement and velocity
        along with harmonic derivatives.
        
        Parameters
        ----------
        unl : (Nnl,) numpy.ndarray
            Local nonlinear displacements to evaluate force at.
        unldot : (Nnl,) numpy.ndarray
            Local nonlinear velocities to evaluate force at.
        h : 1D numpy.ndarray, sorted
            List of harmonics used in subsequent analysis. Corresponds
            to `Nhc` harmonic components.
        cst : (Nhc,) numpy.ndarray
            Evaluation of harmonics without coefficients at the given instant 
            in time. 
            If zeroth harmonic is included, the first entry is 1.0. 
            Beyond that, it is cosine and then sine at the appropriate harmonic
            for the given instant in time then the next harmonic etc.
        update_prev : bool, optional
            Flag to store the results of the evaluation for the start of the
            subsequent step. 
            The default is False.
        
        Returns
        -------
        fnl : (1,) numpy.ndarray
            Local nonlinear forces
        dfduh : (1, 1, Nhc) numpy.ndarray
            Derivative of `fnl` with respect to displacement harmonic
            coefficients.
        dfdudh : (1, 1, Nhc) numpy.ndarray
            Derivative of `fnl` with respect to velocities harmonic
            coefficients.

        Notes
        -----
        
        Starts calculation based on `init_history_harmonic`.
        
        Only implemented for a single nonlinear element or `Nnl == 1`.
        
        """
        
        # Number of nonlinear DOFs
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Nhc))
        # dfdudh = np.zeros((Nhc))
        # dfslidersduh = np.zeros((Nhc))
        
        fnl, dfnldunl, dfnlsliders_dunl = self.instant_force(unl, unldot, update_prev=update_prev)
        
        fnl = np.atleast_1d(fnl)
        
        dfnlsliders_duh = np.einsum('i,j->ij', dfnlsliders_dunl, cst-self.dupduh) \
                + dfnlsliders_dunl.reshape(-1,1)*self.dfpslidersduh # this line is dfnlsliders_dfslidersp*...
        
        dfduh = np.einsum('ij,i->j', dfnlsliders_duh, self.sliderweights) 
        
        dfduh = dfduh.reshape((1,1,-1))
        dfdudh = np.zeros_like(dfduh)
        
        # Save derivatives into history for next call. 
        self.dupduh = cst
        self.dfpduh = dfduh
        self.dfpslidersduh = dfnlsliders_duh 
                
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
        
        # Slider state storage
        fsliders = np.zeros((Nt, self.Nsliders+1))
        dfslidersduh = np.zeros((Nt,self.Nsliders+1,hutils.Nhc(h)))
        
        # Only initialize before the loop. History is propogated through 
        # repeated loops over the period
        self.init_history_harmonic(unlth0, h)
        fp = self.fp
        
        while( (its == 0) or (acheck > atol and rcheck > rtol and its < max_repeats) ):
            
            # Time Loop                
            for ti in range(Nt):
                # Update this to immediately save into array without tmps
                fttmp,dfdutmp,dfdudtmp = \
                    self.instant_force_harmonic(unlt[ti, :], unltdot[ti, :], \
                                                h, cst[ti, :], update_prev=True)
                
                ft[ti,:] = fttmp
                dfduh[ti,:,:,:] = dfdutmp
                dfdudh[ti,:,:,:] = dfdudtmp
                
                fsliders[ti, :] = self.fpsliders
                dfslidersduh[ti, :, :] = self.dfpslidersduh

                
            its = its + 1
            
            acheck = np.abs(ft[ti, :] - fp)
            rcheck = np.abs(acheck / (ft[ti, :]+np.finfo(float).eps) )
            
            fp = ft[ti, :]
        
        return ft, dfduh, dfdudh, fsliders, dfslidersduh
    
        
    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2, \
                            atol=1e-10, rtol=1e-10):
        """
        Evaluate the local forces for steady-state harmonic motion used in AFT.
        
        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unldot : (Nt,Nnl) numpy.ndarray
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
            This is included for compatibility, but is ignored.
            Two repeats of the hysteresis loop are used by default
            to ensure convergence since this is a slider model that
            converges with two repeats.
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
            Derivative of forces with respect to velocities harmonic coefficients.
            First two indices correspond to `ft`. Third index corresponds to
            which local nonlinear displacement. 
            Fourth index corresponds to which of the `Nhc` harmonic 
            components.
        
        Notes
        -----
        
        Convergence criteria is atol or rtol passes. To require a choice, pass 
        in -1 for the other. Convergence should be exact within two cycles
        since this is a slider based model.
        
        This function is reimplemented from `Iwan4Force` with the more
        efficient vectorized algorithm.

        """
        
        # Initialize output memory - Assumption on shape is reasonable for mechanical 
        # systems, but may not be perfect. Use Q and T to get more exact shapes.
        Nt,Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
        
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        
        # Identify reversal points 
        dup = unlt - np.roll(unlt, 1, axis=0) # du to current
        dun = np.roll(unlt, -1, axis=0) - unlt # du to next
        
        vector_set = np.equal(np.sign(dup), np.sign(dun))
        vector_set[0] = False # This makes it much easier to write the loop below and is assumed.
        
        # Critical points that must be evaluated serially
        unlt_crit = unlt[np.logical_not(vector_set).reshape(-1), :]
        unltdot_crit = unltdot[np.logical_not(vector_set).reshape(-1), :]
        cst_crit = cst[np.logical_not(vector_set).reshape(-1), :]
        
        
        ft_crit, dfduh_crit, dfdudh_crit, fsliders_crit, dfslidersduh_crit\
                        = self.local_force_history_crit(unlt_crit, unltdot_crit, h, \
                                                   cst_crit, unlth0, max_repeats=2, \
                                                   atol=1e-10, rtol=1e-10)
        
        # If one rewrote the Iwan4Force class, it may be faster to recalculate
        # this here instead of doing the weight integration in the loop. However,
        # it is not worth that rewrite now.
        ft[np.logical_not(vector_set).reshape(-1), :] = ft_crit
        
        dfduh[np.logical_not(vector_set).reshape(-1), :] = dfduh_crit
        
        # No velocity dependence for Iwan
        # dfdudh[np.logical_not(vector_set).reshape(-1), :] = dfdudh_crit
        
        crit_inds = np.asarray(np.logical_not(vector_set)).nonzero()[0]
        crit_inds = np.append(crit_inds, crit_inds[0]) # Wrap around without logic in for loop below
        
        # EVERYTHING BELOW HERE IS TRIVIALLY PARALLELIZABLE
        
        # Loop over the set of all crit points and evaluate their subsequent history points.
        # Alternative try doing something fancy with creating an index array, 
        # but that's just as likely to either add a bunch of memory or mess up vectorization.
        for i in range( len(crit_inds)-1 ):
            start = crit_inds[i]+1
            stop  = crit_inds[i+1] # want to end on the previous index (i.e., this minus 1)
            
            stop = stop + Nt*(stop == 0) # Wrap at end
            
            if(stop > start): # Skip case of stop == start
                
                # Apply standard Jenkins from the critical point to the current point 
                # for the full vector_set at once.
        
                # Previous States for readability
                up = unlt[start-1, :]
                fpsliders = fsliders_crit[i, :]
                
                dupduh = cst[start-1, :]
                dfpslidersduh = dfslidersduh_crit[i, :, :]
                
                # Stuck Force
                # Ntimes x Nsliders where Ntimes = stop-start-1
                fnlsliders = (unlt[start:stop, :] - up) + fpsliders.reshape(1,-1)
                
                # Mask of stuck sliders == places with unit derivative
                dfnlsliders_dunl = np.less_equal(np.abs(fnlsliders), \
                                                 self.phisliders.reshape(1,-1))

                # Slipped Force
                # This line does some unnecessary multiplication for the False 
                # case
                fnlsliders = np.where(dfnlsliders_dunl, fnlsliders, \
                                      self.phisliders.reshape(1,-1)*np.sign(fnlsliders))
                
                ft[start:stop, :] = (fnlsliders @ self.sliderweights).reshape(-1,1)
                
                # Derivative of Force Calculation
                delta_cst = cst[start:stop, :] - dupduh

                # Second line is dfnlsliders_dfslidersp*...
                dfnlsliders_duh = np.einsum('ti,tj->tij', dfnlsliders_dunl, \
                                            delta_cst) \
                        + np.einsum('ti,ij->tij', dfnlsliders_dunl, dfpslidersduh)
                
                dfduh[start:stop, 0, 0, :] = np.einsum('tij,i->tj', \
                                                       dfnlsliders_duh, \
                                                       self.sliderweights) 
                
        
        return ft, dfduh, dfdudh


