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

# Harmonic Functions for AFT
from ..utils import harmonic as hutils
from ..nlutils import *


class ArcStiffness(HystereticForce):
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
        self.d = (1 + 2/np.pi * np.arctan(self.b*self.s))/self.kt
       
        self.C = -np.log1p(self.b**2*self.s**2)/np.pi/self.b + 2*self.s*np.arctan(self.b*self.s)/np.pi
       
        assert self.Q.shape[0] == 1, 'Not tested for simultaneous Iwan elements.'
       
        self.init_history()
        
    def knorm_model(x, b, s):
        """
        x: array of log10(amplitude)
        b, s: scalars (b>0, s>0)
        returns model values same shape as x
        """
        C = 2.0 / np.pi
        A = np.arctan(b * (np.power(10.0, x) - s))   # arctan(b*(10^x - s))
        denom = 1.0 + C * np.arctan(b * s)           # 1 + 2/pi * arctan(bs)
        return (1.0 - C * A) / denom

    def fit_b_s_knorm(x, y, b0=None, s0=None, bounds=None, verbose=False):
        """
        Fit (b,s) so that knorm_model(x,b,s) approx y in least-squares sense.
        Inputs:
          x: 1D array of log10(nlharmnorm)
          y: 1D array of k_nl / k_t (same length)
          b0, s0: optional initial guesses (scalars)
          bounds: optional ((b_lo,s_lo),(b_hi,s_hi)); if None, use defaults
        Returns dict with keys: b, s, y_fit, res (least_squares result)
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have same shape")
    
        # sensible defaults
        if bounds is None:
            bounds = ([1e-9, 1e-12], [1e9, 1e12])   # enforce b>0, s>=small positive; widen as needed
    
        if s0 is None:
            s0 = 10.0**(np.median(x))   # initial pivot guess in linear amplitude space
        if b0 is None:
            # heuristic: width between 10th and 90th percentiles in linear space
            p10, p90 = np.percentile(x, [10, 90])
            width_lin = max(np.power(10.0, p90) - np.power(10.0, p10), 1e-6)
            b0 = max(4.0 / width_lin, 1e-3)
    
        p0 = np.array([float(b0), float(s0)])
    
        def resid(p):
            b, s = p
            return ArcStiffness.knorm_model(x, b, s) - y
    
        if verbose:
            print("initial guess b0, s0:", p0, "bounds:", bounds)
    
        res = least_squares(resid, p0, bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        b_fit, s_fit = float(res.x[0]), float(res.x[1])
        y_fit = ArcStiffness.knorm_model(x, b_fit, s_fit)
    
        return {"b": b_fit, "s": s_fit, "y_fit": y_fit, "res": res}
    
        
    
    @staticmethod
    def gather_parameters_from_backbone(Q, XlamP_full, M, K, mode_select='nearest'):
        """
        Minimal estimator (assumes denominators nonzero and k_nl finite) that also
        computes nonlinear harmonic norms.
    
        Inputs:
          Q : (Nnl, N)
          XlamP_full : (Ncont, N*Nhc + 3)  (omega at index -3)
          M, K : (N, N)
    
        Returns:
          B              : (Nnl,) local slopes dk/dA at crossing interval (or nan if no crossing)
          S              : (Nnl,) interpolated amplitude (nlharmnorm) where k_nl == kt/2 (or nan)
          k_nl           : (Ncont,) scalar stiffness per continuation line
          kt             : scalar chosen as k_nl[0]
          nlharmnorms    : (Ncont, Nnl) nonlinear harmonic norms from nonlinear_harmonic_norm(...)
        """
        # local (tiny) helper to extract real phi_nl from X row (first-harmonic phasor)
        def _phi_nl_from_row_real(Xrow, N):
            coeff_len = Xrow.size - 3
            Nhc = coeff_len // N
            blocks = [Xrow[i*N:(i+1)*N] for i in range(Nhc)]
            if Nhc >= 3:
                # blocks: [h0, h1c, h1s, h2c, h2s, ...]
                return np.real(blocks[1] + 1j * blocks[2])
            elif Nhc == 2:
                return np.real(blocks[1])
            else:
                return np.real(blocks[0])
    
        # eigenpairs once (generalized problem K phi = lambda M phi)
        try:
            from scipy.linalg import eigh
            evals, evecs = eigh(K, M)
        except Exception:
            # fallback (requires M invertible)
            evals_c, evecs = np.linalg.eig(np.linalg.solve(M, K))
            evals = np.real_if_close(evals_c, tol=1000)
            evecs = np.real_if_close(evecs, tol=1000)
    
        omegas_lin = np.sqrt(np.maximum(evals, 0.0))
    
        Ncont = XlamP_full.shape[0]
        N = M.shape[0]
        Nnl = Q.shape[0]
    
        # precompute Q^T Q
        QTQ = Q.T @ Q
    
        # extract omegas from continuation rows (assumed at -3)
        omegas = XlamP_full[:, -3].astype(float)
    
        k_nl = np.empty(Ncont, dtype=float)
    
        for idx in range(Ncont):
            omega_nl = float(omegas[idx])
    
            # choose linear mode index (nearest frequency)
            if mode_select == 'nearest':
                l = np.argmin(np.abs(omegas_lin - omega_nl))
            else:
                l = 0
    
            omega_l = omegas_lin[l]
            phi_l = evecs[:, l].astype(float)
    
            # reconstruct phi_nl as real first-harmonic phasor
            Xrow = XlamP_full[idx, :]
            phi_nl = _phi_nl_from_row_real(Xrow, N).astype(float)
    
            # normalize to modal mass = 1
            mnorm_nl = phi_nl.T @ (M @ phi_nl)
            if np.abs(mnorm_nl) > 0:
                phi_nl = phi_nl / np.sqrt(mnorm_nl)
            mnorm_l = phi_l.T @ (M @ phi_l)
            if np.abs(mnorm_l) > 0:
                phi_l = phi_l / np.sqrt(mnorm_l)
    
            # residual and scalar k_nl (real arithmetic)
            R = M @ (omega_nl**2 * phi_nl - omega_l**2 * phi_l) - (K @ phi_nl) + (K @ phi_l)
            num = phi_nl.T @ R
            den = float(phi_nl.T @ (QTQ @ phi_nl))   # = || Q phi_nl ||^2
    
            # (per your instruction assume den != 0 and k_nl finite)
            k_nl[idx] = float(num / den)
    
        # scalar reference stiffness
        kt = k_nl[0]
        knl_normalized = k_nl / kt
    
        # compute nonlinear harmonic norms (user-supplied function)
        nlharmnorms = nonlinear_harmonic_norm(XlamP_full, Q)   # shape (Ncont, Nnl)
    
        # Prepare outputs S and B
        S = np.full(Nnl, np.nan, dtype=float)
        B = np.full(Nnl, np.nan, dtype=float)

        # For each nonlinear DOF, find where k_nl crosses target and interpolate S from nlharmnorms
        for nl in range(Nnl):
            nlharmnorm_log = np.log10(nlharmnorms[:, nl].reshape(-1))
            sol = ArcStiffness.fit_b_s_knorm(nlharmnorm_log, knl_normalized)
            
            #Continue this
            
            B[nl] = sol['b']
            S[nl] = sol['s']
            
            print('please do this')
    
        return B, S, kt



        
       
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
    
        fnl = signx * (
            (np.log(self.b**2 * (x - self.s)**2 + 1) / (np.pi * self.b)
             - 2 * (x - self.s) * np.arctan(self.b * (x - self.s)) / np.pi
             + x + self.C) / self.d
        )

        fnl /= (1 + initial_loading)
        
        fnl += self.fp
       
        dfnldunl = (1 - (2 / np.pi) * np.arctan(self.b * (x - self.s))) / self.d

       
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
            
            dupduh = cst[start-1, :]
    
            # Get previous state from critical point
            up = unlt[start-1, :]
            f0 = ft[start-1, :]
            
            self.up = up
            self.fp = f0;
    
            # Current segment data
            segment_cst = cst[start:stop, :]
    
            delta_cst = segment_cst - dupduh
            signx = np.sign(unlt[start:stop] - up)
            x = np.abs(unlt[start:stop] - up) # Prevent log(0) in stiffness calc
               
            # Force calculation
            ft[start:stop] = signx * (
                (np.log(self.b**2 * (x - self.s)**2 + 1) / (np.pi * self.b)
                 - 2 * (x - self.s) * np.arctan(self.b * (x - self.s)) / np.pi
                 + x + self.C) / self.d
            ) + f0
            
            # Key Fix: Use raw harmonic basis (cst) without reversal subtraction
            dfnldunl = (1 - (2 / np.pi) * np.arctan(self.b * (x - self.s))) / self.d

            dfduh_segment = dfnldunl.reshape(-1, 1, 1, 1) * delta_cst.reshape(-1, 1, 1, Nhc)
            dfduh_segment[:, 0, 0, 0] = np.zeros(dfduh_segment.shape[0])
            
            dfduh_segment += dfduh_crit[i, :]
            
            dfduh[start:stop] = dfduh_segment
    
        # Set the DC harmonic to zero because idk 
        #dfduh[:, :, :, 0] = np.zeros((Nt, Ndnl, Ndnl))
        #dfduh -= np.mean(dfduh, axis = 0)
        
        
        # Reset harmonic information i think
        self.init_history_harmonic(unlth0, h)
        return ft, dfduh, dfdudh
        
    
    
    