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
        
    def knorm_model(x, logb, s):
        """
        x: array of log10(amplitude)
        b, s: scalars (b>0, s>0)
        returns model values same shape as x
        """
        C = 2.0 / np.pi
        b = np.power(10.0, logb)
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
            width_lin = max(p90 - p10, 1e-6)
            b0 = max(4.0 / width_lin, 1e-3)
    
        p0 = np.array([float(b0), float(s0)])
    
        def resid(p):
            b, s = p
            return ArcStiffness.knorm_model(x, b, s) - y
    
        if verbose:
            print("initial guess b0, s0:", p0, "bounds:", bounds)
    
        res = least_squares(resid, p0, bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        logb_fit, s_fit = float(res.x[0]), float(res.x[1])
        y_fit = ArcStiffness.knorm_model(x, logb_fit, s_fit)
    
        return {"b": np.power(10.0, logb_fit), "s": s_fit, "y_fit": y_fit, "res": res}
    
    @staticmethod
    def stiffnesses_from_backbone(Uwxa_full, M, K, QN, QT, TN, TT, verbose = False):
        def _phi_nl_from_row_real_out(Xrow, N, normalization=None):
            """
            Extracts and mass-normalizes the first-harmonic mode shape (phi_nl)
            from a single HB solution row.
            """
            coeff_len = Xrow.size - 3
            Nhc = coeff_len // N
            blocks = [Xrow[i*N:(i+1)*N] for i in range(Nhc)]
        
            # Determine which harmonic block to use
            if Nhc >= 3:
                # h1c + j*h1s → real-space mode shape
                phi_non_norm = np.real(blocks[1] + 1j * blocks[2])
            elif Nhc == 2:
                phi_non_norm = np.real(blocks[1])
            else:
                phi_non_norm = np.real(blocks[0])
        
            # Proper normalization (default: mass-normalization)
            if normalization is not None:
                norm_factor = np.sqrt(phi_non_norm.T @ normalization @ phi_non_norm)
                phi_norm = phi_non_norm / norm_factor
        
            return phi_norm
    
        def amplitude_displacement_relations(Uwxa_full, QT):
            """
            Extract unscaled h1c and h1s per physical DOF, convert to nonlinear
            coordinates using QT, compute first-harmonic magnitude, and average
            over continuation points.
        
            Returns:
                avg_mag_nl : shape (n_nl_dof,)
            """
        
            N, M = Uwxa_full.shape
            n_nl, n_phys = QT.shape
        
            # Strip off [omega, xi, logA]
            H = Uwxa_full[:, :M-3]
        
            # Number of harmonic coefficients per physical DOF
            coeffs_per_dof = (M - 3) // n_phys
            if coeffs_per_dof < 3:
                raise ValueError("Need at least h0, h1c, h1s per DOF.")
        
            # Indices inside each DOF block
            idx_h1c = 1
            idx_h1s = 2
        
            # Extract shape (N, n_phys)
            h1c_phys = H[:, idx_h1c::coeffs_per_dof]
            h1s_phys = H[:, idx_h1s::coeffs_per_dof]
        
            # Project into nonlinear coordinates
            # shapes: (N, n_phys) @ (n_phys, n_nl)ᵀ → (N, n_nl)
            u1c_nl = h1c_phys @ QT.T
            u1s_nl = h1s_phys @ QT.T
        
            # Harmonic magnitude per nonlinear DOF, per continuation point
            mag_nl = np.sqrt(u1c_nl**2 + u1s_nl**2)   # (N, n_nl)
        
            # Average over continuation points
            avg_mag_nl = mag_nl.mean(axis=0)          # (n_nl,)
        
            return avg_mag_nl

            
        
        f = lambda x, wf, bw, sw : wf - (wf - omegas[0]) / (2) * (1 - 2/np.pi * np.arctan(bw*(x - sw))) #x is logA
        
        log_amplitude = Uwxa_full[:, -1]
        omegas = Uwxa_full[:, -3]
        
        amplitude_relations = amplitude_displacement_relations(Uwxa_full, QT)
        
        sol = curve_fit(f, 
                        log_amplitude, omegas, 
                        np.array([omegas[-1], 1 / (log_amplitude[-1] - log_amplitude[0]), np.median(log_amplitude)])
                        )
        
        wf, bw, sw = sol[0] #bw, sw are meaningless, all they do are to fit an arctan curve to find wf
        
        wi = Uwxa_full[0, -3]
        
        if verbose:
            plt.plot(log_amplitude, omegas, label = 'Perscribed')
            plt.plot(log_amplitude, wf - (wf - Uwxa_full[0, -3])/2 * (1 - 2/np.pi * np.arctan(bw * (log_amplitude - sw))), label = 'Arctangent Approximated')
            plt.legend()
            plt.show()
            
        phi_i = _phi_nl_from_row_real_out(Uwxa_full[0, :], M.shape[0], M)
        phi_f = _phi_nl_from_row_real_out(Uwxa_full[-1, :], M.shape[0], M)
        Ndof = M.shape[0]
        phi_pre = Uwxa_full[0, Ndof:2*Ndof]
        
        kn = (phi_f.T @ (wf**2 * M - K) @ phi_f) / (phi_f.T @ QN.T @ QN @ phi_f)
        
        K_effect_from_n = TN @ np.diag(kn * np.ones(TN.shape[1])) @ QN
        
        KN = K + K_effect_from_n
        
        
        
        omega_pre = Uwxa_full[0, -3]
        
        kt = (phi_pre.T @ (omega_pre**2 * M - KN) @ phi_pre) / (phi_pre.T @ (QT.T @ QT) @ phi_pre)
        
        #Testing this 
        S = amplitude_relations + sw
        
        return kt, kn, wf, S
    
    @staticmethod
    def gather_parameters_from_backbone(Q, XlamP_full, M, K, mode_select='nearest', verbose = False, QN = np.array([[]]), kn = None):
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
        
        
        # Optionally include normal (unilateral spring) DOFs in the effective stiffness matrix
        if QN.size != 0 and kn is not None:
            # Make sure kn is an array
            kn = np.atleast_1d(kn).astype(float)
        
            # Shape check
            if kn.size == 1:
                # Scalar stiffness for all QN rows
                Kcorr = (kn.item()) * (QN.T @ QN)
            elif kn.size == QN.shape[0]:
                # Elementwise stiffness per nonlinear normal DOF
                Kcorr = QN.T @ np.diag(kn) @ QN
            else:
                raise ValueError(
                    f"Incompatible shapes: QN has {QN.shape[0]} rows, kn has {kn.size} entries."
                )
        
            # Add correction to linear stiffness matrix
            K_eff = K + Kcorr / 2
        
            if verbose:
                print(f"Added unilateral spring correction: ΔK shape = {Kcorr.shape}")
        else:
            K_eff = K
    
    
    
        def _phi_nl_from_row_real(Xrow, N, normalization=M):
            """
            Extracts and mass-normalizes the first-harmonic mode shape (phi_nl)
            from a single HB solution row.
            """
            coeff_len = Xrow.size - 3
            Nhc = coeff_len // N
            blocks = [Xrow[i*N:(i+1)*N] for i in range(Nhc)]
        
            # Determine which harmonic block to use
            if Nhc >= 3:
                # h1c + j*h1s → real-space mode shape
                phi_non_norm = np.real(blocks[1] + 1j * blocks[2])
            elif Nhc == 2:
                phi_non_norm = np.real(blocks[1])
            else:
                phi_non_norm = np.real(blocks[0])
        
            # Proper normalization (default: mass-normalization)
            norm_factor = np.sqrt(phi_non_norm.T @ normalization @ phi_non_norm)
            phi_norm = phi_non_norm / norm_factor
        
            return phi_norm
    
        # eigenpairs once (generalized problem K phi = lambda M phi)
        try:
            from scipy.linalg import eigh
            evals, evecs = eigh(K_eff, M)
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
            phi_nl = _phi_nl_from_row_real(Xrow, N, normalization = M).astype(float)
            
            phi_nl  = phi_nl / np.sqrt(phi_nl.T @ M @ phi_nl)
    
            # residual and scalar k_nl (real arithmetic)
            den = float(phi_nl.T @ (QTQ @ phi_nl))   # = || Q phi_nl ||^2
            num_simpl = float(phi_nl.T @ (omega_nl**2 * M - K_eff) @ phi_nl)            
            

    
            k_nl[idx] = float(num_simpl / den)
            
        
        # scalar reference stiffness
        kt = k_nl[0]
        knl_normalized = k_nl / kt
    
        # compute nonlinear harmonic norms (user-supplied function)
        nlharmnorms = nonlinear_harmonic_norm(XlamP_full, Q)   # shape (Ncont, Nnl)
        
        if verbose:
            plt.plot(XlamP_full[:, -1], k_nl)
            plt.xlabel("Logarithm of Amplitude")
            plt.ylabel("Average Nonlinear Stiffness Effect")
            plt.show()
        
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
            if nl == 0 and verbose:
                y_fit = sol['y_fit']
                plt.plot(nlharmnorm_log, knl_normalized)
                plt.plot(nlharmnorm_log, y_fit)
                plt.xlabel("Logarithm of Harmonic Norm")
                plt.ylabel("Normalized Nonlinear Stiffness")
                plt.legend(('Iwan Calculated', 'Model Approximated'))
                plt.show()
                
            
            print('please do this')
    
        return B, S, kt



    def incomplete_slip_parameters_from_backbone(Qt, XlamP_full, M, K, mode_select='nearest', verbose = False):
        """
        Minimal estimator (assumes denominators nonzero and k_nl finite) that also
        computes nonlinear harmonic norms.
    
        Inputs:
          Qt : (Nnl, N), tangential nonlinear degrees of freedom
          XlamP_full : (Ncont, N*Nhc + 3)  (omega at index -3)
          M, K : (N, N)
    
        Returns:
          B              : (Nnl,) local slopes dk/dA at crossing interval (or nan if no crossing)
          S              : (Nnl,) interpolated amplitude (nlharmnorm) where k_nl == kt/2 (or nan)
          k_nl           : (Ncont,) scalar stiffness per continuation line
          kt             : scalar chosen as k_nl[0]
          nlharmnorms    : (Ncont, Nnl) nonlinear harmonic norms from nonlinear_harmonic_norm(...)
        """
        f = lambda x, wf, bw, sw : wf - (wf - omegas[0])/(2) * (1 - 2/np.pi * np.arctan(bw*(x - sw))) #x is logA
        
        log_amplitude = XlamP_full[:, -1]
        omegas = XlamP_full[:, -3]
        
        sol = curve_fit(f, log_amplitude, omegas, np.array([omegas[0]/2, 1, log_amplitude[-1]]))
        
        wf, bw, sw = sol[0] #bw, sw are meaningless, all they do are to fit an arctan curve to find wf
        
        wi = XlamP_full[0, -3]
        
        def _phi_nl_from_row_real(Xrow, N, normalization=M):
            """
            Extracts and mass-normalizes the first-harmonic mode shape (phi_nl)
            from a single HB solution row.
            """
            coeff_len = Xrow.size - 3
            Nhc = coeff_len // N
            blocks = [Xrow[i*N:(i+1)*N] for i in range(Nhc)]
        
            # Determine which harmonic block to use
            if Nhc >= 3:
                # h1c + j*h1s → real-space mode shape
                phi_non_norm = np.real(blocks[1] + 1j * blocks[2])
            elif Nhc == 2:
                phi_non_norm = np.real(blocks[1])
            else:
                phi_non_norm = np.real(blocks[0])
        
            # Proper normalization (default: mass-normalization)
            norm_factor = np.sqrt(phi_non_norm.T @ normalization @ phi_non_norm)
            phi_norm = phi_non_norm / norm_factor
        
            return phi_norm
        
        phi_i = _phi_nl_from_row_real(XlamP_full[0], M.shape[0])
        
        #Assume phi_i approximately phi_f
        
        kt = (wi**2 - wf**2) / (phi_i.T @ Qt.T @ Qt @ phi_i)
        
        return kt
        
        
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
        self.maxdup = 0 
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
        return (np.log(self.b**2 * (x - self.s)**2 + 1) / (np.pi * self.b)
         - 2 * (x - self.s) * np.arctan(self.b * (x - self.s)) / np.pi
         + x + self.C) / self.d
        
        
        
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
        
        if np.abs(unl - self.unlth0) >= self.maxup: #There will be force scaling relating to initial loading
            fnl = np.sign(x) * 1/2 * self.absolute_force(np.abs(2*(unl - self.unlth0))) #This new loading
            dfnldunl =(1-2/np.pi*np.arctan(self.b*(2 * np.abs(unl - self.unlth0)-self.s)))/ self.d
        else: #This is prior loading where minor loop hysteresis occurs
            fnl = np.sign(x) * self.absolute_force(np.abs(x))
            fnl += fp
            dfnldunl = (1-2/np.pi*np.arctan(self.b*(np.abs(x)-self.s)))/ self.d
        
        
        
        if update_prev:
            # Update History
            self.up = unl
            self.fp = fnl
            # Establish new virtual slider limit
            if np.abs(unl) > self.maxup:
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
    
        its = 0
        rcheck = 0
        acheck = 0
    
        Nt, Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
    
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
    
        self.init_history_harmonic(unlth0, h)
        fp = self.fp
    
        while (its == 0) or (acheck > atol and rcheck > rtol and its < max_repeats):
    
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
    
            its += 1
    
            acheck = np.abs(ft[ti, :] - fp)
            rcheck = np.abs(acheck / (ft[ti, :] + np.finfo(float).eps))
            acheck = 10
            rcheck = 10
            fp = ft[ti, :]
    
        return ft, dfduh, dfdudh
    
    
    def local_force_history(self, unlt, unltdot, h, cst, unlth0,
                            max_repeats=2, atol=1e-10, rtol=1e-10):
    
        Nt, Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)
    
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
    
        b = self.b
        s = self.s
        d = self.d
        C = self.C
    
        inv_d = 1.0 / d
        inv_pi = 1.0 / np.pi
    
        velo_reversal = (unltdot * np.roll(unltdot, -1) <= 0).flatten()
        crit_mask = np.copy(velo_reversal)
        crit_mask[0] = True
    
        unlt_crit = unlt[crit_mask, :]
        unltdot_crit = unltdot[crit_mask, :]
        cst_crit = cst[crit_mask, :]
    
        ft_crit, dfduh_crit, dfdudh_crit = self.local_force_history_crit(
            unlt_crit,
            unltdot_crit,
            h,
            cst_crit,
            unlth0,
            max_repeats=max_repeats,
            atol=atol,
            rtol=rtol
        )
    
        ft[crit_mask] = ft_crit
        dfduh[crit_mask] = dfduh_crit
        dfdudh[crit_mask] = dfdudh_crit
    
        crit_inds = np.where(crit_mask)[0]
        crit_inds = np.append(crit_inds, crit_inds[0])
    
        for i in range(len(crit_inds) - 1):
            start = crit_inds[i] + 1
            stop = crit_inds[i + 1]
            stop = stop + Nt if stop == 0 else stop
    
            if stop <= start:
                continue
    
            if i == 0 and abs(unlt[start]) < np.max(abs(unlt)) and not velo_reversal[0]:
                dupduh = cst[crit_inds[0], :]
                up = unlt[crit_inds[-2], :]
                f0 = ft[crit_inds[-2], :]
            else:
                dupduh = cst[start - 1, :]
                up = unlt[start - 1, :]
                f0 = ft[start - 1, :]
    
            segment_cst = cst[start:stop, :]
            delta_cst = segment_cst - dupduh
    
            dx_raw = unlt[start:stop] - up
            signx = np.sign(dx_raw)
            x = np.abs(dx_raw)
            dx = x - s
    
            ft_segment = f0 + signx * (
                (
                    np.log(b * b * dx * dx + 1.0) * (inv_pi / b)
                    - 2.0 * dx * np.arctan(b * dx) * inv_pi
                    + x + C
                ) * inv_d
            )
    
            dfnldunl = (1.0 - (2.0 * inv_pi) * np.arctan(b * dx)) * inv_d
    
            dfduh[start:stop, 0, 0, :] = (
                dfnldunl[:, 0, None] * delta_cst
                + dfduh[start - 1, 0, 0, :]
            )
    
            ft[start:stop] = ft_segment
    
        dfduh = dfduh - np.mean(dfduh, axis=0)
        
        ft -= np.mean(ft)
        
        return ft, dfduh, dfdudh

        
    
    
    