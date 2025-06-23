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

# Harmonic Functions for AFT
from ..utils import harmonic as hutils

class SigmoidStiffness(HystereticForce):
    
    def __init__(self, Q, T, kt, b, s):
        self.Q = Q
        self.T = T
        self.kt = kt
        self.b = b
        self.s = s
        
        self.init_history()
        self.init_history_harmonic()
    
    @classmethod
    def from_experiment(cls, Q, T, kt, log_amplitude, omega, D):
        """
        

        Parameters
        ----------
        Q : (Nnl, N)
            Array to turn physical DOFs to nonlinear DOFs
        T : (N, Nnl)
            Array to turn nonlinear DOFs to linear DOFs
        log_amplitude : (Nc, )
            Amplitudes tested in nonlinear experiment under base-10 logarithm
        omega : (Nc, )
            Natural frqeuencies in nonlinear experiment, in rad/s or Hz
        D : (Nc, )
            Energy loss per loop in nonlinear experiment, in J

        Returns
        -------
        cls(b, s) : SigmoidStiffness
            SigmoidStiffness force model with nonlinear parameters resulting
            from solution of nonlinear experiment
        """
        return 0    
    
    @staticmethod
    def incomplete_slip_parameters(log_amplitude, omegas, alphas, returnwf = False, verbose = False):
        '''
        Solves the curve fitting methods necessary to find b, s
        for when full arctangent curve is not known
        
        f(logamplitude, p) = p[2] - (p[2] - omega[0])/(2) * (1 - np.special.erf(p[0]*logamplitude - p[1]))
        
        
        '''
        
        f = lambda x, wf, b, s : wf - (wf - omegas[0])/(2) * (1 - erf(b*(x - s)))
        #xdata = log_amplitude
        #ydata = omegas
        #p0 = [omegas[0]/2, 1, 1]
        
        sol = curve_fit(f, log_amplitude, omegas, np.array([omegas[0]/2, 1, log_amplitude[-1]]))
        
        wf, b, s = sol[0]
        
        if verbose:
            print(f"wf:{wf}")
            print(f"b: {b}")
            print(f"s: {s}")
            lamp_pred = np.linspace(-20, 5)
            omegas_prediction = wf - (wf - omegas[0])/2*(1-erf(b*(lamp_pred - s)))
            
            plt.plot(log_amplitude, omegas)
            plt.plot(lamp_pred, omegas_prediction)
            plt.legend(("Data", "Predicted"))
            plt.show()
        
        if returnwf:
            return wf, b, s
        else:
            return b, s
    
    @staticmethod
    def linear_frequency_error(Q, T, log_amplitude, omega, kt, b, s, verbose = False, weights = None):
        
        #Unused, this process is long and does not give good results. Instead, 
        #s can be found without an optimization method, and one constraint variable
        #can be used for optimization (damping error)
        
        kappa = kt/(omega[0] - omega[-1])
        
        test_force = SigmoidStiffness(Q, T, kt, b, s)
        ks = [test_force.secant_stiffness(10**log_amplitude[line]) for line in range(log_amplitude.shape[0])]
        linear_ks = [kappa*(w - omega[-1])/(omega[0] - omega[-1]) for w in omega]
        errors = (np.array(ks)-np.array(linear_ks))**2
        
        if verbose:
            print(f"omega[0]: {omega[0]}")
            print(f"omega[-1]: {omega[-1]}")
            print(f"kappa: {kappa}")
            plt.plot(omega, ks)
            plt.title("ks vs. natural frequency")
            plt.show()
            
            plt.plot(log_amplitude, ks)
            plt.title("ampltiude, ks")
            plt.show()
            
            plt.plot(omega, errors)
            plt.title("Ks errors")
            plt.show()
        
        if weights is not None:
            errors = errors * weights
            
        return np.sum(errors)
    
    @staticmethod
    def damping_error(Q, T, amplitude_final, alpha_final, omega_final, kt, b, s, m, c, verbose = False):
        test_force = SigmoidStiffness(Q, T, kt, b, s)
        x = np.linspace(0, 2*np.pi, 128, endpoint=False)
        Ut = np.atleast_2d(amplitude_final * np.sin(x)).T
        Utdot = np.atleast_2d(amplitude_final * omega_final * np.cos(x)).T
        
        force_history = test_force.local_force_history(Ut, Utdot, np.array([0, 1]), np.ones((128, 3)), np.mean(Ut))
        
        
        upper_half_indices = np.where((x <= np.pi/2) | (x >= 3*np.pi/2))

        
        Ut_upper_half = Ut[upper_half_indices]
        force_upper_half = force_history[0][upper_half_indices]
        nonhysteretic_force = test_force.secant_stiffness(2*amplitude_final)*Ut_upper_half    
        
        difference_force = force_upper_half - nonhysteretic_force
        quadratic_interp = interp1d(np.squeeze(Ut_upper_half), np.squeeze(difference_force), kind="quadratic", fill_value="extrapolate")
        integral, error = quad(quadratic_interp, Ut_upper_half.min(), Ut_upper_half.max())
            
        target_half_hysteresis = 0.5 * np.pi * omega_final * (m * alpha_final - c) * amplitude_final ** 2
        
        if verbose:
            print(f'Half hysteresis: {integral}')
            print(f'Target half hysteresis: {target_half_hysteresis}')
            
            plt.plot(Ut, force_history[0], '.')
            plt.plot(Ut_upper_half, nonhysteretic_force)
            plt.title("Hystereic Loop")
            plt.show()
            
            plt.plot(Ut_upper_half, force_upper_half, '.')
            plt.plot(Ut_upper_half, nonhysteretic_force)
            plt.title("Upper Half")
            plt.show()
        
        
        return np.abs(target_half_hysteresis - integral)
    
    
    @staticmethod
    
    def solve_for_b(Q, T, amplitude_final, alpha_final, omega_final, kt, s, m, c, verbose=False):
        
        cost = lambda b: SigmoidStiffness.damping_error(Q, T, amplitude_final, alpha_final, omega_final, kt, b, s, m, c, verbose)
        opt = minimize(cost, 1, method='nelder-mead')
        print(opt)
        return opt.x
    
    def init_history(self, up = 0, fp = 0):
        """
        Method to initialize history variables for the hysteretic model.
        
        This consists of setting previous displacements and forces
        to be zero.

        Returns
        -------
        None.

        """
        
        self.up = up
        self.fp = fp
        
        return
    
    def init_history_harmonic(self, unlth0=0, h=np.array([0])):
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
        
        fnl, dfnldunl= self.instant_force(unl, 
                                          np.zeros_like(unl),
                                          update_prev=update_hist)
        
        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)
            
        F = self.T @ fnl
        
        dFdX = self.T @ dfnldunl @ self.Q
        
        return F, dFdX
    '''
    
    def secant_stiffness(self, dunl):
        return self.kt / 2 *(1 - \
                             erf(self.b*(np.log10(dunl)-self.s)))
    
    
    def secant_stiffness_derivative(self, dunl):
        # Derivative of secant_stiffness w.r.t. dunl
        ks = self.secant_stiffness(dunl)
        return ks - self.kt * self.b * np.exp(-(self.b*(np.log(dunl)/np.log(10) - self.s))**2) / (np.sqrt(np.pi) * np.log(10))
    
    '''
    
    def secant_stiffness(self, dunl):
        epsilon = 1e-12
        return self.kt / 2 * (1 - erf(self.b * (np.log10(dunl + epsilon) - self.s)))
    
    def secant_stiffness_derivative(self, dunl):
        epsilon = 1e-12
        log_term = np.log10(dunl + epsilon)
        arg = self.b * (log_term - self.s)
        exp_term = np.exp(-arg**2)
        dlog_ddunl = 1 / ((dunl + epsilon) * np.log(10))
        
        return -self.kt * self.b * exp_term * dlog_ddunl / np.sqrt(np.pi)
    
    def instant_force(self, unl, unldot, update_prev=False, initial_loading=False):
        unl0 = self.up
        f0 = self.fp
        dunl = np.abs(unl - unl0) * (1 + initial_loading)
        
        ks = self.secant_stiffness(dunl) #/ (1 + initial_loading)
        dks_ddunl = self.secant_stiffness_derivative(dunl) #/ (1 + initial_loading)
        
        if unldot == 0:
            sign_unldot = np.sign(unl - unl0)
        else:
            sign_unldot = np.sign(unldot)
        
        # Compute dfnldunl correctly
        ddunl_dunl = (1 + initial_loading) #* np.sign(unl - unl0), I'm pretty sure this is a mistake
        #dfnldunl = sign_unldot * (ks + dunl * dks_ddunl) * ddunl_dunl
        dfnldunl = (ks + dunl * dks_ddunl) * ddunl_dunl
        
        
        # ChatGPT said this is better 
        #fnl = sign_unldot * dunl * ks + f0
        fnl = (1 + initial_loading) * (unl - self.up) * ks +f0
        fnl /= (1 + initial_loading)
        # fnl = sign_unldot * (unl - self.up) * ks + f0
        
        if update_prev:
            self.up = unl
            self.fp = fnl
        #Derivative information (i'm pretty sure) is ignored but this will prob be wrong
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
            segment_displacements = unlt[start:stop, :]
            segment_cst = cst[start:stop, :]
    
            # Signed displacement difference (preserve directionality)
            delta_unl = (segment_displacements - up)
            dunl = np.abs(delta_unl) + 1e-15  # Prevent log(0) in stiffness calc
    
            # Stiffness calculations
            ks = self.secant_stiffness(dunl)
            dks_ddunl = self.secant_stiffness_derivative(dunl)
    
            # Force calculation
            ft[start:stop] = delta_unl * ks + f0
    
            # Key Fix: Use raw harmonic basis (cst) without reversal subtraction
            dfnldunl = ks + delta_unl * dks_ddunl
            dfduh_segment = dfnldunl.reshape(-1, 1, 1, 1) * segment_cst.reshape(-1, 1, 1, Nhc)
            
            dfduh[start:stop] = dfduh_segment
    
        # Set the DC harmonic to zero because idk 
        #dfduh[:, :, :, 0] = np.zeros((Nt, Ndnl, Ndnl))
        #dfduh -= np.mean(dfduh, axis = 0)
        
        
        # Reset harmonic information i think
        self.init_history_harmonic(unlth0, h)
        return ft, dfduh, dfdudh
        


    '''


    def local_force_history(self, unlt, unltdot, h, cst, unlth0, 
                            max_repeats=2, atol=1e-10, rtol=1e-10):
        """
        Optimized evaluation of the local forces for steady-state harmonic motion.
        In this version the history (i.e. the previous displacement and force)
        is updated only at reversal points (when the velocity changes sign).
        
        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Time series of local displacements.
        unltdot : (Nt,Nnl) numpy.ndarray
            Time series of local velocities.
        h : 1D numpy.ndarray
            List of harmonic components.
        cst : (Nt,Nhc) numpy.ndarray
            Evaluation of each harmonic component (columns) at each time step.
        unlth0 : (Nnl,) numpy.ndarray
            Zeroth harmonic contributions (initial displacement).
        max_repeats : int, optional
            Maximum number of repeats for convergence (usually 2 for slider models).
        atol : float, optional
            Absolute tolerance on force convergence.
        rtol : float, optional
            Relative tolerance on force convergence.
            
        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear force time series.
        dfduh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces w.r.t. displacement harmonic coefficients.
        dfdudh : (Nt,Nnl,Nnl,Nhc) numpy.ndarray
            Derivative of forces w.r.t. velocity harmonic coefficients.
        """
        # Get dimensions and initialize memory
        Nt, Nnl = unlt.shape
        # Use the number of harmonics from the shape of cst to avoid mismatch
        Nhc = cst.shape[1]
        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Nnl, Nnl, Nhc))
        dfdudh = np.zeros((Nt, Nnl, Nnl, Nhc))
        
        ft_repeats = np.zeros((max_repeats*Nt, Nnl))
        dfduh_repeats = np.zeros((max_repeats*Nt, Nnl, Nnl, Nhc))
        dfdudh_repeats = np.zeros((max_repeats*Nt, Nnl, Nnl, Nhc))
        unlt_repeats = np.tile(unlt, (max_repeats, 1))
        unltdot_repeats = np.tile(unltdot, (max_repeats, 1))
        cst_repeats = np.tile(cst, (max_repeats, 1))
        
        # Identify reversal points based on velocity sign changes.
        # A reversal occurs if (for a given DOF) the product of consecutive velocity
        # values is negative. Here we assume that if any DOF reverses, we update history.
        reversal = np.zeros(max_repeats*Nt, dtype=bool)
        reversal[0] = True  # Always update at the first time step
        # For simplicity, use the first DOF (or you can combine across DOFs as needed)
        #reversal[:-1] |= (unltdot_repeats[:-1, 0] * unltdot_repeats[1:, 0] < 0) #Identifies all of the reversal points.
        
        self.init_history_harmonic(unlth0, h)
        # Figure out when revesal occurs
        
        
        v = unltdot_repeats[:,0]           # shape (max_repeats*Nt,)
        sgn = np.sign(v)                   # gives -1, 0, or +1
        
        # 1) Forward-fill zeros with the last non-zero sign:
        for i in range(1, len(sgn)):
            if sgn[i] == 0:
                sgn[i] = sgn[i-1]
        
        # (Optional) if the very first entries can be zero, you might
        # want to back-fill those with the first non-zero sign.)
        
        # 2) Now detect any sign change in adjacent samples:
        cross = sgn[:-1] != sgn[1:]        # True where a reversal occurred
        reversal[:-1] |= cross

        reversal_indices = np.nonzero(reversal)[0]
        # Ensure the last time step is processed as a reversal.
        if reversal_indices[-1] != max_repeats*Nt - 1:
            reversal_indices = np.append(reversal_indices, max_repeats*Nt - 1)
        
        # Process the reversal (critical) points sequentially.
        for idx in reversal_indices:
            #5/22 tried something: update previous might be false to ensure symmetry across unlth0
            ftmp, dfdtmp, dfdutmp = self.instant_force_harmonic(
                unlt_repeats[idx, :], unltdot_repeats[idx, :], h, cst_repeats[idx, :], update_prev=False
                , initial_loading = idx == reversal_indices[1])
            ft_repeats[idx, :] = ftmp
            dfduh_repeats[idx, :, :, :] = dfdtmp
            dfdudh_repeats[idx, :, :, :] = dfdutmp
    
        # Now, between reversal points the history remains fixed.
        # Process each segment in a vectorized manner.
        for i in range(len(reversal_indices) - 1):
            start = reversal_indices[i] + 1
            stop = reversal_indices[i + 1]
            raw_vel = unltdot_repeats[start:stop, :]

            if stop <= start:
                continue
            
            # Reference state from the most recent reversal:
            ref = reversal_indices[i]
            # Use broadcasting to treat the reference as a row vector.
            ref_unl = unlt_repeats[ref, :][None, :]  # shape (1, Nnl)
            ref_fp = ft_repeats[ref, :][None, :]       # shape (1, Nnl)
            
            # Compute the displacement difference (using absolute difference)
            # Note: (ref_unl == 0) is used to avoid division by zero.
            dunl_seg = np.abs(unlt_repeats[start:stop, :] - ref_unl) #* (1 + (i == 0))
            
            # Compute the secant stiffness for the segment.
            ks_seg = self.secant_stiffness(dunl_seg) #/ (1 + (i == 0))
            
            # Use the sign of the velocity to determine the force direction.
            sign_seg = np.sign(unltdot_repeats[start:stop, :])
            sign_seg = np.sign(raw_vel)  
            # now sign_seg[0] is the old sign, because unltdot_repeats[start] is still the “just before flip” velocity


            # Compute the force vectorized for the entire segment.
            f_seg = ref_fp + sign_seg * dunl_seg * ks_seg
            ft_repeats[start:stop, :] = f_seg
            
            # Compute the derivative with respect to displacement.
            dfnldunl_seg = ks_seg + dunl_seg * self.secant_stiffness_derivative(dunl_seg)      
            
            # For harmonic derivatives, mimic the instant_force_harmonic logic:
            # For each time step in the segment and for each DOF, we set
            # dfduh[t, d, d, :] = dfnldunl_seg[t, d] * cst[t, :]
            for t, global_t in enumerate(range(start, stop)):
                for d in range(Nnl):
                    dfduh_repeats[global_t, d, d, :] = dfnldunl_seg[t, d] * cst_repeats[global_t, :]
                    # dfdudh remains zero (or could be set if needed)
        
        
            ft = ft_repeats[-Nt:, :]
            
            dfduh = dfduh_repeats[-Nt:, :, :, :]
            
        # The 0th harmonic derivative should be zero, but this might affect convergence
        #dfduh[:, 0, 0 ,0] = 0
        #dfduh -= np.mean(dfduh, axis=0, keepdims=True)
        

        return ft, dfduh, dfdudh

'''
    
    
    
    
        
        
        