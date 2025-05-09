import numpy as np

class ContinuationLite:
    """
    Simplified continuation method for parameter tracking with robust error handling.
    Implements basic pseudo-arclength continuation without prediction-correction.
    """

    def __init__(self, solver, ds0=0.01, CtoP=None, RPtoC=None, config={}):
        self.solver = solver
        
        # Initialize conditioning vectors
        if CtoP is None:
            self.setCtoPto1 = True
            self.CtoP = None  # Initialized during first solve
        else:
            assert CtoP.ndim == 1, 'CtoP must be 1D array'
            self.setCtoPto1 = False
            self.CtoP = np.abs(CtoP)
            
        # Residual conditioning
        if RPtoC is None:
            self.setRPtoCto1 = True
            self.RPtoC = 1.0
        else:
            self.RPtoC = RPtoC
            self.setRPtoCto1 = False
            
        # Configuration with safe defaults
        default_config = {
            'ds0': ds0,
            'dsmax': max(5*ds0, 1e-6),  # Prevent invalid ranges
            'dsmin': max(ds0/5, 1e-12),
            'MaxSteps': 500,
            'TargetNfev': 20,
            'DynamicCtoP': False,
            'verbose': 100,
            'xtol': None,
            'nsolve_verbose': False,
            'callback': None,
            'MaxIncrease': 1.2,
            'MaxRetries': 5  # New safety parameter
        }
        default_config.update(config)
        self.config = default_config

        # Validate configuration post-merge
        assert self.config['dsmax'] > self.config['dsmin'], \
            "dsmax must be greater than dsmin"
        assert self.config['MaxIncrease'] >= 1.0, \
            "MaxIncrease must be >= 1.0"

    def continuation_lite(self, fun, XlamP0, lam0, lam1, return_grad=False):
        """Main continuation routine with error resilience"""
        assert XlamP0.ndim == 1, "Initial guess must be 1D vector"
        assert not return_grad, "Gradient return not implemented"
        
        silent = self.config['verbose'] < 0
        direction = np.sign(lam1 - lam0)
        if direction == 0:
            raise ValueError("lam0 and lam1 must be different")

        # Initialize solution storage
        max_steps = self.config['MaxSteps']
        XlamP_full = np.zeros((max_steps, XlamP0.size))
        step = 0
        
        # Initialize conditioning vectors
        if self.setCtoPto1:
            self.CtoP = np.ones_like(XlamP0)
        if self.setRPtoCto1:
            self.RPtoC = np.ones(XlamP0.size - 1)  # Match X dimension
            
        # Dynamic conditioning baseline
        if self.config['DynamicCtoP']:
            self.CtoP0 = np.copy(self.CtoP)
            assert self.CtoP0 is not None, "DynamicCtoP requires initialized CtoP"

        # ----- Initial Solve -----
        try:
            fun0 = lambda X, cg=True: _initial_wrapper(fun, X, lam0, cg)
            fun0_cond = self.solver.conditioning_wrapper(
                fun0, self.CtoP[:-1], RPtoC=self.RPtoC
            )
            Xc, _, _, sol = self.solver.nsolve(
                fun0_cond,
                XlamP0[:-1]/self.CtoP[:-1],
                xtol=self.config['xtol'],
                verbose=self.config['nsolve_verbose']
            )
        except Exception as e:
            raise RuntimeError(f"Initial solve failed: {str(e)}") from e
            
        if not sol['success']:
            raise RuntimeError(
                f"Initial point convergence failed: {sol['message']}"
            )

        # Store initial solution
        XlamP_full[0] = np.hstack((Xc * self.CtoP[:-1], lam0))
        step = 1
        ds = self.config['ds0']
        retry_count = 0

        # ----- Continuation Loop -----
        while (step < max_steps 
               and (direction * XlamP_full[step-1, -1] < direction * lam1)):
            
            current_lam = XlamP_full[step-1, -1]
            target_lam = current_lam + direction * ds
            X_prev = XlamP_full[step-1, :-1]

            # Update dynamic conditioning
            if self.config['DynamicCtoP']:
                self.CtoP = np.maximum(
                    np.abs(XlamP_full[step-1]), 
                    self.CtoP0
                )

            # ----- Solve Attempt -----
            try:
                fun_step = lambda X, cg=True: _initial_wrapper(
                    fun, X, target_lam, cg
                )
                X_guess = X_prev / self.CtoP[:-1]
                Xc, _, _, sol = self.solver.nsolve(
                    fun_step, X_guess,
                    xtol=self.config['xtol'],
                    verbose=self.config['nsolve_verbose']
                )
            except Exception as e:
                if not silent:
                    print(f"Step {step} solver error: {str(e)}")
                sol = {'success': False}

            # ----- Handle Convergence -----
            if sol['success']:
                # Store successful solution
                X_physical = Xc * self.CtoP[:-1]
                XlamP_full[step] = np.hstack((X_physical, target_lam))
                
                # Adapt step size
                nfev = max(sol.get('nfev', 1), 1)  # Prevent division by zero
                ds_factor = min(
                    self.config['TargetNfev'] / nfev,
                    self.config['MaxIncrease']
                )
                ds = min(ds * ds_factor, self.config['dsmax'])
                ds = max(ds, self.config['dsmin'])
                
                # Reset retry counter
                retry_count = 0
                step += 1

                # Callback and logging
                if self.config['callback']:
                    self.config['callback'](XlamP_full[step-1], np.nan)
                if self.config['verbose'] and (step % self.config['verbose'] == 0):
                    print(f"Step {step-1}: λ={target_lam:.4e}, ds={ds:.2e}")
            else:
                # Handle failure
                retry_count += 1
                ds = max(ds / 2, self.config['dsmin'])
                if retry_count > self.config['MaxRetries']:
                    if not silent:
                        print(f"Aborting after {self.config['MaxRetries']} retries")
                    break
                if not silent:
                    print(f"Reducing ds to {ds:.2e} after failure")

        # Final cleanup
        XlamP_full = XlamP_full[:step]  # Trim unused storage
        
        if not silent:
            final_lam = XlamP_full[-1, -1] if step > 0 else lam0
            print(f"Completed {step-1} steps. Final λ: {final_lam:.4e}")
            if abs(final_lam - lam1) > 1e-6 * abs(lam1 - lam0):
                print(f"Warning: Did not reach target λ ({lam1:.4e})")

        return XlamP_full


def _initial_wrapper(fun, X, lam, calc_grad=True):
    """Wrapper function for residual evaluations"""
    try:
        if calc_grad:
            R, dRdX, dRdlam = fun(np.hstack((X, lam)))
            return R, dRdX
        else:
            return fun(np.hstack((X, lam)),)
    except Exception as e:
        raise RuntimeError(f"Residual function failed: {str(e)}") from e