#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:55:07 2024

@author: samuelmccaulley
"""
import numpy as np
from scipy.integrate import solve_ivp

from .nonlinear_force import HystereticForce

# Harmonic Functions for AFT
from ..utils import harmonic as hutils
import time

import signal
import warnings


# Define a custom timeout exception
class TimeoutException(Exception):
    pass

# Define the timeout handler
def timeout_handler(signum, frame):
    print("Timeout occurred!")
    # Access local variables in the frame
    local_vars = frame.f_locals
    print("Local variables in the function at timeout:")
    for var, val in local_vars.items():
        print(f"{var}: {val}")
    raise TimeoutError("Function execution exceeded the time limit!")


class BoucWenForce(HystereticForce):
    
    
    def __init__(self, Q, T, A, beta, gamma, n):
        
        self.Q = Q
        self.T = T
        self.A = A
        self.beta = beta
        self.gamma = gamma
        self.n = n
    
        #Derived Parameters
        
        self.z0 = (self.A/(self.beta + self.gamma))**(1/self.n)
        

        self.rho = self.A/self.z0
        
        assert self.rho > 0, "Incorrect Formulation of rho"
        
        self.sigma = self.beta/(self.beta + self.gamma)
        
        assert self.sigma >= 0, "Incorrect formulation of sigma"
        
        self.init_history()
        
    def init_history(self, u0=0, udot0=0):
        """
        Method to initialize history variables for the hysteretic model.
        
        This consists of setting previous displacements, velocities and forces
        to be zero.

        Returns
        -------
        None.

        """
        
        self.up = u0
        self.udotp = udot0
        self.fp = 0
        
        return
    
    
    def init_history_harmonic(self, unlth0, h=np.array([0])):
        
        self.up = unlth0
        self.fp = 0
        self.dupduh = np.zeros((hutils.Nhc(h)))
        self.dupduh[0] = 1 #?
        self.dfpduh = np.zeros((1, 1, hutils.Nhc(h)))
    
    def force(self, X, update_hist = False):
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
       
    def dzetadunl_fun(self, unl, zeta, unldot):
            
        if zeta == 0: return self.rho
        df = self.rho * (1 - (self.sigma * np.sign(zeta) * np.sign(unldot) + (1 - self.sigma)) * np.abs(zeta)**self.n)
        
        return df
    
    def dfnldunl_fun(self, unl, fnl, unldot):
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            print(f"RuntimeWarning captured: {message}")
            print(f"Problematic values in fnl: {fnl}")
            print(f"Problematic exponent: {self.n}")
            raise RuntimeError("Overflow encountered in np.power")

        # Temporarily catch warnings during this computation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            try:
                # Compute the power operation
                df = self.A - (self.beta *np.sign(fnl)*np.sign(unldot) - self.gamma)*np.abs(fnl)**self.n
            except RuntimeWarning as e:
                warning_handler(str(e), RuntimeWarning, __file__, None)
        
        return df

    def instant_force(self, unl, unldot, update_prev=False):
        # Initial conditions for unl and f
        unl_0 = self.up
        f0 = self.fp
        
        # ode_fun = lambda unl, fnl: self.dfnldunl_fun(unl, fnl, unldot)
        # solution = solve_ivp(ode_fun, [unl_0, unl], [f0], dense_output = True)
        # fnl = solution.y[0][-1]
        
        # Define the ODE as a lambda to fix the argument issue
        ode_zeta_fun = lambda unl, zeta: self.dzetadunl_fun(unl, zeta, unldot)
        
        # Solve the ODE from unl_0 to unl with initial condition f0
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        solution_zeta = solve_ivp(ode_zeta_fun, [unl_0, unl], [f0/self.z0], dense_output=True)
        signal.alarm(0)
        # Extract the final value of f at unl
        zeta = solution_zeta.y[0][-1]
        fnl = zeta * self.z0

        
        # Compute dfnldunl at the final value of unl
        dfnldunl = self.dfnldunl_fun(unl, fnl, unldot)
        
        # Update previous state if necessary
        if update_prev:
            self.up = unl
            self.fp = fnl
        
        return fnl, dfnldunl
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False):
        """
        Evaluates the force at instantaneous displacement and velocity
        with harmonic derivatives for the Bouc-Wen model.
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
        dfduh = np.einsum('i,j->ij', dfnldunl, cst).reshape((1, 1, Nhc))
    
        # Derivative wrt velocity harmonics (assumes zero; modify as needed)
        dfdudh = np.zeros_like(dfduh)
    
        # Store results for continuity between calls
        self.dupduh = cst
        self.dfpduh = dfduh
    
        return fnl, dfduh, dfdudh
        
    

        