#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:54:54 2024

@author: samuelmccaulley
"""

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


import signal #handle timeout errors

# Define a custom timeout exception
class TimeoutException(Exception):
    pass

# Define the timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException



class BoucWenForceNew(HystereticForce):
    
    
    def __init__(self, Q, T, A, alpha, beta, n):
        self.Q = Q
        self.T = T
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.n = n
        
        assert Q.shape[1] == 1, 'Not tested for simultaneous Bouc-Wen Elements'
        
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
       
    def bouc_wen_slope(self, u, f, udot):
        '''
            Function that returns the function used for bouc-wen ODE solving
            
            Not sure why this is written like this in matlab
        '''
        slope = self.A - (self.alpha*np.sign(f*udot) - self.beta)*np.abs(f)**self.n
        
        
        return slope

    def instant_force(self, unl, unldot, update_prev=False):
        # Initial conditions for unl and f
        
        
        unl_curr = unl
        unldot_curr = unldot
        
        unl = unl_curr - self.up
        unldot = unldot_curr - self.udotp
        
        function_g = lambda u, f : self.bouc_wen_slope(u, f, unldot_curr)
        
        
        y = solve_ivp(function_g, [0, unl], np.atleast_1d(0))['y']
        
        if update_prev:
            self.up = unl_curr
            self.udotp = unldot_curr
        
        fnl = y[-1, -1] + self.fp
        
        if update_prev:
            self.fp = fnl
            
        dfnldunl = function_g(unl, fnl)
        
        return fnl, dfnldunl
    
    
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev = False):
        
        #Number of nonlinear DOFs
        Ndnl = unl.shape[0]
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Ndnl, Ndnl, Nhc))
        
        fnl, dfnldunl = self.instant_force(unl, unldot, update_prev = update_prev)
        
        return fnl, dfnldunl, np.zeros_like(dfnldunl)
        
    

        