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
        self.sigma = self.beta/(self.beta + self.gamma)
        
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
       
    def dfnldunl_fun(self, unl, f, unldot):
        df = self.z0 * self.rho * (1 - (self.sigma * np.sign(f) * np.sign(unldot) + (1 - self.sigma)) * np.abs(f)**self.n)
        return df

    def instant_force(self, unl, unldot, update_prev=False):
        # Initial conditions for unl and f
        unl_0 = self.up
        f0 = self.fp
        
        # Define the ODE as a lambda to fix the argument issue
        ode_fun = lambda unl, f: self.dfnldunl_fun(unl, f, unldot)
        
        # Solve the ODE from unl_0 to unl with initial condition f0
        solution = solve_ivp(ode_fun, [unl_0, unl], [f0], dense_output=True)
        
        # Extract the final value of f at unl
        fnl = solution.y[0][-1]
        
        # Compute dfnldunl at the final value of unl
        dfnldunl = self.dfnldunl_fun(unl, fnl, unldot)
        
        # Update previous state if necessary
        if update_prev:
            self.up = unl
            self.fp = fnl
        
        return fnl, dfnldunl
    
    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev = False):
        
        #Number of nonlinear DOFs
        Ndnl = unl.shape[0]
        Nhc = hutils.Nhc(h)
        
        dfduh = np.zeros((Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Ndnl, Ndnl, Nhc))
        
        fnl, dfnldunl = self.instant_force(unl, unldot, update_prev = update_prev)
        
        return fnl, dfnldunl, np.zeros_like(dfnldunl)
        
    

        