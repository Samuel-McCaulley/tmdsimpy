#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bouc-Wen hysteretic nonlinearity.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .nonlinear_force import HystereticForce
from ..utils import harmonic as hutils


class BoucWenForce(HystereticForce):
    def __init__(
        self,
        Q,
        T,
        A,
        beta,
        gamma,
        n,
        integration_method="rk4",
        integration_substeps=1,
    ):
        self.Q = Q
        self.T = T
        self.A = A
        self.beta = beta
        self.gamma = gamma
        self.n = n

        self.z0 = (self.A / (self.beta + self.gamma)) ** (1 / self.n)
        self.rho = self.A / self.z0
        self.sigma = self.beta / (self.beta + self.gamma)

        assert self.rho > 0, "Incorrect Formulation of rho"
        assert self.sigma >= 0, "Incorrect formulation of sigma"
        assert integration_substeps >= 1, "integration_substeps must be >= 1"

        self.integration_method = integration_method
        self.integration_substeps = int(integration_substeps)

        self.init_history()

    def init_history(self, u0=0, udot0=0):
        self.up = float(np.atleast_1d(u0)[0])
        self.udotp = float(np.atleast_1d(udot0)[0])
        self.fp = 0.0

    def init_history_harmonic(self, unlth0, h=np.array([0])):
        self.up = float(np.atleast_1d(unlth0)[0])
        self.fp = 0.0
        self.dupduh = np.zeros((hutils.Nhc(h)))
        self.dupduh[0] = 1.0
        self.dfpduh = np.zeros((1, 1, hutils.Nhc(h)))

    def force(self, X, update_hist=False):
        unl = self.Q @ X
        fnl, dfnldunl = self.instant_force(
            unl,
            np.zeros_like(unl),
            update_prev=update_hist,
        )

        fnl = np.atleast_1d(fnl)
        dfnldunl = np.atleast_2d(dfnldunl)

        F = self.T @ fnl
        dFdX = self.T @ dfnldunl @ self.Q

        return F, dFdX

    def _rhs_zeta(self, zeta, sign_udot):
        if zeta == 0.0:
            return self.rho

        coeff = self.sigma * np.sign(zeta) * sign_udot + (1.0 - self.sigma)
        return self.rho * (1.0 - coeff * np.abs(zeta) ** self.n)

    def _advance_zeta(self, zeta0, du, sign_udot):
        if du == 0.0:
            return zeta0

        if self.integration_method == "solve_ivp":
            ode = lambda _u, z: self._rhs_zeta(z[0], sign_udot)
            sol = solve_ivp(ode, [0.0, du], [zeta0], dense_output=False)
            return float(sol.y[0, -1])

        if self.integration_method == "rk4":
            z = zeta0
            h = du / self.integration_substeps
            for _ in range(self.integration_substeps):
                k1 = self._rhs_zeta(z, sign_udot)
                k2 = self._rhs_zeta(z + 0.5 * h * k1, sign_udot)
                k3 = self._rhs_zeta(z + 0.5 * h * k2, sign_udot)
                k4 = self._rhs_zeta(z + h * k3, sign_udot)
                z += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return z

        raise ValueError(
            "Unsupported integration_method. Use 'rk4' or 'solve_ivp'."
        )

    def dfnldunl_fun(self, unl, fnl, unldot):
        fnl = float(np.atleast_1d(fnl)[0])
        unldot = float(np.atleast_1d(unldot)[0])

        sign_term = np.sign(fnl) * np.sign(unldot)
        df = self.A - (self.beta * sign_term - self.gamma) * np.abs(fnl) ** self.n
        return np.array([df])

    def instant_force(self, unl, unldot, update_prev=False):
        unl = float(np.atleast_1d(unl)[0])
        unldot = float(np.atleast_1d(unldot)[0])

        du = unl - self.up
        sign_udot = np.sign(unldot)

        zeta0 = self.fp / self.z0
        zeta = self._advance_zeta(zeta0, du, sign_udot)
        fnl = zeta * self.z0

        dfnldunl = self.dfnldunl_fun(unl, fnl, unldot)

        if update_prev:
            self.up = unl
            self.fp = fnl
            self.udotp = unldot

        return np.array([fnl]), dfnldunl

    def instant_force_harmonic(self, unl, unldot, h, cst, update_prev=False):
        Nhc = len(cst)

        fnl, dfnldunl = self.instant_force(unl, unldot, update_prev=update_prev)

        dfduh = np.zeros((1, 1, Nhc))
        dfduh[0, 0, :] = dfnldunl[0] * cst
        dfdudh = np.zeros_like(dfduh)

        self.dupduh = cst
        self.dfpduh = dfduh

        return fnl, dfduh, dfdudh

    def local_force_history(
        self,
        unlt,
        unltdot,
        h,
        cst,
        unlth0,
        max_repeats=2,
        atol=1e-10,
        rtol=1e-10,
    ):
        Nt, Ndnl = unlt.shape
        Nhc = hutils.Nhc(h)

        ft = np.zeros_like(unlt)
        dfduh = np.zeros((Nt, Ndnl, Ndnl, Nhc))
        dfdudh = np.zeros((Nt, Ndnl, Ndnl, Nhc))

        self.init_history_harmonic(unlth0, h)
        fp = self.fp

        its = 0
        acheck = 0.0
        rcheck = 0.0

        while (its == 0) or (acheck > atol and rcheck > rtol and its < max_repeats):
            for ti in range(Nt):
                fnl, dfnldunl = self.instant_force(
                    unlt[ti, 0], unltdot[ti, 0], update_prev=True
                )

                ft[ti, 0] = fnl[0]
                dfduh[ti, 0, 0, :] = dfnldunl[0] * cst[ti, :]

            its += 1
            acheck = np.abs(ft[ti, 0] - fp)
            rcheck = np.abs(acheck / (ft[ti, 0] + np.finfo(float).eps))
            fp = ft[ti, 0]

        return ft, dfduh, dfdudh
