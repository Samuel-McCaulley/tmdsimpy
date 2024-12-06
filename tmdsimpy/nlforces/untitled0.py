from scipy.integrate import solve_ivp
import numpy as np
rho = 42064095.286401354
u_0 = -1.21313156e-08
zeta_0= 0.0
sigma = 0.08565882405293729
n = 1.02588743


def dzetadunl_fun(unl, zeta, unldot):
        
    if zeta == 0: return rho
    df = rho * (1 - (sigma * np.sign(zeta) * np.sign(unldot) + (1 - sigma)) * np.abs(zeta)**n)
    
    return df

