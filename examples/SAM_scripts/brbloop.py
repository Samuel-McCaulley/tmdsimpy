import sys
import numpy as np
from scipy import io as sio
from scipy.interpolate import LinearNDInterpolator
import warnings
import time
import argparse # parse command line arguments
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import jax.numpy as jnp
from jax import device_get

    
sys.path.append('../..')
import tmdsimpy.utils.harmonic as hutils

from tmdsimpy.jax.solvers import NonlinearSolverOMP

from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.continuation as cont_utils

from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.jax.nlforces.roughcontact.rough_contact import RoughContactFriction

def evaluate_harmonic_displacement(harmonic_coeffs, Ndof, fundamental_freq, time_steps, Nhc):
    num_harmonics = (Nhc-1)//2
    """
    Evaluates the harmonic displacement for multiple degrees of freedom over time.

    Parameters:
    harmonic_coeffs : list or array-like
        List of harmonic coefficients structured by harmonics and degrees of freedom.
    Ndof : int
        Number of degrees of freedom.
    fundamental_freq : float
        Fundamental frequency.
    time_steps : array-like
        Array of time steps at which to evaluate the displacement.
    num_harmonics : int
        The number of harmonics (excluding the zeroth harmonic).

    Returns:
    displacement_matrix : np.ndarray
        Matrix of shape (Ndof, len(time_steps)) containing the evaluated displacement for each degree of freedom.
    """
    Nt = len(time_steps)  # Number of time steps
    displacement_matrix = np.zeros((Ndof, Nt))  # Initialize the displacement matrix
    velocity_matrix = np.zeros((Ndof, Nt)) #Inititalize the velocity matrix
    
    index = 0
    
    # Zeroth harmonic (constant term)
    for dof in range(Ndof):
        h_0 = harmonic_coeffs[index]
        displacement_matrix[dof, :] += h_0  # Add the constant term to all time steps
        index += 1
    
    # Higher-order harmonics
    for n in range(1, num_harmonics + 1):
        # Cosine terms for n-th harmonic
        for dof in range(Ndof):
            h_cn = harmonic_coeffs[index]
            displacement_matrix[dof, :] += h_cn * np.cos(n * fundamental_freq * time_steps)
            velocity_matrix[dof, :] -= h_cn * fundamental_freq * n * np.sin(n * fundamental_freq * time_steps)
            index += 1
        
        # Sine terms for n-th harmonic
        for dof in range(Ndof):
            h_sn = harmonic_coeffs[index]
            displacement_matrix[dof, :] += h_sn * np.sin(n * fundamental_freq * time_steps)
            velocity_matrix[dof, :] += h_cn * fundamental_freq * n * np.cos(n * fundamental_freq * time_steps)
            index += 1
    
    return displacement_matrix, velocity_matrix



###############################################################################
####### 1. Command Line Defaults                                        #######
###############################################################################

# These defaults can be changed if running in an IDE without giving command
# line arguments to look at different systems

# Set this to 1 to use mesoscale or 0 to not use mesoscale by default
# Command line input will override this if given. Example: 
# python3 -u brb_epmc.py -meso 1
default_mesoscale = 1 

# default filename for a .mat file that contains the matrices appropriately 
# formatted to be able to describe the structural system. 
# Command line input will override this if given
# python3 -u brb_epmc.py -system './data/BRB_ROM_U_232ELS4py.mat'
default_sys_fname = './data/BRB_ROM_U_122ELS4py.mat'

# Flag for running profiler. Command line argument will override this flag
# If flag is not zero, then the code simulation will be profiled. Otherwise, 
# it will not be profiled. Example:
# python3 -u brb_epmc.py -p 1
default_run_profilers = 0 

# Default value for fast solution flag. If not zero, then a fast solution will
# be calculated using reduced harmonics and AFT steps. Example
# python3 -u brb_epmc.py -f 1
default_fast_sol = 0

###############################################################################
####### 2. User Inputs                                                  #######
###############################################################################

# These can be modified, but likely are fine as they are given here. Modifying
# these parameters will likely mean that the results cannot be verified
# with the script 'compare_brb_epmc.py' after running. 

# Log10 Amplitude Start (mass normalized modal amplitude)
Astart = -7.0

# Log10 Amplitude End
Aend = -5.5

# Choose speed or accuracy - edit other parameters based on which value
# of this flag you are using.
fast_sol = False 

if fast_sol:
    # Run with reduced harmonics and AFT time steps to keep time within a 
    # few minutes
    h_max = 1 # harmonics 0 and 1
    Nt = 1<<3 # 2**3 = 8 AFT steps 
    FracLam = 0.5 # Continuation weighting
    
    ds = 0.1
    dsmax = 0.2
    dsmin = 0.02
else:
    # Normal - settings for higher accuracy as used in previous papers
    h_max = 3 # harmonics 0, 1, 2, 3
    Nt = 1<<7 # 2**7 = 128 AFT steps 
    
    ds = 0.08
    dsmax = 0.125*1.4
    dsmin = 0.02
    
    # Adjust weighting of amplitude v. other in continuation to hopefully 
    # reduce turning around. Higher puts more emphasis on continuation 
    # parameter (amplitude)
    FracLam = 0.50     

###############################################################################
####### 3. Friction Model Parameters                                    #######
###############################################################################

# These can be modified, but likely are fine as they are given here. Modifying
# these parameters will likely mean that the results cannot be verified
# with the script 'compare_brb_epmc.py' after running. 

# Surface Parameters for the rough contact model - from ref [1]
surface_fname = './data/brb_surface_data.mat'

surface_pars = sio.loadmat(surface_fname)

ElasticMod = 192.31e9 # Pa
PoissonRatio = 0.3
Radius = surface_pars['Re'][0, 0] # m
TangentMod = 620e6 # Pa
YieldStress = 331.7e6 # Pa 
mu = 0.03

area_density = surface_pars['area_density'][0,0] # Asperities / m^2
max_gap = surface_pars['z_max'][0,0] # m

normzinterp = surface_pars['normzinterp'][0]
pzinterp    = surface_pars['pzinterp'][0]

gaps = np.linspace(0, 1.0, 101) * max_gap

trap_weights = np.ones_like(gaps)
trap_weights[1:-1] = 2.0
trap_weights = trap_weights / trap_weights.sum()

gap_weights = area_density * trap_weights * np.interp(gaps/max_gap, 
                                                      normzinterp, pzinterp)

prestress = (12002+12075+12670)*1.0/3; # N per bolt

mesoscale_xygap = surface_pars['mesoscale_xygap']

sys_fname = './data/BRB_ROM_U_122ELS4py.mat'
system_matrices = sio.loadmat(sys_fname)



damp_ab = [0.087e-2*2*(168.622*2*np.pi), 0.0]

vib_sys = VibrationSystem(system_matrices['M'], system_matrices['K'], 
                          ab=damp_ab)

###############################################################################
####### 9. Add Nonlinear Forces to System                               #######
###############################################################################

# Number of nonlinear frictional elements, Number of Nodes
Nnl,Nnodes = system_matrices['Qm'].shape 

# Need to convert sparse loads into arrays so that operations are expected shapes
# Sparse matrices from matlab are loaded as matrices rather than numpy arrays
# and behave differently than numpy arrays.
Qm = np.array(system_matrices['Qm'].todense()) 
Tm = np.array(system_matrices['Tm'].todense())

# Pull out for reference convenience - null space transformation matrix
L  = system_matrices['L']
Ndof = L.shape[1]

QL = np.kron(Qm, np.eye(3)) @ L[:3*Nnodes, :]
LTT = L[:3*Nnodes, :].T @ np.kron(Tm, np.eye(3))


# Calculate the mesoscale gaps of each node point
interp_obj = LinearNDInterpolator(mesoscale_xygap[:, :2], # x, y
                                  mesoscale_xygap[:, 2]) # gaps

meso_gap_nodes = interp_obj(system_matrices['node_coords'][:, 0], # node x
                            system_matrices['node_coords'][:, 1]) # node y

# interpolate mesoscale at nodes to quadrature points
meso_gap_quads = Qm @ meso_gap_nodes

# move so something is initially in contact
meso_gap_quads = meso_gap_quads - meso_gap_quads.min() 

# Set mesoscale to zero if not using it
mesoscale_TF = 1 #From SAM --- Mesoscale is being used.
meso_gap_quads = mesoscale_TF * meso_gap_quads 

Uwxa_full = np.load("./results/Uwxa_full.npy")
sel = 10

Uwxa_harm, frequency, damping, amplitude_log = Uwxa_full[sel, :-3], Uwxa_full[sel, -3], Uwxa_full[sel, -2], Uwxa_full[sel, -1]

Nhc = 7


U_harm = Uwxa_harm.reshape(Ndof, Nhc)

time_steps = np.linspace(0,2*np.pi/frequency,200)

X, Xdot = evaluate_harmonic_displacement(Uwxa_harm, Ndof, frequency, time_steps, Nhc)
F_sum = np.zeros(X.shape)
print(F_sum.shape)
for i in range(Nnl):
    print(i)
    Ls = (QL[i*3:(i*3+3),   :])
    Lf = (LTT[:, i*3:(i*3+3)])

    tmp_nl_force = RoughContactFriction(Ls, Lf, ElasticMod, PoissonRatio, 
                                        Radius, TangentMod, YieldStress, mu,
                                        gaps=gaps, gap_weights=gap_weights,
                                        meso_gap=meso_gap_quads[i])
    
    
    for t in range(len(time_steps)):
        F, _ = tmp_nl_force.force(X[:, t])
        F = jnp.array(F.addressable_data(0)).copy()
        F_sum[:, t] += F

U = QL @ X
F_nl = QL @ F_sum

for i in range(3*Nnl):
    plt.plot(U[i, :], F_nl[i, :])
    plt.title(f"Nonlinear force vs. dof {i}")
    plt.show()
    
np.savez("hysteretic_loops.npz", X=X, Xdot = Xdot, F=F_sum)


