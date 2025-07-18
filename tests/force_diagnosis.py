import numpy as np
import sys
# Python Utilities
sys.path.append('../..')
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy import nlforces
import matplotlib.pyplot as plt

def time_series_forces(Unl, h, Nt, w, nl_force):
   
    # Unl = np.reshape(Unl, ((-1,1)))

    # Nonlinear displacements, velocities in time
    unlt = hutils.time_series_deriv(Nt, h, Unl, 0) # Nt x Ndnl
    unltdot = w*hutils.time_series_deriv(Nt, h, Unl, 1) # Nt x Ndnl
   
    Nhc = hutils.Nhc(h)
    cst = hutils.time_series_deriv(Nt, h, np.eye(Nhc), 0)
   
    unlth0 = Unl[0]
   
    fnl, dfduh, dfdudh = nl_force.local_force_history(unlt, unltdot, h, cst, unlth0)
   
    fnl = np.einsum('ij -> i', fnl)
    dfduh = np.einsum('ijkl -> il', dfduh)
   
    return fnl, dfduh


def force_diagnosis(ref_nlforces, test_nlforces, U, Nt, w):
    '''
    ref_nlforces and test_nlforces are numpy arrays of NonlinearForce objects, which should be the same size
   
    U is a harmonic coefficient matrix of physical degrees of freedom
    '''
   
    # Number of nonlinear degrees of freedom
    Ndnl = ref_nlforces.size  # Size of the vector of NonlinearForce objects

    # Determine the number of rows and columns for a square-ish layout
    ncols = int(np.ceil(np.sqrt(Ndnl)))  # Number of columns
    nrows = int(np.ceil(Ndnl / ncols))   # Number of rows

    # Create a figure for plotting
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
   
    # Flatten the axes array for easy indexing
    axs = axs.flatten()
   
    for i in range(Ndnl):
        # Extract the Q matrix from the NonlinearForce objects
        Q_ref = ref_nlforces[i].Q  # Q matrix for reference
        Q_test = test_nlforces[i].Q  # Q matrix for test
       
        # Convert physical dofs to nonlinear dofs using Q matrix
        ref_nl_dof = np.dot(Q_ref, U)  # Assuming U is a vector of physical dofs
        test_nl_dof = np.dot(Q_test, U)  # Assuming U is a vector of physical dofs
       
        # Calculate forces and derivatives for reference and test
        fnl_ref, dfduh_ref = time_series_forces(ref_nl_dof, U, Nt, w, ref_nlforces[i])
        fnl_test, dfduh_test = time_series_forces(test_nl_dof, U, Nt, w, test_nlforces[i])
       
        # Plot displacements and forces
        axs[i].plot(fnl_ref, label='Reference Force', color='blue')
        axs[i].plot(fnl_test, label='Test Force', color='orange')
        axs[i].set_title(f'Displacements for DOF {i+1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Force')
        axs[i].legend()

    # Hide any unused subplots
    for j in range(Ndnl, nrows * ncols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()