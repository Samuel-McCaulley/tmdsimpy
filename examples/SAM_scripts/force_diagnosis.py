import numpy as np
import sys
# Python Utilities
sys.path.append('../..')
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy import nlforces
import matplotlib.pyplot as plt
import pickle

def time_series_forces(Unl, h, Nt, w, nl_force):
   
    Unl = np.reshape(Unl, ((-1,1)))

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

def apply_Q_to_U(Q, U, Nhc):
    """
    Apply nonlinear transformation Q to harmonic vector U.

    Q: (Nnl, N_dof)
    U: (N_dof * Nhc,)
    Returns: U_nl (Nnl * Nhc,)
    """
    Nnl, Ndof = Q.shape
    U_reshaped = U.reshape((Nhc, Ndof)).T  # Shape: (Ndof, Nhc)
    U_nl = Q @ U_reshaped  # Shape: (Nnl, Nhc)
    return U_nl.T.reshape(-1)  # Back to (Nnl * Nhc,)


def force_diagnosis(ref_nlforces, test_nlforces, U, Nt, w, h):
    '''
    Compares forces and force derivatives (dfduh) for each nonlinear DOF.
    '''
    plt.close('all')
    Ndnl = ref_nlforces.size
    Nhc = hutils.Nhc(h)

    # --- Force comparison figure ---
    fig, axs = plt.subplots(Ndnl, 1, figsize=(10, 4 * Ndnl), squeeze=False)

    for i in range(Ndnl):
        Q_ref = ref_nlforces[i].Q
        Q_test = test_nlforces[i].Q

        ref_nl_dof = apply_Q_to_U(Q_ref, U, Nhc)
        test_nl_dof = apply_Q_to_U(Q_test, U, Nhc)
        
        #test_nl_dof = np.array([0, 1e-8, 0, 0, 0, 0, 0])

        fnl_ref, dfduh_ref = time_series_forces(ref_nl_dof, h, Nt, w, ref_nlforces[i])
        fnl_test, dfduh_test = time_series_forces(test_nl_dof, h, Nt, w, test_nlforces[i])

        # --- Force plot ---
        axs[i, 0].plot(fnl_ref, label='Reference Force', color='blue')
        axs[i, 0].plot(fnl_test, label='Test Force', color='orange')
        axs[i, 0].set_title(f'DOF {i+1} - Force')
        axs[i, 0].set_xlabel('Time Step')
        axs[i, 0].set_ylabel('Force')
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # --- Comparative dfduh plots (one subplot per harmonic) ---
        fig_dfduh, axarr = plt.subplots(Nhc, 1, figsize=(10, 3 * Nhc), sharex=True)
        fig_dfduh.suptitle(f'DOF {i+1} - dfduh Comparison (Ref vs Test)', fontsize=14)

        # Handle case when Nhc == 1 (axarr isn't iterable)
        if Nhc == 1:
            axarr = [axarr]

        for j in range(Nhc):
            ax = axarr[j]
            ax.plot(dfduh_ref[:, j], label='Reference', color='blue')
            ax.plot(dfduh_test[:, j], label='Test', color='orange')
            ax.set_ylabel(f'df/duh {j}')
            ax.grid(True)
            ax.legend()

        axarr[-1].set_xlabel('Time Step')
        fig_dfduh.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        
        dFdUnl_test = hutils.get_fourier_coeff(h, dfduh_test)
        dFdUnl_ref = hutils.get_fourier_coeff(h, dfduh_ref)

        # Define a tolerance for what counts as "approximately zero"
        tol = 1e-12  
        
        # Mask to identify "small" values in denominator and numerator
        small_ref = np.isclose(dFdUnl_ref, 0, atol=tol)
        small_test = np.isclose(dFdUnl_test, 0, atol=tol)
        
        # Initialize output with NaN
        ratio = np.full_like(dFdUnl_ref, np.nan, dtype=float)
        
        # Case 1: both are ~0 → set ratio = 1 (or 0 depending on your meaning)
        ratio[small_ref & small_test] = 1.0  
        
        # Case 2: ref ~0 but test not ~0 → mark as inf
        ratio[small_ref & ~small_test] = np.inf  
        
        # Case 3: regular division
        ratio[~small_ref] = dFdUnl_test[~small_ref] / dFdUnl_ref[~small_ref]
        
        print(ratio)
        

    plt.tight_layout()
    plt.show()



    
if __name__ == "__main__":
    # Load debugging variables from pickle file
    with open("debug_variables.pkl", "rb") as f:
        debug_data = pickle.load(f)

    # Expected keys in the pickle file
    # 'ref_nlforces', 'test_nlforces', 'U', 'Nt', 'w'
    try:
        ref_nlforces = np.array(debug_data['ref_nlforces'])
        test_nlforces = np.array(debug_data['test_nlforces'])
        U = debug_data['X']
        Nt = debug_data['Nt']
        w = debug_data['w']
        h = debug_data['h']
    except KeyError as e:
        print(f"Missing key in pickle file: {e}")
        sys.exit(1)

    # Run the diagnostic plot
    force_diagnosis(ref_nlforces, test_nlforces, U, Nt, w, h)

    
