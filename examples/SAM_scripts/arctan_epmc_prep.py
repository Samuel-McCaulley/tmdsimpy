
#Standard Imports
import numpy as np
import sys
import matplotlib.pyplot as plt

#Nonlinear Force Imports
import tmdsimpy.nlforces.iwan4_element as Iwan
from tmdsimpy.nlforces.arctangent_stiffness import ArctangentStiffness

#Utility Imports
import tmdsimpy.nlutils as nlutils
import tmdsimpy.harmonic_utils as hutils
import tmdsimpy.continuation as Continutation


#%% Part 1: Gather BRB Variables and Iwan Solve Data

solve_data = np.load('data/iwan_epmc.npz')
Uwxa_full = solve_data['Uwxa_full']
Uwxa_nl_full = solve_data['Uwxa_nl_full']

Q = solve_data['Q']

h = solve_data['h']
N = solve_data['Ndof']
Nnl = Q.shape[0]

Fs_raw, Kt_raw, chi, beta, Fn_raw = solve_data['iwan_parameters']
patch_areas = solve_data['patch_areas']


chi = -0.1 #EXPERIMENT

Nt = 1 << 8
Nc = Uwxa_full.shape[0]
cst = np.ones((2*h[-1] + 1, Nt)).T


#%% Part 2: Print Iwan Hysteresis Loops
print_iwan_hysteresis_secant = True
print_arctan_secant_guesses = True

if print_iwan_hysteresis_secant:
    # Create a list to store all subplot data
    plot_data = []
    if print_arctan_secant_guesses:
        arc_plot_data = []
    
    for nldof in range(Q.shape[0]):
        if nldof % 3 == 0 or nldof % 3 == 1:
            iwan_ks = np.zeros((Uwxa_nl_full.shape[0]))
            print(f"Processing nldof {nldof}")
            for line in range(Uwxa_nl_full.shape[0]):
                # Original Iwan force calculation
                iwan_force = Iwan.Iwan4Force(
                    np.array([[1]]), np.array([[1]]),
                    patch_areas[nldof // 3] * Kt_raw, 
                    patch_areas[nldof // 3] * Fs_raw, 
                    chi, beta
                )
                Ut = hutils.time_series_deriv(
                    Nt, h, 
                    np.atleast_2d(Uwxa_nl_full[line, nldof:-3:Nnl]*10**Uwxa_nl_full[line, -1]).reshape(-1, 1), 
                    order=0
                )
                Utdot = hutils.time_series_deriv(
                    Nt, h,
                    np.atleast_2d(Uwxa_nl_full[line, nldof:-3:Nnl]*10**Uwxa_nl_full[line, -1]).reshape(-1, 1),
                    order=1
                )
                force = iwan_force.local_force_history(
                    Ut, Utdot, h, cst, Uwxa_nl_full[line, nldof]
                )[0]
                
                # Calculate secant stiffness
                secant_stiffness = nlutils.calculate_secant_stiffness(Ut, force)
                iwan_ks[line] = secant_stiffness[0]
            
            if print_arctan_secant_guesses:
                
                Unorm = np.zeros(Nc)
                for line in range(Nc):
                    Unorm[line] = np.linalg.norm(Uwxa_nl_full[line, nldof:-3:Nnl])*10**Uwxa_nl_full[line, -1]
                b, s = ArctangentStiffness.incomplete_slip_parameters(np.log10(Unorm), Uwxa_nl_full[:, -3],
                                                                          Uwxa_nl_full[:, -2])
                
                phi_max = Fs_raw * patch_areas[nldof // 3]*(1 + beta)/(Kt_raw * patch_areas[nldof // 3] * (beta + (chi + 1)/(chi + 2)))
                print(f'Phi_max: {phi_max}')
                beginning_amplitude = Unorm[0]
                print(f'beginning amplitude: {beginning_amplitude}')
                Kt_arctangent = Kt_raw * patch_areas[nldof // 3] * (1 - (beginning_amplitude/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))
                print(f"Kt_arctangent: {Kt_arctangent}")
                '''
                    Doing some random s***
                '''

                #Kt_arctangent /= 3
                arctan_force = ArctangentStiffness(np.array([[1]]), np.array([[1]]), Kt_arctangent, b, s)
                
                
                arctan_ks = np.zeros(Nc)
                for line in range(Nc):
                    Ut = hutils.time_series_deriv(
                        Nt, h, 
                        np.atleast_2d(Uwxa_nl_full[line, nldof:-3:Nnl]*10**Uwxa_nl_full[line, -1]).reshape(-1, 1), 
                        order=0
                    )
                    Utdot = hutils.time_series_deriv(
                        Nt, h,
                        np.atleast_2d(Uwxa_nl_full[line, nldof:-3:Nnl]*10**Uwxa_nl_full[line, -1]).reshape(-1, 1),
                        order=1
                    )
                    force = arctan_force.local_force_history(
                        Ut, Utdot, h, cst, Uwxa_nl_full[line, nldof]
                    )[0]
                    # Calculate secant stiffness
                    secant_stiffness = nlutils.calculate_secant_stiffness(Ut, force)
                    arctan_ks[line] = secant_stiffness[0]
                    
            # Store data for plotting
            plot_data.append((nldof, iwan_ks))
            if print_arctan_secant_guesses:
                arc_plot_data.append((nldof, arctan_ks))
            

    # Create subplots in specified arrangement
    nrows = len(plot_data) // 2 + len(plot_data) % 2
    fig, axs = plt.subplots(nrows, 2, figsize=(12, 3*nrows))
    
    # Flatten axes array for easy indexing
    axs = axs.flatten()
    
    # Plot configuration
    for idx, (nldof, stiffness) in enumerate(plot_data):
        ax = axs[idx]
        ax.plot(Uwxa_full[:, -3], stiffness, linewidth=1.5) 
        if print_arctan_secant_guesses:
            ax.plot(Uwxa_full[:, -3], arc_plot_data[idx][1])
            plt.legend(['iwan', 'arctan'])
        ax.set_title(f'nldof {nldof}', fontsize=10)
        ax.set_ylabel('Secant Stiffness', fontsize=8)
        ax.set_xlabel('Frequency', fontsize = 8)
        ax.tick_params(axis='both', labelsize=8)
        
        
    # Hide unused axes
    for idx in range(len(plot_data), nrows*2):
        axs[idx].axis('off')
        
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(nrows, 2, figsize=(12, 3*nrows))
    
    # Flatten axes array for easy indexing
    axs = axs.flatten()
    
    # Plot configuration
    for idx, (nldof, stiffness) in enumerate(plot_data):
        ax = axs[idx]
        ax.plot(Uwxa_full[:, -1], stiffness, linewidth=1.5) 
        if print_arctan_secant_guesses:
            ax.plot(Uwxa_full[:, -1], arc_plot_data[idx][1])
            plt.legend(['iwan', 'arctan'])
        ax.set_title(f'nldof {nldof}', fontsize=10)
        ax.set_ylabel('Secant Stiffness', fontsize=8)
        ax.set_xlabel('Amplitude', fontsize = 8)
        ax.tick_params(axis='both', labelsize=8)
        
        
    # Hide unused axes
    for idx in range(len(plot_data), nrows*2):
        axs[idx].axis('off')
        
    plt.tight_layout()
    plt.show()
    
#%% Iwan Hysteresis Loops
nldof = 0
for line in range(Uwxa_nl_full.shape[0]):
    # Original Iwan force calculation
    iwan_force = Iwan.Iwan4Force(
        np.array([[1]]), np.array([[1]]),
        patch_areas[nldof // 3] * Kt_raw, 
        patch_areas[nldof // 3] * Fs_raw, 
        chi, beta
    )
    Ut = hutils.time_series_deriv(
        Nt, h, 
        np.atleast_2d(Uwxa_nl_full[line, nldof:-3:Nnl]*10**Uwxa_nl_full[line, -1]).reshape(-1, 1), 
        order=0
    )
    Utdot = hutils.time_series_deriv(
        Nt, h,
        np.atleast_2d(Uwxa_nl_full[line, nldof:-3:Nnl]*10**Uwxa_nl_full[line, -1]).reshape(-1, 1),
        order=1
    )
    force = iwan_force.local_force_history(
        Ut, Utdot, h, cst, Uwxa_nl_full[line, nldof]
    )[0]
    
    if line % 10 == 1:
        plt.plot(Ut, force)
        plt.title(f"Hysterisis Loop for line {line}")
        plt.show()

