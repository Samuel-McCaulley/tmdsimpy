import numpy as np
import pygad
import os
from scipy import io as sio
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from tmdsimpy.nlforces.bouc_wen import BoucWenForce

#Load data from brbloop.py
data = np.load('hysteretic_loops.npz')
X = data['X']
Xdot= data['Xdot']
F = data['F']
Nt = X.shape[1]

sys_fname = './data/BRB_ROM_U_122ELS4py.mat'
system_matrices = sio.loadmat(sys_fname)

Nnl,Nnodes = system_matrices['Qm'].shape 

Qm = np.array(system_matrices['Qm'].todense()) 
Tm = np.array(system_matrices['Tm'].todense())

L  = system_matrices['L']
Ndof = L.shape[1]

QL = np.kron(Qm, np.eye(3)) @ L[:3*Nnodes, :]
LTT = L[:3*Nnodes, :].T @ np.kron(Tm, np.eye(3))

U = QL @ X
Udot = QL @ Xdot
F_nl_ref = QL @ F



def error(theta, weights = None):
    print(theta)
    ''' 
    Parameters
    ----------
    theta : (4, ) numpy.ndarray
        List of Bouc-Wen Force Parameters in order:
        A
        beta
        gamma
        n

    Returns
    -------
    error : scalar
        Error between rough contact force and Iwan force element

    '''
    if weights is None: weights = np.ones_like(U_selected_dof)
    
    A = theta[0]
    beta = theta[1]
    gamma = theta[2]
    n = theta[3]
    
    Ls = np.reshape(QL[i,:], (1, Ndof))
    Lf = np.reshape(LTT[:, i], (Ndof, 1))

    bouc_wen_force = BoucWenForce(Ls, Lf, A, beta, gamma, n)
    
    h_max = 3
    h = np.array(range(h_max + 1))
    bouc_wen_force.init_history_harmonic(np.mean(U_selected_dof), h)

    X_from_U = Lf @ U_selected_dof.T
    
    physical_F = np.zeros_like(X_from_U)
    
    for index in range(X_from_U.shape[1]):
        
        X_from_U_slice = X_from_U[:, index]
        physical_F[:, index], _ = bouc_wen_force.force(X_from_U_slice)
    
    F_nl_est = Ls @ physical_F
    print(F_nl_est)

    error = np.linalg.norm((F_nl_est - F_nl_selected_dof) * weights)
    print(error)
    if np.isnan(error): error = np.iinfo(np.int64).max
    return error

def fitness(ga_instance, parameters, parameters_idx):
    w = np.zeros_like(U_selected_dof)
    for i in range(len(w)):
        if U_selected_dof[i] < 0.0001055 or U_selected_dof[i] > 0.0001070:
            w[i] = 1
        else:
            w[i] = 1
    return -error(parameters, w)


# =============================================================================
# ga_instance.run()
# 
# best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
# 
# print("Best Solution: ", best_solution)
# print("Best Solution Fitness: ", best_solution_fitness)
# 
# kt, Fs, chi, beta = best_solution
# 
# Ls = np.reshape(QL[i,:], (1, Ndof))
# Lf = np.reshape(LTT[:, i], (1, Ndof))
# 
# iwan_best = VectorIwan4(Ls, Lf, kt, Fs, chi, beta)
# 
# U = Ls @ X
# Udot = Ls @ Xdot
# 
# fnl_ref = Ls @ F
# 
# plt.plot(U, fnl_ref)
# plt.title("Reference Force Nonlinear")
# 
# h_max = 3
# Nhc = 2*h_max + 1
# h = np.array(range(h_max + 1))
# cst = np.ones((Nhc, X.shape[1]))
# 
# unlth0 = np.mean(U, axis=1)
# force_est = iwan_best.local_force_history(U.T, Udot.T, h, cst.T, unlth0)
# =============================================================================
global i 
global U_selected_dof
global Udot_selected_dof
global F_nl_selected_dof

i = 329
U_selected_dof = np.reshape(U[i, :], (Nt, 1))
Udot_selected_dof = np.reshape(Udot[i, :], (Nt, 1))
F_nl_selected_dof = np.reshape(F_nl_ref[i, :], (Nt, 1))

good_solution = np.array([1.5e9, 1700, 2.7, 0.9])

sol_per_pop = 100
num_genes = 4
initial_population = np.random.uniform(low=[-1e7, 0, 0.5, 0], high=[3e9, 2550, 3, 1.5], size=(sol_per_pop, num_genes))
initial_population[0] = good_solution

ga_instance = pygad.GA(
    #initial_population = initial_population,
    num_generations=100,                    # Number of generations
    num_parents_mating=5,                   # Number of parents mating
    fitness_func=fitness,          # Fitness function
    sol_per_pop=sol_per_pop,  # Population size
    mutation_probability= 0.2,                  
    mutation_percent_genes=40,  # Mutate 10% of the genes
    mutation_type="random",  # Use adaptive mutation
    num_genes=num_genes,                            # Number of parameters in the model
    gene_space=[(-1e7, 3000000000), (0, 1700*1.5), (0.5, 3), (0, 1.5)], # Ranges for each parameter
)


#ga_instance.run()

#best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()

#print("Best Solution: ", best_solution)
#print("Best Solution Fitness: ", best_solution_fitness)


#-----------Evaluate best solution------------


#kt, Fs, chi, beta = best_solution[0], best_solution[1], best_solution[2], best_solution[3]

# =============================================================================
# kt = 1500000000
# Fs = 1700
# chi = 2.7
# beta = 0.9
# reasonable_error = error(np.array((kt, Fs, chi, beta)))
# print(f"fitness of reasonably good solution: {-reasonable_error}")
# =============================================================================


# =============================================================================
# plt.plot(U_selected_dof, F_nl_selected_dof)
# plt.title("Reference Force Nonlinear")
# 
# Ls = np.reshape(QL[i, :], (1, Ndof))
# Lf = np.reshape(LTT[:, i], (Ndof, 1))
# 
# iwan_best = VectorIwan4(Ls, Lf, kt, Fs, chi, beta)
# 
# h_max = 3
# Nhc = 2*h_max + 1
# h = np.array(range(h_max + 1))
# cst = np.ones((Nhc, X.shape[1]))
# 
# unlth0 = np.mean(U_selected_dof)
# 
# force_est_return = iwan_best.local_force_history(U_selected_dof, Udot_selected_dof, h, cst.T, unlth0)
# force_est = force_est_return[0]
# plt.plot(U_selected_dof, force_est)
# plt.title("Predicted Force Nonlinear")
# 
# =============================================================================


plt.plot(U_selected_dof, F_nl_selected_dof)
plt.title("Reference Force Nonlinear")

Ls = np.reshape(QL[i, :], (1, Ndof))
Lf = np.reshape(LTT[:, i], (Ndof, 1))

A = 1e10
beta = 1
gamma = 1
n = 0.7

h_max = 3
h = np.array(range(h_max + 1))
bouc_best = BoucWenForce(Ls, Lf, A, beta, gamma, n)
bouc_best.init_history_harmonic(np.mean(U_selected_dof), h)

print(Lf.shape)
print(U_selected_dof.T.shape)

X_from_U = Lf @ U_selected_dof.T

physical_F = np.zeros_like(X_from_U)

for index in range(X_from_U.shape[1]):
    
    X_from_U_slice = X_from_U[:, index]
    physical_F[:, index], _ = bouc_best.force(X_from_U_slice)

F_nl_est = (Ls @ physical_F).T

plt.plot(U_selected_dof, F_nl_est)
plt.title("Predicted Force Nonlinear")

