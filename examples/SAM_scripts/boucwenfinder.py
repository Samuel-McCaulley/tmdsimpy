import numpy as np
import sys
sys.path.append('../..')
import tmdsimpy.nlutils as nlutils
from tmdsimpy.hysteretic_loop_fitness import *
import matplotlib.pyplot as plt
import pygad


solve_data = np.load('data/iwan_epmc.npz')

Uwxa_full = solve_data['Uwxa_full']
Uwxa_nl_full = solve_data['Uwxa_nl_full']
h = solve_data['h']
Ndof = solve_data['Ndof']
Q = solve_data['Q']
iwan_parameters = solve_data['iwan_parameters']
patch_area = solve_data['patch_area']

Nnl = Q.shape[0]

dof = 9
line = 93
X0 = np.atleast_2d(Uwxa_full[line, dof:-3:Ndof]).T*(10**Uwxa_full[line, -1])
U0 = np.atleast_2d(Uwxa_nl_full[line, dof:-3:Nnl]).T*(10**Uwxa_nl_full[line, -1])

# U0_plural = np.reshape(Uwxa_nl_full[line, :-3].T*(10**Uwxa_nl_full[line, -1]), (hutils.Nhc(h), Nnl))
# X_full_harmonic = np.reshape(Uwxa_full[line, :-3].T*(10**Uwxa_full[line, -1]), (hutils.Nhc(h), Ndof))

lam = Uwxa_full[line, -3]

ref_times, ref_disp, ref_forces = nlutils.hysteresis_loop(1 << 12, h, U0, lam, 'iwan', iwan_parameters[:-1])

min_disp = np.min(ref_disp)
max_disp = np.max(ref_disp)

# Define thresholds for 10% of the min and max
min_threshold = min_disp * 1.2
max_threshold = max_disp * 0.8

# Initialize weights vector
weights = np.ones_like(ref_disp)

# Assign weight 10 to indices within 10% of the min and max
# weights[(ref_disp <= min_threshold) | (ref_disp >= max_threshold)] = 20

# # Assign high weight to minimum and maximum

# weights[(ref_disp == min_disp) | (ref_disp == max_disp)] = 150

lines = [2, 60, 85]

U = np.atleast_2d(Uwxa_nl_full[lines, dof:-3:Nnl]).T*(10**Uwxa_nl_full[lines, -1])
lams = Uwxa_full[lines, -3]

fitness_config = {
    'Nt'    : 1<<12,
    'h'     : h,
    'X0'    : U0,
    'lam'   : lam,
    'weights': weights,
    'verbose': True,
    'testloop': 6,
    'doftitle': line
    }



plural_fitness_config = {
    'Nt'    : 1<<7,
    'h'     : h,
    'X0_plural'    : U,
    'lam'   : lams,
    'weights': weights,
    'verbose': True,
    'testloop': 6,
    'doftitles': lines
    }

bouc_lpsci = [1, 1, 1, 0]

hysteretic_fitness = lambda ga_instance, theta, idx: hysteretic_loop_plural_fitness('iwan', 
                                                           iwan_parameters[:-1], 
                                                           [0, 0, 0, 0], 'bouc-wen', 
                                                           theta, bouc_lpsci, plural_fitness_config, print_fitness = True)

# hysteretic_fitness_total = lambda ga_instance, theta, idx: hysteretic_loop_plural_fitness(1<<11, 
#                                                                 h, U0_plural, lam, 'iwan', 
#                                                                 iwan_parameters[:-1], [0, 0, 0, 0], 
#                                                                 'bouc-wen', theta, [1, 1, 1, 0], False, True)



good_solution = [9.28655569, 5.66776462, 0.05721373, 1.59373539]
good_solution = [9.52789669, 4.85045283, 5.70009417, 1.86747747]
good_solution = [9.66307232, 7.99999021, 8.71325276, 0.40387107]
good_solution = [9.64519515, 6.51131523, 7.5396514,  1.02588743]
good_solution = [9.64105747, 7.2339903,  8.10953879, 0.71738319]
good_solution_pure = nlutils.paramexp(good_solution, bouc_lpsci)
good_solution_pure[0] /= patch_area
good_solution_invariant_log = nlutils.paramlog(good_solution_pure, bouc_lpsci)

print(f"Good solution to pass: {good_solution_invariant_log}")

hysteretic_fitness(None, good_solution, None)





#%% PyGaD

def on_generation(ga_instance):
    print(f"Generation: {ga_instance.generations_completed}")

#8, 3, 2, 2 is an okay solution
ga_instance = pygad.GA(
    num_generations=75,              # Number of generations
    num_parents_mating=6,            # Number of parents for mating
    fitness_func=hysteretic_fitness,       # Fitness function
    sol_per_pop=250,                  # Number of solutions in the population
    num_genes=4,             # Number of genes (4 in this case)
    gene_space=[{'low': 8, 'high': 10},
                {'low': 0, 'high': 10},
                {'low': 0, 'high': 10},
                {'low': 0, 'high': 2}], # Range for each gene
    mutation_percent_genes=25   ,     # Percentage of genes to mutate
    on_generation = on_generation
)




ga_instance.run()

# Get the best solution
solution, solution_fitness, _ = ga_instance.best_solution()
print("Best solution:", solution)
print("Fitness value of the best solution:", solution_fitness)

# Plot the fitness evolution
ga_instance.plot_fitness()





        
        