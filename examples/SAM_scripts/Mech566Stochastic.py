import numpy as np
import sys
sys.path.append('../..')
import tmdsimpy.nlutils as nlutils
from tmdsimpy.nlforces.bouc_wen import BoucWenForce
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az



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
line = 85
X0 = np.atleast_2d(Uwxa_full[line, dof:-3:Ndof]).T*(10**Uwxa_full[line, -1])
U0 = np.atleast_2d(Uwxa_nl_full[line, dof:-3:Nnl]).T*(10**Uwxa_nl_full[line, -1])


Nt = 1 << 12

lam = Uwxa_full[line, -3]

ref_times, ref_disp, ref_forces = nlutils.hysteresis_loop(Nt, h, U0, lam, 'iwan', iwan_parameters[:-1])


SNR = 10**0


ref_forces += np.random.normal(scale = np.std(ref_forces)/np.sqrt(SNR), size = ref_forces.shape) 

plt.plot(ref_disp, ref_forces)
plt.title(f"Experimentally Simulated Data, SNR = {SNR}")
plt.show()



nominal_solution = [9.64105747, 7.2339903,  8.10953879, 0.71738319]

bw_lpsci = [1, 1, 1, 0]
A_nom, beta_nom, gamma_nom, n_nom = nlutils.paramexp(nominal_solution, bw_lpsci)


def convert_tensor_to_scalar(value):
    """
    Converts a PyTensor variable to a scalar (if it's a tensor).
    If it's already a scalar (like a numpy float), it simply returns it.
    """
    if isinstance(value, pt.TensorVariable):
        # Use .eval() to get the scalar value if it's a tensor
        return np.float64(value.eval())  # Converts tensor to float64 scalar
    else:
        # If it's already a scalar, return it directly
        return np.float64(value)

def hysteresis_loop_with_scalars(Nt, h, U0, lam, bw_params, loops=1):
    """
    Wrapper around the hysteresis_loop to ensure all parameters are scalars.
    """
    # Convert all the Bouc-Wen parameters to scalars
    bw_params_scalars = [convert_tensor_to_scalar(param) for param in bw_params]

    # Now call the original hysteresis_loop function with scalar values
    _, _, F_pred = nlutils.hysteresis_loop(Nt, h, U0, lam, 'bouc-wen', bw_params_scalars, loops=loops)
    
    F_pred = F_pred[-Nt:] #Only selects last loop

    return F_pred



# Bayesian model
with pm.Model() as model:
    A = pm.Normal('A', mu=A_nom, sigma=A_nom/10)
    beta = pm.Normal('beta', mu=beta_nom, sigma=beta_nom/10)
    gamma = pm.Normal('gamma', mu=gamma_nom, sigma=gamma_nom/10)
    n = pm.Normal('n', mu=n_nom, sigma=n_nom/10)

    # Likelihood
    bw_params = np.array([A, beta, gamma, n])
    F_pred = hysteresis_loop_with_scalars(Nt, h, U0, lam, bw_params, loops=12)
    y_obs = pm.Normal('y_obs', 
                      mu=F_pred, 
                      sigma=0.1, 
                      observed=ref_forces)

    # Inference
    trace = pm.sample(
        5000, 
        return_inferencedata=True, 
        progressbar=True, 
        tune=1000,
        nuts_sampler_kwargs ={"nuts": {"max_treedepth": 50, "target_accept": 0.95}}
    )
    
    print("here")




# Analyze results
pm.summary(trace)
pm.plot_trace(trace)
plt.show()


summary = az.summary(trace, round_to=2)
print(summary)