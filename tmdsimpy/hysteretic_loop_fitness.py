import numpy as np


import tmdsimpy.utils.harmonic as hutils
import tmdsimpy.nlutils as nlutils
import matplotlib.pyplot as plt
import signal

# Define a custom timeout exception
class TimeoutException(Exception):
    pass

# Define the timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException

def hysteretic_loop_fitness(ref_model, ref_lparameters, ref_lpsci, test_model, test_lparameters, test_lpsci, config):
    
    Nt = config['Nt']
    h = config['h']
    X0 = config['X0']
    lam = config['lam']
    
    weights = None
    try:
        weights = config['weights']
        assert weights.shape[0] == Nt, "Weights is the wrong shape. Try transposing?"
    except:
        weights = np.ones((Nt, 1))
    
    verbose = None
    try:
        verbose = config['verbose']
    except:
        verbose = False
        
    refloops = None
    try:
        refloop = config['refloop']
    except:
        refloop = 1
    
    testloop = None
    try:
        testloop = config['testloop']
    except:
        testloop = 1
        
    if verbose:
        print(f"Testing {test_lparameters}")
    ref_parameters = nlutils.paramexp(ref_lparameters, ref_lpsci)
    test_parameters = nlutils.paramexp(test_lparameters, test_lpsci)
    
    try:
    
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(25)
        
        ref_time, ref_displacement, ref_forces = nlutils.hysteresis_loop(Nt, h, X0, lam, ref_model, ref_parameters, loops = refloop)
        test_time, test_displacement, test_forces = nlutils.hysteresis_loop(Nt, h, X0, lam, test_model, test_parameters, loops = testloop)
        
        
        assert (ref_time[1] - ref_time[0]) == (test_time[1] - test_time[0]), 'Timings went wrong'
        
        ref_force_loop = ref_forces[-Nt:]
        
        test_force_loop = test_forces[-Nt:]
        weighted_norm = np.linalg.norm((ref_force_loop - test_force_loop) * weights)
        if verbose:
            plt.plot(ref_displacement[-Nt:], ref_force_loop)
            plt.plot(test_displacement[-Nt:], test_force_loop)
            plt.show()
            plt.legend(("Reference Loop", "Test Loop"))
            
            print(f"Weighted norm of error: {weighted_norm}")
        
        signal.alarm(0)
            
        return -1 * weighted_norm
        
    except TimeoutException:
        print("Timed Out!")
        return -1e10
    except Exception as e:
        # Handle other errors
        print(f"Error with solution {test_lparameters}: {e}")
        return -1e10
        
        
def hysteretic_loop_plural_fitness(ref_model, ref_lparameters, ref_lpsci, test_model, test_lparameters, test_lpsci, config, print_fitness):
  
    Nt = config['Nt']
    h = config['h']
    X0_plural = config['X0_plural']
    lams = config['lam']
    
    assert len(lams) == X0_plural.shape[1], "Not enough lambdas for X0_plural"
    
    weights = None
    try:
        weights = config['weights']
        assert weights.shape[0] == Nt, "Weights is the wrong shape. Try transposing?"
    except:
        weights = np.ones((Nt, 1))
    
    verbose = None
    try:
        verbose = config['verbose']
    except:
        verbose = False
        
    refloops = None
    try:
        refloop = config['refloop']
    except:
        refloop = 1
    
    testloop = None
    try:
        testloop = config['testloop']
    except:
        testloop = 1
    
    ref_parameters = nlutils.paramexp(ref_lparameters, ref_lpsci)
    test_parameters = nlutils.paramexp(test_lparameters, test_lpsci)
    total_fitness = 0
    for dof in range(X0_plural.shape[1]):
        X0 = np.atleast_2d(X0_plural[:, dof]).T
        singular_config = {
            'Nt'    : Nt,
            'h'     : h,
            'X0'    : X0,
            'lam'   : lams[dof],
            'weights': weights,
            'verbose': True,
            'testloop': 12
            }

        dof_fitness = hysteretic_loop_fitness(ref_model, ref_lparameters, ref_lpsci, test_model, test_lparameters, test_lpsci, singular_config)
        
        if dof_fitness == -1e10:
            total_fitness = -1e10
            break
        
        ref_time, ref_displacement, ref_forces = nlutils.hysteresis_loop(Nt, h, X0, lams[dof], ref_model, ref_parameters)
        force_norm = np.linalg.norm(ref_forces)
        
        dof_fitness /= force_norm
        
        total_fitness += dof_fitness
    if print_fitness: print(f"Fitness: {total_fitness}")
    return total_fitness
    
    

