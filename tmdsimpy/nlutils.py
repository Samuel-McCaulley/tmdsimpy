import numpy as np

from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.nlforces.bouc_wen import BoucWenForce
import tmdsimpy.utils.harmonic as hutils



def hysteresis_loop(Nt, h, X0, lam, model, parameters, loops = 1):
    """
    Generates a hysteresis loop based on the specified model and parameters.

    Parameters:
    - Nt (int): Number of time steps to evaluate the forcing function at. 
      Must be an even power of 2 for FFT operations.
    - h (np.ndarray): Sorted numpy array of harmonics to use in the forcing function.
    - X0 : (Nhc, N) numpy.ndarray
        Harmonic Coefficients for columns corresponding to degrees of freedom
        and rows corresponding to different harmonic components.
    - lam (float): Fundamental frequency 
    - model (str): Specifies the hysteresis model to use. 
      Should be either 'iwan' or 'bouc-wen'.
    - parameters (np.ndarray): Array of 5 model-specific parameters for tuning the hysteresis behavior.
    - loops (int, optional): Number of cycles to get rid of transient effects, default = 1

    Returns:
    - result (dict): Dictionary containing the computed hysteresis loop data, including:
      - 'displacement': Array of displacements over the hysteresis loop.
      - 'force': Array of forces corresponding to each displacement.
      - 'time': Array of time steps corresponding to each data point.
    
    Notes:
    - The function requires Nt to be an even power of 2 for compatibility with FFT-based computations.
    - The choice of model ('iwan' or 'bouc-wen') determines how the hysteresis is computed based on 
      the input parameters.
    """
    
    assert X0.shape[1] ==1, 'Only tested for one nonlinear degree of freedom'
    assert (model == 'iwan' or model == 'bouc-wen'), 'Not tested for other models'
    
    oneloop_displacement = hutils.time_series_deriv(Nt, h, X0, 0)
    oneloop_velocity = hutils.time_series_deriv(Nt, h, X0, 1)
    
    
    displacement = np.tile(oneloop_displacement, (loops, 1))
    velocity = np.tile(oneloop_velocity,(loops, 1))
    
    Nhc = hutils.Nhc(h)
    
    tau = np.linspace(0,loops,loops*Nt+1)[:-1]
    time = tau * 2*np.pi/lam

    if model == 'iwan':
        iwan_force = VectorIwan4(np.array([[1]]), np.array([[1]]), parameters[0], 
                                 parameters[1], parameters[2], parameters[3])
        
        cst = np.ones([Nt*loops, Nhc])
        
        forces = iwan_force.local_force_history(displacement, velocity, h, cst, X0[0])[0]
    elif model == 'bouc-wen':
        bw_force = BoucWenForce(np.array([[1]]), np.array([[1]]), parameters[0], 
                                 parameters[1], parameters[2], parameters[3])
                
        cst = np.ones([Nt*loops, Nhc])
        
        forces = bw_force.local_force_history(displacement, velocity, h, cst, X0[0])[0]

    return time, displacement, forces

def paramexp(lparams, lpsci, base = 10):
    return [base ** lparams[i] if lpsci[i] == 1 else lparams[i] for i in range(len(lparams))]

def paramlog(params, lpsci, base = 10):
    return [np.emath.logn(base, params[i]) if lpsci[i] == 1 else params[i] for i in range(len(params))]


def transform_to_nonlinear(matrix, Q, Ndof, Nnl):
    """
    Transform blocks of Ndof columns in the input matrix into nonlinear blocks using the Q matrix.
    
    Parameters:
        matrix (numpy.ndarray): Input matrix of shape (Ncont, Ndof*Nhc + 3).
        Q (numpy.ndarray): Transformation matrix of shape (Nnl, Ndof).
        Ndof (int): Number of degrees of freedom (width of each block to transform).
        Nnl (int): New width of transformed blocks (15 in this case).
    
    Returns:
        numpy.ndarray: Transformed matrix with blocks replaced and the last 3 columns untouched.
    """
    Ncont, total_columns = matrix.shape
    Nhc = (total_columns - 3) // Ndof  # Calculate number of blocks (Nhc)
    transformed_matrix = np.zeros((Ncont, Nhc * Nnl + 3))  # Initialize output matrix
    
    for i in range(Nhc):
        # Extract block of width Ndof
        block_start = i * Ndof
        block_end = (i + 1) * Ndof
        block = matrix[:, block_start:block_end]
        
        # Apply the linear transformation
        transformed_block = Q @ block.T  # Q has shape (Nnl, Ndof), block.T has shape (Ndof, Ncont)
        transformed_block = transformed_block.T  # Resulting shape: (Ncont, Nnl)
        
        # Place the transformed block in the new matrix
        new_block_start = i * Nnl
        new_block_end = (i + 1) * Nnl
        transformed_matrix[:, new_block_start:new_block_end] = transformed_block
    
    # Copy the last three columns untouched
    transformed_matrix[:, -3:] = matrix[:, -3:]
    
    return transformed_matrix

def process_eigenpairs(eigvals, eigevs):
    """
    Process eigenvalues and eigenvectors:
    1. Make all eigenvalues positive.
    2. Sort eigenvalues in ascending order.
    3. Arrange eigenvectors to match the sorted eigenvalues.

    Parameters:
        eigvals (numpy.ndarray): Eigenvalues, shape (n,).
        eigevs (numpy.ndarray): Eigenvectors, shape (m, n).

    Returns:
        sorted_eigvals (numpy.ndarray): Processed eigenvalues, shape (n,).
        sorted_eigevs (numpy.ndarray): Eigenvectors corresponding to sorted eigenvalues, shape (m, n).
    """
    # Make eigenvalues positive
    eigvals = np.abs(eigvals)
    
    # Sort eigenvalues and get sorted indices
    sorted_indices = np.argsort(eigvals)
    sorted_eigvals = eigvals[sorted_indices]
    
    # Sort eigenvectors to match sorted eigenvalues
    sorted_eigevs = eigevs[:, sorted_indices]
    
    return sorted_eigvals, sorted_eigevs
    