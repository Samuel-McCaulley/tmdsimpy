import numpy as np

from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.nlforces.bouc_wen import BoucWenForce
from tmdsimpy.nlforces.arctangent_stiffness import ArctangentStiffness
from tmdsimpy.nlforces.sigmoid_stiffness import SigmoidStiffness
import tmdsimpy.utils.harmonic as hutils
from scipy.stats import linregress



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
      Should be either 'iwan' or 'bouc-wen' or 'arctangent'.
    - parameters (np.ndarray): Array of model-specific parameters for tuning the hysteresis behavior.
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
    assert (model == 'iwan' or model == 'bouc-wen' or model == 'arctangent' or model == 'sigmoid'), 'Not tested for other models'
    
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
    
    elif model == 'arctangent':
        arc_force = ArctangentStiffness(np.array([[1]]), np.array([[1]]), parameters[0], parameters[1], parameters[2])

        cst = np.ones([Nt*loops, Nhc])
        
        forces = arc_force.local_force_history(displacement, velocity, h, cst, X0[0])[0]
    elif model == 'sigmoid':
        sig_force = SigmoidStiffness(np.array([[1]]), np.array([[1]]), parameters[0], parameters[1], parameters[2])

        cst = np.ones([Nt*loops, Nhc])
        
        forces = sig_force.local_force_history(displacement, velocity, h, cst, X0[0])[0]
    return time, displacement, forces

def paramexp(lparams, lpsci, base = 10):
    return [base ** lparams[i] if lpsci[i] == 1 else lparams[i] for i in range(len(lparams))]

def paramlog(params, lpsci, base = 10):
    return [np.emath.logn(base, params[i]) if lpsci[i] == 1 else params[i] for i in range(len(params))]


def first_order_exponential(u0, uI, tau, t, negative_control = True):
    '''
    Returns a first order exponential system of form 
    
    u(t) = u0 + (uI - u0)exp(-t/tau)
    
    negative control requires inputs less than 0 are assigned to u0
    '''
    if negative_control and t < 0: return u0
    
    return uI + (u0 - uI)* np.exp(-t/tau)


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


def identify_linear_regime(x, y, tolerance=0.01):
    """
    Identify the largest linear regime in the data where x and y are affine.

    Parameters:
        x (numpy.ndarray): Data for the x-axis.
        y (numpy.ndarray): Data for the y-axis.
        tolerance (float): Maximum allowable deviation (R^2 threshold) from linearity.

    Returns:
        slope (float): The slope of the largest linear regime.
        bounds (tuple): A tuple (start_idx, end_idx) representing the index bounds of the largest linear regime.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length.")

    best_r2 = 0
    best_bounds = None
    best_slope = None

    # Test all possible subarrays
    for start_idx in range(n):
        for end_idx in range(start_idx + 2, n + 1):  # Minimum size of 2 for regression
            x_window = x[start_idx:end_idx]
            y_window = y[start_idx:end_idx]

            # Check for duplicate x values in the window
            if len(np.unique(x_window)) < 2:
                continue  # Skip this window as regression is undefined

            # Perform linear regression
            slope, intercept, r_value, _, _ = linregress(x_window, y_window)
            r2 = r_value**2

            # Check if this window is linear enough and larger than the current best
            if r2 > best_r2 and r2 > 1 - tolerance:
                best_r2 = r2
                best_bounds = (start_idx, end_idx - 1)  # Use inclusive indices
                best_slope = slope

    if best_bounds is None:
        raise ValueError("No linear regime found within the specified tolerance.")

    return best_slope, best_bounds


def harmonic_norm(coeffs: np.ndarray, ndof: int, logbase=10):
    """
    Compute per-DOF norms from harmonic coefficients.

    Parameters:
    - coeffs: numpy array of shape (Ncont, M), where M = total coefficients per row
    - ndof: number of degrees of freedom
    - logbase: base of logarithm used in amplitude column (10 for log10, np.e for natural)

    Returns:
    - norms: array of shape (Nhc, Ndof) with per-DOF norms
    """
    coeffs = np.atleast_2d(coeffs)
    Ncont, total_len = coeffs.shape
    metadata = coeffs[:, -3:]  # frequency, excitation_ratio, amplitude
    data = coeffs[:, :-3]

    # Determine number of harmonics
    nharmonics = (data.shape[1] - ndof) // (2 * ndof)

    # Initialize per-DOF energy array
    energy = np.zeros((Ncont, ndof))

    # h0 terms
    h0 = data[:, :ndof]
    energy += h0**2

    # harmonic terms
    for h in range(nharmonics):
        base = ndof + h * 2 * ndof
        hcos = data[:, base : base + ndof]
        hsin = data[:, base + ndof : base + 2 * ndof]
        energy += 0.5 * (hcos**2 + hsin**2)

    # sqrt of energy gives base norm per DOF
    base_norms = np.sqrt(energy)

    # apply amplitude scaling per row
    log_amp = metadata[:, -1]  # shape (Nhc,)
    if logbase == 10:
        amp_scale = 10 ** log_amp
    else:
        amp_scale = np.exp(log_amp)

    # Reshape amp_scale for broadcasting: (Nhc, 1)
    amp_scale = amp_scale[:, np.newaxis]

    return base_norms * amp_scale

def nonlinear_harmonic_norm(phys_coeffs, Q, logbase=10):
    """
    Compute nonlinear harmonic norms from physical harmonic coefficients and transformation Q.

    Parameters:
        phys_coeffs (np.ndarray): (Ncont, Ndof * Nhc + 3) array in physical coordinates.
        Q (np.ndarray): (Nnl, Ndof) nonlinear transformation matrix.
        Ndof (int): Number of physical DOFs.
        Nnl (int): Number of nonlinear DOFs (rows of Q).
        logbase (float): Log base for amplitude (10 by default).

    Returns:
        np.ndarray: (Nhc, Nnl) matrix of norms in nonlinear coordinates.
    """
    Nnl = Q.shape[0]
    Ndof = Q.shape[1]
    nonlin_coeffs = transform_to_nonlinear(phys_coeffs, Q, Ndof, Nnl)
    return harmonic_norm(nonlin_coeffs, Nnl, logbase=logbase)


def calculate_secant_stiffness(displacement, force):
    """
    Calculate single secant stiffness value from force-displacement data
    
    Parameters:
    displacement (array-like): Array of displacement values
    force (array-like): Array of corresponding force values
    
    Returns:
    float: Secant stiffness value (slope from origin to maximum displacement point)
    """
    d = np.asarray(displacement)
    f = np.asarray(force)
    
    # Find index of maximum absolute displacement
    max_idx = np.argmax(d)
    min_idx = np.argmin(d)
    
    # Get corresponding displacement and force
    d_max = d[max_idx]
    d_min = d[min_idx]
    f_max = f[max_idx]
    f_min = f[min_idx]

    return (f_max - f_min)/ (d_max - d_min)
