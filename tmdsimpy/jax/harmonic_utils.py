"""
Subset of `tmdsimpy.utils.harmonic` using JAX

Autodiff is applied mainly for AFT at this point, thus only some functions
need to be converted to JAX. Other functions are not updated at this point to 
use JAX.

See Also
--------
tmdsimpy.utils.harmonic :
    Baseline implementation of harmonic utility methods without JAX.

"""

import numpy as np

import jax.numpy as jnp

# Imports for decorating functions with jit calls at this level
# See specific function comments for where this cannot be done.
#
# import jax
# from functools import partial

# @partial(jax.jit, static_argnums=(0,1,3)) # May cause excessive recompile, do not use here.
def time_series_deriv(Nt, htuple, X0, order):
    """
    Returns derivative of a time series defined by a set of harmonics
    
    Parameters
    ----------
    Nt : int, power of 2
        Number of times considered, must be even.
        Must be greater than `2*np.array(h).max()`.
    htuple : (H,) tuple of sorted int
        Harmonics considered, 0th harmonic must be first if included.
    X0 : (Nhc, N) numpy.ndarray
        Harmonic Coefficients for columns corresponding to degrees of freedom
        and rows corresponding to different harmonic components.
    order : int
        Order of the derivative returned. 0 is generally displacement, 1 
        is velocity, 2 is acceleration.
    
    Returns
    -------
    x_t : (Nt, N) numpy.ndarray
        Time series of each DOF. Rows are time instants and columns are
        DOFs.
    
    See Also
    --------
    tmdsimpy.utils.harmonic.time_series_deriv :
        Implementation of this function without JAX.
    
    Notes
    -----
    For JAX/JIT calls to this function, the top level function will likely need
    to have `htuple` and `Nt` set to static.
    
    The number of harmonic components is 
    `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`
    
    The normalized time instants between [0,1) for a cycle can be calculated as
    `tau = numpy.linspace(0,1,Nt+1)[:-1]`.
    
    If you have `h` as a numpy.ndarray of harmonic components, you can use
    `htuple = tuple(h)`. The tuple is used here to allow for static arguments
    in JAX compilation.
    
    The following notes are speculation about JIT compilation about this
    function. Unit tests ensure that this function gives appropriate results.
    
    This function cannot be compiled with partial for the current 
    implementation. Rather this function gets different compiled versions for
    the top level function being compiled when AFT uses both displacement
    and velocity. 
    
    If this function was compiled with partial, it would be forced to recompile
    every time that order changed. This could cause excessive recompilation. 
    
    If the top level function gets compiled, it appears that two versions of 
    this are created or this code is inlined into the top level function. 
    This could be verified by profiling the final code to ensure that 
    there are not lots of recompiles during evaluations.
    
    """
    
    h = np.array(htuple)
    
    assert ((np.equal(h, 0)).sum() == 0 or h[0] == 0), 'Zeroth harmonic must be first'
    
    nd = X0.shape[1] # Degrees of Freedom
    Nh = np.max(h)
    
    # Create list including all harmonic components
    X0full = jnp.zeros((2*Nh+1, nd), np.float64)
    if h[0] == 0:
        X0full = X0full.at[0, :].set(X0[0, :])
        X0full = X0full.at[2*h[1:]-1, :].set(X0[1::2, :])
        X0full = X0full.at[2*h[1:], :].set(X0[2::2, :])
        
    else:
        X0full = X0full.at[2*h-1, :].set(X0[0::2, :])
        X0full = X0full.at[2*h, :].set(X0[1::2, :])
        
    # Check that sufficient time is considered
    assert Nt > 2*Nh + 1, 'More times are required to avoid truncating harmonics.'
    
    if order > 0:
        D1 = np.zeros((2*Nh+1, 2*Nh+1))
        
        # If order is static, this for loop should be safe since it can be 
        # calculated at compile time (uses numpy operations)
        # if order is not static, JIT will throw an error.
        
        # Note that if top level functions are compiled, multiple versions of 
        # this can be compiled for different order cases
        
        for k in h[h != 0]:
            # Only rotates the derivatives for the non-zero harmonic components
            cosrows = (k-1)*2 + 1
            sinrows = (k-1)*2 + 2
            
            D1[cosrows, sinrows] = k
            D1[sinrows, cosrows] = -1*k
            
        # This is not particularly fast, consider optimizing this portion.
        #   D could be constructed just be noting if rows flip for odd/even
        #   and sign changes as appropriate.
        D = np.linalg.matrix_power(D1, order)
        
        X0full = D @ X0full
    
    # Extend X0full to have coefficients corresponding to Nt times for ifft
    #   Previous MATLAB implementation did this before rotating harmonics, but
    #   that seems rather inefficient in increasing the size of the matrix 
    #   multiplication
    Nht = int(Nt/2 -1)
    X0full = jnp.vstack((X0full,np.zeros((2*(Nht-Nh), nd)) ))
    Nt = 2*Nht+2

    # Fourier Coefficients    
    Xf = jnp.vstack((2*X0full[0, :], \
         X0full[1::2, :] - 1j*X0full[2::2], \
         jnp.zeros((1, nd)), \
         X0full[-2:0:-2, :] + 1j*X0full[-1:1:-2]))
        
    Xf = Xf * (Nt/2)
         
    assert Xf.shape[0] == Nt, 'Unexpected length of Fourier Coefficients'
    
    x_t = jnp.real(jnp.fft.ifft(Xf, axis=0))
    
    return x_t

# @partial(jax.jit, static_argnums=(0))
def get_fourier_coeff(htuple, x_t):
    """
    Calculates a specific set of Fourier coefficients of a time series.

    Parameters
    ----------
    htuple : (H,) tuple of sorted int
        Harmonics considered, 0th harmonic must be first if included.
    x_t : (Nt, N) numpy.ndarray
        Time series of each DOF. Rows are time instants over a cycle 
        (see Notes). 
        Columns are DOFs.

    Returns
    -------
    v : (Nhc, N) numpy.ndarray
        Containing Fourier coefficients of harmonics `h` (rows) and 
        DOFs (columns).

    See Also
    --------
    tmdsimpy.utils.harmonic.get_fourier_coeff :
        Implementation without JAX support.
    
    Notes
    -----
    The number of harmonic components is 
    `Nhc = tmdsimpy.utils.harmonic.Nhc(h)`
    
    The normalized time instants between [0,1) for a cycle can be calculated as
    `tau = numpy.linspace(0,1,Nt+1)[:-1]`.
    
    If you have `h` as a numpy.ndarray of harmonic components, you can use
    `htuple = tuple(h)`. The tuple is used here to allow for static arguments
    in JAX compilation.
    
    For JAX and JIT, the top level function that calls this should likely be
    compiled with `htuple` as a static argument.
        
    """
    
    h = np.array(htuple)

    Nt, nd = x_t.shape
    Nhc = 2*(h != 0).sum() + (h == 0).sum() # Number of Harmonic Components
    n = h.shape[0] - (h[0] == 0)
    
    assert ((h == 0).sum() == 0 or h[0] == 0), 'Zeroth harmonic must be first'
    
    v = jnp.zeros((Nhc, nd))
    
    xf = jnp.fft.fft(x_t, axis=0)
        
    if h[0] == 0:
        v = v.at[0, :].set(jnp.real(xf[0, :])/Nt)
        zi = 1
    else:
        zi = 0

    # As long as h is treated as static, this is safe for this for loop
    for i in range(n):
        hi = h[i + zi]
        v = v.at[2*i+zi].set(jnp.real(xf[hi, :]) / (Nt/2))
        v = v.at[2*i+1+zi].set(-jnp.imag(xf[hi, :]) / (Nt/2))
    
    return v
