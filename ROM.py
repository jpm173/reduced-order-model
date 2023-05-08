# This package is capable of building a reduced basis with a modified Gram-Schmidt algorithm; using empirical interpolation to build a
# B matrix that can be used to rebuild a waveform from nodes and a reduced basis; and finally use a polynomial fit to build waveforms from input
# parameters outside of the training space.

import numpy as np
import scipy.integrate as integrate
from itertools import repeat

def get_norm(u, x):
    """Gets the norm of a vectors

    Args:
        u (array): vector to be normed
        x (array): space over which to norm vector

    Returns:
        float: norm of vector
    """    
    return np.sqrt(get_inner(u,u,x))


def get_normalized(u, x):
    """Normalizes vector

    Args:
        u (array): vector to be normalized
        x (array): space over which to normalize

    Returns:
        array: normalized vector
    """    
    return u / get_norm(u,x)


def get_inner(u, v, x):
    """Gets the inner product of two vectors using integration techniques

    Args:
        u (array): first vector to be used in inner product         
        v (array): second vector to be used in inner product
        x (array): space to perform inner product over

    Returns:
        float : inner product of two vectors given a space to integrate over
    """    
    return np.real(integrate.simpson(u*np.conjugate(v), x))   


def get_projection(u, v, x):
    """Projects one vector onto another using an inner product

    Args:
        u (array): first vetcor
        v (array): second vector
        x (array): space to project over

    Returns:
        array: projected vector
    """    
    proj = get_inner(u, v, x) * u / get_inner(u, u, x)

    return proj


def get_error(RB, function, x):
    func_norm = get_norm(function, x).real
    error = 0.0
    for k in range(len(RB)):
        error += np.abs(get_inner(RB[k], function, x))**2.0
    return func_norm**2 - error


def get_greedy_errors(RB, functions, x, pool=None):
    """Generates greedy errors from a reduced basis (RB) and input functions

    Args:
        RB (array): reduced basis
        functions (array): input functions
        x (array): domain of functions

    Returns:
        array: greedy errors
    """
    if pool:
        errors = np.array(pool.starmap(get_error, zip(repeat(RB), functions, repeat(x))))
    else:
        errors = np.array([get_error(RB, f, x) for f in functions])
        
    return errors


def get_reduced_basis(input_funcs, x, error=1e-10, verbose=False, pool=None, rel_error=True):
    """Performs a Gram-Schmidt algorithm on a set of input functions

    Args:
        input_funcs (array): array of input functions to be orthonormalised
        error (float): the desired level of accuracy when performing a Gram-Schmidt algorithm
        x (array): domain of the input functions
        verbose (bool, optional): If TRUE:. Defaults to False.

    Returns:
        array: array of orthonormalised basis functions
    """    
    verboseprint = print if verbose else lambda *a: None
    
    functions    = np.copy(input_funcs)
    greedy_error = error

    # We select the first (arbitrary) function and normalise it. This is our first basis vector in the reduced basis
    RB = [get_normalized(functions[0], x)]

    vectors = functions
    for j in range(len(vectors)):
        vectors[j] -= get_projection(RB[0], vectors[j], x)

    k                 = 0
    potential_gerror  = get_greedy_errors(RB, functions, x, pool=pool)
    greedy_error      = np.max(potential_gerror)
    if rel_error:
        greedy_error_init = greedy_error
    else:
        greedy_error_init = 1.0
    greedy_error /= greedy_error_init
    verboseprint("Iteration {k} : {err}".format(k=k, err=greedy_error))

    while greedy_error > error:    
        # Do Gram-Schmidt orthonormalisation
        new_BV = vectors[np.argmax(potential_gerror)]
        RB.append(get_normalized(new_BV, x))
        k     += 1

        # Regenerate vectors
        for j in range(len(vectors)):
            vectors[j] -= get_projection(RB[k], vectors[j], x)
        
        potential_gerror = get_greedy_errors(RB, functions, x, pool=pool)
        greedy_error     = np.max(potential_gerror) / greedy_error_init
        
        verboseprint("Iteration {k} : {err}".format(k=k, err=greedy_error))

    return np.asarray(RB)


def empirical_interpolation(basis, verbose=False):
    """Executes an empirical interpolation algorithm that is able to return a matrix of elements that encodes a compressed waveform

    Args:
        basis (array): an orthogonal reduced basis that will be used to reconstruct the training space
        verbose (bool, optional): TRUE if extra computation information is required (for debugging purposes). Defaults to False.

    Returns:
        array: B-matrix which can be used to form the interpolant (a compressed waveform)
    """    
    verboseprint = print if verbose else lambda *a: None

    nodes = [np.argmax(np.abs(basis[0]))]
    V = basis[0]
    V = V[nodes[0]]

    for i in range(1, len(basis)):
        verboseprint("Iteration {i}: {idx}".format(i=i, idx=nodes[i-1]))
        #Compute B elements
        if i == 1:
            iV = np.reciprocal(V)
        else:
            iV = np.linalg.pinv(V)  # Pseudo-inverse (pinv instead of inv) seems to be more stable for our data

        B = np.dot(np.transpose(basis[:i]), iV)
        
        #Interpolant
        e_nodes = basis[i]
        e_nodes = e_nodes[nodes]
        interpolant = np.dot(B, e_nodes)

        #Maximum residual selection
        residuals = interpolant - basis[i]
        arg_max = np.argmax(np.abs(residuals))
        nodes.append(arg_max)

        #Regenerates V-matrix
        e = np.array(basis[:i+1])
        V = e[:,nodes]
        V = np.transpose(V)
    
    #Determine final B matrix that can rec recreate total interpolant
    final_V = np.linalg.inv(V)
    B = np.dot(np.transpose(basis), final_V)
    
    #Returning both B and nodes; B encodes all the information but nodes are useful for extra computations
    return np.asarray(nodes), B


def get_polyfit(parameter, function, nodes, order=2):
    """Generates a polynomial for a specific parameter

    Args:
        parameter (float): input parameter
        function (array): function data used to generate fit
        nodes (array): nodes (mostly time-like) where function is evaluated
        order (int, optional): order of the polynomial fit. Defaults to 2.

    Returns:
        array: array with the polynomial coefficients
    """    
    fits = []
    for j in nodes:
        fit = np.polyfit(parameter, function[:, j], order)
        fits.append(fit)
    
    return np.asarray(fits)


def fit_waveform(parameter, fit):
    """Builds a waveform from a polynomial fit

    Args:
        parameter (float): input parameter to build waveform for
        fit (array): coefficients of polynomial fit

    Returns:
        float: interpolated value of function at given node
    """    
    x = 0
    order = len(fit)-1
    for i in range(len(fit)):
        x += fit[i]*parameter**(order-i)
    
    return x 

if __name__ == '__main__':
    main()