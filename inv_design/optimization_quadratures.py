import jax 
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import numpy as np
import jaxopt

@partial(jit, static_argnums=(1,))
def moving_window(theta, size: int):
    """
    Returns a matrix of size (size+1, size) with each ij element being equal to mn mn element if
    i+j = m+n.
    
    Args: 
        theta[float]: problem parameters. Length is 2*size
        size (int): length of pump vector
    returns:
        array[float]: Hankel matrix representing the pump
    """
    starts = jnp.arange(len(theta) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(theta, (start,), (size,)))(starts)
def get_propagators(theta, size:int, alpha, H, l):
    """
    Returns the propagators for the equations of dynamics with pseudo-quadratures
    
    Args:
        theta[float]: problem parameters. Length is 2*size
        size (int): length of pump vector
        alpha(float): constant including power of pump, group velocity of all modes, etc.
        H[[float]]: matrix containing the elements that are not in the integral in the dynamics equations
        l (float): length of the waveguide
    """
    W_plus = jscipy.linalg.expm(l*(1.j*H + alpha*moving_window(theta, size)[:-1]))
    W_minus = jscipy.linalg.expm(l*(1.j*H - alpha*moving_window(theta, size)[:-1]))
    return W_plus, W_minus
def get_observables(theta, size:int, alpha, H, l):
    """
    Returns the mean number of photon pairs per pulse and the Schmidt number
    
    Args:
        theta[float]: problem parameters. Length is 2*size
        size(int): length of pump vector
        alpha(float): constant including power of pump, group velocity of all modes, etc.
        H[[float]]: matrix containing the elements that are not in the integral in the dynamics equations
        l(float): length of the waveguide
    """
    W_plus, W_minus = get_propagators(theta, size, alpha, H, l)
    J = 0.25*(W_plus.conj().T@W_plus + W_minus.conj().T@W_minus - 2*jnp.eye(size))
    N = jnp.trace(J)
    K = (N**2)/jnp.trace(J.conj().T@J)
    return N, K
def get_loss_N(theta, size:int, alpha, H, l, y_N):
    """
    Gives the closeness of the system to the constraint measured in Euclidean distance.

    Args:
        theta (array[float]): problem parameters. Length is 2*size
        size (int): length of pump vector
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        H [float]: matrix containing the elements that are not in the integral in the dynamics equations
        l (float): length of the waveguide
        y_N (float): desired value for photon pairs
    returns:
        float: value of the difference between the mean photon pair number of system and the 
        value wished by the user
    """
    N_value, schmidt_number = get_observables(theta, size, alpha, H, l)
    loss = (jnp.real(N_value) - y_N)**2
    return loss
def get_loss_K(theta, size:int, alpha, H, l, omega):
    """
    Gives the value of the objective function.

    Args:
        theta (array[float]): problem parameters. Length is 2*size
        size (int): length of pump vector
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        H [float]: matrix containing the elements that are not in the integral in the dynamics equations
        l (float): length of the waveguide
        omega (array[float]): frequency domain of the pump
    returns:
        float: value of the objective function
    """
    N_value, schmidt_number = get_observables(theta, size, alpha, H, l)
    # mean_loss adds another objective function that is minimizing the variance.
    # Without this objective function, optimizing a random seed would give a pump located at a distance from
    # central frequency
    mean_loss = jnp.sum(((jnp.abs(omega[1] - omega[0])*omega*jnp.abs(theta))/jnp.linalg.norm(theta))**2)
    loss = jnp.real(schmidt_number) - 1 + 40*mean_loss
    return loss
def get_penalty_loss(theta, size:int, alpha, H, l, omega, y_N, sigma):
    """
    Gives the loss value when using the penalty method.

    Args:
        theta (array[float]): problem parameters. Length is 2*size
        size (int): length of pump vector
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        H [float]: matrix containing the elements that are not in the integral in the dynamics equations
        l (float): length of the waveguide
        omega (array[float]): frequency domain of the pump
        sigma (float): weight of the penalty
    returns:
        float: value of the loss when using penalty method
    """
    loss_K = get_loss_K(theta, size, alpha, H, l, omega)
    loss_N = get_loss_N(theta, size, alpha, H, l, y_N)
    penalty_loss = loss_K + sigma*((jnp.maximum(0, loss_N))**2 + (jnp.maximum(0, - loss_N))**2)
    return penalty_loss
def optimize(theta, size:int, alpha, H, l, omega, y_N, sigma, optimizer, penalty_terms):
    """
    Maximize the spectral purity of heralded photons with the constraint that parameters are elements
    of Lie algebra of the propagator.

    Args:
        theta (array[float]): problem parameters. Length is 2*size
        size (int): length of pump vector
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        H [float]: matrix containing the elements that are not in the integral in the dynamics equations
        l (float): length of the waveguide
        omega (array[float]): frequency domain of the pump
        sigma (float): weight of the penalty
        optimizer (abc.ABCMeta): optimizer used to effect gradient descent
        penalty_terms [float]: penalty weights for constraints
    returns:
        float: value of the loss when using penalty method
    """
    for i in range()