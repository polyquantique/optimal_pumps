import jax 
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import numpy as np
import jaxopt
import jax.scipy as jscipy

def get_gaussian(theta, omega):
    """
    Returns the real Gaussian pump from parameters theta

    Args:
        theta(array[float]): amplitude and width of Gaussian
        omega(array[float]): frequency range of interest
    returns:
        array[float]: Gaussian pump of size 2*N_omega - 1
    """
    amplitude, width = theta
    N_omega = len(omega)
    w_i = omega[0]
    w_f = omega[-1]
    pump = amplitude*jnp.exp(-jnp.linspace(w_i, w_f, 2*N_omega - 1)**2/width)
    return pump

def get_hankel_matrix(theta, omega):
    """
    Returns the real valued matrix representing the pump.
    
    Args: 
        theta(array[float]): amplitude and width of Gaussian
        omega(array[float]): frequency range of interest
    returns:
        array[[float]]: Hankel matrix representing the pump
    """
    pump = get_gaussian(theta, omega)
    N_omega = len(omega)
    starts = np.arange(0, len(pump) - N_omega + 1)
    return vmap(lambda start:jax.lax.dynamic_slice(pump, (start,), (N_omega,)))(starts)

@jit
def get_propagators(theta, omega, delta_k, l):
    """
    Returns the propagators for the equations of dynamics with pseudo-quadratures
    
    Args:
        theta(array[float]): amplitude and width of Gaussian
        omega(array[float]): frequency range of interest
        delta_k [[complex]]: matrix containing the elements about the symmetric-group-velocity matching
        l (float): length of the waveguide
    """
    W_plus = jscipy.linalg.expm(l*(delta_k + get_hankel_matrix(theta, omega)))
    W_minus = jscipy.linalg.expm(l*(delta_k - get_hankel_matrix(theta, omega)))
    return W_plus, W_minus

def get_observables(theta, omega, delta_k, l):
    """
    Returns the mean number of photon pairs per pulse and the Schmidt number
    
    Args:
        theta(array[float]): amplitude and width of Gaussian
        omega(array[float]): frequency range of interest
        delta_k [[complex]]: matrix containing the elements about the symmetric-group-velocity matching
        l(float): length of the waveguide
    """
    N_omega = len(omega)
    W_plus, W_minus = jax.jit(get_propagators)(theta, omega, delta_k, l)
    J = 0.25*(W_plus.conj().T@W_plus + W_minus.conj().T@W_minus - 2*jnp.eye(N_omega))
    N = jnp.real(jnp.trace(J))
    K = jnp.real((N**2)/jnp.trace(J.conj().T@J))
    return N, K

def problem(theta, omega, delta_k, l, y_N, k):
    """
    Gives the penalized objective function

    Args:
        theta(array[float]): amplitude and width of Gaussian
        omega(array[float]): frequency range of interest        alpha(float): constant including power of pump, group velocity of all modes, etc.
        H[[float]]: matrix containing the elements that are not in the integral in the dynamics equations
        l(float): length of the waveguide
    """
    _, width = theta
    N_value, schmidt_number = get_observables(theta, omega, delta_k, l)
    loss_n = (jnp.real(N_value) - y_N)**2
    loss = jnp.real(schmidt_number) - 1 
    return loss + k*(loss_n) + (1/k)*(1/width**2)


