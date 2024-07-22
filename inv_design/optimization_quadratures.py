import jax 
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import numpy as np
import jaxopt
import jax.scipy as jscipy

@partial(jit, static_argnums=(1,))
def get_hankel_matrix(pump, N_omega):
    """
    Returns the real valued matrix representing the pump.
    
    Args: 
        pump[float]: problem parameters. Length is 2*N_omega - 1
        N_omega(int): dimension of the propagators
    returns:
        array[float]: Hankel matrix representing the pump
    """
    starts = np.arange(0, len(pump) - N_omega + 1)
    return vmap(lambda start:jax.lax.dynamic_slice(pump, (start,), (N_omega,)))(starts)

def get_propagators(pump, N_omega, delta_k, l):
    """
    Returns the propagators for the equations of dynamics with pseudo-quadratures
    
    Args:
        pump[float]: problem parameters. Length is 2*N_omega - 1
        N_omega(int): dimension of the propagators
        delta_k [[complex]]: matrix containing the elements about the symmetric-group-velocity matching
        l (float): length of the waveguide
    """
    W_plus = jscipy.linalg.expm(l*(delta_k + get_hankel_matrix(pump, N_omega)))
    W_minus = jscipy.linalg.expm(l*(delta_k - get_hankel_matrix(pump, N_omega)))
    return W_plus, W_minus

def get_observables(pump, N_omega, delta_k, l):
    """
    Returns the mean number of photon pairs per pulse and the Schmidt number
    
    Args:
        pump[float]: problem parameters. Length is 2*N_omega - 1
        N_omega(int): dimension of the propagators
        delta_k [[complex]]: matrix containing the elements about the symmetric-group-velocity matching
        l(float): length of the waveguide
    """
    W_plus, W_minus = get_propagators(pump, N_omega, delta_k, l)
    J = 0.25*(W_plus.conj().T@W_plus + W_minus.conj().T@W_minus - 2*jnp.eye(N_omega))
    N = jnp.real(jnp.trace(J))
    K = jnp.real((N**2)/jnp.trace(J.conj().T@J))
    return N, K

def problem(pump, omega, delta_k, l, y_N, k):
    """
    Gives the penalized objective function

    Args:
        pump[float]: problem parameters. Length is 2*N_omega - 1
        omega[float]: discretized frequency domain
        alpha(float): constant including power of pump, group velocity of all modes, etc.
        H[[float]]: matrix containing the elements that are not in the integral in the dynamics equations
        l(float): length of the waveguide
    """
    N_omega = len(omega)
    N_value, schmidt_number = get_observables(pump, N_omega, delta_k, l)
    loss_n = (jnp.real(N_value) - y_N)**2
    mean_loss = (jnp.linspace(omega[0], omega[-1], 2*N_omega - 1))@jnp.abs(pump)/jnp.sum(jnp.abs(pump))
    var_loss = ((jnp.linspace(omega[0], omega[-1], 2*N_omega - 1))**2)@jnp.abs(pump)/jnp.sum(jnp.abs(pump))
    loss = jnp.real(schmidt_number) - 1 + .01*var_loss
    return loss + k*(loss_n + mean_loss**2)


