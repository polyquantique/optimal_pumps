import jax 
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import numpy as np

@partial(jit, static_argnums=(1,))
def get_hankel_matrix(pump, N_omega):
    """
    Returns the complex valued matrix representing the pump.
    
    Args: 
        pump (array[float]): vector including the real and imaginary part of the pump, its size is 4*N_omega - 2
        N_omega (int): size of the matrices
    returns:
        array[float]: Hankel matrix representing the pump
    """
    real_proj = jnp.hstack([jnp.eye(2*N_omega - 1), jnp.zeros((2*N_omega - 1, 2*N_omega - 1))])
    imag_proj = jnp.hstack([jnp.zeros((2*N_omega - 1, 2*N_omega - 1)), jnp.eye(2*N_omega - 1)])
    complex_pump = real_proj@pump + 1.j*imag_proj@pump
    starts = np.arange(0, len(complex_pump) - N_omega + 1)
    return vmap(lambda start:jax.lax.dynamic_slice(complex_pump, (start,), (N_omega,)))(starts)

def get_submatrices(pump, N_omega, G, H, l):
    """
    Returns the blocks of the matrix to effect the Bogoliubov transform on the annihilation and creation operators
    of the signal and idler modes

    Args:
        pump (array[float]): vector including the real and imaginary part of the pump, its size is 4*N_omega - 2
        N_omega (int): size of the matrices
        G (array[complex]): matrix containing phase matching conditions on signal state
        H (array[complex]): matrix containing phase matching conditions on idler state
        l (float): length of the waveguide
    returns:
        array[complex]: output matrix
    """
    F = get_hankel_matrix(pump, N_omega)
    Q = jnp.block([[G, F], [-jnp.conj(F).T, -jnp.conj(H).T]])
    U = jax.scipy.linalg.expm(1j*Q*l)
    U_ss = U[:N_omega, :N_omega]
    U_is = U[:N_omega, N_omega:]
    U_si = U[N_omega:, :N_omega]
    U_ii = U[N_omega:,N_omega:]
    return U_ss, U_is, U_si, U_ii

def get_observables(pump, N_omega, G, H, l):
    """
    Gives the observables, namely the Schmidt number and the average photon pair number
    to optimize.

    Args:
        pump (array[float]): vector including the real and imaginary part of the pump, its size is 4*N_omega - 2
        N_omega (int): size of the matrices
        G (array[complex]): matrix containing phase matching conditions on signal state
        H (array[complex]): matrix containing phase matching conditions on idler state
        l (float): length of the waveguide
    returns:
        N_value (float): the average number of photon pairs created
        schmidt_number (float): the schmidt number corresponding to all the parameters
    """
    _, U_is, _, _ = jax.jit(get_submatrices, static_argnums=(1,))(pump, N_omega, G, H, l)
    N_matrix = jnp.matmul(jnp.conj(U_is), U_is.T)
    N_value = jnp.trace(N_matrix)
    schmidt_number = (N_value**2)/(jnp.trace(jnp.matmul(N_matrix, N_matrix)))
    return jnp.real(N_value), jnp.real(schmidt_number)

def get_centre_freq_pump(pump, omega):
    """
    Gives the expected value of the pump

    Args:
        pump (array[float]): vector including the real and imaginary part of the pump, its size is 4*N_omega - 2
        omega (array[float]): vector of discretized frequency domain
    returns:
        (float): expected value of the pump
    """
    N_omega = len(omega)
    extended_omega = jnp.linspace(omega[0], omega[-1], 2*N_omega - 1)
    real_pump = pump[:2*N_omega - 1]
    imag_pump = pump[2*N_omega - 1:]
    return extended_omega.conj().T@(jnp.abs(real_pump)/jnp.sum(jnp.abs(real_pump))), extended_omega.conj().T@(jnp.abs(imag_pump)/jnp.sum(jnp.abs(imag_pump)))

def problem(pump, omega, G, H, l, y_N, k):
    """
    Gives the objective function with the penalty terms

    Args:
        pump (array[float]): vector including the real and imaginary part of the pump, its size is 4*N_omega - 2
        omega (array[float]): vector of discretized frequency domain
        G (array[complex]): matrix containing phase matching conditions on signal state
        H (array[complex]): matrix containing phase matching conditions on idler state
        l (float): length of the waveguide
        y_N (float): targeted mean photon number
        k (int): penalty value for penalty method
    returns:
        (float): value of objective function with the penalty term
    """
    N_omega = len(omega)
    extended_omega = jnp.linspace(omega[0], omega[-1], 2*N_omega - 1)
    N_value, schmidt_number = get_observables(pump, N_omega, G, H, l)
    penalty_y = (jnp.real(N_value) - y_N)**2
    real_pump = pump[:len(pump)//2]
    imag_pump = pump[len(pump)//2:]
    mean_loss_real = (extended_omega**2)@jnp.abs(real_pump)/jnp.sum(jnp.abs(real_pump))
    mean_loss_imag = (extended_omega**2)@jnp.abs(imag_pump)/jnp.sum(jnp.abs(imag_pump))
    loss = jnp.real(schmidt_number) - 1 + .05*(mean_loss_imag+mean_loss_real)
    loss_centre_real, loss_centre_imag = get_centre_freq_pump(pump, omega)
    return loss + k*(penalty_y+ loss_centre_real**2 + loss_centre_imag**2)