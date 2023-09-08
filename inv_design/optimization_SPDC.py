import jax 
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import numpy as np

def get_complex_array(theta):
    """
    Returns a complex array with real part the first half of the input vector
    and the imaginary part the second half of the input vector.
    
    Args:
        theta (array[float]): vector of parameters
    returns:
        array([complex]): complex vector initializing the optimization
    """
    real = theta[:len(theta)//2]
    imag = theta[len(theta)//2:]
    return real + 1j*imag
@partial(jit, static_argnums=(1,))
def moving_window(theta, size: int):
    """
    Returns a matrix of size (size+1, size) with each element ij being an
    addition of the terms a_i and a_j.
    
    Args: 
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
    returns:
        array[float]: Hankel matrix representing the pump
    """
    a = get_complex_array(theta)
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)
def get_U_matrix(theta, size: int, alpha, G, H, l):
    """
    Find the U matrix that is the gives the relationship between annihilation operator
    of the signal mode and the creation operator of the idler mode for at a position z
    and at an initial position.

    Args:
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[complex]: output matrix
    """
    F = alpha*moving_window(theta, size)[:-1]
    Q = jnp.block([[G, F], [-jnp.conj(F).T, -jnp.conj(H).T]])
    U = jax.scipy.linalg.expm(1j*Q*l)
    return U
def get_submatrices(theta, size: int, alpha, G, H, l):
    """
    Gives submatrix from U matrix to calculate the Schmidt number and the mean photon number.

    Args:
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        U_ss,U_is, U_si, U_ii (array[complex]): Submatrices of the U matrix
    """
    U = get_U_matrix(theta, size, alpha, G, H, l)
    N = len(U)
    U_ss = U[:N//2, :N//2]
    U_is = U[:N//2, N//2:N]
    U_si = U[N//2:N, :N//2]
    U_ii = U[N//2:N,N//2:N]
    return U_ss, U_is, U_si, U_ii
def get_observables(theta, size: int, alpha, G, H, l):
    """
    Gives the observables, namely the Schmidt number and the average photon pair number
    to optimize.

    Args:
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        N_value (float): the average number of photon pairs created
        schmidt_number (float): the schmidt number corresponding to all the parameters
    """
    U_ss, U_is, U_si, U_ii = get_submatrices(theta, size, alpha, G, H, l)
    N_matrix = jnp.matmul(jnp.conj(U_is), U_is.T)
    N_value = jnp.trace(N_matrix)
    schmidt_number = (N_value**2)/(jnp.trace(jnp.matmul(N_matrix, N_matrix)))
    return jnp.real(N_value), jnp.real(schmidt_number)
def get_loss_N(theta, size: int, alpha, G, H, l, y_N):
    """
    Gives the closeness of the system to the constraint measured in Euclidean distance.

    Args:
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value for photon pairs
    returns:
        float: value of the difference between the mean photon pair number of system and the 
        value wished by the user
    """
    N_value, schmidt_number = get_observables(theta, size, alpha, G, H, l)
    loss = (jnp.real(N_value) - y_N)**2
    return loss
def get_loss_K(theta, size: int, alpha, G, H, l, omega):
    """
    Gives the value of the objective function.

    Args:
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        omega (array[float]): frequency domain of the pump
    returns:
        float: value of the objective function
    """
    N_value, schmidt_number = get_observables(theta, size, alpha, G, H, l)
    real_theta = theta[:len(theta)//2]
    imag_theta = theta[len(theta)//2:]
    # mean_loss_real and mean_loss_imag add another objective function that is minimizing the variance.
    # Without this objective function, optimizing a random seed would give a pump located at a distance from
    # central frequency
    mean_loss_real = jnp.sum(((jnp.abs(omega[1] - omega[0])*omega*jnp.abs(real_theta))/jnp.linalg.norm(real_theta))**2)
    mean_loss_imag = jnp.sum(((jnp.abs(omega[1] - omega[0])*omega*jnp.abs(imag_theta))/jnp.linalg.norm(imag_theta))**2)
    loss = jnp.real(schmidt_number) - 1 + 40*(mean_loss_imag+mean_loss_real)
    return loss
def get_penalty_loss(theta, size: int, alpha, G, H, l, y_N, omega, sigma):
    """
    Gives the loss value when using the penalty method.

    Args:
        theta (array[float]): problem parameters. Length is 4 times size
        size (int): resolution of the matrices
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value for photon pairs
        omega (array[float]): frequency of the pump
        sigma (float): weight of the penalty
    returns:
        float: value of the loss when using penalty method
    """
    loss_K = get_loss_K(theta, size, alpha, G, H, l, omega)
    loss_N = get_loss_N(theta, size, alpha, G, H, l, y_N)
    penalty_loss = loss_K + sigma*((jnp.maximum(0, loss_N))**2 + (jnp.maximum(0, - loss_N))**2)
    return penalty_loss