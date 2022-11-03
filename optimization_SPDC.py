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
        theta (array[float]): problem parameters. Length and values will depend on pump_shape
        size (int): length of pump vector
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
        theta (array[float]): problem parameters. Length and values will depend on pump_shape
        size (int): length of a divided by 2
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
        theta (array[float]): problem parameters. Length and values will depend on pump_shape
        size (int): length of a divided by 2
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
        theta (array[float]): problem parameters. Length and values will depend on pump_shape
        size (int): length of a divided by 2
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
def get_loss(theta, size: int, alpha, G, H, l, y_N):
    """
    Gives the euclidean distance between the Schmidt number ot the system and 
    the desired value. 

    Args:
        theta (array[float]): problem parameters. Length and values will depend on pump_shape
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value for photon pairs
    returns:
        float: loss of objective function
    """
    N_value, schmidt_number = get_observables(theta, size, alpha, G, H, l)
    loss = jnp.real(schmidt_number) - 1 + (jnp.real(N_value) - y_N)**2
    return loss
def get_JSA(x, theta, size: int, alpha, G, H, l):
    """
    Gives the JSA matrix by singular value decomposition and extracting 
    the weights of all Schmidt modes.

    Args:
        x (array[float]): vector the contains all of desired frequencies
        theta (array[float]): problem parameters. Length and values will depend on pump_shape
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[float]: JSA matrix
    """
    U = get_U_matrix(theta, size, alpha, G, H, l)
    N = len(U)
    dw = (x[len(x) - 1] - x[0]) / (len(x) - 1)
    Uss = U[0 : N // 2, 0 : N // 2]
    Uiss = U[N // 2 : N, 0 : N // 2]
    M = jnp.matmul(Uss, (jnp.conj(Uiss).T))
    L, s, Vh = jax.scipy.linalg.svd(M)
    Sig = np.diag(s)
    D = np.arcsinh(2 * Sig) / 2
    JSA = np.abs(L @ D @ Vh) / dw
    return JSA