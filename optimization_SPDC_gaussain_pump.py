import jax 
import jax.numpy as jnp
import numpy as np
import scipy 
from functools import partial
from jax import jit, vmap

@partial(jit, static_argnums=(1,))
def moving_window(w, size: int):
    """
    Returns a matrix of size (size+1, size) with each element ij being an
    addition of the terms a_i and a_j.
    
    Args: 
        w (array[float]): frequency vector that create the pump amplitude
        size (int): length of vector divided by 2
    returns:
        array[float]: output matrix
    """
    starts = jnp.arange(len(w) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(w, (start,), (size,)))(starts)
def get_gaussian(theta, w):
    """
    Returns a matrix that represents pulse function f, such as every element of
    the matrix corresponds to f(x_n, y_m), where n and m are the column and row 
    positions.
    
    Args: 
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
    returns:
        array[float]: output matrix
    """
    a, tau, phi = theta
    gaussian = a*jnp.exp(-(tau*w)**2)*jnp.exp(1j*phi)
    return moving_window(gaussian, len(gaussian)//2)[:-1]
def get_gaussian_pump(theta, w):
    """
    Returns a vector that represents the gaussian pump created by the parameters
    
    Args: 
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
    returns:
        array[float]: output pump vector
    """
    a, tau, phi = theta
    gaussian = a*jnp.exp(-(tau*w)**w)*jnp.exp(1j*phi)
    return gaussian
def get_U_matrix(theta, w, alpha, G, H, l):
    """
    Find the U matrix that is the gives the relationship between annihilation operator
    of the signal mode and the creation operator of the idler mode for at a position z
    and at an initial position.

    Args:
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[complex]: output matrix
    """
    F = alpha*get_gaussian(theta, w)
    Q = jnp.block([[G, F], [-jnp.conj(F).T, -jnp.conj(H).T]])
    U = jax.scipy.linalg.expm(1j*Q*l)
    return U
def get_submatrix(theta, w, alpha, G, H, l):
    """
    Gives submatrix from U matrix to calculate the Schmidt number and the mean photon number.

    Args:
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        U_ss,U_is, U_si, U_ii (array[complex]): Submatrices of the U matrix
    """
    U = get_U_matrix(theta, w, alpha, G, H, l)
    N = len(U)
    U_ss = U[0:N//2, 0:N//2]
    U_is = U[0:N//2, N//2:N]
    U_si = U[N//2:N, 0:N//2]
    U_ii = U[N//2:N, N//2:N]
    return U_ss, U_is, U_si, U_ii
def get_observable(theta, w, alpha, G, H, l):
    """
    Gives the observables, namely the Schmidt number and the average photon pair number
    to optimize.

    Args:
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        N_value (float): the average number of photon pairs created
        schmidt_number (float): the schmidt number corresponding to all the parameters
    """
    U_ss, U_is, U_si, U_ii = get_submatrix(theta, w, alpha, G, H, l)
    N_matrix = jnp.matmul(jnp.conj(U_is), U_is.T)
    N_value = jnp.trace(N_matrix)
    schmidt_number = (N_value**2)/(jnp.trace(jnp.matmul(N_matrix, N_matrix)))
    return jnp.real(N_value), jnp.real(schmidt_number)
def get_loss(theta, w, alpha, G, H, l, y_N):
    """
    Gives the euclidean distance between the Schmidt number ot the system and 
    the desired value. 

    Args:
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value of pair number
    returns:
        float: loss value
    """
    N_value, schmidt_number = get_observable(theta, w, alpha, G, H, l)
    loss = jnp.real(schmidt_number) - 1 + (jnp.real(N_value) - y_N)**2
    return loss
def get_JSA(theta, w, alpha, G, H, l):
    """
    Gives the JSA matrix by singular value decomposition and extracting 
    the weights of all Schmidt modes.

    Args:
        theta (array[float]): vector that contains the parameters to optimize
        w (array[float]): vector containing frequencies that will go into the gaussian
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[float]: JSA matrix
    """
    U = get_U_matrix(theta, w, alpha, G, H, l)
    N = len(U)
    x = jnp.linspace(w[0], w[-1], len(w)//2)
    dw = (x[len(x) - 1] - x[0]) / (len(x) - 1)
    Uss = U[0 : N // 2, 0 : N // 2]
    Usi = U[0 : N // 2, N // 2 : N]
    Uiss = U[N // 2 : N, 0 : N // 2]
    M = jnp.matmul(Uss, (jnp.conj(Uiss).T))
    L, s, Vh = jax.scipy.linalg.svd(M)
    Sig = np.diag(s)
    D = np.arcsinh(2 * Sig) / 2
    J = np.abs(L @ D @ Vh) / dw
    return J