import jax 
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import numpy as np

def get_complex_array(v):
    """
    Returns a complex array with real part the first half of the input vector
    and the imaginary part the second half of the input vector.
    
    Args:
        v (array[float]): vector to be transformed into complex
    returns:
        array([complex]): output vector
    """
    real = v[:len(v)//2]
    imag = v[len(v)//2:]
    return real + 1j*imag
@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    """
    Returns a matrix of size (size+1, size) with each element ij being an
    addition of the terms a_i and a_j.
    
    Args: 
        a (array[float]): vector that spawns the output matrix
        size (int): length of vector divided by 2
    returns:
        array[float]: output matrix
    """
    a = get_complex_array(a)
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)
def get_U_matrix(a, size: int, alpha, G, H, l):
    """
    Find the U matrix that is the gives the relationship between annihilation operator
    of the signal mode and the creation operator of the idler mode for at a position z
    and at an initial position.

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[complex]: output matrix
    """
    F = alpha*moving_window(a, size)[:-1]
    Q = jnp.block([[G, F], [-jnp.conj(F).T, -jnp.conj(H).T]])
    U = jax.scipy.linalg.expm(1j*Q*l)
    return U
def get_submatrix(a, size: int, alpha, G, H, l):
    """
    Gives submatrix from U matrix to calculate the Schmidt number and the mean photon number.

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[complex]: Submatrix of the U matrix
    """
    U = get_U_matrix(a, size, alpha, G, H, l)
    N = len(U)
    U_is = U[0:N//2, N//2:N]
    return U_is
def get_observables(a, size: int, alpha, G, H, l):
    """
    Gives the observables, namely the Schmidt number and the average photon pair number
    to optimize.

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        N_value (float): the average number of photon pairs created
        schmidt_number (float): the schmidt number corresponding to all the parameters
    """
    U_is = get_submatrix(a, size, alpha, G, H, l)
    N_matrix = jnp.matmul(jnp.conj(U_is), U_is.T)
    N_value = jnp.trace(N_matrix)
    schmidt_number = (N_value**2)/(jnp.trace(jnp.matmul(N_matrix, N_matrix)))
    return jnp.real(N_value), jnp.real(schmidt_number)
def get_euclidean_loss_K(a, size: int, alpha, G, H, l, y_K):
    """
    Gives the euclidean distance between the Schmidt number ot the system and 
    the desired value. 

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_K (float): desired value for Schmidt number
    returns:
        float: euclidean distance
    """
    N_value, schmidt_number = get_observables(a, size, alpha, G, H, l)
    loss = (jnp.real(schmidt_number) - y_K)**2
    """
    if jnp.nan_to_num(loss) != 0:
        return loss
    else: 
        raise ValueError("Loss is nan")
        """
    return loss
def get_euclidean_loss_N(a, size: int, alpha, G, H, l, y_N):
    """
    Gives the euclidean distance between the number of pairs ot the system and 
    the desired value. 

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value for number of pairs
    returns:
        float: euclidean distance
    """
    N_value, schmidt_number = get_observables(a, size, alpha, G, H, l)
    # Replace by a loss that gies the hard penalty (inf for loss!=0 and min for loss = 0)
    loss = (N_value-y_N)**2
    """
    if jnp.nan_to_num(loss) != 0:
        return loss
    else: raise ValueError("Loss is nan")
    """
    return loss
def get_total_loss(a, size: int, alpha, G, H, l, y_N, y_K):
    """
    Gives the total loss. 

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value for number of pairs
        y_K (float): desired value for Schmidt number
    returns:
        float:  total loss
    """
    loss_K = get_euclidean_loss_K(a, size, alpha, G, H, l, y_K)
    loss_N = get_euclidean_loss_N(a, size, alpha, G, H, l, y_N)
    return loss_K + loss_N
def update(a, size: int, alpha, G, H, l, y_N, y_K, lr = 0.1): 
    """
    Updates the a vector through back-propagation

    Args:
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
        y_N (float): desired value for number of pairs
        y_K (float): desired value for Schmidt number
        lr (float): learninng rate for backpropagation
    returns:
        array[float]: the updated a vector 
    """
    return a - lr*(jax.grad(get_euclidean_loss_K)(a, size, alpha, G, H, l, y_K) + jax.grad(get_euclidean_loss_N, argnums=(0))(a, size, alpha, G, H, l, y_N))
def get_JSA(x, a, size: int, alpha, G, H, l):
    """
    Gives the JSA matrix by singular value decomposition and extracting 
    the weights of all Schmidt modes.

    Args:
        x (array[float]): vector the contains all of desired frequencies
        a (array[float]): vector that spawns the F matrix
        size (int): length of a divided by 2
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        array[float]: JSA matrix
    """
    U = get_U_matrix(a, size, alpha, G, H, l)
    N = len(U)
    dw = (x[len(x) - 1] - x[0]) / (len(x) - 1)
    Uss = U[0 : N // 2, 0 : N // 2]
    Uiss = U[N // 2 : N, 0 : N // 2]
    M = jnp.matmul(Uss, (jnp.conj(Uiss).T))
    L, s, Vh = jax.scipy.linalg.svd(M)
    Sig = np.diag(s)
    D = np.arcsinh(2 * Sig) / 2
    J = np.abs(L @ D @ Vh) / dw
    return J