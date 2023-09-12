import jax
import jax.numpy as jnp
import scipy
import optimization_SPDC as opt
import numpy as np

def get_V_W_matrices(theta, size:int, alpha, G, H, l):
    """
    Find the U matrix that is the answer to the dynamics and output the V/W matrices that are
    the propagators of the system in time domain when uncoupled.

    Args:
        theta (array[float]): problem parameters. Here would be every element of the pump vector
        size (int): length of the frequency domain
        alpha (float): constant including power of pump, group velocity of all modes, etc.
        G (array[complex]): matrix giving the dependency of a_s(z) on a_z(z_o)
        H (array[complex]): matrix giving the dependency of a_i(z) dagger on a_i(z_o) dagger
        l (float): length of the waveguide
    returns:
        V (array[complex]): propagator for the r"$X_ +$" operator in time
        W (array[complex]): propagator for the r"$X_ -$" operator in time
    """
    U_ss, U_is, U_si, U_ii = opt.get_submatrices(theta, size, alpha, G, H, l)
    F = scipy.fft.fft(np.eye(size))
    F_inv = scipy.fft.ifft(np.eye(size))
    V = F@(U_ss + U_si)@F_inv
    W = F@(U_ss - U_si)@F_inv
    #V = U_ss + U_si
    #W = U_ss - U_si
    return V, W