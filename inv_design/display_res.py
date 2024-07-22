import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import scipy

def get_JSA(x, U):
    """
    Gives the JSA matrix by singular value decomposition and extracting 
    the weights of all Schmidt modes.

    Args:
        x (array[float]): vector the contains all of desired frequencies
        U (array[float]): element of group SU(1,1) 
    returns:
        array[float]: JSA matrix
    """
    N = len(U)
    dw = (x[len(x) - 1] - x[0]) / (len(x) - 1)
    Uss = U[0 : N // 2, 0 : N // 2]
    Uiss = U[N // 2 : N, 0 : N // 2]
    M = jnp.matmul(Uss, (jnp.conj(Uiss).T))
    L, s, Vh = scipy.linalg.svd(M)
    Sig = jnp.diag(s)
    D = jnp.arcsinh(2 * Sig) / 2
    JSA = jnp.abs(L @ D @ Vh) / dw
    return JSA
    
def plot_pump(pumps, omega, pump_labels):
    """
    Plots the real and negative part of the pump
    
    Args:
        pump (array[complex]): complex valued vector containing the pump
        omega (array[float]): vector containing the frequencies
        pump_labels [str]: list of strings labeling each pump
    returns:
        None
    """
    plt.figure(figsize = (12, 8))
    plt.title("Pump as a function of the frequency")
    plt.xlabel("Frequency (arbitrary units)")
    plt.ylabel("Amplitude of the pump (arbitrary units)")
    for i in range(len(pumps)):
        plt.plot(omega, jnp.real(pumps[i]), label = pump_labels[i])
    plt.legend()
    return