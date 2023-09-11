import numpy as np
import jax.numpy as jnp
import scipy

def get_constants(vp, l, wi, wf, Np, alpha_phase, N = 401):
    """
    Gives the values of the U matrix that do not change with backpropagation.
    All nonlinear interactions beyond second order are ignored.
    
    Args:
        vp(float): pump group velocity
        l(float): length of the waveguide
        wi(float): starting frequency difference from center frequency
        wf(float): ending frequency difference from center frequency
        Np(float): initial power of the pump
        alpha_phase(float): the phase of the coefficient multiplying the pump
        N(int): resolution of the F matrix
        
    returns:
        alpha(float): term to be multiplied with pump envelop
        G(array[float]): upper left submatrix for Q block matrix
        H(array[float]): lower right submatrix for Q block matrix
    """
    x = np.linspace(wi, wf, N)
    sigma = 1
    a = 1.61/1.13
    vi = vp / (1 - 2 * a * vp / (l * sigma))
    vs = vp / (1 + 2 * a * vp / (l * sigma))
    alpha = jnp.exp(1.j*alpha_phase)*jnp.sqrt(Np)*(x[len(x) - 1] - x[0])/(len(x) - 1)/(np.sqrt(2 * np.pi * vs * vi * vp))
    G = jnp.diag((1/vs - 1/vp)*x)
    H = jnp.diag((1/vi - 1/vp)*x)
    return alpha, G, H
def get_initialization_array(init_params, wi, wf, method = "hermite", N=401):
    """
    Gives array that will initialize the pump for backpropagation. Depending
    on the method, different types of initialization parameters must be used.
    
    Args:
        method(str): shape of initialization pump. Choices are:
            -"hermite": initialize hermitian polynomials multiplied by
                        a gaussian. Order of polynomial, gaussian amplitude,
                        width and complex phase must be contained in init_params
            -"constant": initialize a constant pump with amplitude equal to init_params
        init_params(list): list of initialization parameters specific to each method parameters
        vp(float): pump group velocity
        l(float): length of the waveguide
        wi(float): starting frequency difference from center frequency
        wf(float): ending frequency difference from center frequency
        Np(float): initial power of the pump
        N(int): resolution of the F matrix
        
    returns:
        array[complex]: initial pump guess
    """
    a = []
    if method == "hermite":
        order, amplitude, width, phase = init_params
        real_amplitude = amplitude*np.cos(phase)
        imag_amplitude = amplitude*np.sin(phase)
        x = jnp.linspace(2*wi, 2*wf, 2*N)
        hermite_poly = scipy.special.hermite(order, monic = True)(x)
        a.append(real_amplitude*hermite_poly*jnp.exp(-(x**2)/width))
        a.append(imag_amplitude*hermite_poly*jnp.exp(-(x**2)/width))
        a = jnp.array(a)
        a = jnp.reshape(a, 2*len(a[0]))
    if method == "constant":
        a = init_params*jnp.ones(4*N)
    return a