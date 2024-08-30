import numpy as np
import jax.numpy as jnp
import scipy

def get_constants(vp, l, wi, wf, N = 401):
    """
    Gives the values of the U matrix that do not change with backpropagation.
    All nonlinear interactions beyond second order are ignored.
    
    Args:
        vp(float): pump group velocity
        l(float): length of the waveguide
        wi(float): starting frequency difference from center frequency
        wf(float): ending frequency difference from center frequency
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
    G = jnp.diag((1/vs - 1/vp)*x)
    H = jnp.diag((1/vi - 1/vp)*x)
    return G, H
    