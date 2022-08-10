import numpy as np

def symmetric_v(vp, sigma, l, a):
    """
    Gives group velocity matching conditions for a SPDC process. Note that it gives idler 
    and signal group velocity depending on what pulse width is used. Closer look at those
    conditions can be viewed in the paper by Graffitti et al. 
    (Design considerations for high-purity heralded single-photon sources)
    
    Args:
        vp(float): pump group velocity
        sigma(float): width of the pump
        l(float): length of the waveguide
        a(float): term to get the best pump at sigma width
        
    returns:
        vs(float): signal group velocity
        vi(float): idler group velocity
    """
    vi = vp / (1 - 2 * a * vp / (l * sigma))
    vs = vp / (1 + 2 * a * vp / (l * sigma))
    return vs, vi
def get_constants(l, wi, wf, Np, N = 401):
    """
    Gives the values of the U matrix that do not change with backpropagation.
    All nonlinear interactions beyond second order are ignored.
    
    Args:
        l(float): length of the waveguide
        wi(float): starting frequency difference from center frequency
        wf(float): ending frequency difference from center frequency
        Np(float): initial power of the pump
        N(int): resolution of the F matrix
        
    returns:
        alpha(float): term to be multiplied with pump envelop
        G(array[float]): upper left submatrix for Q block matrix
        H(array[float]): lower right submatrix for Q block matrix
    """
    x = np.linspace(wi, wf, N)
    vp = 0.1
    sigma = 1
    a = 1.61/1.13
    vs, vi = symmetric_v(vp, sigma, l, a)
    alpha = np.sqrt(Np)*(x[len(x) - 1] - x[0])/(len(x) - 1)/(np.sqrt(2 * np.pi * vs * vi * vp))
    G = np.diag((1/vs - 1/vp)*x)
    H = np.diag((1/vi - 1/vp)*x)
    return alpha, G, H