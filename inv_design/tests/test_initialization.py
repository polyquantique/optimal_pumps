import get_initialization_cond as init
import numpy as np

np.random.seed(0)

def test_GVD():
    """
    Test for the function get_constants from get_initialization for inverse design.
    Fails if the group velocity matching of signal and idler is not respected
    """
    vp = 5*np.random.rand(1)
    l = np.random.rand(1)
    wi = -3
    wf = 3
    Np = np.random.rand(1)
    alpha_phase = 2*np.pi*np.random.rand(1)
    alpha, G, H = init.get_constants(vp, l, wi, wf, Np, alpha_phase)
    assert np.allclose(G + H, np.diag((-2/vp)*np.linspace(wi, wf, len(G))))