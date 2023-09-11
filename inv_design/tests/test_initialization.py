from inv_design import get_initialization_cond as init
import numpy as np
import scipy
import jax

np.random.seed(0)

def test_GVD():
    """
    Test for the function get_constants from get_initialization for inverse design.
    Fails if the group velocity matching of signal and idler is not respected
    """
    vp = -5*jax.random.uniform(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,))[0]
    l = jax.random.uniform(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,))[0]
    wi = -3
    wf = 3
    Np = jax.random.uniform(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,))[0]
    alpha_phase = 2*np.pi*jax.random.uniform(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,))[0]
    alpha, G, H = init.get_constants(vp, l, wi, wf, Np, alpha_phase)
    assert np.allclose(G, -H)

def test_initial_seed_hermite():
    """
    Test to see if the get_initialization_array function in get_initialization_cond 
    gives indeed the Hermitian polynomial of the right order multiplied by a
    Gaussian.
    """
    polyn_order = jax.random.randint(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,), minval=1, maxval=10)[0]
    amplit = jax.random.uniform(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,))[0]
    width = jax.random.uniform(key = jax.random.PRNGKey(np.random.randint(999)),shape = (1,))[0]
    phase = 2*0.2*np.pi
    omega = np.linspace(-5, 5, 402)
    real_pump = np.cos(phase)*amplit*scipy.special.hermite(polyn_order, monic = True)(omega)*np.exp(-((omega)**2)/width)
    imag_pump = np.sin(phase)*amplit*scipy.special.hermite(polyn_order, monic = True)(omega)*np.exp(-((omega)**2)/width)
    pump = np.concatenate((real_pump ,imag_pump))
    test_pump = init.get_initialization_array([polyn_order, amplit, width, phase], -2.5, 2.5, N = 201)
    assert np.allclose(pump, test_pump, rtol = 0.0001)