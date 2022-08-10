import optimization_SPDC
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.scipy.optimize as jax_opt
import get_initialization_cond as init_value

Np = 0.2
N = 401
wi = -7
wf = 7
x = np.linspace(wi, wf, N)
l = 1
alpha, G, H = init_value.get_constants(l, wi, wf, Np, N)

def test_low_gain_width(wi, wf, l, x, N, Np, alpha, G, H):
    
    y_N = 0.001
    y_K = 1
    initial_params = []
    initial_params.append(0.02*jnp.exp(-(jnp.linspace(2*wi, 2*wf, 2*N))**2)/2)
    initial_params.append(0.1*jnp.exp(-(jnp.linspace(2*wi, 2*wf, 2*N))**2)/2)
    initial_params = jnp.reshape(jnp.array(initial_params), 2*len(initial_params[0]))
    optimized = (jax_opt.minimize(optimization_SPDC.get_total_loss, initial_params, args = (N, alpha, G, H, l, y_N, y_K), method = "BFGS")).x
    complex_opt = optimization_SPDC.get_complex_array(optimized)
    complex_init = optimization_SPDC.get_complex_array(initial_params)
    if len(np.nonzero(complex_opt)[0]) < len(np.nonzero(complex_init)[0]):
        indices = np.nonzero(complex_opt)
    else:
        indices = np.nonzero(complex_init)
    
    