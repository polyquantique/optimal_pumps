import jax 
import scipy
import numpy as np
import optimization_SPDC as opt

def test_hankel():
    N = 10
    vector_real = np.random.random(2*N - 1)
    vector_imag = np.random.random(2*N - 1)
    vector = list(vector_real) + list(vector_imag)
    hankel_mat = opt.get_hankel_matrix(vector, N)
    true_hankel = scipy.linalg.hankel((vector_real + 1.j*vector_imag)[:N], (vector_real + 1.j*vector_imag)[N - 1:])
    assert np.allclose(hankel_mat, true_hankel)

def test_submatrices():
    """
    Test fails if U_ss, U_si, U_is or U_ii are not of size N
    """
    N = 10
    vector_real = np.random.random(2*N - 1)
    vector_imag = np.random.random(2*N - 1)
    vector = list(vector_real) + list(vector_imag)
    U_ss, U_si, U_is, U_ii = opt.get_submatrices