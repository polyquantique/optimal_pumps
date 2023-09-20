import numpy as np
import scipy.sparse as sparse

def diag_proj_unity(N_omega, N_proj, real = True):
    """
    Gives a list of projection matrices that have 1 or i on 
    a diagonal and 0 everywhere else

    Args:
        N_omega(int): The size of discretized frequency domain
        N_proj(int): The number of projection matrices
        real(bool): True if the the diagonals have 1. False if they have i 

    returns:
        a[[complex64]]: List of projection matrices 
        b[[complex64]]: List of Hermitian conjugate of projection matrices
    """
    zero = N_proj//2
    if real == True:
        proj_matrices = [sparse.csr_matrix(np.eye(N_omega, k= i - zero).astype("complex64")) for i in range(N_proj)]
        proj_matrices_conj = [sparse.csr_matrix(np.eye(N_omega, k= zero - i).astype("complex64")) for i in range(N_proj)]
    else:
        proj_matrices = [sparse.csr_matrix(1.j*np.eye(N_omega, k= i - zero)) for i in range(N_proj)]
        proj_matrices_conj = [sparse.csr_matrix(-1.j*np.eye(N_omega, k= zero - i).astype("complex64")) for i in range(N_proj)]
    return proj_matrices, proj_matrices_conj

def element_proj_unity(N_omega, real = True):
    """
    Gives a list of projection matrices that have 1 or i at
    one position of the matrix and 0 everywhere else

    Args:
        N_omega(int): The size of discretized frequency domain
        real(bool): True if the the diagonals have 1. False if they have i 

    returns:
        a[[complex64]]: List of projection matrices 
        b[[complex64]]: List of Hermitian conjugate of projection matrices
    """
    proj_matrices = []
    proj_matrices_conj = []
    for i in range(N_omega):
        for j in range(N_omega):
            proj = np.zeros((N_omega, N_omega)).astype("complex64")
            proj_conj = np.zeros((N_omega, N_omega)).astype("complex64")
            if real == True:
                proj[i][j] = 1
                proj_conj[j][i] = 1
            else:
                proj[i][j] = 1.j
                proj_conj[j][i] = -1.j
            proj_matrices.append(sparse.csr_matrix(proj))
            proj_matrices_conj.append(sparse.csr_matrix(proj_conj))
    return proj_matrices, proj_matrices_conj

def get_green_functions(omega, vp, z):
    """
    Gives the Green's function in frequency domain.

    Args:
        omega[float]: discretized frequency domain
        vp(float): pump group velocity
        z[float]: discretized position arguments of waveguide

    returns:
        [[complex]]: list containing the Green's function evaluated at
                    different points of waveguide
    """
    a = 1.61/1.13
    vs = vp / (1 + 2 * a * vp / z[-1])
    delta_v = vp*vs/(vp - vs)
    green_f = [np.diag(np.exp(1.j*(omega/delta_v)*z[i])) for i in range(len(z))]
    return green_f