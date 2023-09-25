import numpy as np
import scipy.sparse as sparse

def basic_proj(N_omega):
    """
    Most basic projection. The Hermitian is the identity and the 
    skew-Hermitian is the identity times sqrt(-1).
    
    Args:
        N_omega(int): The size of discretized frequency domain

    returns:
        a[[complex64]]: Hermitian projection matrix
        b[[complex64]]: Skew-Hermitian projection matrix
    """
    return sparse.csr_matrix(np.eye(N_omega).astype("complex64")), sparse.csr_matrix(1.j*np.eye(N_omega))

def diag_proj_unity(N_omega, N_proj, real = True):
    """
    Gives a list of projection matrices that have 1 or i on 
    a diagonal and 0 everywhere else

    Args:
        N_omega(int): The size of discretized frequency domain
        N_proj(int): The number of projection matrices
        real(bool): True if the the diagonals have 1. False if they have i 

    returns:
        a[[complex64]]: List of hermitian projection matrices
        b[[complex64]]: List of skew-Hermitian projection matrices
    """
    hermitian_proj_matrices = []
    antiherm_proj_matrices = []
    for i in range(N_proj):
        half_projection = sparse.csr_matrix(np.eye(N_omega, k= i).astype("complex64"))
        antiherm_proj_matrices.append(1.j*half_projection + 1.j*half_projection.T)
        if real == True:
            hermitian_proj_matrices.append(half_projection + half_projection.T)
        else:
            hermitian_proj_matrices.append(1.j*half_projection - 1.j*half_projection.T)
    return hermitian_proj_matrices, antiherm_proj_matrices

def element_proj_unity(N_omega, real = True):
    """
    Gives a list of projection matrices that have 1 or i at
    one position of the matrix and 0 everywhere else

    Args:
        N_omega(int): The size of discretized frequency domain
        real(bool): True if the the diagonals have 1. False if they have i 

    returns:
        a[[complex64]]: List of Hermitian projection matrices 
        b[[complex64]]: List of anti-Hermitian projection matrices
    """
    proj_matrices = []
    conj_proj_matrices = []
    for i in range(N_omega):
        for j in range(i):
            proj_mat = np.zeros((N_omega, N_omega)).astype("complex64")
            proj_mat[i][j] = 1
            proj_antiherm = 1.j*proj_mat + 1.j*proj_mat.T
            if real == True:
                proj_herm = proj_mat + proj_mat.T
            else:
                proj_herm = 1.j*proj_mat - 1.j*proj_mat.T
            proj_matrices.append(sparse.csr_matrix(proj_herm))
            conj_proj_matrices.append(sparse.csr_matrix(conj_proj_matrices))
    return proj_matrices, conj_proj_matrices

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
    green_f = [sparse.csr_matrix(np.diag(np.exp(1.j*(omega/delta_v)*z[i]))) for i in range(len(z))]
    return green_f

def matmul_green_f_basic_proj(omega, vp, z, N_omega):
    """
    Gives the matrix product between the projection matrix and individual 
    elements of the Green's function to be used in dynamics constraints.
    Note that the Green's function is placed in the reverse order compared
    to how they are generated.

    Args:
        omega[float]: discretized frequency domain
        vp(float): pump group velocity
        z[float]: discretized position arguments of waveguide
        N_omega(int): The size of discretized frequency domain
    
    returns:
        a[[complex64]]: List containing N_z terms that are matrix 
                        product between Hermitian projection matrices
                        and Green's function at every space step.
        b[[complex64]]: List containing N_z terms that are matrix 
                        product between skew-Hermitian projection matrices
                        and Green's function at every space step.
    """
    green_f = get_green_functions(omega, vp, z)
    herm_proj, antiherm_proj = basic_proj(N_omega)
    N_z = len(z)
    herm_mat_mul = []
    anti_mat_mul = []
    for i in range(N_z):
        herm_mat_mul.append(green_f[i]@herm_proj)
        anti_mat_mul.append(green_f[i]@antiherm_proj)
    return herm_mat_mul, anti_mat_mul