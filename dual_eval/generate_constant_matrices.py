import numpy as np
import scipy.sparse as sparse
import generate_proj_mat as proj

def get_proj_lin_basic(N_omega, N_z):
    """
    Gives a list of Hermitian projection matrices that is the dimension of linear 
    terms of the Lagrangian and a list of skew-Hermitian projection matrices.

    Args:
        N_omega(int): Size of the discretized frequency domain
        N_z(int): Size of the vector z

    returns:
        a[[complex64]]: List of Hermitian projection matrices
        b[[complex64]]: List of skew-Hermitian projection matrices
    """
    herm_list_proj = []
    skew_herm_list_proj = []
    for i in range(3*N_z):
        herm_proj, skew_herm_proj = proj.basic_proj(N_omega)
        herm_proj.resize((N_omega, (3*N_z + 1 - i)*N_omega))
        skew_herm_proj.resize((N_omega, (3*N_z + 1 - i)*N_omega))
        left = sparse.csr_matrix((N_omega, i*N_omega))
        herm_list_proj.append(sparse.hstack([left, herm_proj]))
        skew_herm_list_proj.append(sparse.hstack([left, skew_herm_proj]))
    return herm_list_proj, skew_herm_list_proj

def get_proj_quad_basic(N_omega, N_z):
    """
    Gives a list of Hermitian and skew-Hermitian matrices the dimension 
    suitable for quadratic term of the Lagrangian.

    Args:
        N_omega(int): Size of the discretized frequency domain
        N_z(int): Size of the vector z

    returns:
        a[[complex64]]: List of Hermitian projection matrices
        b[[complex64]]: List of skew-Hermitian projection matrices
    """
    herm_list_proj = []
    skew_herm_list_proj = []
    for i in range(3*N_z):
        herm_proj, skew_herm_proj = proj.basic_proj(N_omega)
    return

def get_dynamics_matrices(N_z, N_omega, N_proj, omega, vp, z, proj_type = "diagonal", real = True):
    """
    Gives the off-diagonal block matrices describing the dynamics of the problem


    """
    hermitian_proj, antiherm_proj = get_lin_proj_mat(N_z, N_omega, N_proj, proj_type = "diagonal", real = True)
    green_f = proj.get_green_functions(omega, vp, z)
    
    return
# Todo: Change get_lin_proj_mat to quad_proj_mid
def get_lin_proj_mat(N_z, N_omega, N_proj, proj_type = "diagonal", real = True):
    """
    Gives a set of matrices of size (3*N_z + 1)*N_omega \times\ (3*N_z + 1)*N_omega
    that are the projection matrices, but scaled to the dimension containing all of
    the variables

    Args:
        N_z(int): Size of the vector z
        N_omega(int): Size of the discretized frequency domain
        N_proj(int): Number of projections
        proj_type(string): The type of projection needed. 2 are supported for now:
            - diagonal: projection matrices are on small diagonals
            - element: projection is on every element of the matrix
        real(bool): True if the the diagonals have 1. False if they have i 

    returns:
        a[[complex64]]: List of matrices containing projection matrices for
                        each block of variables
    """
    if proj_type == "diagonal":
        proj_list, conj_proj_list = proj.diag_proj_unity(N_omega, N_proj, real)
    elif proj_type == "element":
        proj_list, conj_proj_list = proj.element_proj_unity(N_omega, real)
    proj_mat_diff_proj = []
    proj_mat_diff_proj_antiherm = [] 
    for j in range(len(proj_list)):
        proj_mat_for_diff_z = []
        proj_mat_antiherm_for_diff_z = []
        # The size of the matrix sould be 3*N_z + 1...
        for k in range(3*N_z):
            zero_right = sparse.csr_matrix(((3*N_z + 1)*N_omega, N_omega))
            zero_left_lower = sparse.csr_matrix((N_omega, 3*N_z*N_omega))
            proj_list[j].resize(((3*N_z - k)*N_omega, (3*N_z - k)*N_omega))
            conj_proj_list[j].resize(((3*N_z - k)*N_omega, (3*N_z - k)*N_omega))
            if k == 0:
                zero_left = sparse.vstack([proj_list[j], zero_left_lower])
                zero_left_antiherm = sparse.vstack([conj_proj_list[j], zero_left_lower])
                zero = sparse.hstack([zero_left, zero_right])
                zero_conj = sparse.hstack([zero_left_antiherm, zero_right])
                proj_mat_for_diff_z.append(zero)
                proj_mat_antiherm_for_diff_z.append(zero_conj)
            else:
                left_upper_corner_zero = sparse.csr_matrix((k*N_omega,k*N_omega))
                left_lower_corner_zero = sparse.csr_matrix(((3*N_z - k)*N_omega, k*N_omega))
                right_upper_corner_zero = sparse.csr_matrix((k*N_omega, (3*N_z - k)*N_omega))
                right_mat = sparse.vstack([right_upper_corner_zero, proj_list[j]])
                right_mat_conj = sparse.vstack([right_upper_corner_zero, conj_proj_list[j]])
                left_mat = sparse.vstack([left_upper_corner_zero, left_lower_corner_zero])
                proj_mat = sparse.hstack([left_mat, right_mat])
                antiherm_proj_mat = sparse.hstack([left_mat, right_mat_conj])
                proj_mat = sparse.vstack([proj_mat, zero_left_lower])
                proj_mat = sparse.hstack([proj_mat, zero_right])
                antiherm_proj_mat = sparse.vstack([antiherm_proj_mat, zero_left_lower])
                antiherm_proj_mat = sparse.hstack([antiherm_proj_mat, zero_right])
                proj_mat_for_diff_z.append(proj_mat)
                proj_mat_antiherm_for_diff_z.append(antiherm_proj_mat)
        proj_mat_diff_proj.append(proj_mat_for_diff_z)
        proj_mat_diff_proj_antiherm.append(proj_mat_antiherm_for_diff_z)
    return proj_mat_diff_proj, proj_mat_diff_proj_antiherm