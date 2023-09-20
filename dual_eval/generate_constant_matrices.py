import numpy as np
import scipy.sparse as sparse
import generate_proj_mat as proj

def get_lin_proj_mat(N_z, N_omega, N_proj, proj_type = "diagonal", real = True):
    """
    Gives a set of matrices of size (3*N_z + 1)*N_omega \times\ (3*N_z + 1)*N_omega
    that are linear terms of the Lagrangian of the optimization problem. This set
    includes only the projection matrices, since linear matrices in the Lagrangian
    only have projection matrices.

    Args:
        N_z(int): Size of the vector z
        N_omega(int): Size of the discretized frequency domain
        N_proj(int): Number of projections
        proj_type(string): The type of projection needed. 2 are supported for now:
            - diagonal: projection matrices are on small diagonals
            - element: projection is on every element of the matrix

    returns:
        a[[complex64]]: List of matrices containing linear projection matrix for each 
    """
    if proj_type == "diagonal":
        proj_list, proj_list_conj = proj.diag_proj_unity(N_omega, N_proj, real)
    elif proj_type == "element":
        proj_list, proj_list_conj = proj.element_proj_unity(N_omega, real)
    proj_mat_diff_proj = []
    proj_conj_mat_diff_proj = []
    for j in range(N_proj):
        proj_mat_for_diff_z = []
        proj_conj_mat_for_diff_z = []
        # The size of the matrix sould be 3*N_z + 1...
        for k in range(3*N_z):
            zero_right = sparse.csr_matrix(((3*N_z + 1)*N_omega, N_omega))
            zero_left_lower = sparse.csr_matrix((N_omega, 3*N_z*N_omega))
            if k == 0:
                proj_list[j].resize((3*N_z*N_omega, 3*N_z*N_omega))
                proj_list_conj[j].resize((3*N_z*N_omega, 3*N_z*N_omega))
                zero_left = sparse.vstack([proj_list[j], zero_left_lower])
                zero_left_conj = sparse.vstack([proj_list_conj[j], zero_left_lower])
                zero = sparse.hstack([zero_left, zero_right])
                zero_conj = sparse.hstack([zero_left_conj, zero_right])
                proj_conj_mat_for_diff_z.append(zero_conj)
                proj_mat_for_diff_z.append(zero)
            else:
                proj_list[j].resize(((3*N_z - k)*N_omega, (3*N_z - k)*N_omega))
                proj_list_conj[j].resize(((3*N_z - k)*N_omega, (3*N_z - k)*N_omega))
                left_upper_corner_zero = sparse.csr_matrix((k*N_omega,k*N_omega))
                left_lower_corner_zero = sparse.csr_matrix(((3*N_z - k)*N_omega, k*N_omega))
                right_upper_corner_zero = sparse.csr_matrix((k*N_omega, (3*N_z - k)*N_omega))
                right_mat = sparse.vstack([right_upper_corner_zero, proj_list[j]])
                right_mat_conj = sparse.vstack([right_upper_corner_zero, proj_list_conj[j]])
                left_mat = sparse.vstack([left_upper_corner_zero, left_lower_corner_zero])
                proj_mat = sparse.hstack([left_mat, right_mat])
                proj_mat_conj = sparse.hstack([left_mat, right_mat_conj])
                proj_mat = sparse.vstack([proj_mat, zero_left_lower])
                proj_mat = sparse.hstack([proj_mat, zero_right])
                proj_mat_conj = sparse.vstack([proj_mat_conj, zero_left_lower])
                proj_mat_conj = sparse.hstack([proj_mat_conj, zero_right])
                proj_mat_for_diff_z.append(proj_mat)
                proj_conj_mat_for_diff_z.append(proj_mat_conj)
        proj_mat_diff_proj.append(proj_mat_for_diff_z)
        proj_conj_mat_diff_proj.append(proj_conj_mat_for_diff_z)
    return proj_mat_diff_proj, proj_conj_mat_diff_proj

def get_quad_terms_diag(N_z, N_omega, N_proj, proj_type = "diagonal", real = True):
    """
    Gives the diagonal matrix blocks of the quadratic matrix. The terms affecting
    the propagators are made of 
    """
    return