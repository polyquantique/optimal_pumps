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
    low_dim_proj, conj_low_dim_proj = proj.basic_proj(N_omega)
    herm_list_proj = []
    skew_herm_list_proj = []
    for k in range(3*N_z):
        zero_right = sparse.csr_matrix(((3*N_z + 1)*N_omega, N_omega))
        zero_left_lower = sparse.csr_matrix((N_omega, 3*N_z*N_omega))
        low_dim_proj.resize(((3*N_z - k)*N_omega, (3*N_z - k)*N_omega))
        conj_low_dim_proj.resize(((3*N_z - k)*N_omega, (3*N_z - k)*N_omega))
        if k == 0:
            zero_left = sparse.vstack([low_dim_proj, zero_left_lower])
            zero_left_antiherm = sparse.vstack([conj_low_dim_proj, zero_left_lower])
            zero = sparse.hstack([zero_left, zero_right])
            zero_conj = sparse.hstack([zero_left_antiherm, zero_right])
            herm_list_proj.append(zero)
            skew_herm_list_proj.append(zero_conj)
        else:
            left_upper_corner_zero = sparse.csr_matrix((k*N_omega,k*N_omega))
            left_lower_corner_zero = sparse.csr_matrix(((3*N_z - k)*N_omega, k*N_omega))
            right_upper_corner_zero = sparse.csr_matrix((k*N_omega, (3*N_z - k)*N_omega))
            right_mat = sparse.vstack([right_upper_corner_zero, low_dim_proj])
            right_mat_conj = sparse.vstack([right_upper_corner_zero, conj_low_dim_proj])
            left_mat = sparse.vstack([left_upper_corner_zero, left_lower_corner_zero])
            proj_mat = sparse.hstack([left_mat, right_mat])
            antiherm_proj_mat = sparse.hstack([left_mat, right_mat_conj])
            proj_mat = sparse.vstack([proj_mat, zero_left_lower])
            proj_mat = sparse.hstack([proj_mat, zero_right])
            antiherm_proj_mat = sparse.vstack([antiherm_proj_mat, zero_left_lower])
            antiherm_proj_mat = sparse.hstack([antiherm_proj_mat, zero_right])
            herm_list_proj.append(proj_mat)
            skew_herm_list_proj.append(antiherm_proj_mat)
    return herm_list_proj, skew_herm_list_proj

def get_dynamics_matrices(N_omega, omega, vp, z):
    """
    Gives a list of off-diagonal block matrices describing the dynamics of the problem.
    The vertical block matrices are multiplied by Hermitian projections and the horizontal
    block matrices multiplied by anti-Hermitian projections.

    Args:
        N_z(int): Size of the vector z
        N_omega(int): Size of the discretized frequency domain
        omega[float]: discretized frequency domain
        vp(float): pump group velocity
        z[float]: discretized position arguments of waveguide

    returns:
        [[complex64]]: List of quadratic terms for the dynamics of the problem
    """
    herm_mat_mul, anti_herm_mat_mul = proj.matmul_green_f_basic_proj(omega, vp, z, N_omega)
    N_z = len(z)
    top = sparse.csr_matrix((3*N_z*N_omega, (3*N_z + 1)*N_omega))
    left = sparse.csr_matrix(((3*N_z + 1)*N_omega, 3*N_z*N_omega))
    real_dynamics_matrices = []
    imag_dynamics_matrices = []
    for i in range(N_z - 1):
        horiz_low_right = herm_mat_mul[i]
        vert_low_right = anti_herm_mat_mul[i].conj().T
        horiz_low_left = sparse.csr_matrix((N_omega, (3*N_z - i - 1)*N_omega))
        vert_top_right = sparse.csr_matrix(((3*N_z - i - 1)*N_omega, N_omega))
        if i > 0:
            for j in range(i):
                horiz_low_right = sparse.hstack([horiz_low_right, herm_mat_mul[i - j]])
                vert_low_right = sparse.vstack([vert_low_right, herm_mat_mul[i - j].conj().T])
                imag_horiz_low_right = sparse.hstack([horiz_low_right, anti_herm_mat_mul[i - j]])
                imag_vert_low_right = sparse.vstack([vert_low_right, anti_herm_mat_mul[i - j].conj().T])
        horiz_low_right = sparse.hstack([horiz_low_right, sparse.csr_matrix((N_omega, N_omega))])
        vert_low_right = sparse.vstack([vert_low_right, sparse.csr_matrix((N_omega, N_omega))])
        imag_horiz_low_right = sparse.hstack([imag_horiz_low_right, sparse.csr_matrix((N_omega, N_omega))])
        imag_vert_low_right = sparse.vstack([imag_vert_low_right, sparse.csr_matrix((N_omega, N_omega))])
        horiz_low = sparse.hstack([horiz_low_left, horiz_low_right])
        vert_right = sparse.vstack([vert_top_right, vert_low_right])
        imag_horiz_low = sparse.hstack([horiz_low_left, imag_horiz_low_right])
        imag_vert_right = sparse.vstack([vert_top_right, imag_vert_low_right])
        horiz = sparse.vstack([top, horiz_low])
        vert = sparse.hstack([left, vert_right])
        imag_horiz = sparse.vstack([top, imag_horiz_low])
        imag_vert = sparse.hstack([left, imag_vert_right])
        real_dynamics_matrices.append(vert + horiz)
        imag_dynamics_matrices.append(imag_horiz + imag_vert)
    return real_dynamics_matrices, imag_dynamics_matrices
