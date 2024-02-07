import generate_constant_matrices as gen_mat
import generate_proj_mat as proj
import numpy as np
import scipy.sparse as sparse

def generate_dynamics_matrices(omega, z, projection):
    """
    Gives 4 lists of matrices; one for real values of positive quadrature, one for
    imaginary values of positive quadrature, one for real values of negative quadrature
    and one for imaginary values of negative quadrature
    """
    N_omega = len(omega)
    N_z = len(z)
    green_fs = proj.get_green_functions(omega, 0, z)
    plus_real_dyn, plus_imag_dyn = gen_mat.proj_dynamics_mat_propagator(omega, z, "plus", 0, projection)
    minus_real_dyn, minus_imag_dyn = gen_mat.proj_dynamics_mat_propagator(omega, z, "minus", 0, projection)
    real_lin_proj_mat, imag_lin_proj_mat = gen_mat.get_proj_lin_basic(N_omega, N_z, projection)
    cst_terms = [(projection@green_fs[i]).trace() for i in range(N_z)]
    real_plus_M_mat = [sparse.bmat([[plus_real_dyn[i], -0.5*real_lin_proj_mat[N_z + i]], [-0.5*real_lin_proj_mat[N_z + i].conj().T, (np.real(cst_terms[i])/N_omega)*sparse.eye(N_omega)]]) for i in range(N_z)]
    imag_plus_M_mat = [sparse.bmat([[plus_imag_dyn[i], -0.5*imag_lin_proj_mat[N_z + i]],[-0.5*imag_lin_proj_mat[N_z + i].conj().T, (np.imag(cst_terms[i])/N_omega)*sparse.eye(N_omega)]]) for i in range(N_z)]
    real_minus_M_mat = [sparse.bmat([[minus_real_dyn[i], -0.5*real_lin_proj_mat[2*N_z + i]], [-0.5*real_lin_proj_mat[2*N_z + i].conj().T, (np.real(cst_terms[i])/N_omega)*sparse.eye(N_omega)]]) for i in range(N_z)]
    imag_minus_M_mat = [sparse.bmat([[minus_imag_dyn[i], -0.5*imag_lin_proj_mat[2*N_z + i]], [-0.5*imag_lin_proj_mat[2*N_z + i].conj().T, (np.imag(cst_terms[i])/N_omega)*sparse.eye(N_omega)]]) for i in range(N_z)]
    return real_plus_M_mat, imag_plus_M_mat, real_minus_M_mat, imag_minus_M_mat

def generate_base_matrices(N_omega, N_z, n):
    """
    Gives a list of matrices for the inequalities governing square of Frobenius norm of J for all z
    and matrix giving the mean number of photon pair per pulse
    """
    real_lin_J, _ = gen_mat.get_proj_lin_basic(N_omega, N_z, sparse.eye(N_omega))
    diag_mat = gen_mat.get_proj_quad_diag(N_omega, N_z, sparse.eye(N_omega))
    lin_J = 0.5*real_lin_J[N_z - 1]
    quad_J = diag_mat[:N_z]
    M_mat_ineq_constr = [sparse.bmat([[quad_J[i], sparse.csc_matrix(((3*N_z + 1)*N_omega, N_omega))],[sparse.csc_matrix(((3*N_z + 1)*N_omega, N_omega)).conj().T, -((n**2)/N_omega)*sparse.eye(N_omega)]]) for i in range(N_z)]
    M_mat_photon_nbr = sparse.bmat([[sparse.csc_matrix(((3*N_z + 1)*N_omega,(3*N_z + 1)*N_omega)), lin_J],[lin_J.conj().T, -(n/N_omega)*sparse.eye(N_omega)]])
    return M_mat_ineq_constr, M_mat_photon_nbr

def generate_matrices_herm(N_omega, N_z, projection):
    """
    Gives list of matrices giving the Hermitian constraints of J matrices
    """
    _, imag_lin_J = gen_mat.get_proj_lin_basic(N_omega, N_z, projection)
    lin_J = imag_lin_J[:N_z]
    M_mat_herm = [sparse.bmat([[sparse.csc_matrix(((3*N_z + 1)*N_omega,(3*N_z + 1)*N_omega)), 0.5*lin_J[i]],[0.5*lin_J[i].conj().T, sparse.csc_matrix((N_omega, N_omega))]]) for i in range(N_z)]
    return M_mat_herm

def generate_def_J(N_omega, N_z, projection):
    """
    Generate matrix corresponding to the definition of the J matrix
    """
    W_plus_diag = gen_mat.get_proj_quad_diag(N_omega, N_z, projection)[N_z:2*N_z]
    W_minus_diag = gen_mat.get_proj_quad_diag(N_omega, N_z, projection)[2*N_z:3*N_z]
    lin_generator, _ = gen_mat.get_proj_lin_basic(N_omega, N_z, projection)
    P_mat = [-4*lin_generator[i] for i in range(N_z)]
    Q_mat = [W_plus_diag[i] + W_minus_diag[i] for i in range(N_z)]
    M_mat = [sparse.bmat([[Q_mat[i], 0.5*P_mat[i]],[0.5*P_mat[i].conj().T, (-2*projection.trace()/N_omega)*sparse.eye(N_omega)]]) for i in range(N_z)]
    return M_mat

def generate_beta_cov_constraint(omega, N_z, covariance):
    """
    Gives the matrix corresponding to the constraint on the covariance of the pump
    """
    N_omega = len(omega)
    omega_mat = sparse.csc_matrix(np.diag(omega**2))
    beta_quad = sparse.bmat([[sparse.csc_matrix((3*N_z*N_omega, 3*N_z*N_omega)), sparse.csc_matrix((3*N_z*N_omega, N_omega))],[sparse.csc_matrix((N_omega, 3*N_z*N_omega)), omega_mat]])
    beta_quad_constr = sparse.bmat([[beta_quad, sparse.csc_matrix(((3*N_z + 1)*N_omega, N_omega))],[sparse.csc_matrix((N_omega, (3*N_z + 1)*N_omega)), (-covariance/N_omega)*sparse.eye(N_omega)]])
    return beta_quad_constr

def generate_constr_Z_mat(N_omega, N_z):
    """
    Make sure that matrices Z are unitary
    """
    N_matrices = []
    kron_delta = []
    for i in range(N_omega):
        for j in range(i + 1):
            mat = np.zeros((N_omega, N_omega))
            mat[i, j] = 1
            N_matrices.append(sparse.bmat([[sparse.csc_matrix(((3*N_z + 1)*N_omega, (3*N_z + 1)*N_omega)), np.zeros(((3*N_z + 1)*N_omega, N_omega))],[np.zeros((N_omega, (3*N_z + 1)*N_omega)), sparse.csc_matrix(mat + mat.T)]]))
            if i == j:
                kron_delta.append(2)
            else:
                kron_delta.append(0)
    return N_matrices, kron_delta