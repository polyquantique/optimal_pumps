import numpy as np
import scipy.sparse as sparse
import scipy

np.random.seed(0)

def quad_proj(N_omega, N_z, x_index, y_index, proj):
    """
    Gives a matrix with the proj matrix on the x_index block on the vertical
    and y_index block on the horizontal
    """
    if x_index >= 2*N_z - 2 or y_index >= 2*N_z - 2:
        raise ValueError("Invalid indices")
    left_upper = sparse.csc_matrix((x_index*N_omega, y_index*N_omega))
    upper_right = sparse.csc_matrix((x_index*N_omega, (2*N_z - 2 - y_index)*N_omega))
    left_lower = sparse.csc_matrix(((2*N_z - 2 - x_index)*N_omega, y_index*N_omega))
    right_lower = sparse.bmat([[proj, sparse.csc_matrix((N_omega, (2*N_z - 3 - y_index)*N_omega))],
                                [sparse.csc_matrix(((2*N_z - 3 - x_index)*N_omega, N_omega)), sparse.csc_matrix(((2*N_z - 3 - x_index)*N_omega,(2*N_z - 3 - y_index)*N_omega))]])
    mat = sparse.bmat([[left_upper, upper_right],
                        [left_lower, right_lower]])
    return mat

def get_lin_matrices(N_z, N_omega, proj):
    """
    List of projection matrices for every propagator
    """
    lin = []
    for i in range(2*N_z - 2):
        lin_mat = proj.copy()
        lin_mat.resize(((2*N_z - 2 - i)*N_omega, N_omega))
        lin_mat = sparse.vstack([sparse.csc_matrix((i*N_omega, N_omega)), lin_mat])
        lin.append(sparse.csc_matrix(lin_mat))
    return lin

def sdr_def_constr(N_omega, N_z, proj):
    """
    Give the constraint that the SDR is linked with the QCQP
    """
    constr_sdr_def = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)), sparse.csc_matrix(((2*N_z - 2)*N_omega,N_omega))],
                                  [sparse.csc_matrix((N_omega,(2*N_z - 2)*N_omega)), proj + proj.conj().T]])
    return constr_sdr_def

def symplectic_constr(N_omega, N_z, project, n, cst):
    """
    Gives matrices for the symplectic constraints. The term cst is the constant associated to the project matrix, 
    which is a Kronecker delta
    """
    mats = []
    terms = [1., 1.j]
    quads = []
    for i in range(N_z - 1):
        x_index = i
        y_index = N_z - 1 + i
        for j in range(len(terms)):
            proj = terms[j]*project.copy()
            if j == 0:
                cst_mat = (0.5/n)*(cst/N_omega)*sparse.eye(N_omega)
            else:
                cst_mat = sparse.csc_matrix((N_omega, N_omega))
            mat = quad_proj(N_omega, N_z, x_index, y_index, proj)
            mat = 0.5*(mat + mat.conj().T)
            quad = mat.copy()
            quads.append(quad)
            mat = sparse.bmat([[mat, sparse.csc_matrix(((2*N_z - 2)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z - 2)*N_omega)), -cst_mat]])
            mats.append(mat)
    return mats

def quad_to_lin_prop_upper(N_omega, N_z, n, project):
    """
    Gives the relation between the quadratic product of U_+i and
    U_-j dagger when j > i, the cst term is just the Kronecker Delta
    """
    mats = []
    terms = [1., 1.j]
    x_indices = []
    y_indices = []
    lins = []
    for i in range(N_z - 1):
        x_index = i
        for j in range(N_z + x_index, 2*N_z - 2):
            for k in range(len(terms)):
                proj = terms[k]*project
                y_index = j
                y_indices.append(y_index)
                x_indices.append(x_index)
                quad = quad_proj(N_omega, N_z, x_index, y_index, proj)
                quad = 0.5*(quad + quad.conj().T)
                lin = (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, proj.conj().T)[y_index - 1 - x_index])
                lins.append(lin)
                mat = sparse.bmat([[quad, -0.5*lin],
                                   [-0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
                mats.append(mat)
    return mats

def quad_to_lin_prop_lower(N_omega, N_z, n, project):
    """
    Gives the relation between the quadratic product of U_+i and
    U_-j dagger when j > i, the cst term is just the Kronecker Delta
    """
    mats = []
    terms = [1., 1.j]
    x_indices = []
    y_indices = []
    for i in range(1, N_z - 1):
        x_index = i
        for j in range(N_z - 1, N_z - 1 + x_index):
            for k in range(len(terms)):
                proj = terms[k]*project
                y_index = j
                y_indices.append(y_index)
                x_indices.append(x_index)
                quad = quad_proj(N_omega, N_z, x_index, y_index, proj)
                quad = 0.5*(quad + quad.conj().T)
                lin = (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, proj)[x_index - y_index - N_z])
                mat = sparse.bmat([[quad, -0.5*lin],
                                    [-0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
                mats.append(mat)
    return mats

def mean_photon_numb_constr(N_omega, N_z, n):
    """
    Gives the photon pair number constraint at the end of waveguide
    """
    quad = quad_proj(N_omega, N_z, N_z - 2, N_z - 2, sparse.eye(N_omega)) + quad_proj(N_omega, N_z, 2*N_z - 3, 2*N_z - 3, sparse.eye(N_omega))
    cst = -(2*N_omega/n) - 4
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z - 2)*N_omega,N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z - 2)*N_omega)), (cst/N_omega)*sparse.eye(N_omega)]])
    return mat

def mean_photon_numb_prev_constr(N_omega, N_z):
    """
    Gives the matrices representing mean photon number pair inequality between the 
    present propagators and the future propagators. The trace should be always less or equal to 0
    """
    mats = []
    for i in range(N_z - 2):
        quad_present = quad_proj(N_omega, N_z, i, i, sparse.eye(N_omega)) + quad_proj(N_omega, N_z, i + N_z - 1, i + N_z - 1, sparse.eye(N_omega))
        quad_future = quad_proj(N_omega, N_z, i + 1, i + 1, sparse.eye(N_omega)) + quad_proj(N_omega, N_z, i + N_z, i + N_z, sparse.eye(N_omega))
        quad = quad_present - quad_future
        mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z - 2)*N_omega,N_omega))],
                           [sparse.csc_matrix((N_omega, (2*N_z - 2)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
        mats.append(mat)
    return mats

def dynamics_proj_mat(N_omega, z):
    """
    Gives the matrices to project on the variable to obtain the matrix to equate
    delta_k + pump
    """
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    I_proj_on_Z = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 3]
    I_proj_on_Z = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), I_proj_on_Z])
    proj_plus = (1/delta_z)*sparse.vstack([0.75*sparse.eye(N_omega), -(3/20)*sparse.eye(N_omega), 
                                           (1/60)*sparse.eye(N_omega), sparse.csc_matrix(((N_z)*N_omega, N_omega))])
    proj_minus = (1/delta_z)*sparse.vstack([sparse.csc_matrix(((N_z - 1)*N_omega, N_omega)),-0.75*sparse.eye(N_omega), (3/20)*sparse.eye(N_omega),
                                            -(1/60)*sparse.eye(N_omega), sparse.csc_matrix((N_omega,N_omega))])
    return I_proj_on_Z, proj_plus, proj_minus

def obj_f(N_omega, N_z):
    """
    Gives the matrices to apply on the variable to obtain the matrix
    to be used in the objective function
    """
    left = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 2] - get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 3]
    left.resize(((2*N_z - 1)*N_omega, N_omega))
    right = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 3]
    right = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), right])
    return left, right