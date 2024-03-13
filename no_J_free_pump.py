import numpy as np
import scipy.sparse as sparse
import scipy

np.random.seed(0)

def diag_mat(N_omega, N_z, proj):
    """
    List of diagonal matrices with fully sized proj
    """
    diag_mats = []
    for i in range(2*N_z + 2):
        full_proj = sparse.csc_matrix(0.5*(proj + proj.conj().T))
        full_proj.resize(((2*N_z + 2 - i)*N_omega, (2*N_z + 2 - i)*N_omega))
        full_proj = sparse.bmat([[sparse.csc_matrix((i*N_omega, i*N_omega)), sparse.csc_matrix((i*N_omega, (2*N_z + 2 - i)*N_omega))],[sparse.csc_matrix(((2*N_z + 2 - i)*N_omega, i*N_omega)), full_proj]])
        diag_mats.append(full_proj)
    return diag_mats

def get_lin_matrices(N_z, N_omega, proj):
    """
    List of projection matrices for every propagator
    """
    lin = []
    for i in range(2*N_z + 2):
        lin_mat = proj.copy()
        lin_mat.resize(((2*N_z + 2 - i)*N_omega, N_omega))
        lin_mat = sparse.vstack([sparse.csc_matrix((i*N_omega, N_omega)), lin_mat])
        lin.append(sparse.csc_matrix(lin_mat))
    return lin

def get_green_f(omega, z):
    return [np.diag(np.exp(1.j*omega*z[i])) for i in range(len(z))]

def dynamics_W(omega, z, proj, prop_sign, pump_power):
    """
    delta_v = 1, prop_sign either "plus" or "minus"
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    dynamics_real_list = [sparse.csc_matrix(((2 + 2*N_z)*N_omega, (2 + 2*N_z)*N_omega))]
    dynamics_imag_list = [sparse.csc_matrix(((2 + 2*N_z)*N_omega, (2 + 2*N_z)*N_omega))]
    for i in range(1, N_z):
        green_fs = get_green_f(omega, z[:i + 1])
        green_fs.reverse()
        green_fs[0] = green_fs[0]/2
        green_fs[-1] = green_fs[-1]/2
        projected_green_fs = [sparse.csc_matrix(green_fs[i]@proj.conj()) for i in range(len(green_fs))]
        full_proj_green_fs = [delta_z*projected_green_fs[i] for i in range(len(green_fs))]
        # Good until here
        stacked_dynamics = pump_power*sparse.hstack(full_proj_green_fs)
        if prop_sign == "plus":
            stacked_dynamics.resize((N_omega, (2*N_z + 1)*N_omega))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((N_omega, N_omega)), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z + 1)*N_omega, (2*N_z + 2)*N_omega)), stacked_dynamics])
            dynamics_real_list.append(0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(0.5*(-1.j*stacked_dynamics + 1.j*(stacked_dynamics).conj().T))
        if prop_sign == "minus":
            stacked_dynamics.resize((N_omega, (N_z + 1)*N_omega))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((N_omega, (N_z + 1)*N_omega)), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z + 1)*N_omega, (2*N_z + 2)*N_omega)), stacked_dynamics])
            dynamics_real_list.append(-0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(0.5*(1.j*stacked_dynamics  + (1.j*stacked_dynamics).conj().T))
    return dynamics_real_list, dynamics_imag_list

def get_dynamics_sdr(omega, z, proj, n, pump_power):
    """
    Gives the semidefinite relaxation matrices for the dynamics constraints
    """
    N_omega = len(omega)
    N_z = len(z)
    dynamics_real_plus, dynamics_imag_plus = dynamics_W(omega, z, proj, "plus", pump_power)
    dynamics_real_minus, dynamics_imag_minus = dynamics_W(omega, z, proj, "minus", pump_power)
    lin_list = get_lin_matrices(N_z, N_omega, proj)
    dynamics_real_plus_sdr = []
    dynamics_imag_plus_sdr = []
    dynamics_real_minus_sdr = []
    dynamics_imag_minus_sdr = []
    green_fs = get_green_f(omega,z)
    for i in range(N_z):
        dynamics_real_plus_sdr.append(sparse.bmat([[dynamics_real_plus[i], -0.5*lin_list[1 + i]],[-0.5*lin_list[1 + i].conj().T, (1/np.sqrt(n))*np.real((np.trace(proj@green_fs[i])/N_omega))*sparse.eye(N_omega)]]))
        dynamics_imag_plus_sdr.append(sparse.bmat([[dynamics_imag_plus[i], -0.5*(1.j*lin_list[1 + i])],[-0.5*(1.j*lin_list[i + 1]).conj().T, (1/np.sqrt(n))*np.imag((np.trace(proj@green_fs[i])/N_omega))*sparse.eye(N_omega)]]))
        dynamics_real_minus_sdr.append(sparse.bmat([[dynamics_real_minus[i], -0.5*lin_list[1 + i + N_z]],[-0.5*lin_list[1 + i + N_z].conj().T, (1/np.sqrt(n))*np.real((np.trace(proj@green_fs[i])/N_omega))*sparse.eye(N_omega)]]))
        dynamics_imag_minus_sdr.append(sparse.bmat([[dynamics_imag_minus[i], -0.5*(1.j*lin_list[1 + i + N_z])],[-0.5*(1.j*lin_list[i + 1 + N_z]).conj().T, (1/np.sqrt(n))*np.imag((np.trace(proj@green_fs[i])/N_omega))*sparse.eye(N_omega)]]))
    return dynamics_real_plus_sdr, dynamics_imag_plus_sdr, dynamics_real_minus_sdr, dynamics_imag_minus_sdr

def sympl_constr_sdr(N_omega, N_z, proj, n):
    """
    Get full symplectic matrices for the given projection matrix
    """
    # Symplective constraints
    real_symplectic = []
    imag_symplectic = []
    real_cst_proj = -0.5*(proj + proj.conj().T).trace()
    for i in range(N_z):
        proj_copy = proj.copy()
        proj_copy.resize(((2*N_z - i + 1)*N_omega, (N_z - i + 1)*N_omega))
        proj_copy = sparse.bmat([[sparse.csc_matrix(((i + 1)*N_omega, (N_z + 1 + i)*N_omega)), sparse.csc_matrix(((i + 1)*N_omega, (N_z + 1 - i)*N_omega))],
                            [sparse.csc_matrix(((2*N_z + 1 - i)*N_omega,(N_z + 1 + i)*N_omega)), proj_copy]])
        real_proj = 0.5*(proj_copy + proj_copy.conj().T)
        real_proj = sparse.bmat([[real_proj, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), (1/n)*(real_cst_proj/N_omega)*sparse.eye(N_omega)]])
        imag_proj = 0.5*(1.j*proj_copy + (1.j*proj_copy).conj().T)
        imag_proj = sparse.bmat([[imag_proj, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
        real_symplectic.append(real_proj)
        imag_symplectic.append(imag_proj)
    return real_symplectic, imag_symplectic

def photon_nbr_prev_points(N_omega, N_z):
    """
    Gives the matrix for the constraint that says mean photon number increases through waveguide
    """
    constraint = []
    diags = diag_mat(N_omega, N_z, sparse.eye(N_omega))
    for i in range(1, N_z):
        quad = 0.25*(diags[i + 1] + diags[N_z + i + 1] - diags[i] - diags[N_z + i])
        constraint.append(sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                                       [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]]))
    return constraint

def photon_nbr_constr(N_omega, N_z, n):
    """
    Fixes the photon number at end of waveguide
    """
    diags = 0.25*(diag_mat(N_omega, N_z, sparse.eye(N_omega))[N_z] + diag_mat(N_omega, N_z, sparse.eye(N_omega))[2*N_z])
    diags = sparse.bmat([[diags, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                         [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), -((N_omega/(2*n) + 1)/N_omega)*sparse.eye(N_omega)]])
    return diags

def unitary_constr(N_omega, N_z, proj):
    """
    Gives constraint that the matrix in objective function is unitary
    """
    diag_real = diag_mat(N_omega, N_z, 0.5*(proj + proj.conj().T))[0]
    diag_real.resize(((2*N_z + 2)*N_omega, (2*N_z + 2)*N_omega))
    diag_real = sparse.bmat([[diag_real, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                             [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), -0.5*(((proj.conj().T + proj).trace())/N_omega)*sparse.eye(N_omega)]])
    diag_imag = diag_mat(N_omega, N_z, 0.5*(1.j*proj + (1.j*proj).conj().T))[0]
    diag_imag.resize(((2*N_z + 2)*N_omega, (2*N_z + 2)*N_omega))
    diag_imag = sparse.bmat([[diag_imag, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                             [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
    return diag_real, diag_imag

def trace_unitary_mat(N_omega, N_z):
    """
    Gives the constraint on trace of unitary matrix to be equal to 1
    """
    lin = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[0]
    lin.resize(((2*N_z + 2)*N_omega, N_omega))
    mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin],
                       [0.5*lin.conj().T, sparse.eye(N_omega)]])
    return mat

def obj_f_sdr(N_omega, N_z, n):
    """
    Gives the matrix for the objective function in frame of SDR
    """
    quad_U_plus = sparse.eye(N_omega)
    quad_U_plus.resize(((2*N_z + 2)*N_omega,(N_z + 2)*N_omega))
    quad_U_plus = sparse.hstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, N_z*N_omega)), quad_U_plus])
    quad_U_minus = -sparse.eye(N_omega)
    quad_U_minus.resize(((2*N_z + 2)*N_omega, 2*N_omega))
    quad_U_minus = sparse.hstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, 2*N_z*N_omega)), quad_U_minus])
    quad = 0.5*(quad_U_plus + quad_U_minus + quad_U_plus.conj().T + quad_U_minus.conj().T)
    return sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))], [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])

def sdr_def_constr(N_omega, N_z, proj):
    """
    Give the constraint that the SDR is linked with the QCQP
    """
    constr_sdr_def = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), sparse.csc_matrix(((2*N_z + 2)*N_omega,N_omega))],
                                  [sparse.csc_matrix((N_omega,(2*N_z + 2)*N_omega)), proj + proj.conj().T]])
    return constr_sdr_def

def sdr_fixed_pump(N_omega, N_z, beta, proj):
    """
    Gives the linear SDR constraint on fixing the pump
    """
    lin_real = 0.5*(get_lin_matrices(N_z, N_omega, proj)[2*N_z + 1])
    lin_imag = 0.5*1.j*(get_lin_matrices(N_z, N_omega, proj)[2*N_z + 1])
    mat_real = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), lin_real],
                            [lin_real.conj().T, -(np.real(np.trace(proj.conj().T@beta))/N_omega)*sparse.eye(N_omega)]])
    mat_imag = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), lin_imag],
                            [lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat_real, mat_imag

def get_standard_Q(N_omega):
    """
    Generate list of standard unitary matrices
    """
    mat_list = [sparse.eye(N_omega)]
    upper_hankel = np.zeros(N_omega)
    lower_hankel = np.zeros(N_omega)
    upper_hankel[-1] = 1
    mat_list.append(sparse.csc_matrix(scipy.linalg.hankel(upper_hankel, lower_hankel)))
    for i in range(2*N_omega):
        mat_list.append(sparse.csc_matrix(np.diag(np.exp(2*np.pi*1.j*np.random.random(N_omega)))))
    return mat_list

def get_Q_list(N_omega, len_Q):
    """
    Gives a list of complex diagonal matrix with entries of unit norm 
    """
    Q_list = []
    for i in range(len_Q):
        rand = np.random.random((N_omega, N_omega)) + 1.j*np.random.random((N_omega, N_omega))
        V, _, W = scipy.linalg.svd(rand)
        Q_list.append(sparse.csc_matrix(V@W))
    return Q_list

def ineq_real_any_unitary(N_omega, N_z, Q_list):
    """
    Gives the constraint that real part of trace with unitary B is the greatest out of all 
    unitary matrices
    """
    quad_plus = sparse.eye(N_omega)
    quad_minus = -sparse.eye(N_omega)
    quad_plus.resize((N_omega, (N_z + 2)*N_omega))
    quad_plus = sparse.bmat([[sparse.csc_matrix((N_omega, N_z*N_omega)), quad_plus],
                             [sparse.csc_matrix(((2*N_z + 1)*N_omega, N_z*N_omega)), sparse.csc_matrix(((2*N_z + 1)*N_omega, (N_z + 2)*N_omega))]])
    quad_minus.resize((N_omega, 2*N_omega))
    quad_minus = sparse.bmat([[sparse.csc_matrix((N_omega, 2*N_z*N_omega)), quad_minus],
                              [sparse.csc_matrix(((2*N_z + 1)*N_omega, 2*N_z*N_omega)), sparse.csc_matrix(((2*N_z + 1)*N_omega, 2*N_omega))]])
    quad = 0.25*(quad_plus + quad_minus + quad_plus.conj().T + quad_minus.conj().T)
    len_Q = len(Q_list)
    mat_list = []
    for i in range(len_Q):
        lin = - 0.5*(get_lin_matrices(N_z, N_omega, Q_list[i])[N_z] - get_lin_matrices(N_z, N_omega, Q_list[i])[2*N_z])
        mat = sparse.bmat([[quad, 0.5*lin],
                           [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_list.append(mat)
    return mat_list

def imag_obj_f(N_omega, N_z):
    """
    Gives matrix that makes the imaginary part of the objective function 0
    """
    quad_U_plus = sparse.eye(N_omega)
    quad_U_plus.resize(((2*N_z + 2)*N_omega,(N_z + 2)*N_omega))
    quad_U_plus = sparse.hstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, N_z*N_omega)), quad_U_plus])
    quad_U_minus = -sparse.eye(N_omega)
    quad_U_minus.resize(((2*N_z + 2)*N_omega, 2*N_omega))
    quad_U_minus = sparse.hstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, 2*N_z*N_omega)), quad_U_minus])
    quad = 0.25*(1.j*quad_U_plus + 1.j*quad_U_minus + (1.j*quad_U_plus).conj().T + (1.j*quad_U_minus).conj().T)
    return sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))], [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])


def constr_Q_cst(N_omega, N_z, Q_list):
    """
    Gives a list of constraints telling the imaginary part of Tr(B(U_+ - U_-)Q) is 0 for
    some fixed unitaries Q
    """
    len_Q = len(Q_list)
    real = []
    imag = []
    for i in range(len_Q):
        plus =  Q_list[i].copy()
        plus.resize((N_omega, (N_z + 3)*N_omega))
        plus = sparse.bmat([[sparse.csc_matrix((N_omega, N_z*N_omega)), plus],
                            [sparse.csc_matrix(((2*N_z + 2)*N_omega, N_z*N_omega)), sparse.csc_matrix(((2*N_z + 2)*N_omega, (N_z + 3)*N_omega))]])
        minus = Q_list[i].copy()
        minus.resize((N_omega, 3*N_omega))
        minus = sparse.bmat([[sparse.csc_matrix((N_omega, 2*N_z*N_omega)), minus],
                            [sparse.csc_matrix(((2*N_z + 2)*N_omega, 2*N_z*N_omega)), sparse.csc_matrix(((2*N_z + 2)*N_omega, 3*N_omega))]])
        real.append(0.5*((plus - minus) + (plus - minus).conj().T))
        imag.append(0.5*(1.j*(plus - minus) + (1.j*(plus - minus)).conj().T))
    return real, imag, Q_list

def limit_pump_power(N_omega, N_z):
    """
    Constraint limiting the trace of beta dagger beta to be 1
    """
    quad = sparse.eye(N_omega)
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                        [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), quad]])
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z + 2)*N_omega)), -(1/N_omega)*sparse.eye(N_omega)]])
    return mat

def hankel_constr(N_omega, N_z):
    """
    Constraint making sure the pump is a Hankel matrix
    """
    mat_list = []
    constr_len = 2*sum(np.linspace(1, N_omega - 2, N_omega - 2)) + N_omega - 1
    for i in range(constr_len):
        prior_mat = np.zeros((N_omega, N_omega))
        after_mat = np.zeros((N_omega, N_omega))
        
    return 