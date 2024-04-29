import numpy as np
import scipy.sparse as sparse
import scipy

np.random.seed(0)

def diag_mat(N_omega, N_z, proj):
    """
    List of diagonal matrices with fully sized proj
    """
    diag_mats = []
    for i in range(2*N_z):
        full_proj = sparse.csc_matrix(0.5*(proj + proj.conj().T))
        full_proj.resize(((2*N_z - i)*N_omega, (2*N_z - i)*N_omega))
        full_proj = sparse.bmat([[sparse.csc_matrix((i*N_omega, i*N_omega)), sparse.csc_matrix((i*N_omega, (2*N_z - i)*N_omega))],[sparse.csc_matrix(((2*N_z - i)*N_omega, i*N_omega)), full_proj]])
        diag_mats.append(full_proj)
    return diag_mats

def get_lin_matrices(N_z, N_omega, proj):
    """
    List of projection matrices for every propagator
    """
    lin = []
    for i in range(2*N_z):
        lin_mat = proj.copy()
        lin_mat.resize(((2*N_z - i)*N_omega, N_omega))
        lin_mat = sparse.vstack([sparse.csc_matrix((i*N_omega, N_omega)), lin_mat])
        lin.append(sparse.csc_matrix(lin_mat))
    return lin

def get_green_f(omega, z):
    return [np.diag(np.exp(1.j*omega*z[i])) for i in range(len(z))]

def sdr_def_constr(N_omega, N_z, proj):
    """
    Give the constraint that the SDR is linked with the QCQP
    """
    constr_sdr_def = sparse.bmat([[sparse.csc_matrix(((2*N_z)*N_omega,(2*N_z)*N_omega)), sparse.csc_matrix(((2*N_z)*N_omega,N_omega))],
                                  [sparse.csc_matrix((N_omega,(2*N_z)*N_omega)), proj + proj.conj().T]])
    return constr_sdr_def

def dynamics_W(omega, z, proj, prop_sign, pump_power):
    """
    delta_v = 1, prop_sign either "plus" or "minus"
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    dynamics_real_list = []
    dynamics_imag_list = []
    for i in range(1, N_z):
        green_fs = get_green_f(omega, z[:i + 1])
        green_fs.reverse()
        green_fs[0] = green_fs[0]/2
        green_fs[-1] = green_fs[-1]/2
        projected_green_fs = [sparse.csc_matrix(green_fs[i]@proj.conj()) for i in range(1, len(green_fs))]
        full_proj_green_fs = [delta_z*projected_green_fs[i] for i in range(len(green_fs) - 1)]
        # Good until here
        stacked_dynamics = pump_power*sparse.hstack(full_proj_green_fs)
        if prop_sign == "plus":
            stacked_dynamics.resize((N_omega, (2*N_z - 1)*N_omega))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((N_omega, N_omega)), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z - 1)*N_omega, (2*N_z)*N_omega)), stacked_dynamics])
            dynamics_real_list.append(0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(0.5*(-1.j*stacked_dynamics + 1.j*(stacked_dynamics).conj().T))
        if prop_sign == "minus":
            stacked_dynamics.resize((N_omega, (N_z)*N_omega))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((N_omega, (N_z)*N_omega)), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z - 1)*N_omega, (2*N_z)*N_omega)), stacked_dynamics])
            dynamics_real_list.append(-0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(0.5*(1.j*stacked_dynamics  + (1.j*stacked_dynamics).conj().T))
    return dynamics_real_list, dynamics_imag_list

def get_dynamics_sdr(omega, z, proj, n, pump_power):
    """
    Gives the semidefinite relaxation matrices for the dynamics constraints
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    dynamics_real_plus, dynamics_imag_plus = dynamics_W(omega, z, proj, "plus", pump_power)
    dynamics_real_minus, dynamics_imag_minus = dynamics_W(omega, z, proj, "minus", pump_power)
    U_lin_list = get_lin_matrices(N_z, N_omega, proj)
    dynamics_real_plus_sdr = []
    dynamics_imag_plus_sdr = []
    dynamics_real_minus_sdr = []
    dynamics_imag_minus_sdr = []
    green_fs = get_green_f(omega,z)
    for i in range(len(dynamics_real_plus)):
        beta_lin = get_lin_matrices(N_z, N_omega, (0.5*pump_power*delta_z/np.sqrt(n))*green_fs[1 + i].conj()@proj)[2*N_z - 1]
        dynamics_real_plus_sdr.append(sparse.bmat([[dynamics_real_plus[i], 0.5*(-U_lin_list[1 + i] + beta_lin)],[0.5*(-U_lin_list[1 + i] + beta_lin).conj().T, (1/np.sqrt(n))*np.real((np.trace(proj@green_fs[1])/N_omega))*sparse.eye(N_omega)]]))
        dynamics_imag_plus_sdr.append(sparse.bmat([[dynamics_imag_plus[i], 0.5*(1.j*(-U_lin_list[1 + i] + beta_lin))],[0.5*(1.j*(-U_lin_list[1 + i] + beta_lin)).conj().T, (1/np.sqrt(n))*np.imag((np.trace(proj@green_fs[1])/N_omega))*sparse.eye(N_omega)]]))
        dynamics_real_minus_sdr.append(sparse.bmat([[dynamics_real_minus[i], 0.5*(-U_lin_list[N_z + i] - beta_lin)],[0.5*(-U_lin_list[N_z + i] - beta_lin).conj().T, (1/np.sqrt(n))*np.real((np.trace(proj@green_fs[1])/N_omega))*sparse.eye(N_omega)]]))
        dynamics_imag_minus_sdr.append(sparse.bmat([[dynamics_imag_minus[i], 0.5*(1.j*(-U_lin_list[N_z + i] - beta_lin))],[0.5*(1.j*(-U_lin_list[N_z + i] - beta_lin)).conj().T, (1/np.sqrt(n))*np.imag((np.trace(proj@green_fs[1])/N_omega))*sparse.eye(N_omega)]]))
    return dynamics_real_plus_sdr, dynamics_imag_plus_sdr, dynamics_real_minus_sdr, dynamics_imag_minus_sdr

def sympl_constr_sdr(N_omega, N_z, proj, n):
    """
    Get full symplectic matrices for the given projection matrix
    """
    # Symplective constraints
    real_symplectic = []
    imag_symplectic = []
    real_cst_proj = -0.5*(proj + proj.conj().T).trace()
    for i in range(N_z - 1):
        proj_copy = proj.copy()
        proj_copy.resize(((N_z - i + 1)*N_omega, (N_z - i)*N_omega))
        proj_copy = sparse.vstack([sparse.csc_matrix((N_omega, (N_z - i)*N_omega)), proj_copy])
        proj_copy = sparse.hstack([sparse.csc_matrix((2*N_z*N_omega, N_z*N_omega)), proj_copy])
        real_proj = 0.5*(proj_copy + proj_copy.conj().T)
        real_proj = sparse.bmat([[real_proj, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), (1/n)*(real_cst_proj/N_omega)*sparse.eye(N_omega)]])
        imag_proj = 0.5*(1.j*proj_copy + (1.j*proj_copy).conj().T)
        imag_proj = sparse.bmat([[imag_proj, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
        real_symplectic.append(real_proj)
        imag_symplectic.append(imag_proj)
    return real_symplectic, imag_symplectic

def photon_nbr_constr(N_omega, N_z, n):
    """
    Fixes the photon number at end of waveguide
    """
    diags = 0.25*(diag_mat(N_omega, N_z, sparse.eye(N_omega))[N_z - 1] + diag_mat(N_omega, N_z, sparse.eye(N_omega))[2*N_z - 2])
    diags = sparse.bmat([[diags, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                         [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), -((N_omega/(2*n) + 1)/N_omega)*sparse.eye(N_omega)]])
    return diags

def unitary_constr(N_omega, N_z, proj):
    """
    Gives constraint that the matrix in objective function is unitary
    """
    diag_real = diag_mat(N_omega, N_z, 0.5*(proj + proj.conj().T))[0]
    diag_real.resize(((2*N_z)*N_omega, (2*N_z)*N_omega))
    diag_real = sparse.bmat([[diag_real, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                             [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), -0.5*(((proj.conj().T + proj).trace())/N_omega)*sparse.eye(N_omega)]])
    diag_imag = diag_mat(N_omega, N_z, 0.5*(1.j*proj + (1.j*proj).conj().T))[0]
    diag_imag.resize(((2*N_z)*N_omega, (2*N_z)*N_omega))
    diag_imag = sparse.bmat([[diag_imag, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                             [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
    return diag_real, diag_imag

def limit_pump_power(N_omega, N_z):
    """
    Constraint limiting the trace of beta dagger beta to be 1
    """
    quad = sparse.eye(N_omega)
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                        [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), quad]])
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), -(1/N_omega)*sparse.eye(N_omega)]])
    return mat

def hankel_constr_list(N_omega):
    """
    Build list of matrices in N_omega dimension to make sure the pump is Hankel
    """
    first_list = list((np.linspace(1, N_omega - 2, N_omega - 2)).astype("int32"))
    second_list = list((np.linspace(1, N_omega - 2, N_omega - 2)).astype("int32"))
    second_list.reverse()
    constr_anti_diags = first_list + [N_omega - 1] + second_list
    prior_mats = []
    after_mats = []
    for position, nbr_constr in enumerate(constr_anti_diags):
        for j in range(nbr_constr):
            prior_mat = np.zeros((N_omega, N_omega))
            after_mat = np.zeros((N_omega, N_omega))
            if position < N_omega - 2:
                prior_mat[j, nbr_constr - j] = 1
                after_mat[j + 1, nbr_constr - j - 1] = 1
            elif position == N_omega - 2:
                prior_mat[j, nbr_constr - j] = 1
                after_mat[j + 1, nbr_constr - j - 1] = 1
            else:
                prior_mat[position - (N_omega - 2 - j), N_omega - 1 - j] = 1
                after_mat[position - (N_omega - 3 - j), N_omega - 2 - j] = 1
            prior_mats.append(prior_mat)
            after_mats.append(after_mat)
    return prior_mats, after_mats

def constr_hankel_sdr(N_omega, N_z):
    """
    Gives the matrices telling the pump to be Hankel in SDR frame
    """
    prior_mats, after_mats = hankel_constr_list(N_omega)
    mat_list = []
    for i in range(len(prior_mats)):
        prior_lin = get_lin_matrices(N_z, N_omega, prior_mats[i])[2*N_z - 1]
        after_lin = -get_lin_matrices(N_z, N_omega, after_mats[i])[2*N_z - 1]
        lin = prior_lin + after_lin
        mat = sparse.bmat([[sparse.csc_matrix(((2*N_z)*N_omega,(2*N_z)*N_omega)), 0.5*lin],
                           [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_list.append(mat)
    return mat_list
    
def sdr_fixed_pump(N_omega, N_z, beta, proj):
    """
    Gives the linear SDR constraint on fixing the pump
    """
    lin_real = 0.5*(get_lin_matrices(N_z, N_omega, proj)[2*N_z - 1])
    lin_imag = 0.5*1.j*(get_lin_matrices(N_z, N_omega, proj)[2*N_z - 1])
    mat_real = sparse.bmat([[sparse.csc_matrix(((2*N_z)*N_omega,(2*N_z)*N_omega)), lin_real],
                            [lin_real.conj().T, -(np.real(np.trace(proj.conj().T@beta))/N_omega)*sparse.eye(N_omega)]])
    mat_imag = sparse.bmat([[sparse.csc_matrix(((2*N_z)*N_omega,(2*N_z)*N_omega)), lin_imag],
                            [lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat_real, mat_imag

def obj_f_psd(N_omega, N_z):
    """
    Gives matrices to get constraint that matrix in the objective function should be positive semi definite
    """
    right_side_plus = 0.5*get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 1]
    right_side_plus.resize(((2*N_z + 1)*N_omega, N_omega))
    right_side_minus = 0.5*get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 2]
    right_side_minus.resize(((2*N_z + 1)*N_omega, N_omega))
    left = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[0]
    left.resize(((2*N_z + 1)*N_omega, N_omega))
    right = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 1] - get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 2]
    right.resize(((2*N_z + 1)*N_omega, N_omega))
    return left, right, right_side_plus, right_side_minus

def obj_f_psd_parts(N_omega, N_z):
    """
    Gives the matrices that makes both parts of the matrix in the objective function
    positie semidefinite
    """
    left = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[0]
    left.resize(((2*N_z + 1)*N_omega, N_omega))
    right_pos = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 1]
    right_pos.resize(((2*N_z + 1)*N_omega, N_omega))
    right_neg = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 2]
    right_neg.resize(((2*N_z + 1)*N_omega, N_omega))
    return left, right_pos, right_neg

def obj_f_mat_hermit(N_omega, N_z, proj):
    """
    Matrix making the objective matrix positive
    """
    mat_plus_real = proj - proj.conj().T
    mat_plus_real.resize(((2*N_z + 1)*N_omega, (N_z + 2)*N_omega))
    mat_plus_real = sparse.hstack([sparse.csc_matrix(((2*N_z + 1)*N_omega, (N_z - 1)*N_omega)), mat_plus_real])
    mat_plus_real = 0.5*(mat_plus_real + mat_plus_real.conj().T)
    mat_minus_real = proj.copy() - proj.copy().conj().T
    mat_minus_real.resize(((2*N_z + 1)*N_omega, 3*N_omega))
    mat_minus_real = sparse.hstack([sparse.csc_matrix(((2*N_z + 1)*N_omega, (2*N_z - 2)*N_omega)), mat_minus_real])
    mat_minus_real = 0.5*(mat_minus_real + mat_minus_real.conj().T)
    mat_plus_imag = 1.j*proj - (1.j*proj).conj().T
    mat_plus_imag.resize(((2*N_z + 1)*N_omega, (N_z + 2)*N_omega))
    mat_plus_imag = sparse.hstack([sparse.csc_matrix(((2*N_z + 1)*N_omega, (N_z - 1)*N_omega)), mat_plus_imag])
    mat_plus_imag = 0.5*(mat_plus_imag + mat_plus_imag.conj().T)
    mat_minus_imag = 1.j*proj.copy() - (1.j*proj).copy().conj().T
    mat_minus_imag.resize(((2*N_z + 1)*N_omega, 3*N_omega))
    mat_minus_imag = sparse.hstack([sparse.csc_matrix(((2*N_z + 1)*N_omega, (2*N_z - 2)*N_omega)), mat_minus_imag])
    mat_minus_imag = 0.5*(mat_minus_imag + mat_minus_imag.conj().T)
    return mat_plus_real, mat_plus_imag, mat_minus_real, mat_minus_imag

def constr_B_equal_Z(N_omega, N_z, proj, kron_delta):
    """
    Gives constraint makin BZ_dagger is identity
    """
    quad = proj.copy()
    quad.resize((2*N_z*N_omega, N_omega))
    quad = sparse.hstack([sparse.csc_matrix(((2*N_z)*N_omega, (2*N_z - 1)*N_omega)), quad])
    quad_real = 0.5*(quad + quad.conj().T)
    quad_imag = 0.5*(1.j*quad + (1.j*quad).conj().T)
    real = sparse.bmat([[quad_real, sparse.csc_matrix((2*N_z*N_omega, N_omega))],[sparse.csc_matrix((N_omega, 2*N_z*N_omega)), -(kron_delta/N_omega)*sparse.eye(N_omega)]])
    imag = sparse.bmat([[quad_imag, sparse.csc_matrix((2*N_z*N_omega, N_omega))],[sparse.csc_matrix((N_omega, 2*N_z*N_omega)), sparse.csc_matrix((N_omega,N_omega))]])
    return real, imag

def PSD_B_plus_B_dagger(N_omega, N_z):
    """
    Gives the matrices to isolate Z_B_dagger and B_Z_dagger to make identity minus their multiplication positive
    semidefintie
    """
    right_B = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[-1]
    left_B = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[0]
    right_B = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), right_B])
    left_B.resize(((2*N_z + 1)*N_omega, N_omega))
    return left_B, right_B

def obj_f_sdr(N_omega, N_z):
    """
    Gives the matrix for the objective function in frame of SDR
    """
    quad_U_plus = sparse.eye(N_omega)
    quad_U_plus.resize(((2*N_z)*N_omega,(N_z + 1)*N_omega))
    quad_U_plus = sparse.hstack([sparse.csc_matrix(((2*N_z)*N_omega, (N_z - 1)*N_omega)), quad_U_plus])
    quad_U_minus = -sparse.eye(N_omega)
    quad_U_minus.resize(((2*N_z)*N_omega, 2*N_omega))
    quad_U_minus = sparse.hstack([sparse.csc_matrix(((2*N_z)*N_omega, (2*N_z - 2)*N_omega)), quad_U_minus])
    quad = 0.25*(quad_U_plus + quad_U_minus + quad_U_plus.conj().T + quad_U_minus.conj().T)
    return sparse.bmat([[quad, sparse.csc_matrix(((2*N_z)*N_omega, N_omega))], [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
