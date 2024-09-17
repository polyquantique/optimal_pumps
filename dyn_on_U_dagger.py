import numpy as np
import scipy.sparse as sparse
import scipy

np.random.seed(0)

def get_green_f(omega, z):
    return [np.diag(np.exp(1.j*omega*z[i])) for i in range(len(z))]

def sdr_def_constr(N_omega, N_z, proj):
    """
    Give the constraint that the SDR is linked with the QCQP
    """
    constr_sdr_def = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), sparse.csc_matrix(((2*N_z + 2)*N_omega,N_omega))],
                                  [sparse.csc_matrix((N_omega,(2*N_z + 2)*N_omega)), proj + proj.conj().T]])
    return constr_sdr_def

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

def get_dynamics(omega, z, n, proj, prop_sign, pump_power):
    """
    Gives the SDR matrices for plus or minus propagator dynamics constraints
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    real_proj = proj.copy()
    imag_proj = 1.j*(proj.copy())
    dynamics_real_list = [sparse.bmat([[sparse.csc_matrix(((2 + 2*N_z)*N_omega, (2 + 2*N_z)*N_omega)), 0.5*(-get_lin_matrices(N_z, N_omega, real_proj.conj().T)[1] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, real_proj))[0])],
                                       [0.5*(-get_lin_matrices(N_z, N_omega, real_proj.conj().T)[1] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, real_proj))[0]).conj().T, sparse.csc_matrix((N_omega, N_omega))]])]
    dynamics_imag_list = [sparse.bmat([[sparse.csc_matrix(((2 + 2*N_z)*N_omega, (2 + 2*N_z)*N_omega)), 0.5*(-get_lin_matrices(N_z, N_omega, imag_proj.conj().T)[1] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, imag_proj))[0])],
                                       [0.5*(-get_lin_matrices(N_z, N_omega, imag_proj.conj().T)[1] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, imag_proj))[0]).conj().T, sparse.csc_matrix((N_omega, N_omega))]])]
    if prop_sign == "plus":
        for i in range(1, N_z):
            green_fs = get_green_f(omega, z[:i + 1])
            green_fs.reverse()
            green_fs[0] = green_fs[0]/2
            green_fs[-1] = green_fs[-1]/2
            real_weighted_green_f = [delta_z*real_proj.conj().T@green_fs[i].conj().T for i in range(len(green_fs))]
            quad = pump_power*sparse.csc_matrix(np.vstack(real_weighted_green_f))
            quad.resize(((2*N_z + 1)*N_omega, N_omega))
            quad = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega, (2*N_z + 1)*N_omega)), sparse.csc_matrix((N_omega, N_omega))],
                                [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), quad]])
            quad_real = 0.5*(quad + quad.conj().T)
            quad_imag = 1.j*0.5*(-quad + quad.conj().T)
            lin_real = -(get_lin_matrices(N_z, N_omega, real_proj.conj().T))[i + 1] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, 2*green_fs[0]@real_proj))[0]
            lin_imag = -(get_lin_matrices(N_z, N_omega, imag_proj.conj().T))[i + 1] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, 2*green_fs[0]@imag_proj))[0]
            dynamics_real_list.append(sparse.bmat([[quad_real, 0.5*lin_real],
                                                    [0.5*lin_real.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
            dynamics_imag_list.append(sparse.bmat([[quad_imag, 0.5*lin_imag],
                                                    [0.5*lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
    if prop_sign == "minus":
        for i in range(1, N_z):
            green_fs = get_green_f(omega, z[:i + 1])
            green_fs.reverse()
            green_fs[0] = green_fs[0]/2
            green_fs[-1] = green_fs[-1]/2
            real_weighted_green_f = [delta_z*real_proj.conj().T@green_fs[i].conj().T for i in range(len(green_fs))]
            quad = pump_power*sparse.csc_matrix(np.vstack(real_weighted_green_f))
            quad.resize(((N_z + 1)*N_omega, N_omega))
            quad = -sparse.bmat([[sparse.csc_matrix(((N_z + 1)*N_omega, (2*N_z + 1)*N_omega)), sparse.csc_matrix(((N_z + 1)*N_omega, N_omega))],
                                [sparse.csc_matrix(((N_z + 1)*N_omega, (2*N_z + 1)*N_omega)), quad]])
            quad_real = 0.5*(quad + quad.conj().T)
            quad_imag = 1.j*0.5*(-quad + quad.conj().T)
            lin_real = -(get_lin_matrices(N_z, N_omega, real_proj.conj().T))[i + 1 + N_z] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, 2*green_fs[0]@real_proj))[0]
            lin_imag = -(get_lin_matrices(N_z, N_omega, imag_proj.conj().T))[i + 1 + N_z] + (1/np.sqrt(n))*(get_lin_matrices(N_z, N_omega, 2*green_fs[0]@imag_proj))[0]
            dynamics_real_list.append(sparse.bmat([[quad_real, 0.5*lin_real],
                                                    [0.5*lin_real.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
            dynamics_imag_list.append(sparse.bmat([[quad_imag, 0.5*lin_imag],
                                                    [0.5*lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
    return dynamics_real_list, dynamics_imag_list

def get_dynamics_sdr(omega, z, n, proj, pump_power):
    """
    Gives SDR matrices for dynamics constraints
    """
    real_plus, imag_plus = get_dynamics(omega, z, n, proj, "plus", pump_power)
    real_minus, imag_minus = get_dynamics(omega, z, n, proj, "minus", pump_power)
    return real_plus, imag_plus, real_minus, imag_minus

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

def ineq_U_and_B_U(N_omega, N_z):
    """
    Gives the constraint that real part of trace of U_+ - U_- is greater or equal to B(U_+ - U_-)
    """
    prop_plus = -sparse.eye(N_omega)
    prop_minus = sparse.eye(N_omega)
    prop_plus.resize(((2*N_z + 2)*N_omega, (N_z + 2)*N_omega))
    prop_minus.resize(((2*N_z + 2)*N_omega, 2*N_omega))
    prop_plus = sparse.hstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, N_z*N_omega)), prop_plus])
    prop_minus = sparse.hstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, (2*N_z)*N_omega)), prop_minus])
    quad = 0.5*(prop_plus + prop_minus + prop_plus.conj().T + prop_minus.conj().T)
    lin = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z] - get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z]
    mat = sparse.bmat([[quad, 0.5*lin],
                       [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat

def obj_f(N_omega, N_z):
    """
    Gives the matrix for the objective function
    """
    lin = 0.5*(get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z] - get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z])
    mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin],
                       [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat

def gen_standard_Q_mat(N_omega):
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

def gen_Q_mat(N_omega, len_Q):
    """
    Generate a list of random unitary matrix
    """
    Q_mat_list = []
    for i in range(len_Q):
        random_mat = np.random.random((N_omega, N_omega))
        W, D, V = scipy.linalg.svd(random_mat)
        unitary = W@V
        Q_mat_list.append(unitary)
    return Q_mat_list

def real_ineq_Q(N_omega, N_z, Q_mat_list):
    """
    Gives the constraints that the real part of the trace of difference of modified propagators is greater
    than the trace of it multiplied by any unitary matrix
    """
    len_Q = len(Q_mat_list)
    mat_list = []
    for i in range(len_Q):
        lin_plus = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega) - Q_mat_list[i])[N_z]
        lin_minus = -get_lin_matrices(N_z, N_omega, sparse.eye(N_omega) - Q_mat_list[i])[2*N_z]
        lin = 0.5*(lin_plus + lin_minus)
        mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin],
                           [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_list.append(mat)
    return mat_list

def imag_constr_tr_B_U(N_omega, N_z):
    """
    Gives the constraint that the imaginary part of the trace of difference of modified propagators
    is equal to 0
    """
    lin = 0.5*(get_lin_matrices(N_z, N_omega, 1.j*sparse.eye(N_omega))[N_z] - get_lin_matrices(N_z, N_omega, 1.j*sparse.eye(N_omega))[2*N_z])
    mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin],
                       [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat

def ineq_on_propagators(N_omega, N_z, Q_mat_list):
    """
    Gives the linear constraints on U_plus and U_minus with prefixed unitary matrix
    """
    len_Q = len(Q_mat_list)
    imag_mat_list = []
    real_mat_list = []
    for i in range(len_Q):
        lin_plus_imag = 1.j*get_lin_matrices(N_z, N_omega, Q_mat_list[i])[N_z]
        lin_minus_imag = 1.j*get_lin_matrices(N_z, N_omega, Q_mat_list[i])[2*N_z]
        lin_imag = lin_plus_imag - lin_minus_imag
        mat_imag = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin_imag],
                           [0.5*lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        imag_mat_list.append(mat_imag)
        lin_plus_real = get_lin_matrices(N_z, N_omega, Q_mat_list[i])[N_z]
        lin_minus_real = get_lin_matrices(N_z, N_omega, Q_mat_list[i])[2*N_z]
        lin_real = lin_plus_real - lin_minus_real
        mat_real = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin_real],
                           [0.5*lin_real.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        real_mat_list.append(mat_real)
    return real_mat_list, imag_mat_list

def obj_f_sdr(N_omega, N_z):
    """
    Gives matrices to isolate the matrix in the objective function to make it
    positive semidefinite
    """
    left_plus = sparse.eye(N_omega)
    left_plus.resize(((N_z + 3)*N_omega, N_omega))
    left_plus = sparse.vstack([sparse.csc_matrix((N_z*N_omega, N_omega)), left_plus])
    left_minus = -sparse.eye(N_omega)
    left_minus.resize((3*N_omega, N_omega))
    left_minus = sparse.vstack([sparse.csc_matrix((2*N_z*N_omega, N_omega)), left_minus])
    left = left_plus + left_minus
    right = sparse.eye(N_omega)
    right = sparse.vstack([sparse.csc_matrix(((2*N_z + 2)*N_omega, N_omega)), right])
    quad_plus = 0.5*get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z]
    quad_plus.resize(((2*N_z + 3)*N_omega, N_omega))
    quad_minus = 0.5*get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z]
    quad_minus.resize(((2*N_z + 3)*N_omega, N_omega))
    return left, right, quad_plus, quad_minus

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
        prior_lin = get_lin_matrices(N_z, N_omega, prior_mats[i])[2*N_z + 1]
        after_lin = -get_lin_matrices(N_z, N_omega, after_mats[i])[2*N_z + 1]
        lin = prior_lin + after_lin
        mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega)), 0.5*lin],
                           [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_list.append(mat)
    return mat_list

def constr_obj_f_hermitian(N_omega, N_z, proj):
    """
    Gives SDR matrices to make sure the matrix in the objective function is Hermitian 
    """
    quad = sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega))
    cst = sparse.csc_matrix((N_omega, N_omega))
    real_proj = proj.copy()
    imag_proj = 1.j*proj.copy()
    real_lin = get_lin_matrices(N_z, N_omega, real_proj - real_proj.conj().T)[N_z] - get_lin_matrices(N_z, N_omega, real_proj - real_proj.conj().T)[2*N_z] 
    real_mat = sparse.bmat([[quad, real_lin],[real_lin.conj().T, cst]])
    imag_lin = get_lin_matrices(N_z, N_omega, imag_proj - imag_proj.conj().T)[N_z] - get_lin_matrices(N_z, N_omega, imag_proj - imag_proj.conj().T)[2*N_z]
    imag_mat = sparse.bmat([[quad, 0.5*imag_lin],[0.5*imag_lin.conj().T, cst]])
    return real_mat, imag_mat

def random_rank_one_mat(N_omega, nbr_elements_basis):
    """
    Gives a list of random vectors and first rank matrices associated with them
    """
    rand_mat = []
    rand_vec_list = []
    for i in range(nbr_elements_basis):
        rand_vec = np.random.randint(-1, 2, (N_omega, )) + 1.j*np.random.randint(-1, 2, (N_omega, ))
        rand_vec_list.append(rand_vec)
        rand_mat.append(sparse.csc_matrix(np.outer(rand_vec, rand_vec.conj())))
    return rand_vec_list, rand_mat

def constr_semidefinite_obj_f(N_omega, N_z, rank_one_mat_list):
    """
    Gives list of matrices that give constraints that the matrix in objective function is 
    positive semidefinite
    """
    real_mats = []
    imag_mats = []
    quad = sparse.csc_matrix(((2*N_z + 2)*N_omega,(2*N_z + 2)*N_omega))
    for i in range(len(rank_one_mat_list)):
        lin_real = get_lin_matrices(N_z, N_omega, rank_one_mat_list[i])[N_z] - get_lin_matrices(N_z, N_omega, rank_one_mat_list[i])[2*N_z]
        lin_imag = get_lin_matrices(N_z, N_omega, 1.j*rank_one_mat_list[i])[N_z] - get_lin_matrices(N_z, N_omega, 1.j*rank_one_mat_list[i])[2*N_z]
        mat_real = sparse.bmat([[quad, 0.5*lin_real],
                                [0.5*lin_real.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_imag = sparse.bmat([[quad, 0.5*lin_imag],
                                [0.5*lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        real_mats.append(mat_real)
        imag_mats.append(mat_imag)
    return real_mats, imag_mats