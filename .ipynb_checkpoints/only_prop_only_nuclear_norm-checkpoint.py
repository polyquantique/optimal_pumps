import numpy as np
import scipy.sparse as sparse
import scipy
import math
import findiff

np.random.seed(0)

def quad_proj(N_omega, N_z, x_index, y_index, proj):
    """
    Gives a matrix with the proj matrix on the x_index block on the vertical
    and y_index block on the horizontal
    """
    if x_index >= 2*N_z - 1 or y_index >= 2*N_z - 1:
        raise ValueError("Invalid indices")
    left_upper = sparse.csc_matrix((x_index*N_omega, y_index*N_omega))
    upper_right = sparse.csc_matrix((x_index*N_omega, (2*N_z - 1 - y_index)*N_omega))
    left_lower = sparse.csc_matrix(((2*N_z - 1 - x_index)*N_omega, y_index*N_omega))
    right_lower = sparse.bmat([[proj, sparse.csc_matrix((N_omega, (2*N_z - 2 - y_index)*N_omega))],
                                [sparse.csc_matrix(((2*N_z - 2 - x_index)*N_omega, N_omega)), sparse.csc_matrix(((2*N_z - 2 - x_index)*N_omega,(2*N_z - 2 - y_index)*N_omega))]])
    mat = sparse.bmat([[left_upper, upper_right],
                        [left_lower, right_lower]])
    return mat

def order_sing_vals(N_omega, n, U_plus, U_minus):
    """
    Gives the ordered singular values of U_+N such that the relation
    d_+i = 1/(nd_-i) is true. U_plus and U_minus are matrices
    """
    D_plus = scipy.linalg.svd(U_plus)[1]
    D_minus = scipy.linalg.svd(U_minus)[1]
    ordered_D_plus = np.zeros(N_omega)
    for i in range(N_omega):
        pos = np.where(np.round(D_plus[i], 6) == np.round(1/(n*D_minus), 6))[0]
        ordered_D_plus[pos] = D_plus[i]
    return ordered_D_plus

def diag_mat(N_omega, N_z, proj):
    """
    List of diagonal matrices with fully sized proj
    """
    diag_mats = []
    for i in range(2*N_z - 1):
        full_proj = sparse.csc_matrix(0.5*(proj + proj.conj().T))
        full_proj.resize(((2*N_z - 1 - i)*N_omega, (2*N_z - 1 - i)*N_omega))
        full_proj = sparse.bmat([[sparse.csc_matrix((i*N_omega, i*N_omega)), sparse.csc_matrix((i*N_omega, (2*N_z - 1 - i)*N_omega))],[sparse.csc_matrix(((2*N_z - 1 - i)*N_omega, i*N_omega)), full_proj]])
        diag_mats.append(full_proj)
    return diag_mats

def get_lin_matrices(N_z, N_omega, proj):
    """
    List of projection matrices for every propagator
    """
    lin = []
    for i in range(2*N_z - 1):
        lin_mat = proj.copy()
        lin_mat.resize(((2*N_z - 1 - i)*N_omega, N_omega))
        lin_mat = sparse.vstack([sparse.csc_matrix((i*N_omega, N_omega)), lin_mat])
        lin.append(sparse.csc_matrix(lin_mat))
    return lin

def get_green_f(omega, z):
    return [np.diag(np.exp(1.j*omega*z[i])) for i in range(len(z))]

def sdr_def_constr(N_omega, N_z, proj):
    """
    Give the constraint that the SDR is linked with the QCQP
    """
    constr_sdr_def = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), sparse.csc_matrix(((2*N_z - 1)*N_omega,N_omega))],
                                  [sparse.csc_matrix((N_omega,(2*N_z - 1)*N_omega)), proj + proj.conj().T]])
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
        green_f = get_green_f(omega, z[:i + 1])
        green_f.reverse()
        green_f[0] = 0.5*green_f[0]
        green_f[-1] = 0.5*green_f[-1]
        projected_green_f = [sparse.csc_matrix(green_f[i]@proj.conj()) for i in range(len(green_f))]
        dyn_green_f = [pump_power*delta_z*projected_green_f[i + 1] for i in range(i)]
        stacked_dynamics = sparse.hstack(dyn_green_f)
        if prop_sign == "plus":
            stacked_dynamics.resize((N_omega, (2*N_z - 1)*N_omega))
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z - 2)*N_omega, (2*N_z - 1)*N_omega)), stacked_dynamics])
            dynamics_real_list.append(0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(-0.5*(1.j*stacked_dynamics - 1.j*stacked_dynamics.conj().T))
        if prop_sign == "minus":
            stacked_dynamics.resize((N_omega, (N_z)*N_omega))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((N_omega, (N_z - 1)*N_omega)), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z - 2)*N_omega, (2*N_z - 1)*N_omega)), stacked_dynamics])
            dynamics_real_list.append(-0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(0.5*(1.j*stacked_dynamics - 1.j*stacked_dynamics.conj().T))
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
    for i in range(1, N_z):
        green_f = get_green_f(omega, z[:i + 1])[-1]
        cst_green_f = (1/np.sqrt(n))*np.trace(proj.conj().T@green_f)
        lin_green_f = get_lin_matrices(N_z, N_omega, 0.5*(1/np.sqrt(n))*delta_z*pump_power*(proj.conj().T@green_f).conj().T)[-1]
        dynamics_real_plus_sdr.append(sparse.bmat([[dynamics_real_plus[i - 1], 0.5*(-U_lin_list[i - 1] + lin_green_f)],
                                                   [0.5*(-U_lin_list[i - 1] + lin_green_f).conj().T, np.real(cst_green_f/N_omega)*sparse.eye(N_omega)]]))
        dynamics_imag_plus_sdr.append(sparse.bmat([[dynamics_imag_plus[i - 1], 0.5*(1.j*(-U_lin_list[i - 1] + lin_green_f))],
                                                   [0.5*(1.j*(-U_lin_list[i - 1] + lin_green_f)).conj().T, np.imag(cst_green_f/N_omega)*sparse.eye(N_omega)]]))
        dynamics_real_minus_sdr.append(sparse.bmat([[dynamics_real_minus[i - 1], 0.5*(-U_lin_list[i - 2 + N_z] - lin_green_f)],
                                                    [0.5*(-U_lin_list[i - 2 + N_z] - lin_green_f).conj().T, np.real(cst_green_f/N_omega)*sparse.eye(N_omega)]]))
        dynamics_imag_minus_sdr.append(sparse.bmat([[dynamics_imag_minus[i - 1], 0.5*(1.j*(-U_lin_list[i - 2 + N_z] - lin_green_f))],
                                                    [0.5*(1.j*(-U_lin_list[i - 2 + N_z] - lin_green_f)).conj().T, np.imag(cst_green_f/N_omega)*sparse.eye(N_omega)]]))
    return dynamics_real_plus_sdr, dynamics_imag_plus_sdr, dynamics_real_minus_sdr, dynamics_imag_minus_sdr

def sympl_constr_sdr(N_omega, N_z, proj, n):
    """
    Get full symplectic matrices for the given projection matrix
    """
    real_symplectic = []
    imag_symplectic = []
    real_cst_proj = -0.5*np.trace((proj + proj.conj().T).toarray())
    for i in range(N_z - 1):
        proj_copy = proj.copy()
        proj_copy.resize(((2*N_z - i - 1)*N_omega, (N_z - i)*N_omega))
        proj_copy = sparse.vstack([sparse.csc_matrix((i*N_omega, (N_z - i)*N_omega)), proj_copy])
        proj_copy = sparse.hstack([sparse.csc_matrix(((2*N_z - 1)*N_omega, (N_z + i - 1)*N_omega)), proj_copy])
        real_proj = 0.5*(proj_copy + proj_copy.conj().T)
        real_proj = sparse.bmat([[real_proj, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), (1/n)*(real_cst_proj/N_omega)*sparse.eye(N_omega)]])
        imag_proj = 0.5*(1.j*proj_copy + (1.j*proj_copy).conj().T)
        imag_proj = sparse.bmat([[imag_proj, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
        real_symplectic.append(real_proj)
        imag_symplectic.append(imag_proj)
    return real_symplectic, imag_symplectic

def photon_nbr_constr(N_omega, N_z, n):
    """
    Fixes the photon number at end of waveguide
    """
    diags = 0.25*(diag_mat(N_omega, N_z, sparse.eye(N_omega))[N_z - 2] + diag_mat(N_omega, N_z, sparse.eye(N_omega))[2*N_z - 3])
    diags = sparse.bmat([[diags, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                         [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), -((N_omega/(2*n) + 1)/N_omega)*sparse.eye(N_omega)]])
    return diags

def constr_upper_quadratic_prop_sympl(N_z, N_omega, project):
    """
    Gives the matrices stating the upper part of the symplectic block has every matrices on the diagonals equal
    """
    mats = []
    terms = [1., 1.j]
    for k in range(len(terms)):
        proj = terms[k]*project.copy()
        for i in range(N_z - 3):
            for j in range(N_z - 3 - i):
                x_index_previous = j
                x_index = j + 1
                y_index_previous = i + j + N_z
                y_index = i + j + N_z + 1
                mat = quad_proj(N_omega, N_z, x_index_previous, y_index_previous, proj) - quad_proj(N_omega, N_z, x_index, y_index, proj)
                mat.resize((((2*N_z*N_omega),(2*N_z*N_omega))))
                mats.append(0.5*(mat + mat.conj().T))
    return mats

def constr_lower_quadratic_prop_sympl(N_omega, N_z, project):
    """
    Gives the matrices stating the lower part of the symplectic block has every matrices on the diagonals equal
    """
    mats = []
    terms = [1., 1.j]
    for k in range(len(terms)):
        proj = terms[k]*project.copy()
        for i in range(N_z - 3):
            for j in range(N_z - 3 - i):
                x_index_previous = j + i + 1
                x_index = i + j + 2
                y_index_previous = N_z - 1 + j
                y_index = N_z + j
                mat = quad_proj(N_omega, N_z, x_index_previous, y_index_previous, proj) - quad_proj(N_omega, N_z, x_index, y_index, proj)
                mat.resize((((2*N_z*N_omega),(2*N_z*N_omega))))
                mats.append(0.5*(mat + mat.conj().T))
    return mats

def constr_sympl_minus(N_z, N_omega, n, project):
    """
    Gives the matrices building the relation between the propagators and
    the symplectic quadratic form of propagators for the minus propagators
    """
    mats = []
    terms = [1., 1.j]
    for j in range(len(terms)):
        proj = terms[j]*project
        for i in range(N_z - 2):
            x_index = 0
            y_index = N_z + i
            quad = 0.5*(quad_proj(N_omega, N_z, x_index, y_index, proj.conj().T) + quad_proj(N_omega, N_z, x_index, y_index, proj.conj().T).conj().T)
            lin = -(1/np.sqrt(n))*get_lin_matrices(N_z, N_omega, proj)[N_z - 1 + i]
            mat = sparse.bmat([[quad, 0.5*lin],
                               [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
            mats.append(mat)
    return mats

def constr_sympl_plus(N_z, N_omega, n, project):
    """
    Gives the matrices building the relation between the propagators and
    the symplectic quadratic form of propagators for the plus propagators
    """
    mats = []
    terms = [1., 1.j]
    for j in range(len(terms)):
        proj = terms[j]*project
        for i in range(N_z - 2):
            x_index = i + 1
            y_index = N_z - 1
            quad = 0.5*(quad_proj(N_omega, N_z, x_index, y_index, proj) + quad_proj(N_omega, N_z, x_index, y_index, proj).conj().T)
            lin = -(1/np.sqrt(n))*get_lin_matrices(N_z, N_omega, proj)[i]
            mat = sparse.bmat([[quad, 0.5*lin],
                               [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
            mats.append(mat)
    return mats

def backwards_fin_diff_quad_plus(N_omega, z, beta_weight, delta_k, proj):
    """
    Gives the matrices for a projection that apply the backwards finite difference constraint
    on the quadratic parts of the plus propagators and on the quadratic part with the pump
    """
    N_z = len(z)
    coeffs = findiff.coefficients(deriv = 1, offsets=list(np.arange(0, N_z)))["coefficients"]
    delta_z = np.abs(z[1] - z[0])
    real_mat = []
    imag_mat = []
    for j in range(N_z - 1):
        quad_j_prop = sum([(1/delta_z)*coeffs[i + 1]*quad_proj(N_omega, N_z, j, i, proj) for i in range(len(coeffs) - 1)])
        quad_j_pump = beta_weight*quad_proj(N_omega, N_z, j, 2*N_z - 2, proj)
        quad = quad_j_prop - quad_j_pump
        lin = (get_lin_matrices(N_omega=N_omega, N_z = N_z, proj = (coeffs[0]/delta_z)*proj - proj@delta_k)[j])
        real_mat.append(sparse.bmat([[0.5*(quad + quad.conj().T), 0.5*lin],
                                    [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
        imag_mat.append(sparse.bmat([[-0.5*1.j*(quad - quad.conj().T), -0.5*1.j*lin],
                                     [0.5*1.j*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
    return real_mat + imag_mat

def backwards_fin_diff_quad_minus(N_omega, z, beta_weight, delta_k, proj):
    """
    Gives the matrices for a projection that apply the backwards finite difference constraint
    on the quadratic parts of the minus propagators and on the quadratic part with the pump
    """
    N_z = len(z)
    coeffs = findiff.coefficients(deriv = 1, offsets=list(np.arange(0, N_z)))["coefficients"]
    delta_z = np.abs(z[1] - z[0])
    real_mat = []
    imag_mat = []
    for j in range(N_z - 1):
        quad_j_prop = sum([(1/delta_z)*coeffs[i + 1]*quad_proj(N_omega, N_z, N_z - 1 + j, N_z - 1 + i, proj) for i in range(len(coeffs) - 1)])
        quad_j_pump = beta_weight*quad_proj(N_omega, N_z, N_z - 1 + j, 2*N_z - 2, proj)
        quad = quad_j_prop + quad_j_pump
        lin = (get_lin_matrices(N_omega=N_omega, N_z = N_z, proj = (coeffs[0]/delta_z)*proj - proj@delta_k)[N_z - 1 + j])
        real_mat.append(sparse.bmat([[0.5*(quad + quad.conj().T), 0.5*lin],
                                    [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
        imag_mat.append(sparse.bmat([[-0.5*1.j*(quad - quad.conj().T), -0.5*1.j*lin],
                                     [0.5*1.j*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
    return real_mat + imag_mat

def limit_pump_power(omega, N_z):
    """
    Constraint limiting the trace of beta dagger beta to be 1
    """
    N_omega = len(omega)
    delta_omega = np.abs(omega[1] - omega[0])
    quad = sparse.eye(N_omega)
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)), sparse.csc_matrix(((2*N_z - 2)*N_omega, N_omega))],
                        [sparse.csc_matrix((N_omega, (2*N_z - 2)*N_omega)), quad]])
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), -((delta_omega**2)/N_omega)*sparse.eye(N_omega)]])
    return mat

def photon_nbr_prev_points(N_omega, N_z):
    """
    Gives the matrix for the constraint that says mean photon number increases through waveguide
    """
    constraint = []
    diags = diag_mat(N_omega, N_z, sparse.eye(N_omega))
    for i in range(N_z - 2):
        quad = 0.25*(diags[i + 1] + diags[N_z + i] - diags[i] - diags[N_z + i - 1])
        constraint.append(sparse.bmat([[quad, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                                       [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]]))
    return constraint


def sdr_fixed_pump(N_omega, N_z, beta, proj):
    """
    Gives the linear SDR constraint on fixing the pump
    """
    lin_real = 0.5*(get_lin_matrices(N_z, N_omega, proj)[2*N_z - 2])
    lin_imag = 0.5*1.j*(get_lin_matrices(N_z, N_omega, proj)[2*N_z - 2])
    mat_real = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), lin_real],
                            [lin_real.conj().T, -(np.real(np.trace(proj.conj().T@beta))/N_omega)*sparse.eye(N_omega)]])
    mat_imag = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), lin_imag],
                            [lin_imag.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat_real, mat_imag

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
        prior_lin = get_lin_matrices(N_z, N_omega, prior_mats[i])[2*N_z - 2]
        after_lin = -get_lin_matrices(N_z, N_omega, after_mats[i])[2*N_z - 2]
        lin = prior_lin + after_lin
        mat = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin],
                           [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_list.append(mat)
    return mat_list

def constr_first_last_row_pump(N_omega, N_z, fixed_beta, free_indices):
    """
    Gives list of matrices to constraint the first row and last row
    of the pump to be fixed. The parameter free_indices means how many
    indices in frequency domain of the pump are free. It should not be 
    greater than 2*N_omega - 1
    """
    mats = []
    lower_row_indices = free_indices//2
    upper_row_indices = free_indices - lower_row_indices
    upper_stop = N_omega - 1 - upper_row_indices
    for i in range(2*N_omega - 1):
        proj = np.zeros((N_omega, N_omega))
        if i <= N_omega - 1 and i <= upper_stop:
            proj[0, i] = 1
        elif i > lower_row_indices + N_omega - 1:
            proj[N_omega - 1, i - N_omega + 1] = 1
        else:
            continue
        cst = 0.5*np.trace(proj.conj().T@fixed_beta + proj@fixed_beta.conj().T)
        lin = get_lin_matrices(N_z, N_omega, proj)[2*N_z - 2]
        mat = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin],
                            [0.5*lin.conj().T, -(cst/N_omega)*sparse.eye(N_omega)]])
        mats.append(mat)
    return mats

def basic_affine_ineq_pump(N_omega, N_z, fixed_pump, margin):
    """
    Gives matrices to enforce affine inequality constraints on the freely
    varying indices of the pump. Taking in a fixed pump, it allows the pump 
    to vary within a certain margin 
    """
    one_hot_list = list(np.eye(N_omega))
    positions = np.arange(0, N_omega)
    upper_hankel_gen = lambda pos:np.vstack((one_hot_list[pos], np.zeros((N_omega - 1, N_omega))))
    lower_hankel_gen = lambda pos:np.vstack((np.zeros((N_omega - 1, N_omega)), one_hot_list[pos]))
    projections = list(map(upper_hankel_gen, positions)) + list(map(lower_hankel_gen, positions[1:]))
    mats = []
    for i in range(len(projections)):
        upper_bound = fixed_pump[i] + margin
        lower_bound = fixed_pump[i] - margin
        lin = get_lin_matrices(N_z, N_omega, projections[i])[2*N_z - 2]
        leq_mat = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega, (2*N_z - 1)*N_omega)), 0.5*lin],
                               [0.5*lin.conj().T, -(upper_bound/N_omega)*sparse.eye(N_omega)]])
        geq_mat =sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega, (2*N_z - 1)*N_omega)), -0.5*lin],
                               [-0.5*lin.conj().T, (lower_bound/N_omega)*sparse.eye(N_omega)]])
        mats += [leq_mat, geq_mat]
    return mats

def pump_quad_ineq(N_omega, N_z):
    """
    Gives the constraint that the sum of square of every
    element of the pump is equal to 1
    """
    proj = np.zeros((N_omega, N_omega))
    proj[0,0] = 1
    proj[N_omega - 1, N_omega - 1] = 1
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)), sparse.csc_matrix(((2*N_z - 2)*N_omega, N_omega))],
                        [sparse.csc_matrix((N_omega, (2*N_z - 2)*N_omega)), proj]])
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), (-1/N_omega)*sparse.eye(N_omega)]])
    return mat

def pump_exp_decay_constr(N_omega, N_z, left_decay_sign, right_decay_sign, position, decay_curve, max_ampli = 0.1):
    """
    Gives matrix representing constraint that the pump will decay exponentially 
    """
    delta_omega = np.log(decay_curve)/position
    left = (max_ampli/decay_curve)*np.exp(np.linspace(0, position*delta_omega, position))
    right = (max_ampli/decay_curve)*np.exp(np.linspace(position*delta_omega, 0, position))
    mat_left = []
    mat_right = []
    for i in range(position):
        if left_decay_sign == "positive":
            proj_left = np.zeros((N_omega, N_omega))
            proj_left[0, i] = 1
        elif left_decay_sign == "negative":
            proj_left = np.zeros((N_omega, N_omega))
            proj_left[0, i] = - 1
        if right_decay_sign == "positive":
            proj_right = np.zeros((N_omega, N_omega))
            proj_right[-1, N_omega - 1 - i] = 1
        elif right_decay_sign == "negative":
            proj_right = np.zeros((N_omega, N_omega))
            proj_right[-1, N_omega - 1 - i] = - 1
        else:
            raise ValueError("Invalid decay sign")
        lin_left = get_lin_matrices(N_z, N_omega, proj_left)[2*N_z - 2]
        lin_right = get_lin_matrices(N_z, N_omega, proj_right)[2*N_z - 2]
        mat_left.append(sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin_left],
                                    [0.5*lin_left.conj().T, -(left[i]/N_omega)*sparse.eye(N_omega)]]))
        mat_right.append(sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin_right],
                                    [0.5*lin_right.conj().T, -(right[position - i - 1]/N_omega)*sparse.eye(N_omega)]]))
    return mat_left, mat_right

def isolate_vect_pump(N_omega, N_z):
    """
    Gives the matrices to isolate the entire first row of the pump and the last row of pump except the first element
    """
    left_first_half = np.eye((2*N_z*N_omega))[(2*N_z - 2)*N_omega]
    right_first_half = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[-1]
    right_first_half = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), right_first_half])
    left_second_half = np.eye((2*N_z*N_omega))[(2*N_z - 1)*N_omega - 1]
    right_second_half = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[-1][:, 1:]
    right_second_half = sparse.vstack([sparse.csc_matrix((N_omega, N_omega - 1)), right_second_half])
    return left_first_half, left_second_half, right_first_half, right_second_half

def get_hermite_polynom_mat(omega, max_order, width):
    """
    Gives the matrix with Hermite basis element on every row. Width is the width of the Gaussian in the basis.
    """
    N_omega = len(omega)
    increase_omega = np.linspace(omega[0], omega[-1], 2*N_omega - 1)
    gauss_modes_mat = (1/np.sqrt(np.sqrt(np.pi)))*np.exp(-increase_omega**2/width)
    for i in range(1, max_order):
        norm_cst = (1/np.sqrt((2**i)*math.factorial(i)*np.sqrt(np.pi)))
        gauss_modes_mat = np.vstack([gauss_modes_mat, norm_cst*scipy.special.hermite(i)(increase_omega)*np.exp(-increase_omega**2/width)])
    return gauss_modes_mat

def obj_f_mat(N_omega, N_z):
    """
    Gives the matrices to isolate Z_U_plus_dagger and Z_U_minus_dagger for the
    objective function
    """
    left = np.sqrt(0.5)*get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 2]
    left = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), left])
    right = np.sqrt(0.5)*(get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 2] - get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 3])
    right.resize((2*N_z*N_omega, N_omega))
    return left, right