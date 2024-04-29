import numpy as np
import scipy.sparse as sparse
import scipy

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
    real_cst_proj = -0.5*(proj + proj.conj().T).trace()
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

def lin_finite_diff_constr(z, N_omega, delta_k, n, beta_weight, project):
    """
    Gives the matrices to constraint the real and imaginary parts of the dynamics when
    expressed as central finite difference for adding or substracting the pump
    """
    terms = [1., 1.j]
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    quad = sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega))
    mats = []
    for i in range(len(terms)):
        proj = terms[i]*project.copy()
        lin_pump = (1/np.sqrt(n))*get_lin_matrices(N_z, N_omega, proj)[-1]
        cst = (0.5/np.sqrt(n))*np.trace(proj.conj().T@delta_k + delta_k.conj().T@proj)
        lin_plus_add = (0.75*get_lin_matrices(N_z, N_omega, proj)[0] - (3/20)*get_lin_matrices(N_z, N_omega, proj)[1] + (1/60)*get_lin_matrices(N_z, N_omega, proj)[2])
        lin_minus_add = -(0.75*get_lin_matrices(N_z, N_omega, proj.conj().T)[3] - (3/20)*get_lin_matrices(N_z, N_omega, proj.conj().T)[4] + (1/60)*get_lin_matrices(N_z, N_omega, proj.conj().T)[5])
        lin_plus_subs = -(0.75*get_lin_matrices(N_z, N_omega, proj.conj().T)[0] - (3/20)*get_lin_matrices(N_z, N_omega, proj.conj().T)[1] + (1/60)*get_lin_matrices(N_z, N_omega, proj.conj().T)[2])
        lin_minus_subs = (0.75*get_lin_matrices(N_z, N_omega, proj)[3] - (3/20)*get_lin_matrices(N_z, N_omega, proj)[4] + (1/60)*get_lin_matrices(N_z, N_omega, proj)[5])
        lin_add = (1/delta_z)*(lin_plus_add + lin_minus_add) - beta_weight*lin_pump
        lin_subs = (1/delta_z)*(lin_plus_subs + lin_minus_subs) + beta_weight*lin_pump
        mat_add = sparse.bmat([[quad, 0.5*lin_add],
                               [0.5*lin_add.conj().T,  -(cst/N_omega)*sparse.eye(N_omega)]])
        mat_subs = sparse.bmat([[quad, 0.5*lin_subs],
                               [0.5*lin_subs.conj().T,  -(cst/N_omega)*sparse.eye(N_omega)]])
        mats += [mat_add, mat_subs]
    return mats

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

def limit_pump_power(N_omega, N_z):
    """
    Constraint limiting the trace of beta dagger beta to be 1
    """
    quad = sparse.eye(N_omega)
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)), sparse.csc_matrix(((2*N_z - 2)*N_omega, N_omega))],
                        [sparse.csc_matrix((N_omega, (2*N_z - 2)*N_omega)), quad]])
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z - 1)*N_omega, N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z - 1)*N_omega)), -(1/N_omega)*sparse.eye(N_omega)]])
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

def constr_relate_quad_lin_min(delta_k, N_omega, z, beta_weight, n, project):
    """
    Constraint relating the quadratic blocks of minus propagators to the linear blocks of minus propagators
    and of the block representing multiplication between the pump and the propagators
    """
    delta_z = np.abs(z[1] - z[0])
    N_z = len(z)
    terms = [1., 1.j]
    mats = []
    for i in range(len(terms)):
        proj = terms[i]*project.copy()
        lin_term = get_lin_matrices(N_z, N_omega, sparse.csc_matrix((-proj.T@delta_k + (1.5/delta_z)*proj.T).conj().T))[N_z]
        diff_diag_term = sparse.csc_matrix((-np.sqrt(n)/delta_z)*proj.conj().T)
        quad_term_pump = sparse.csc_matrix(-0.5*beta_weight*proj.conj().T)
        quad_term = sparse.csc_matrix((np.sqrt(n)/(4*delta_z))*(proj + proj.conj().T))
        mat = sparse.bmat([[sparse.csc_matrix((N_omega, N_omega)), diff_diag_term.conj().T],
                           [diff_diag_term, quad_term]])
        mat = sparse.bmat([[sparse.csc_matrix(((N_z - 1)*N_omega,(N_z - 1)*N_omega)), sparse.csc_matrix(((N_z - 1)*N_omega, 2*N_omega))],
                           [sparse.csc_matrix((2*N_omega, (N_z - 1)*N_omega)), mat]])
        lower_block = sparse.bmat([[sparse.csc_matrix(((N_z - 3)*N_omega, N_z*N_omega)), sparse.csc_matrix(((N_z - 3)*N_omega, N_omega))],
                                   [sparse.csc_matrix((N_omega, N_z*N_omega)), quad_term_pump.conj().T]])
        mat = sparse.bmat([[mat, lower_block.conj().T],
                           [lower_block, sparse.csc_matrix(((N_z - 2)*N_omega, (N_z - 2)*N_omega))]])
        mat = np.sqrt(n)*sparse.bmat([[mat, 0.5*lin_term],
                       [0.5*lin_term.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mats.append(mat)
    return mats

def constr_relate_quad_lin_plus(delta_k, N_omega, z, beta_weight, n, project):
    """
    Constraint relating the quadratic blocks of minus propagators to the linear blocks of minus propagators
    and of the block representing multiplication between the pump and the propagators
    """
    delta_z = np.abs(z[1] - z[0])
    N_z = len(z)
    terms = [1., 1.j]
    mats = []
    for i in range(len(terms)):
        proj = terms[i]*project.copy()
        lin_term = get_lin_matrices(N_z, N_omega, sparse.csc_matrix((-proj.T@delta_k + (1.5/delta_z)*proj.T).conj().T))[N_z - 2]
        diff_diag_term = sparse.csc_matrix((-np.sqrt(n)/delta_z)*proj.conj().T)
        quad_term_pump = sparse.csc_matrix(0.5*beta_weight*proj.conj().T)
        quad_term = sparse.csc_matrix((np.sqrt(n)/(4*delta_z))*(proj + proj.conj().T))
        mat = sparse.bmat([[sparse.csc_matrix((N_omega, N_omega)), diff_diag_term.conj().T],
                           [diff_diag_term, quad_term]])
        mat = sparse.bmat([[mat, sparse.csc_matrix((2*N_omega, (2*N_z - 4)*N_omega))],
                           [sparse.csc_matrix(((2*N_z - 4)*N_omega, 2*N_omega)), sparse.csc_matrix(((2*N_z - 4)*N_omega,(2*N_z - 4)*N_omega))]])
        lower_block = sparse.hstack([sparse.csc_matrix((N_omega, N_omega)), quad_term_pump.conj().T, sparse.csc_matrix((N_omega, (2*N_z - 4)*N_omega))])
        mat = sparse.bmat([[mat, lower_block.conj().T],
                           [lower_block, sparse.csc_matrix((N_omega, N_omega))]])
        mat = np.sqrt(n)*sparse.bmat([[mat, 0.5*lin_term],
                           [0.5*lin_term.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mats.append(mat)
    return mats

def constr_lin_quad_pump_minus(N_omega, z, beta_weight, n, delta_k, project):
    """
    Gives the matrices to fix the linear terms of minus propagators with the
    block representing multiplication between the pump and U_minus(2)
    """
    delta_z = np.abs(z[1] - z[0])
    N_z = len(z)
    terms = [1., 1.j]
    mats = []
    for i in range(len(terms)):
        proj = terms[i]*project.copy()
        quad = sparse.hstack([sparse.csc_matrix((N_omega,(2*N_z - 3)*N_omega)), beta_weight*proj])
        quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)),  sparse.csc_matrix(((2*N_z - 2)*N_omega, N_omega))],
                            [quad, sparse.csc_matrix((N_omega, N_omega))]])
        quad = 0.5*(quad + quad.conj().T)
        lin_second = get_lin_matrices(N_z, N_omega, (1.5/delta_z)*proj.conj().T + proj.conj().T@delta_k)[2*N_z - 3]
        lin_first = get_lin_matrices(N_z, N_omega, -(2/delta_z)*proj.conj().T)[2*N_z - 4]
        lin = lin_second + lin_first
        cst = np.real((0.5/(delta_z*np.sqrt(n)))*(proj.conj().T).trace())
        mat = sparse.bmat([[quad, 0.5*lin],
                           [0.5*lin.conj().T, (cst/N_omega)*sparse.eye(N_omega)]])
        mats.append(mat)
    return mats

def constr_lin_quad_pump_plus(N_omega, z, beta_weight, n, delta_k, project):
    """
    Gives the matrices to fix the linear terms of plus propagators with the
    block representing multiplication between the pump and U_pluus(2)
    """
    delta_z = np.abs(z[1] - z[0])
    N_z = len(z)
    terms = [1., 1.j]
    mats = []
    for i in range(len(terms)):
        proj = terms[i]*project.copy()
        quad = sparse.hstack([sparse.csc_matrix((N_omega,N_omega)), -beta_weight*proj, sparse.csc_matrix((N_omega, (2*N_z - 4)*N_omega))])
        quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)),  sparse.csc_matrix(((2*N_z - 2)*N_omega, N_omega))],
                            [quad, sparse.csc_matrix((N_omega, N_omega))]])
        quad = 0.5*(quad + quad.conj().T)
        lin_second = get_lin_matrices(N_z, N_omega, ((1.5/delta_z)*proj.conj().T + proj.conj().T@delta_k))[1]
        lin_first = get_lin_matrices(N_z, N_omega, (-(2/delta_z)*proj.conj().T))[0]
        lin = lin_second + lin_first
        cst = np.real((0.5/(delta_z*np.sqrt(n)))*(proj.conj().T).trace())
        mat = sparse.bmat([[quad, 0.5*lin],
                            [0.5*lin.conj().T, (cst/N_omega)*sparse.eye(N_omega)]])
        mats.append(mat)
    return mats


def constr_relate_quad_mid_lin_finite_diff(N_omega, N_z, beta_weight, n, delta_z, delta_k, proj):
    """
    Gives the matrix relating the linear part of the pump to the quadratic parts of 
    pump and of the propagators
    """
    plus_pump = (0.5*np.sqrt(n)*beta_weight/delta_z)*proj.conj().T
    minus_pump = -0.5*(np.sqrt(n)*beta_weight/delta_z)*proj.conj()
    lin_pump = proj.conj().T@delta_k
    quad_pump = -proj*beta_weight**2
    vert = sparse.vstack([plus_pump, sparse.csc_matrix(((2*N_z - 3)*N_omega, N_omega))])
    horiz = sparse.hstack([sparse.csc_matrix((N_omega, (N_z - 1)*N_omega)), minus_pump, sparse.csc_matrix((N_omega, (N_z - 2)*N_omega))])
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z - 2)*N_omega,(2*N_z - 2)*N_omega)), vert],
                       [horiz, quad_pump]])
    quad = 0.5*(quad + quad.conj().T)
    lin = get_lin_matrices(N_z, N_omega, -beta_weight*delta_k@proj)[2*N_z - 2]
    mat = sparse.bmat([[quad, 0.5*lin],
                    [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
    return mat

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

def basic_affine_ineq_pump(N_omega, N_z, free_indices, high_bound):
    """
    Gives matrices to enforce affine inequality constraints on the freely
    varying indices of the pump. The inequality is a box inequality, so values
    of the pump is in between +high_bound and -high_bound
    """
    lower_row_indices = free_indices//2
    upper_row_indices = free_indices - lower_row_indices
    high_mats = []
    low_mats = []
    for i in range(upper_row_indices):
        hankel_vec_upper = np.zeros(N_omega)
        hankel_vec_lower = np.zeros(N_omega)
        if i == 0:
            hankel_vec_upper[-1] = 1
        else:
            hankel_vec_upper[N_omega - 1 - i] = 1
            hankel_vec_lower[i] = 1
        proj_upper = sparse.csc_matrix(scipy.linalg.hankel(hankel_vec_upper, np.zeros(N_omega)))
        proj_lower = sparse.csc_matrix(scipy.linalg.hankel(np.zeros(N_omega), hankel_vec_lower))
        lin_upper = get_lin_matrices(N_z, N_omega, proj_upper)[(2*N_z - 2)]
        lin_lower = get_lin_matrices(N_z, N_omega, proj_lower)[(2*N_z - 2)]
        high_mat_upper = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin_upper],
                                [0.5*lin_upper.conj().T, -(high_bound/N_omega)*sparse.eye(N_omega)]])
        high_mat_lower = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin_lower],
                                [0.5*lin_lower.conj().T, -(high_bound/N_omega)*sparse.eye(N_omega)]])
        low_mat_upper = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin_upper],
                                [0.5*lin_upper.conj().T, (high_bound/N_omega)*sparse.eye(N_omega)]])
        low_mat_lower = sparse.bmat([[sparse.csc_matrix(((2*N_z - 1)*N_omega,(2*N_z - 1)*N_omega)), 0.5*lin_lower],
                                [0.5*lin_lower.conj().T, (high_bound/N_omega)*sparse.eye(N_omega)]])
        high_mats += [high_mat_upper, high_mat_lower]
        low_mats += [low_mat_upper, low_mat_lower]
    return high_mats, low_mats

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