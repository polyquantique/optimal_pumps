import numpy as np
import scipy.sparse as sparse
import scipy
import math

np.random.seed(0)

def quad_proj(N_omega, N_z, x_index, y_index, proj):
    """
    Gives a matrix with the proj matrix on the x_index block on the vertical
    and y_index block on the horizontal
    """
    if x_index >= 2*N_z + 1 or y_index >= 2*N_z + 1:
        raise ValueError("Invalid indices")
    left_upper = sparse.csc_matrix((x_index*N_omega, y_index*N_omega))
    upper_right = sparse.csc_matrix((x_index*N_omega, (2*N_z + 1 - y_index)*N_omega))
    left_lower = sparse.csc_matrix(((2*N_z + 1 - x_index)*N_omega, y_index*N_omega))
    right_lower = sparse.bmat([[proj, sparse.csc_matrix((N_omega, (2*N_z - y_index)*N_omega))],
                                [sparse.csc_matrix(((2*N_z - x_index)*N_omega, N_omega)), sparse.csc_matrix(((2*N_z - x_index)*N_omega,(2*N_z - y_index)*N_omega))]])
    mat = sparse.bmat([[left_upper, upper_right],
                        [left_lower, right_lower]])
    return mat

def get_lin_matrices(N_z, N_omega, proj):
    """
    List of projection matrices for every propagator
    """
    lin = []
    for i in range(2*N_z + 1):
        lin_mat = proj.copy()
        lin_mat.resize(((2*N_z + 1 - i)*N_omega, N_omega))
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

def get_dynamics_matrices(omega, z, beta_weight, proj):
    """
    Gives the matricees for the dynamics relating U_plus - U_minus and U_plus + U_minus
    """
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    mats = []
    N_omega = len(omega)
    for end in range(2, N_z + 1):
        green_fs = get_green_f(omega, z[:end])
        green_fs.reverse()
        quad_plus = quad_proj(N_omega, N_z, end - 1, 2*N_z, 0.5*beta_weight*delta_z*proj)+ quad_proj(N_omega, N_z, 0, 2*N_z, 0.5*beta_weight*delta_z*proj@green_fs[0].conj().T)
        quad_minus = quad_proj(N_omega, N_z, N_z + end - 1, 2*N_z, 0.5*beta_weight*delta_z*proj) + quad_proj(N_omega, N_z, N_z, 2*N_z, 0.5*beta_weight*delta_z*proj@green_fs[0].conj().T)
        lin = get_lin_matrices(N_z, N_omega, proj)[end - 1] - get_lin_matrices(N_z, N_omega, proj)[N_z + end - 1]
        for i in range(1, end - 1):
            quad_plus += quad_proj(N_omega, N_z, i, 2*N_z, beta_weight*delta_z*proj@green_fs[i].conj().T)
            quad_minus += quad_proj(N_omega, N_z, i + N_z, 2*N_z, beta_weight*delta_z*proj@green_fs[i].conj().T)
        quad_real = 0.5*(quad_plus + quad_minus + (quad_plus + quad_minus).conj().T)
        quad_imag = -0.5*1.j*(quad_plus + quad_minus - (quad_plus + quad_minus).conj().T)
        mat_real = sparse.bmat([[quad_real, -0.5*lin],
                                [-0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_imag = sparse.bmat([[quad_imag, 0.5*1.j*lin],
                                [-0.5*1.j*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mats += [mat_real, mat_imag]
    return mats

def sympl_constr_sdr(N_omega, N_z, proj, n):
    """
    Gives matrices stating the symplectic properties of propagators
    """
    mats = []
    cst = (1/n)*np.trace(proj@np.eye(N_omega))
    for i in range(N_z):        
        quad_sympl = quad_proj(N_omega, N_z, i, N_z + i, proj)
        real_sympl = 0.5*(quad_sympl + quad_sympl.conj().T)
        imag_sympl = 0.5*1.j*(quad_sympl - quad_sympl.conj().T)
        mat_real = sparse.bmat([[real_sympl, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                                [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), -(cst/N_omega)*sparse.eye(N_omega)]])
        mat_imag = sparse.bmat([[imag_sympl, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                                [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
        mats += [mat_real, mat_imag]
    return mats

def photon_nbr_constr(N_omega, N_z, n):
    """
    Fixes the photon number at end of waveguide
    """
    diags = 0.25*(quad_proj(N_omega, N_z, N_z - 1, N_z - 1, sparse.eye(N_omega)) + quad_proj(N_omega, N_z, 2*N_z - 1, 2*N_z - 1, sparse.eye(N_omega)))
    diags = sparse.bmat([[diags, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                         [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), -((N_omega/(2*n) + 1)/N_omega)*sparse.eye(N_omega)]])
    return diags

def photon_nbr_prev_points(N_omega, N_z):
    """
    Gives the matrix for the constraint that says mean photon number increases through waveguide
    """
    constraint = []
    diags = [quad_proj(N_omega, N_z, j, j, sparse.eye(N_omega)) for j in range(2*N_z + 1)]
    for i in range(N_z - 1):
        quad = 0.25*(diags[i + 1] + diags[N_z + i + 1] - diags[i] - diags[N_z + i])
        constraint.append(sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                                       [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]]))
    return constraint

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
        prior_lin = get_lin_matrices(N_z, N_omega, prior_mats[i])[2*N_z]
        after_lin = -get_lin_matrices(N_z, N_omega, after_mats[i])[2*N_z]
        lin = prior_lin + after_lin
        mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*lin],
                           [0.5*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]])
        mat_list.append(mat)
    return mat_list

def limit_pump_power(omega, N_z):
    """
    Constraint limiting the trace of beta dagger beta to be 1
    """
    N_omega = len(omega)
    delta_omega = np.abs(omega[1] - omega[0])
    quad = sparse.eye(N_omega)
    quad = sparse.bmat([[sparse.csc_matrix(((2*N_z)*N_omega,(2*N_z)*N_omega)), sparse.csc_matrix(((2*N_z)*N_omega, N_omega))],
                        [sparse.csc_matrix((N_omega, (2*N_z)*N_omega)), quad]])
    mat = sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                       [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), -((delta_omega**2)/N_omega)*sparse.eye(N_omega)]])
    return mat

def isolate_vect_pump(N_omega, N_z):
    """
    Gives the matrices to isolate the entire first row of the pump and the last row of pump except the first element
    """
    left_first_half = np.eye(((2*N_z + 2)*N_omega))[(2*N_z)*N_omega]
    right_first_half = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[-1]
    right_first_half = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), right_first_half])
    left_second_half = np.eye(((2*N_z + 2)*N_omega))[(2*N_z + 1)*N_omega - 1]
    right_second_half = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[-1][:, 1:]
    right_second_half = sparse.vstack([sparse.csc_matrix((N_omega, N_omega - 1)), right_second_half])
    return left_first_half, left_second_half, right_first_half, right_second_half

def real_pump_constr(N_omega, N_z):
    """
    Gives matrices saying the imaginary part of the pump is 0
    """
    mat = []
    for i in range(2*N_omega - 1):
        proj = np.zeros((N_omega, N_omega))
        proj[-i//N_omega, i%N_omega + i//N_omega] = 1.
        lin = get_lin_matrices(N_z, N_omega, proj)[2*N_z]
        mat.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*1.j*lin],
                                [-0.5*1.j*lin.conj().T, sparse.csc_matrix((N_omega, N_omega))]]))
    return mat

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

def isolate_propagators(N_omega, N_z, index, propagator):
    """
    Isolate the positive or negative propagator whether the value of propagator
    is plus or minus and whose index is the index value
    """
    if propagator == "plus":
        left = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[index - 1]
    elif propagator == "minus":
        left = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z + index - 1]
    left.resize(((2*N_z + 2)*N_omega, N_omega))
    right = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z]
    right = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), right])
    return left, right