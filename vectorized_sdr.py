import numpy as np
import scipy.sparse as sparse

def mat_to_vec(matrix):
    vec = []
    for i in range(len(matrix)):
        vec += list(matrix[i])
    return sparse.csc_array(vec).T

def diag_mat(N_omega, N_z, proj):
    diag_mats = []
    for i in range(2*N_z + 1):
        full_proj = sparse.kron(sparse.eye(N_omega), proj)
        full_proj.resize(((2*N_z + 1 - i)*(N_omega**2) + 2*N_omega - 1, (2*N_z + 1 - i)*(N_omega**2) + 2*N_omega - 1))
        full_proj = sparse.bmat([[sparse.csc_matrix((i*(N_omega**2), i*(N_omega**2))), sparse.csc_matrix((i*(N_omega**2), (2*N_z + 1 - i)*(N_omega**2) + 2*N_omega - 1))],[sparse.csc_matrix(((2*N_z + 1 - i)*(N_omega**2) + 2*N_omega - 1, i*(N_omega**2))), full_proj]])
        diag_mats.append(full_proj)
    return diag_mats

def vectorized_proj(N_omega, N_z, proj):
    vect_proj_list = []
    for i in range(2*N_z + 1):
        vect_proj = mat_to_vec(proj)
        vect_proj.resize(((2*N_z + 1 - i)*(N_omega**2) + 2*N_omega - 1, 1))
        vect_proj = sparse.vstack([sparse.csc_matrix((i*(N_omega**2), 1)), vect_proj])
        vect_proj_list.append(vect_proj)
    return vect_proj_list

def get_green_f(omega, z):
    return [np.diag(np.exp(1.j*omega*z[i])) for i in range(len(z))]

def omega_vec_to_omega_dof(N_omega):
    beta_len = 2*N_omega - 1
    vec_to_mat_hankel = []
    for i in range(N_omega):
        vec_to_hankel_line = sparse.eye(N_omega)
        vec_to_hankel_line.resize((N_omega, beta_len - i))
        vec_to_hankel_line = sparse.hstack([sparse.csc_matrix((N_omega, i)), vec_to_hankel_line])
        vec_to_mat_hankel.append(vec_to_hankel_line)
    hankel = sparse.vstack(vec_to_mat_hankel)
    return hankel

def dynamics_W(omega, z, proj, vec_to_hankel, prop_sign):
    """
    delta_v = 1, prop_sign either "plus" or "minus"
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    dynamics_real_list = [sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1, (2*N_z + 1)*(N_omega**2) + 2*N_omega - 1))]
    dynamics_imag_list = [sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1, (2*N_z + 1)*(N_omega**2) + 2*N_omega - 1))]
    for i in range(1, N_z):
        green_fs = get_green_f(omega, z[:i + 1])
        projected_green_fs = [sparse.csc_matrix(green_fs[i]@proj) for i in range(len(green_fs))]
        full_proj_green_fs = [delta_z*sparse.kron(sparse.eye(N_omega), projected_green_fs[i]) for i in range(len(green_fs))]
        rect_proj_green_fs = [vec_to_hankel.conj().T@full_proj_green_fs[i] for i in range(len(green_fs))]
        rect_proj_green_fs.reverse()
        rect_proj_green_fs[0] = 0.5*rect_proj_green_fs[0]
        rect_proj_green_fs[-1] = 0.5*rect_proj_green_fs[-1]
        # Good until here
        stacked_dynamics = sparse.hstack(rect_proj_green_fs)
        if prop_sign == "plus":
            stacked_dynamics.resize((2*N_omega - 1, (2*N_z*(N_omega**2) + 2*N_omega - 1)))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((2*N_omega - 1, N_omega**2)), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z + 1)*(N_omega**2), (2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), stacked_dynamics])
            dynamics_real_list.append(0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(-0.5*1.j*(stacked_dynamics - stacked_dynamics.conj().T))
        if prop_sign == "minus":
            stacked_dynamics.resize((2*N_omega - 1, (N_z*(N_omega**2) + 2*N_omega - 1)))
            stacked_dynamics = sparse.hstack([sparse.csc_matrix((2*N_omega - 1, (N_z + 1)*(N_omega**2))), stacked_dynamics])
            stacked_dynamics = sparse.vstack([sparse.csc_matrix(((2*N_z + 1)*(N_omega**2), (2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), stacked_dynamics])
            dynamics_real_list.append(-0.5*(stacked_dynamics + stacked_dynamics.conj().T))
            dynamics_imag_list.append(0.5*1.j*(stacked_dynamics  - stacked_dynamics.conj().T))
    return dynamics_real_list, dynamics_imag_list

def obj_f_sdp_mat(N_omega, N_z, n):
    diags = diag_mat(N_omega, N_z, np.eye(N_omega))
    Q_obj_f = -diags[0]
    P_obj_f = sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1,1))
    sdp_mat = sparse.bmat([[Q_obj_f, P_obj_f],[P_obj_f.conj().T, n**2]])
    return sdp_mat

def J_def_sdp_mat(N_omega, N_z, proj):
    diags = diag_mat(N_omega, N_z, proj)
    eye_proj = vectorized_proj(N_omega, N_z, proj)
    Q_J_def = diags[N_z] + diags[2*N_z]
    P_J_def = - 2*eye_proj[0]
    cst_J_def = - 2*np.trace(proj)
    sdp_mat = sparse.bmat([[Q_J_def, P_J_def],[P_J_def.conj().T, cst_J_def]])
    return sdp_mat

def dynamics_mat(omega, z, proj):
    N_omega = len(omega)
    N_z = len(z)
    vec_to_hankel = omega_vec_to_omega_dof(N_omega)
    plus_dyn_real, plus_dyn_imag = dynamics_W(omega, z, proj, vec_to_hankel, "plus")
    minus_dyn_real, minus_dyn_imag = dynamics_W(omega, z, proj, vec_to_hankel, "minus")
    full_proj_plus = vectorized_proj(N_omega, N_z, proj)[1:N_z + 1]
    full_proj_minus = vectorized_proj(N_omega, N_z, proj)[N_z + 1:2*N_z + 1]
    green_f = get_green_f(omega, z)
    real_plus_mats = [sparse.bmat([[plus_dyn_real[i], -0.5*full_proj_plus[i]],[-0.5*full_proj_plus[i].conj().T, np.real(np.trace(proj@green_f[i]))]]) for i in range(N_z)]
    imag_plus_mats = [sparse.bmat([[plus_dyn_imag[i], -0.5*1.j*full_proj_plus[i]],[0.5*1.j*full_proj_plus[i].conj().T, np.imag(np.trace(proj@green_f[i]))]]) for i in range(N_z)]
    real_minus_mats = [sparse.bmat([[minus_dyn_real[i], -0.5*full_proj_minus[i]],[-0.5*full_proj_minus[i].conj().T, np.real(np.trace(proj@green_f[i]))]]) for i in range(N_z)]
    imag_minus_mats = [sparse.bmat([[minus_dyn_imag[i], -0.5*1.j*full_proj_minus[i]],[0.5*1.j*full_proj_minus[i].conj().T, np.imag(np.trace(proj@green_f[i]))]]) for i in range(N_z)]
    return real_plus_mats, imag_plus_mats, real_minus_mats, imag_minus_mats

def photon_numb_mat(N_omega, N_z, n):
    vect_J = vectorized_proj(N_omega, N_z, np.eye(N_omega))[0]
    J_mat = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1, (2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), 0.5*vect_J], [0.5*vect_J.conj().T, -n]])
    return J_mat

def verif_constr(W_plus_list, W_minus_list, beta, proj, omega, z):
    delta_z = np.abs(z[1] - z[0])
    N_z = len(z)
    N_omega = len(omega)
    W_plus_results = [np.trace(proj@np.eye(N_omega) - proj@W_plus_list[0])]
    W_minus_results = [np.trace(proj@np.eye(N_omega) - proj@W_minus_list[0])]
    for i in range(1, N_z):
        green_f = get_green_f(omega, z)[:i + 1]
        green_f[0] = 0.5*green_f[0]
        green_f[-1] = 0.5*green_f[-1]
        green_f.reverse()
        W_plus_results.append(np.trace(- proj@W_plus_list[i] + 2*proj@green_f[0] + delta_z*proj@sum([green_f[j]@beta@W_plus_list[j] for j in range(N_z)])))
        W_minus_results.append(np.trace(- proj@W_minus_list[i] + 2*proj@green_f[0] - delta_z*proj@sum([green_f[j]@beta@W_minus_list[j] for j in range(N_z)])))
    return W_plus_results, W_minus_results