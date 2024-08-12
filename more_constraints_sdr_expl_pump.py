import numpy as np
import scipy.sparse as sparse

def diag_mat(N_omega, N_z, proj):
    """
    List of diagonal matrices with fully sized proj
    """
    diag_mats = []
    for i in range(2*N_z + 1):
        full_proj = sparse.csc_matrix(0.5*(proj + proj.conj().T))
        full_proj.resize(((2*N_z + 1 - i)*N_omega, (2*N_z + 1 - i)*N_omega))
        full_proj = sparse.bmat([[sparse.csc_matrix((i*N_omega, i*N_omega)), sparse.csc_matrix((i*N_omega, (2*N_z + 1 - i)*N_omega))],[sparse.csc_matrix(((2*N_z + 1 - i)*N_omega, i*N_omega)), full_proj]])
        diag_mats.append(full_proj)
    return diag_mats

def get_lin_matrices(N_z, N_omega, proj):
    lin = []
    for i in range(2*N_z + 1):
        lin_mat = np.zeros(((2*N_z + 1)*N_omega, N_omega))
        lin_mat[i*N_omega:(i + 1)*N_omega] = proj.toarray()
        lin.append(sparse.csc_matrix(lin_mat))
    return lin

def get_green_f(omega, z):
    return [np.diag(np.exp(1.j*omega*z[i])) for i in range(len(z))]

def dynamics_fixed_pump(omega, z, beta, proj, prop_sign):
    """
    Gives the linear terms for propagator when pump is hard-coded into the dynamics
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    dynamics_mat = []
    if prop_sign == "plus":
        E_ij = get_lin_matrices(N_z, N_omega, proj)[1:N_z + 1]
        dynamics_mat.append(-E_ij[0])
        for i in range(1, N_z):
            green_fs = get_green_f(omega, z)[:i+1]
            green_fs.reverse()
            green_fs[0] = 0.5*green_fs[0]
            green_fs[-1] = 0.5*green_fs[-1]
            dynamics = sparse.csc_matrix(np.vstack([delta_z*(beta@green_fs[j]@proj.conj()).conj().T for j in range(len(green_fs))]))
            dynamics.resize(((2*N_z)*N_omega,N_omega))
            dynamics = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), dynamics]) 
            dynamics_mat.append(dynamics - E_ij[i])
    if prop_sign == "minus":
        E_ij = get_lin_matrices(N_z, N_omega, proj)[N_z + 1:2*N_z + 1]
        dynamics_mat.append(-E_ij[0])
        for i in range(1, N_z):
            green_fs = get_green_f(omega, z[:i + 1])
            green_fs.reverse()
            green_fs[0] = 0.5*green_fs[0]
            green_fs[-1] = 0.5*green_fs[-1]
            dynamics = sparse.csc_matrix(np.vstack([-delta_z*(beta@green_fs[j]@proj.conj()).conj().T for j in range(len(green_fs))]))
            dynamics.resize(((N_z)*N_omega,N_omega))
            dynamics = sparse.vstack([sparse.csc_matrix(((N_z + 1)*N_omega, N_omega)), dynamics]) 
            dynamics_mat.append(dynamics - E_ij[i])
    return dynamics_mat

def get_dynamics_sdr(omega, z, beta, proj):
    """
    Gives the semidefinite relaxation matrices for the dynamics constraints
    """
    N_omega = len(omega)
    N_z = len(z)
    dynamics_plus = dynamics_fixed_pump(omega, z, beta, proj, "plus")
    dynamics_minus = dynamics_fixed_pump(omega, z, beta, proj, "minus")
    green_fs = get_green_f(omega, z)
    dynamics_real_plus_sdr = []
    dynamics_imag_plus_sdr = []
    dynamics_real_minus_sdr = []
    dynamics_imag_minus_sdr = []
    for i in range(N_z):
        dynamics_real_plus_sdr.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*dynamics_plus[i]]
                                                   ,[0.5*dynamics_plus[i].conj().T, (1/N_omega)*np.real(sparse.csc_matrix(proj.conj().T@green_fs[i]).trace())*sparse.eye(N_omega)]]))
        dynamics_real_minus_sdr.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*dynamics_minus[i]]
                                                   ,[0.5*dynamics_minus[i].conj().T, (1/N_omega)*np.real((sparse.csc_matrix(proj.conj().T@green_fs[i]))).trace()*sparse.eye(N_omega)]]))
        dynamics_imag_plus_sdr.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*1.j*dynamics_plus[i]]
                                                   ,[0.5*(1.j*dynamics_plus[i]).conj().T, (1/N_omega)*np.imag((sparse.csc_matrix(proj.conj().T@green_fs[i])).trace())*sparse.eye(N_omega)]]))
        dynamics_imag_minus_sdr.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*(1.j*dynamics_plus[i])]
                                                   ,[0.5*(1.j*dynamics_plus[i]).conj().T, (1/N_omega)*np.imag((sparse.csc_matrix(proj.conj().T@green_fs[i])).trace())*sparse.eye(N_omega)]]))
    return dynamics_real_plus_sdr, dynamics_imag_plus_sdr, dynamics_real_minus_sdr, dynamics_imag_minus_sdr

def sympl_constr_sdr(N_omega, N_z, proj):
    """
    Get full symplectic matrices for the given projection matrix
    """
    # Symplective constraints
    real_symplectic = []
    imag_symplectic = []
    real_cst_proj = -0.5*(proj + proj.conj().T).trace()
    for i in range(N_z):
        proj_copy = proj.copy()
        proj_copy.resize(((2*N_z - i)*N_omega, (N_z - i)*N_omega))
        proj_copy = sparse.bmat([[sparse.csc_matrix(((i + 1)*N_omega, (N_z + 1 + i)*N_omega)), sparse.csc_matrix(((i + 1)*N_omega, (N_z - i)*N_omega))],
                            [sparse.csc_matrix(((2*N_z - i)*N_omega,(N_z + 1 + i)*N_omega)), proj_copy]])
        real_proj = 0.5*(proj_copy + proj_copy.conj().T)
        real_proj = sparse.bmat([[real_proj, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), (real_cst_proj/N_omega)*sparse.eye(N_omega)]])
        imag_proj = 0.5*(1.j*proj_copy + (1.j*proj_copy).conj().T)
        imag_proj = sparse.bmat([[imag_proj, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                            [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), sparse.csc_matrix((N_omega, N_omega))]])
        real_symplectic.append(real_proj)
        imag_symplectic.append(imag_proj)
    return real_symplectic, imag_symplectic

def def_J_constr(N_omega, N_z, proj):
    """
    Gives the SDR matrix for the definition of J for a given projection matrix
    """
    imag_proj = 1.j*proj
    diags_real = diag_mat(N_omega, N_z, proj)
    diags_imag = diag_mat(N_omega, N_z, imag_proj)
    quads_real = []
    quads_real = diags_real[N_z] + diags_real[2*N_z]
    quads_imag = []
    quads_imag = diags_imag[N_z] + diags_imag[2*N_z]
    lin = -4*get_lin_matrices(N_z, N_omega, proj)[0]
    cst_real = -(proj + proj.conj().T).trace()
    cst_imag = -(1.j*proj + (1.j*proj).conj().T).trace()
    real = sparse.bmat([[quads_real, 0.5*lin],[0.5*lin.conj().T, (cst_real/N_omega)*sparse.eye(N_omega)]])
    imag = sparse.bmat([[quads_imag, 0.5*1.j*lin],[0.5*(1.j*lin).conj().T, ((cst_imag)/N_omega)*sparse.eye(N_omega)]])
    return real, imag

def photon_nbr_prev_points(N_omega, N_z, n):
    """
    Gives the matrix for the constraint that says mean photon number increases through waveguide
    """
    constraint = []
    diags = diag_mat(N_omega, N_z, sparse.eye(N_omega))
    for i in range(N_z):
        quad = 0.25*(diags[1 + i] + diags[1 + N_z + i])
        constraint.append(sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],
                                       [sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), -((N_omega/2 + n)/N_omega)*sparse.eye(N_omega)]]))
    return constraint

def obj_f_sdr(N_omega, N_z, n):
    """
    Gives the matrix for the objective function in frame of SDR
    """
    diags = diag_mat(N_omega, N_z, np.eye(N_omega))[0]
    obj_f_mat = sparse.bmat([[-diags, sparse.csc_matrix(((2*N_z + 1)*N_omega,N_omega))],[sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), ((n**2)/N_omega)*np.eye(N_omega)]])
    return obj_f_mat

def photon_nbr_sdr_constr(N_omega, N_z, n):
    """
    Gives SDR matrix for photon number constraint
    """
    lin = get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[0]
    photon_nbr_cstr = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), 0.5*lin],[0.5*lin.conj().T, (-n/N_omega)*sparse.eye(N_omega)]])
    return photon_nbr_cstr

def sdr_def_constr(N_omega, N_z, proj):
    """
    Give the constraint on the block that is added to the primal variables
    """
    constr_sdr_def = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega)), sparse.csc_matrix(((2*N_z + 1)*N_omega,N_omega))],[sparse.csc_matrix((N_omega,(2*N_z + 1)*N_omega)), proj + proj.conj().T]])
    return constr_sdr_def

def inf_trace_photon_nbr(N_omega, N_z, n):
    cst  = -0.5*N_omega - n
    constr_inf_photon_nbr = []
    diags = diag_mat(N_omega, N_z, np.eye(N_omega))
    for i in range(N_z):
        quad = 0.25*(diags[1 + i] + diags[1 + N_z + i])
        constr_inf_photon_nbr.append(sparse.bmat([[quad, sparse.csc_matrix(((2*N_z + 1)*N_omega, N_omega))],[sparse.csc_matrix((N_omega, (2*N_z + 1)*N_omega)), (cst/N_omega)*sparse.eye(N_omega)]]))
    return constr_inf_photon_nbr