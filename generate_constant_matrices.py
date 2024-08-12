import numpy as np
import scipy.sparse as sparse
import generate_proj_mat as proj

def get_proj_lin_basic(N_omega, N_z, projection):
    """
    Gives a list of projection matrices for linear terms of real constraints 
    and for linear terns if imaginary constraints that is of length 3*N_z + 1

    Args:
        N_omega(int): Size of the discretized frequency domain
        N_z(int): Size of the vector z
        projection[[float]]: Projection matrix

    returns:
        a[[complex64]]: List of projection matrices for linear terms for real constraints
        b[[complex64]]: List of projection matrices for linear terms for imaginary constraints
    """
    real_proj_list = []
    imag_proj_list = []
    for i in range(3*N_z + 1):
        proj_copy = projection.copy()
        proj_copy.resize(((3*N_z + 1 - i)*N_omega, N_omega))
        left = sparse.csc_matrix((i*N_omega, N_omega))
        real_proj_list.append(sparse.vstack([left, proj_copy]))
        imag_proj_list.append(1.j*sparse.vstack([left, proj_copy]))
    return real_proj_list, imag_proj_list

def get_proj_quad_diag(N_omega, N_z, projection):
    """
    Gives a list of matrices of size (3*N_omega*N_z + 1)*(3*N_omega*N_z + 1)
    that have an identity matrix of size N_omega*N_omega at different positions
    of the diagonal.

    Args:
        N_omega(int):
        N_z(int): Size of the vector z
        projection[[float]]: Projection matrix

    returns:
        a[[complex64]]
    """
    quad_proj_list = []
    for i in range(int(3*N_z + 1)):
        proj_copy = projection.copy()
        proj_copy.resize(((3*N_z - i + 1)*N_omega, (3*N_z - i + 1)*N_omega))
        proj_copy = sparse.hstack([sparse.csc_matrix(((3*N_z - i + 1)*N_omega, (i)*N_omega)), proj_copy])
        up = sparse.csc_matrix(((i)*N_omega, (3*N_z + 1)*N_omega))
        proj_copy = sparse.vstack([up, proj_copy])
        quad_proj_list.append(proj_copy)
    return quad_proj_list

def get_proj_quad_symplectic(N_omega, N_z, projection):
    """
    Gives a list of matrices of dimension ((3*N_z + 1)*N_omega)*((3*N_z + 1)*N_omega)
    that is used for the real part of symplectic constraints and another list for the
    imaginary part

    Args:
        N_omega(int): Size of the discretized frequency domain
        N_z(int): Size of the vector z
        projection[[float]]: Projection matrix

    returns:
        a[[complex64]]: List of Hermitian matrices for symplectic constraints
    """
    real_list = []
    imag_list = []
    for i in range(N_z):
        proj_copy = projection.copy()
        proj_copy.resize(((2*N_z + 1 - i)*N_omega, (N_z + 1 - i)*N_omega))
        proj_copy = sparse.vstack([sparse.csc_matrix(((N_z + i)*N_omega, (N_z - i + 1)*N_omega)), proj_copy])
        left = sparse.csr_matrix(((3*N_z + 1)*N_omega, (2*N_z + i)*N_omega))
        proj_copy = sparse.hstack([left, proj_copy])
        real_list.append(0.5*(proj_copy + proj_copy.conj().T))
        imag_list.append(0.5*(- 1.j*proj_copy + 1.j*proj_copy.conj().T))
    return real_list, imag_list

def proj_dynamics_mat_propagator(omega, z, prop_sign, vp, projection):
    """
    Projects a list of matrices of dimension (3*N_z + 1)*N_omega*(3*N_z + 1)*N_omega 
    for quadratic term of the real part of the constraints associated with dynamics
    and another list of matrices of same dimension for the imaginary part associated
    with those same constraints.

    Args:
        N_omega(int): number of discretized frequency domain elements
        N_z[float32]: number of discretized space domain elements
        prop_sign(str): Can be of 2 values:
                            - "plus": the high dimension matrix output will be for the 
                                propagator associated with addition
                            - "minus": the high dimension matrix output will be for the 
                                propagator associated with substraction
        vp(float64): group velocity for the pump mode
        projection[[float]]: Projection matrix

    returns:
        a[[complex64]]: matrix representing the dynamics for the given list of matrices.
    """
    N_omega = len(omega)
    N_z = len(z)
    delta_z = np.abs(z[1] - z[0])
    real_dynamics = []
    imag_dynamics = []
    real_dynamics.append(sparse.csr_matrix(((3*N_z + 1)*N_omega, (3*N_z + 1)*N_omega)))
    imag_dynamics.append(sparse.csr_matrix(((3*N_z + 1)*N_omega, (3*N_z + 1)*N_omega)))
    # The first term should be identity
    for i in range(1, N_z):
        proj_copy = projection.copy()
        green_f_s = proj.get_green_functions(omega, vp, z)[:i + 1]
        green_f_s.reverse()
        green_f_s = [green_f_s[i]@proj_copy for i in range(len(green_f_s))]
        green_f_s[0] = green_f_s[0]/2
        green_f_s[-1] = green_f_s[-1]/2
        if prop_sign == "plus":
            Q_mat = delta_z*sparse.hstack(green_f_s)
            Q_mat.resize((N_omega, (2*N_z + 1)*N_omega))
            Q_mat = sparse.hstack([sparse.csr_matrix((N_omega, N_z*N_omega)), Q_mat])
            Q_mat = sparse.vstack([sparse.csr_matrix((3*N_z*N_omega, (3*N_z + 1)*N_omega)), Q_mat])
            real_dynamics.append(0.5*(Q_mat + Q_mat.conj().T))
            imag_dynamics.append(0.5*(-1.j*Q_mat + 1.j*Q_mat.conj().T))
        elif prop_sign == "minus":
            Q_mat = delta_z*sparse.hstack(green_f_s)
            Q_mat.resize((N_omega, (N_z + 1)*N_omega))
            Q_mat = sparse.hstack([sparse.csr_matrix((N_omega, 2*N_z*N_omega)), Q_mat])
            Q_mat = sparse.vstack([sparse.csr_matrix((3*N_z*N_omega, (3*N_z + 1)*N_omega)), Q_mat])
            real_dynamics.append(-0.5*(Q_mat + Q_mat.conj().T))
            imag_dynamics.append(-0.5*(-1.j*Q_mat + 1.j*Q_mat.conj().T))
    return real_dynamics, imag_dynamics
