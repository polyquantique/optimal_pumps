import vectorized_sdr as vect
import numpy as np
import scipy 
import scipy.sparse as sparse
import cvxpy as cp

np.random.seed(0)
# Initialize omega, z and projection matrices
N_omega = 11
omega = np.linspace(-2, 2, N_omega)
N_z = 10
z = np.linspace(0, 10**-2, N_z)
n = 0.1
vec_to_hankel = vect.omega_vec_to_omega_dof(N_omega)
projections = []
for i in range(N_omega):
    for j in range(N_omega):
        proj = np.zeros((N_omega, N_omega))
        proj[i,j] = 1
        projections.append(proj.conj().T)
real_plus_mats_list = []
imag_plus_mats_list = []
real_minus_mats_list = []
imag_minus_mats_list = []
J_def_mat = []
for i in range(len(projections)):
    real_plus_mats, imag_plus_mats, real_minus_mats, imag_minus_mats = vect.dynamics_mat(omega, z, projections[i])
    real_plus_mats_list += real_plus_mats
    imag_plus_mats_list += imag_plus_mats
    real_minus_mats_list += real_minus_mats
    imag_minus_mats_list += imag_minus_mats
    J_def_mat.append(vect.J_def_sdp_mat(N_omega, N_z, projections[i]))
photon_nbr_constr_mat = vect.photon_numb_mat(N_omega, N_z, n)
obj_f_mat = vect.obj_f_sdp_mat(N_omega, N_z, n)
quad_cov_pump = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*(N_omega**2),(2*N_z + 1)*(N_omega**2))), sparse.csc_matrix(((2*N_z + 1)*(N_omega**2), 2*N_omega - 1))],
             [sparse.csc_matrix((2*N_omega - 1, (2*N_z + 1)*(N_omega**2))), sparse.csc_matrix(np.diag(np.linspace(omega[0], omega[-1], 2*N_omega - 1)**2))]])
constr_cov_pump = sparse.bmat([[quad_cov_pump, sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1,1))],
                               [sparse.csc_matrix((1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), -0.5*sparse.eye(1)]])
sdr_constr = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1, 1))],
                          [sparse.csc_matrix((1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), sparse.eye(1)]])
variable = cp.Variable(shape = ((2*N_z + 1)*(N_omega**2) + 2*N_omega, (2*N_z + 1)*(N_omega**2) + 2*N_omega), complex = True)
constraints = [variable >> 0]
constraints += [cp.real(cp.trace(real_plus_mats_list[i]@variable)) == 0 for i in range(len(real_plus_mats_list))]
constraints += [cp.real(cp.trace(imag_plus_mats_list[i]@variable)) == 0 for i in range(len(imag_plus_mats_list))]
constraints += [cp.real(cp.trace(real_minus_mats_list[i]@variable)) == 0 for i in range(len(real_minus_mats_list))]
constraints += [cp.real(cp.trace(imag_minus_mats_list[i]@variable)) == 0 for i in range(len(imag_minus_mats_list))]
constraints += [cp.real(cp.trace(J_def_mat[i]@variable)) == 0 for i in range(len(J_def_mat))]
constraints.append(cp.real(cp.trace(photon_nbr_constr_mat@variable)) == 0)
constraints.append(cp.real(cp.trace(obj_f_mat@variable)) >= 0)
constraints.append(cp.real(cp.trace(constr_cov_pump@variable)) <= 0)
constraints.append(cp.real(cp.trace(sdr_constr@variable)) == 1.)
problem = cp.Problem(cp.Minimize(cp.real(cp.trace(obj_f_mat@variable))), constraints)
problem.solve()
np.save("cvxpy_var_value_limited_cov.npy",np.array(variable.value))