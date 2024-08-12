import vectorized_sdr as vect
import numpy as np
import scipy 
import scipy.sparse as sparse
import cvxpy as cp

np.random.seed(0)
N_omega = 10
omega = np.linspace(-2, 2, N_omega)
N_z = 20
z = np.linspace(0, 2*10**-1, N_z)
n = 0.004
projections = []
for i in range(N_omega):
    for j in range(N_omega):
        proj = np.zeros((N_omega, N_omega))
        proj[i,j] = 1
        projections.append(proj.conj().T)
# Get the dynamics constraints matrices
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
sdr_constr = sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1, 1))],
                          [sparse.csc_matrix((1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), sparse.eye(1)]])
obj_f_mat = vect.obj_f_sdp_mat(N_omega, N_z, n)
# Fix values of beta
random_beta = np.exp(-(np.linspace(omega[0], omega[-1], 2*N_omega - 1)**2)/0.2)
beta_def_constr_list_real = []
beta_def_constr_list_imag = []
for i in range(2*N_omega - 1):
    beta_proj = np.zeros(2*N_omega - 1)
    beta_proj[i] = 1
    beta_proj = sparse.csc_matrix(beta_proj.reshape((2*N_omega - 1, 1)))
    beta_proj = sparse.vstack([sparse.csc_matrix(((2*N_z + 1)*(N_omega**2),1)), beta_proj])
    beta_def_constr_list_real.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), 0.5*beta_proj],[0.5*beta_proj.conj().T, -random_beta[i]*sparse.eye(1)]]))
    beta_def_constr_list_imag.append(sparse.bmat([[sparse.csc_matrix(((2*N_z + 1)*(N_omega**2) + 2*N_omega - 1,(2*N_z + 1)*(N_omega**2) + 2*N_omega - 1)), 0.5*(beta_proj*1.j)],[0.5*(beta_proj*1.j).conj().T, sparse.csc_matrix((1,1))]]))
variable = cp.Variable(shape = ((2*N_z + 1)*(N_omega**2) + 2*N_omega, (2*N_z + 1)*(N_omega**2) + 2*N_omega), complex = True)
constraints = [variable >> 0]
constraints += [cp.real(cp.trace(real_plus_mats_list[i]@variable)) == 0 for i in range(len(real_plus_mats_list))]
constraints += [cp.real(cp.trace(imag_plus_mats_list[i]@variable)) == 0 for i in range(len(imag_plus_mats_list))]
constraints += [cp.real(cp.trace(real_minus_mats_list[i]@variable)) == 0 for i in range(len(real_minus_mats_list))]
constraints += [cp.real(cp.trace(imag_minus_mats_list[i]@variable)) == 0 for i in range(len(imag_minus_mats_list))]
constraints += [cp.real(cp.trace(J_def_mat[i]@variable)) == 0 for i in range(len(J_def_mat))]
constraints += [cp.real(cp.trace(beta_def_constr_list_real[i]@variable)) == 0 for i in range(2*N_omega - 1)]
constraints += [cp.real(cp.trace(beta_def_constr_list_imag[i]@variable)) == 0 for i in range(2*N_omega - 1)]
constraints.append(cp.real(cp.trace(sdr_constr@variable)) == 1.)
problem = cp.Problem(cp.Minimize(cp.real(cp.trace(-obj_f_mat@variable))), constraints)
problem.solve(complex = True)
np.save("fixed_pump_as_gaussian.npy", np.array(variable.value))