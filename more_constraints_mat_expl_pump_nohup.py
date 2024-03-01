import more_constraints_sdr_expl_pump as api
import numpy as np
import scipy
import scipy.sparse as sparse
import cvxpy as cp

np.random.seed(0)
# Initialize parameters and SDR constraint
N_omega = 5
omega = np.linspace(-2, 2, N_omega)
N_z = 20
z = np.linspace(0, 5*10**-2, N_z)
green_fs = api.get_green_f(omega,z)
projection = np.zeros((N_omega, N_omega))
projections = []
sdr_def_constr = []
sdr_cst = []
for i in range(N_omega):
    for j in range(N_omega):
        proj_copy = projection.copy()
        proj_copy[i, j] = 1
        projections.append(sparse.csc_matrix(proj_copy))
        sdr_def_constr.append(api.sdr_def_constr(N_omega, N_z, sparse.csc_matrix(proj_copy)))
        if i == j:
            sdr_cst.append(2.)
        else:
            sdr_cst.append(0.)
# Propagators for testing out legitimacy of matrices
rand_pump = np.random.random(2*N_omega - 1)
beta = scipy.linalg.hankel(rand_pump[:N_omega], rand_pump[N_omega - 1:])
delta_k = np.diag(1.j*omega)
W_plus = [scipy.linalg.expm((delta_k + beta)*z[i]) for i in range(N_z)]
W_minus = [scipy.linalg.expm((delta_k - beta)*z[i]) for i in range(N_z)]
J = 0.25*(W_plus[-1]@W_plus[-1].conj().T + W_minus[-1]@W_minus[-1].conj().T - 2*np.eye(N_omega))
n = np.trace(J)
stacked_W_plus = np.vstack(W_plus)
stacked_W_minus = np.vstack(W_minus)
X = sparse.csc_matrix(np.vstack([J, stacked_W_plus, stacked_W_minus]))
Y = sparse.vstack([X, sparse.eye(N_omega)])
# Get the dynamics constraints, symplectic and definition of J constraints and constraints that fix the pump
real_plus_constr_mat = []
imag_plus_constr_mat = []
real_minus_constr_mat = []
imag_minus_constr_mat = []
sympl_real_constr = []
sympl_imag_constr = []
def_J_real_constr = []
def_J_imag_constr = []
for i in range(len(projections)):    
    real_plus, imag_plus, real_minus, imag_minus = api.get_dynamics_sdr(omega, z, beta, projections[i])
    real_plus_constr_mat += real_plus
    imag_plus_constr_mat += imag_plus
    real_minus_constr_mat += real_minus
    imag_minus_constr_mat += imag_minus
    sympl_real, sympl_imag = api.sympl_constr_sdr(N_omega, N_z, projections[i])
    sympl_real_constr += sympl_real
    sympl_imag_constr += sympl_imag
    real_def_J, imag_def_J = api.def_J_constr(N_omega, N_z, projections[i])
    def_J_real_constr.append(real_def_J)
    def_J_imag_constr.append(imag_def_J)
# The rest of the constraints (objective function is positive, photon number constraint, constraint that prior photon number is always less than the end)
obj_f = api.obj_f_sdr(N_omega, N_z, n)
photon_nbr_constr = api.photon_nbr_sdr_constr(N_omega, N_z, n)
# List containing all the constraints
constr_list = real_plus_constr_mat + imag_plus_constr_mat + real_minus_constr_mat + imag_minus_constr_mat + sympl_real_constr + sympl_imag_constr + def_J_real_constr + def_J_imag_constr + sdr_def_constr
constr_list.append(photon_nbr_constr)
# Make the cvxpy model
variable = cp.Variable(shape = ((2*N_z + 2)*N_omega, (2*N_z + 2)*N_omega), complex = True)
constraints = [variable >> 0]
constraints += [cp.real(cp.trace(real_plus_constr_mat[i]@variable)) == 0 for i in range(len(real_plus_constr_mat))]
constraints += [cp.real(cp.trace(imag_plus_constr_mat[i]@variable)) == 0 for i in range(len(imag_plus_constr_mat))]
constraints += [cp.real(cp.trace(real_minus_constr_mat[i]@variable)) == 0 for i in range(len(real_minus_constr_mat))]
constraints += [cp.real(cp.trace(imag_minus_constr_mat[i]@variable)) == 0 for i in range(len(imag_minus_constr_mat))]
constraints += [cp.real(cp.trace(sympl_real_constr[i]@variable)) == 0 for i in range(len(sympl_real_constr))]
constraints += [cp.real(cp.trace(sympl_imag_constr[i]@variable)) == 0 for i in range(len(sympl_imag_constr))]
constraints += [cp.real(cp.trace(def_J_real_constr[i]@variable)) == 0 for i in range(len(def_J_real_constr))]
constraints += [cp.real(cp.trace(def_J_imag_constr[i]@variable)) == 0 for i in range(len(def_J_imag_constr))]
constraints += [cp.real(cp.trace(sdr_def_constr[i]@variable)) == sdr_cst[i] for i in range(len(sdr_def_constr))]
constraints.append(cp.real(cp.trace(obj_f@variable)) >= -1.)
constraints.append(cp.real(cp.trace(photon_nbr_constr@variable)) == 0)
problem = cp.Problem(cp.Minimize(cp.real(cp.trace(obj_f@variable))), constraints=constraints)
problem.solve(solver = "SCS", verbose = True, eps_rel = 10**-8, eps_abs = 10**-8)
np.save("explicit_fixed_pump_prob_optim_variable.npy", variable.value)
np.save("pump.npy", beta)