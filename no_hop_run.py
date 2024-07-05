import numpy as np
import scipy
import scipy.sparse as sparse
import cvxpy as cp
import matplotlib.pyplot as plt
import only_prop_only_nuclear_norm as api
import time
import math

np.random.seed(0)

N_omega = 21
free_indices = 2*N_omega - 1 - 12
omega = np.linspace(-10, 10, N_omega)
delta_omega = np.abs(omega[1] - omega[0])
N_z = 6
z = np.linspace(0, 1., N_z)
delta_z = np.abs(z[1] - z[0])
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
        quad = api.diag_mat(N_omega, N_z, proj_copy)[0] + api.diag_mat(N_omega, N_z, proj_copy)[0].conj().T
        quad.resize(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega))
        if i == j:
            sdr_cst.append(2.)
        else:
            sdr_cst.append(0.)
beta_vec = np.exp(-np.linspace(omega[0], omega[-1], 2*N_omega - 1)**2/4.)#3*(np.random.random(2*N_omega - 1) - 0.5*np.ones(2*N_omega - 1))#
#np.array(list((0.1/decay_curve)*np.exp(np.linspace(0, position*delta_omega, position))) + list(0.1 + np.exp(-np.linspace(omega[position], omega[len(omega) - 1 - position], 2*N_omega - 1 - 2*position)**2)) + list((0.1/decay_curve)*np.exp(np.linspace(position*delta_omega, 0, position))))
beta = scipy.linalg.hankel(beta_vec[:N_omega], beta_vec[N_omega - 1:])
new_beta = beta/(np.sqrt(np.trace(beta@beta)))
# Try for n = 0.5
beta_weight = 2.145#0.02531#1.631#0.110747#663.61#1.30664#1.0868#
delta_k = 1.j*np.diag(omega)
Q_plus = delta_k + beta_weight*new_beta
Q_minus = delta_k - beta_weight*new_beta
n = 0.25*np.trace((scipy.linalg.expm(Q_plus*z[-1]) - scipy.linalg.expm(Q_minus*z[-1])).conj().T@(scipy.linalg.expm(Q_plus*z[-1]) - scipy.linalg.expm(Q_minus*z[-1])))
W_plus = [(1/np.sqrt(n))*scipy.linalg.expm(Q_plus*z[i]) for i in range(1, N_z)]
W_minus = [(1/np.sqrt(n))*scipy.linalg.expm(Q_minus*z[i]) for i in range(1, N_z)]
X = np.vstack(W_plus + W_minus + [new_beta])
Y = np.vstack(W_plus + W_minus + [new_beta, np.eye(N_omega)])
# Generate constraints
n = 1.
beta_weight = 0.23#0.02531#0.25#1.5#
dynamics_constr = []
sympl_constr = []
pump_fix_constr = []
lin_dynamics = []
lin_dynamics_accurate = []
quad_symplect = []
gen_symplect = []
fin_diff_quad_lin_plus = []
fin_diff_quad_lin_minus = []
fin_diff_backwards_lin = []
lin_dynamics_more_accurate = []
# Changed the constraint to be tr(VV) = delta_omega**2
pump_power_constr = api.limit_pump_power(omega, N_z)
photon_end_nbr_constr = api.photon_nbr_constr(N_omega, N_z, n)
photon_prev_ineq_constr = api.photon_nbr_prev_points(N_omega, N_z)
pump_hankel_constr = api.constr_hankel_sdr(N_omega, N_z)
fixed_first_last_row = api.constr_first_last_row_pump(N_omega, N_z, new_beta, free_indices)
affine_ineq_pump = api.basic_affine_ineq_pump(N_omega, N_z, np.array(list(new_beta[0]) + list(new_beta[-1][1:])), 10**-6)
constr_left_decay, constr_right_decay = api.pump_exp_decay_constr(N_omega, N_z, "positive", "positive", 5, 20, max_ampli=0.01)
for i in range(len(projections)):
    real_plus_dyn, imag_plus_dyn, real_minus_dyn, imag_minus_dyn = api.get_dynamics_sdr(omega, z, projections[i], n, beta_weight)
    real_sympl, imag_sympl = api.sympl_constr_sdr(N_omega, N_z, projections[i], n)
    real_fixed_pump, imag_fixed_pump = api.sdr_fixed_pump(N_omega, N_z, new_beta, projections[i])
    dynamics_constr += real_plus_dyn + imag_plus_dyn + real_minus_dyn + imag_minus_dyn
    sympl_constr += real_sympl + imag_sympl
    #lin_dynamics += api.lin_finite_diff_constr(z, N_omega, delta_k, n, beta_weight, projections[i])
    #lin_dynamics_accurate += api.lin_finite_diff_constr_accurate(z, N_omega, delta_k, n, beta_weight, projections[i])
    #lin_dynamics_more_accurate += api.lin_finite_diff_constr_more_accurate(z, N_omega, delta_k, n, beta_weight, projections[i])
    quad_symplect += api.constr_lower_quadratic_prop_sympl(N_omega, N_z, projections[i]) + api.constr_upper_quadratic_prop_sympl(N_z, N_omega, projections[i])
    gen_symplect += api.constr_sympl_plus(N_z, N_omega, n, projections[i]) + api.constr_sympl_minus(N_z, N_omega, n, projections[i])
    #fin_diff_quad_lin_plus += api.constr_fin_diff_quad_lin_plus(N_omega, z, beta_weight, n, delta_k, projections[i])
    #fin_diff_quad_lin_minus += api.constr_fin_diff_quad_lin_minus(N_omega, z, beta_weight, n, delta_k, projections[i])
    #fin_diff_backwards_lin += api.finite_diff_constr_V(z, N_omega, delta_k, n, beta_weight, projections[i])
    pump_fix_constr += [imag_fixed_pump]#, real_fixed_pump]
constraints_list = dynamics_constr + sympl_constr + pump_hankel_constr + [photon_end_nbr_constr, pump_power_constr]  + quad_symplect + gen_symplect + pump_fix_constr# + fin_diff_backwards_lin + fin_diff_quad_lin_plus + fin_diff_quad_lin_minus + lin_dynamics_more_accurate
# Get matrices to build objective function
max_order = 5
width = 4.
hermite_mat = api.get_hermite_polynom_mat(omega, max_order, width)
left, right = api.obj_f_mat(N_omega, N_z)
# Set constraints to make the nuclear norm of U_+ and U_- less or equal that of the rank 1 problem
left_both = api.get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 2]
left_both = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), left_both])
right_plus = api.get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 2]
right_plus.resize((2*N_z*N_omega, N_omega))
right_minus =  api.get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 3]
right_minus.resize((2*N_z*N_omega, N_omega))
# Try relaxing the constraint on high purity more to obtain solution with a hermite cutoff
epsilon = .1
cst_plus = (N_omega - 2)/np.sqrt(n) + 1 + np.sqrt(1 + 1/n) +  epsilon/2 + np.sqrt((epsilon**2)/4 + 1/n)
cst_minus = (N_omega - 2)/np.sqrt(n) - .95 + np.sqrt((.95)**2 + 1/n) + epsilon/2 + np.sqrt((epsilon**2)/4 + 1/n)
hermite_coeff = cp.Variable(shape=(max_order,))
variable = cp.Variable(shape = (2*N_z*N_omega, 2*N_z*N_omega), hermitian = True)
constraints = [variable >> 0]
constraints += [cp.real(cp.trace(sdr_def_constr[i]@variable)) == sdr_cst[i] for i in range(len(sdr_def_constr))]
constraints += [cp.real(cp.trace(constraints_list[i]@variable)) == 0 for i in range(len(constraints_list))]
constraints += [cp.real(cp.trace(photon_prev_ineq_constr[i]@variable)) >= 0 for i in range(len(photon_prev_ineq_constr))]
V_plus = cp.Variable(shape = (N_omega, N_omega), hermitian = True)
X_plus = cp.Variable(shape = (N_omega, N_omega), hermitian = True)
V_minus = cp.Variable(shape = (N_omega, N_omega), hermitian = True)
X_minus = cp.Variable(shape = (N_omega, N_omega), hermitian = True)
var_U_plus_dagger = left_both.conj().T@variable@right_plus
var_U_minus_dagger = left_both.conj().T@variable@right_minus
Q_plus = cp.vstack([cp.hstack([V_plus, var_U_plus_dagger]), cp.hstack([var_U_plus_dagger.conj().T, X_plus])])
Q_minus = cp.vstack([cp.hstack([V_minus, var_U_minus_dagger]), cp.hstack([var_U_minus_dagger.conj().T, X_minus])])
constraints.append(Q_plus >> 0)
constraints.append(Q_minus >> 0)
constraints.append(cp.real(cp.trace(V_plus + X_plus)) - 2*cst_plus == 0)
constraints.append(cp.real(cp.trace(V_minus + X_minus)) - 2*cst_minus == 0)
left_first_pump, left_second_pump, right_first_pump, right_second_pump = api.isolate_vect_pump(N_omega, N_z)
constraints.append(cp.real(cp.hstack([left_first_pump.conj().T@variable@right_first_pump, left_second_pump.conj().T@variable@right_second_pump])) == hermite_mat.T@hermite_coeff)
problem = cp.Problem(cp.Minimize(cp.atoms.norm(left.conj().T@variable@right, "nuc")), constraints)
problem.solve(solver = cp.MOSEK, mosek_params = {"MSK_IPAR_NUM_THREADS":8, "MSK_IPAR_INTPNT_MAX_ITERATIONS":2000, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP":10e-3, "MSK_DPAR_INTPNT_CO_TOL_MU_RED":10e-2}, verbose = True)
end_product = variable.value#np.load("n_1_fix_pump.npy")#np.load("n_1_fix_pump_small_margin.npy")#
U_plus = [end_product[(2*N_z - 1)*N_omega:(2*N_z)*N_omega, i*N_omega:(i + 1)*N_omega] for i in range(N_z - 1)]
U_minus = [end_product[(2*N_z - 1)*N_omega:(2*N_z)*N_omega, (N_z - 1 + i)*N_omega:(N_z + i)*N_omega] for i in range(N_z - 1)]
VU_plus = [end_product[(2*N_z - 2)*N_omega:(2*N_z - 1)*N_omega, i*N_omega:(i + 1)*N_omega] for i in range(N_z - 1)]
VU_minus = [end_product[(2*N_z - 2)*N_omega:(2*N_z - 1)*N_omega, (N_z - 1 + i)*N_omega:(N_z + i)*N_omega] for i in range(N_z - 1)]
quad_U_plus = [end_product[i*N_omega:(i + 1)*N_omega, i*N_omega:(i + 1)*N_omega] for i in range(N_z - 1)]
quad_U_minus = [end_product[(N_z - 1 + i)*N_omega:(N_z + i)*N_omega, (N_z - 1 + i)*N_omega:(N_z + i)*N_omega] for i in range(N_z - 1)]
pump_U_plus = [end_product[(2*N_z - 2)*N_omega:(2*N_z - 1)*N_omega, i*N_omega:(i + 1)*N_omega] for i in range(N_z - 1)]
opt_pump = end_product[(2*N_z - 1)*N_omega:(2*N_z)*N_omega, (2*N_z - 2)*N_omega:(2*N_z - 1)*N_omega]















