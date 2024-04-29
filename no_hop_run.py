import numpy as np
import scipy
import scipy.sparse as sparse
import cvxpy as cp
import only_prop_only_nuclear_norm as api

np.random.seed(0)


N_omega = 31
free_indices = 57
omega = np.linspace(-2, 2, N_omega)
N_z = 3
z = np.linspace(0, 4*10**-3, N_z)
delta_z = np.abs(z[1] - z[0])
green_fs = api.get_green_f(omega,z)
projection = np.zeros((N_omega, N_omega))
projections = []
sdr_def_constr = []
B_dagger_B_constr = []
sdr_cst = []
for i in range(N_omega):
    for j in range(N_omega):
        proj_copy = projection.copy()
        proj_copy[i, j] = 1
        projections.append(sparse.csc_matrix(proj_copy))
        sdr_def_constr.append(api.sdr_def_constr(N_omega, N_z, sparse.csc_matrix(proj_copy)))
        quad = api.diag_mat(N_omega, N_z, proj_copy)[0] + api.diag_mat(N_omega, N_z, proj_copy)[0].conj().T
        quad.resize(((2*N_z + 1)*N_omega,(2*N_z + 1)*N_omega))
        B_dagger_B_constr.append(quad)
        if i == j:
            sdr_cst.append(2.)
        else:
            sdr_cst.append(0.)
position = 5
decay_curve = 5
delta_omega = np.log(decay_curve)/position
beta_vec = np.array(list(np.zeros(10)) + list(np.random.random(2*N_omega - 21)) + list(np.zeros(10)))
#1.916*np.exp(-np.linspace(omega[0], omega[-1], 2*N_omega - 1)**2/1.995)#np.array([0., 0., 0.] + list(np.random.random(2*N_omega - 7)) + [0., 0., 0.])
#np.array([2.10900246, 2.10794203, 2.10786931, 2.10829837, 2.10790335,
       #2.1076864 , 2.10778626, 2.1078306 , 2.10778798, 2.10783712,
       #2.10781799, 2.1078467 , 2.10784045, 2.10769124, 2.10781962,
       #2.10769124, 2.10784045, 2.1078467 , 2.10781799, 2.10783712,
       #2.10778798, 2.1078306 , 2.10778626, 2.1076864 , 2.10790335,
       #2.10829837, 2.10786931, 2.10794203, 2.10900246])
#np.array(list((0.1/decay_curve)*np.exp(np.linspace(0, position*delta_omega, position))) + list(0.1 + np.exp(-np.linspace(omega[position], omega[len(omega) - 1 - position], 2*N_omega - 1 - 2*position)**2)) + list((0.1/decay_curve)*np.exp(np.linspace(position*delta_omega, 0, position))))
beta = scipy.linalg.hankel(beta_vec[:N_omega], beta_vec[N_omega - 1:])
new_beta = beta/np.sqrt(np.trace(beta@beta))
beta_weight = np.sqrt(np.trace(beta@beta))
delta_k = 1.j*np.diag(omega)
Q_plus = delta_k + beta_weight*new_beta
Q_minus = delta_k - beta_weight*new_beta
n = 0.25*np.trace((scipy.linalg.expm(Q_plus*z[-1]) - scipy.linalg.expm(Q_minus*z[-1])).conj().T@(scipy.linalg.expm(Q_plus*z[-1]) - scipy.linalg.expm(Q_minus*z[-1])))
W_plus = [(1/np.sqrt(n))*scipy.linalg.expm(Q_plus*z[i]) for i in range(1, N_z)]
W_minus = [(1/np.sqrt(n))*scipy.linalg.expm(Q_minus*z[i]) for i in range(1, N_z)]
X = np.vstack(W_plus + W_minus + [new_beta])
Y = np.vstack(W_plus + W_minus + [new_beta, np.eye(N_omega)])
full_rank = Y@Y.conj().T
# Generate constraints
dynamics_constr = []
sympl_constr = []
pump_fix_constr = []
pump_power_constr = api.limit_pump_power(N_omega, N_z)
photon_end_nbr_constr = api.photon_nbr_constr(N_omega, N_z, n)
photon_prev_ineq_constr = api.photon_nbr_prev_points(N_omega, N_z)
pump_hankel_constr = api.constr_hankel_sdr(N_omega, N_z)
fixed_first_last_row = api.constr_first_last_row_pump(N_omega, N_z, new_beta, free_indices)
basic_ineq_on_pump_high, basic_ineq_on_pump_low = api.basic_affine_ineq_pump(N_omega, N_z, free_indices, 0.3)
constr_left_decay, constr_right_decay = api.pump_exp_decay_constr(N_omega, N_z, "positive", "positive", 5, 20, max_ampli=0.01)
for i in range(len(projections)):
    real_plus_dyn, imag_plus_dyn, real_minus_dyn, imag_minus_dyn = api.get_dynamics_sdr(omega, z, projections[i], n, beta_weight)
    real_sympl, imag_sympl = api.sympl_constr_sdr(N_omega, N_z, projections[i], n)
    real_fixed_pump, imag_fixed_pump = api.sdr_fixed_pump(N_omega, N_z, new_beta, projections[i])
    dynamics_constr += real_plus_dyn + imag_plus_dyn + real_minus_dyn + imag_minus_dyn
    sympl_constr += real_sympl + imag_sympl
    pump_fix_constr += [imag_fixed_pump]#, real_fixed_pump]
constraints_list = dynamics_constr + sympl_constr + pump_hankel_constr + [photon_end_nbr_constr, pump_power_constr] + pump_fix_constr + fixed_first_last_row
left, right = api.obj_f_mat(N_omega, N_z)
left_both = api.get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 2]
left_both = sparse.vstack([sparse.csc_matrix((N_omega, N_omega)), left_both])
right_plus = api.get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[N_z - 2]
right_plus.resize((2*N_z*N_omega, N_omega))
right_minus =  api.get_lin_matrices(N_z, N_omega, sparse.eye(N_omega))[2*N_z - 3]
right_minus.resize((2*N_z*N_omega, N_omega))
epsilon = .6
cst_plus = (N_omega - 2)/np.sqrt(n) + 1 + np.sqrt(1 + 1/n) + epsilon/2 + np.sqrt((epsilon**2)/4 + 1/n)
cst_minus = (N_omega - 2)/np.sqrt(n) -0.9 + np.sqrt((0.9)**2 + 1/n) + epsilon/2 + np.sqrt((epsilon**2)/4 + 1/n)
bounds = []
for i in range(5):
    fixed_first_last_row = api.constr_first_last_row_pump(N_omega, N_z, new_beta, free_indices - i)
    constraints_list = dynamics_constr + sympl_constr + pump_hankel_constr + [photon_end_nbr_constr, pump_power_constr] + pump_fix_constr + fixed_first_last_row
    variable = cp.Variable(shape = (2*N_z*N_omega, 2*N_z*N_omega), hermitian = True)
    constraints = [variable >> 0]
    constraints += [cp.real(cp.trace(sdr_def_constr[i]@variable)) == sdr_cst[i] for i in range(len(sdr_def_constr))]
    constraints += [cp.real(cp.trace(constraints_list[i]@variable)) == 0 for i in range(len(constraints_list))]
    constraints += [cp.real(cp.trace(photon_prev_ineq_constr[i]@variable)) >= 0 for i in range(len(photon_prev_ineq_constr))]
    # Constraints on nuclear norm of U_+ and U_-
    constraints.append((cp.real(cp.atoms.norm(left_both.conj().T@variable@right_plus, "nuc")) - cp.real(cst_plus)) <= 0)
    constraints.append((cp.real(cp.atoms.norm(left_both.conj().T@variable@right_minus, "nuc")) - cp.real(cst_minus)) <= 0)
    problem = cp.Problem(cp.Minimize(cp.atoms.norm(left.conj().T@variable@right, "nuc")), constraints)
    bounds.append(problem.solve(solver = cp.MOSEK, mosek_params = {"MSK_IPAR_INTPNT_MAX_ITERATIONS":10**5}))
np.save("bounds_diff_cst.npy", np.array(bounds))