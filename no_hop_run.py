import numpy as np
import scipy.sparse as sparse
import scipy
import matplotlib.pyplot as plt
import cvxpy as cp
import vectorized_sdr as vect_sdr

np.random.seed(0)
N_omega = 21
N_z = 2
n = 10**-5
omega = np.linspace(-3, 3, N_omega)
z = np.linspace(0, 10**-3, N_z)
projection = np.zeros((N_omega, N_omega))
J_def_constr = []
real_plus_dyn = []
imag_plus_dyn = []
real_minus_dyn = []
imag_minus_dyn = []
for i in range(N_omega):
    for j in range(N_omega):
        proj = projection.copy()
        proj[i, j] = 1
        real_plus_mats, imag_plus_mats, real_minus_mats, imag_minus_mats = vect_sdr.dynamics_mat(omega, z, proj)
        J_def_constr.append(vect_sdr.J_def_sdp_mat(N_omega, N_z, proj))
        real_plus_dyn += real_plus_mats
        imag_plus_dyn += imag_plus_mats
        real_minus_dyn += real_minus_mats
        imag_minus_dyn += imag_minus_mats
photon_numb_constr = vect_sdr.photon_numb_mat(N_omega, N_z, n)
X = cp.Variable(shape=((2*N_z + 1)*(N_omega**2) + 2*N_omega,(2*N_z + 1)*(N_omega**2) + 2*N_omega))
obj_f_mat = vect_sdr.obj_f_sdp_mat(N_omega, N_z, n)
constraints = [X >> 0]
constraints += [cp.trace(J_def_constr[i]@X) == 0 for i in range(len(J_def_constr))]
constraints += [cp.trace(real_plus_dyn[i]@X) == 0 for i in range(len(real_plus_dyn))]
constraints += [cp.trace(imag_plus_dyn[i]@X) == 0 for i in range(len(real_plus_dyn))]
constraints += [cp.trace(real_minus_dyn[i]@X) == 0 for i in range(len(real_plus_dyn))]
constraints += [cp.trace(imag_minus_dyn[i]@X) == 0 for i in range(len(real_plus_dyn))]
constraints.append(cp.trace(photon_numb_constr@X) == 0)
problem = cp.Problem(cp.Minimize(cp.trace(obj_f_mat@X)), constraints)
problem.solve(verbose = True)
np.save()