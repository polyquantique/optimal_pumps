import numpy as np
import matplotlib.pyplot as plt

def save_plot(N, wi, wf, Np, vp, l, yN, path = "plots/"):
    title = path + "N_"+str(N)+"wi_"+str(wi)+"wf_"+str(wf)+"Np_"+str(Np)+"vp_"+str(vp)+"l_"+str(l)+"yN"+str(yN)+".pdf"
    return title
def save_data(N, wi, wf, Np, vp, l, yN, array, pump_shape, initial_guess, number, stable = True, path = "results/"):
    if len(array.shape) == 1:
        to_save = {
            "resolution":N, "wi":wi, "wf":wf, "Np":Np, "vp":vp, "l":l, "yN":yN, "parameters":array, "parameters_initialization":initial_guess
        }
        if stable == True:
            title = path + pump_shape + "_" + "stable" + "_" + str(number) + ".npy"
        else:
            title = path + pump_shape + "_" + "unstable" + "_" + str(number) + ".npy"
        np.save(title, to_save, allow_pickle = True)
    elif len(array.shape) == 2:
        to_save = {
            "resolution":N, "wi":wi, "wf":wf, "Np":Np, "vp":vp, "l":l, "yN_start":yN[0], 
            "yN_end": yN[-1], "parameters":array, "parameters_initialization":initial_guess
        }
        title = path + pump_shape + "_" + "unstable" + "_many_yN_" + str(number) +".npy"
        np.save(title, to_save)
    else:
        raise Exception("Parameters have too many dimensions") 