import numpy as np
import matplotlib.pyplot as plt

def save_plot(N, wi, wf, Np, vp, l, y, path = "plots/"):
    title = path + "N_"+str(N)+"wi_"+str(wi)+"wf_"+str(wf)+"Np_"+str(Np)+"vp_"+str(vp)+"l_"+str(l)+"y_"+str(y)+".pdf"
    return title
def save_data(M, wi, wf, Np, vp, l, y, array, pump_shape, path = "results/"):
    if len(array.shape) == 1:
        title = path + pump_shape
        +"N_"+str(N)+"wi_"+str(wi)+"wf_"+str(wf)+"Np_"+str(Np)+"vp_"+str(vp)+"l_"+str(l)+"y_"+str(y)+".npy"
        np.save(title, array)
    elif len(array.shape) == 2:
        title = path + pump_shape + "N_"+str(N)+"wi_"+str(wi)+"wf_"+str(wf)
        +"Np_"+str(Np)+"vp_"+str(vp)+"l_"+str(l)+"y_"+str(y[0])+"_to_"+str(y[-1])+".npy"
        np.save(title, array)