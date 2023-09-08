import numpy as np
import matplotlib.pyplot as plt

def save_plot_name(N, wi, wf, Np, vp, l, yN, path = "plots/"):
    """
    Returns the path and name of the plot to save. The returned string can be directly used 
    with matplotlib.pyplot.savefig(). This function should be called for plots done 
    for a certain N value.
    
    Args:
        N (int): dimension of the sub-matrices of the Q matrix
        wi (float): starting frequency
        wf (float): ending frequency
        Np (float): amplitude of initialization seed
        vp (float): group velocity of the pump mode
        l (float): length of waveguide
        yN (float): mean number of pairs generated
        path (str): directory to save the plot
        
    returns:
        name and path for plt.savefig to save the plot in
    """
    title = path + "N_"+str(N)+"wi_"+str(wi)+"wf_"+str(wf)+"Np_"+str(Np)+"vp_"+str(vp)+"l_"+str(l)+"yN"+str(yN)+".pdf"
    return title
def save_data(N, wi, wf, Np, vp, l, yN, array, pump_shape, initial_guess, number, stable = True, path = "results/"):
    """
    Saves the results of each optimization experiment with the hyperparameters used,
    optimized results and the initial seed.
    
    Args:
        N (int): dimension of the sub-matrices of the Q matrix
        wi (float): starting frequency
        wf (float): ending frequency
        Np (float): amplitude of initialization seed
        vp (float): group velocity of the pump mode
        l (float): length of waveguide
        yN (float): mean number of pairs generated. Can be a float or an array of float
        array (ndarray[float] or list[ndarray[float]]): the optimized parameters.
            For the arbitrary pump, it is every elements of the pump. For a gaussian
            pump, it is the amplitude, width and phase.
        pump_shape (str): shape of the initial guess
        initial_guess (ndarray[float]): initialization seed used for the optimization
        number (int): the version of the experiment
        stable (bool): whether the optimized pump is stable (physically realisable or not noisy)
        path (str): repository in which the data will be saved
        
    returns:
        None
    """
    if len(np.array(array).shape) == 1:
        to_save = {
            "resolution":N, "wi":wi, "wf":wf, "Np":Np, "vp":vp, "l":l, "yN":yN, "parameters":array, "parameters_initialization":initial_guess
        }
        if stable == True:
            title = path + pump_shape + "_" + "stable" + "_" + str(number) + ".npy"
        else:
            title = path + pump_shape + "_" + "unstable" + "_" + str(number) + ".npy"
        np.save(title, to_save, allow_pickle = True)
    elif len(np.array(array).shape) == 2:
        to_save = {
            "resolution":N, "wi":wi, "wf":wf, "Np":Np, "vp":vp, "l":l, "yN_start":yN[0], 
            "yN_end": yN[-1], "parameters":array, "parameters_initialization":initial_guess
        }
        title = path + pump_shape + "_" + "unstable" + "_many_yN_" + str(number) +".npy"
        np.save(title, to_save)
    else:
        raise Exception("Parameters have too many dimensions") 