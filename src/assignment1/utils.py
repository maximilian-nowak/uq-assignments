import numpy as np

rng = np.random.default_rng()
config = {
    'datadir' : "../data"
}

def get_sample(x0, pert):
    """Sampels a given data vector x0 by a certain perturbation

    Args:
        x0 (np.array): data vector to perturb
        pert (float): percentage value of perturbation

    Returns:
        tuple: tuple of perturbed parameters
    """
    h = [0,0,0,0]
    for i in range(len(h)):
        h[i] = rng.uniform(x0[i]*(1-pert), x0[i]*(1+pert))
    return (h[0], h[1], h[2], h[3])

def run_simulations(model, x0, y0, t, p, N):
    """Generates data for solutions of the SEIR ode with a
    certain amount perturbation. 

    Args:
        seir_model (function): function defining a system of ode
        p (float): percentage of perturbation
        N (int): number of simulations
    """
    datadir = config['datadir']
    with open(datadir + "/C_data_" + str(int(p*100)), "wb") as fh1:
        with open(datadir + "/I_data_" + str(int(p*100)), "wb") as fh2:
            for _ in range(N):
                results = model(get_sample(x0, p)).solve(t, y0)
                C_data = results[:, 4]
                np.savetxt(fh1, np.column_stack(C_data))
                I_data = results[:, 2]
                np.savetxt(fh2, np.column_stack(I_data))

def get_metrics(data):
    """Returns metrics to a given dataset

    Args:
        C_data (np.array): the dataset

    Returns:
        tuple: mean, median, quantile_high, quantile_low
    """
    n = data.shape[1]
    mean = data.mean(0)
    median = np.zeros(n)
    quant_high = np.zeros(n)
    quant_low = np.zeros(n)
    for t in range(n):
        median[t] = np.median(data[:, t])
        quant_high[t] = np.quantile(data[:, t], 0.975)
        quant_low[t] = np.quantile(data[:, t], 0.025)
    return mean, median, quant_high, quant_low