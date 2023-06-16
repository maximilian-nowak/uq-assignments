import numpy as np
from assignment1.seir import seirc_model, G1, G2

def create_sobol_matrices(A, B):
    """ Generates sample matrices by switching the ith column of matrix A and
        the ith column of matrix B.
        
    Returns:
        np.array((n,N,n)): list of matrices with length of number of columns of A.
    """
    # ensure correct input shapes
    if A.shape != B.shape:
        raise Exception("Error: matrices A and B must be of same shape.")
    
    # construct list of matrices A with length of columns
    N, n = A.shape
    samples_Ab = np.zeros((n, N, n))
    
    # create samples by switching ith column of A by ith column of B
    for i in range(n):
        samples_Ab[i,:,:] = A
        samples_Ab[i,:,i] = B[:,i]
        
    return samples_Ab


def compute_total_indices(f_A, f_Ab):
    """computes the total-effect sobol indices using Jansen's estimator.

    Args:
        f_A (np.array((N,))): model function
        f_AB (np.array((n, N))): Model function with swapped columns.

    Returns:
        np.array((N,)): array of total Sobol indices for each input variable.
    """
    N = f_A.shape[0]
    var_f_A = np.var(f_A)
    Ti = (1/(2*N) * np.sum((f_A - f_Ab)**2, axis=1)) / var_f_A
    return Ti

def compute_QoIs_from_samples(X0, args):
    """Runs the seirc model by sampling parameters from X0 and 
       determines quantities of interest (QoI) by applying G1 and G2.

    Args:
        X0 (np.array(N, n)): matrix containing parameter combinations.

    Returns:
        np.array((2, N)): G1 and G2
    """
    (N, S0, E0, i0, R0, C0), t = args
    
    g1_results, g2_results = [], []
    # run model with sampled parameters and apply QoIs
    for beta, alpha, gamma, i0 in X0:
        S0 = N - i0
        y0 = [S0, E0, i0, R0, C0]  # update initial conditions with i0
        sol = seirc_model((beta, alpha, gamma, N)).solve(t, y0)
        g1_results.append(G1(sol[:, 4]))
        g2_results.append(G2(sol[:, 2], t))
    
    return np.array([g1_results, g2_results])