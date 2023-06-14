import numpy as np

def create_sobol_matrices(A, B):
    """ Generates sample matrices by switching the ith column of matrix A and
        the ith column of matrix B.
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