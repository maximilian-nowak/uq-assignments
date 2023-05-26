import numpy as np

def latin_hypercube_uniform(dim, n_samples, low=0, high=1):
    """Returns a latin hypercube whose samples are split into equally spaced intervals.
    Works with multidimensional boundaries.
    
    Args:
        dim (int): dimension
        n_samples (int): number of samples in each dimension.
        low (int, optional): lower boundary of samples. Can be a vector. Defaults to 0.
        high (int, optional): upper boundary of samples. Can be a vector. Defaults to 1.

    Returns:
        np.array: 2d hypercube of samples
    """
    rand_matrix = np.empty((n_samples, dim))
    subintervals, stepsize = np.linspace(0, 1, n_samples+1, retstep=True)
    
    # generate uniformely distributed samples
    for i in range(dim):
        for j, min in enumerate(subintervals[:-1]):
            rand_matrix[j, i] = np.random.uniform(min, min+stepsize)

    # permute the columns
    for i in range(dim):
        rand_matrix[:, i] = np.random.permutation(rand_matrix[:, i])
        
    return rand_matrix * (high - low) + low 