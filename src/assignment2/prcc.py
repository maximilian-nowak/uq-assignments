import numpy as np
import numpy.linalg as lin

def partial_corrcoef(X, y):
    """Returns the partial correlation coefficients to a given matrix X and vector y.
    Args:
        X (np.array): the matrix.
        y (np.array): the vector.
    Returns:
        np.array: vector containing the pcc's.
    """
    dim, N = X.shape
    pcc = []
    X = np.c_[X, np.ones(dim)]
    
    for i in range(N):
        X_partial = np.delete(X, i, 1)
        x_i = X[:, i]
        
        wx_i = lin.lstsq(X_partial, x_i, rcond=-1)[0]
        x_res = x_i -  X_partial @ wx_i

        wy_i = lin.lstsq(X_partial, y, rcond=-1)[0]
        y_res = y - X_partial @ wy_i

        pcc.append(x_res @ y_res / (np.sqrt(np.sum(x_res**2) * np.sum(y_res**2))))

    return np.array(pcc).T

def prcc(X, y):
    """Returns the partial rank correlation coefficient. 
    Args:
        X (np.array): the matrix.
        y (np.array): the vector.
    Returns:
        np.array: vector containing the prcc's.
    """
    X = X.argsort(axis=0).argsort(axis=0)
    y = y.argsort(axis=0).argsort(axis=0)
    return partial_corrcoef(X, y)