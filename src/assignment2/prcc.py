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
        # construct partial matrix without column i
        X_partial = np.delete(X, i, 1)
        x_i = X[:, i]
        
        # remove linear dependencies in xi
        wx_i = lin.lstsq(X_partial, x_i, rcond=-1)[0]  # approximate weights of xi by linear regression
        x_res = x_i -  X_partial @ wx_i  # subtract influence of other variable from xi

        # remove linear dependencies in yi
        wy_i = lin.lstsq(X_partial, y, rcond=-1)[0]   # approximate weights of yi by linear regression
        y_res = y - X_partial @ wy_i  # subtract influence of other variables from y

        # compute pcc of variable xi
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