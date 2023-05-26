import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class seir_model(object):
    """ODE solver for the SEIR model"""
    def __init__(self, x0):
        if len(x0)!=4:
            raise Exception("Error in x0: expected 4 parameters, got %i" % len(x0))
        self.x0 = x0
    
    def _get_ODE(self, y0):
        """Defines the system of ODE for SEIR. Deriving classes can override this method."""
        
        if len(y0)!=4:
            raise Exception("Error in y0: expected 4 initial values for SEIR, got %i" % len(y0))
        S, E, I, R = y0
        beta, alpha, gamma, N = self.x0
        dSdt = -beta*S*I/N
        dEdt = beta*S* I/N - alpha*E
        dIdt = alpha*E - gamma*I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    def __call__(self, y0, t, *args):
        return self._get_ODE(y0)
    
    def solve(self, t, y0):
        """Solves the system of ode.
        Args:
            t (array): time range to for which to solve the ode.
            y0 (tuple): initial values
        Returns:
            array: solution for each compartment of the ode.
        """
        return odeint(self, y0, t, args=(self.x0))
    
class seirc_model(seir_model):
    """ODE solver for the SEIRC as extension of SEIR."""
    
    def _get_ODE(self, y0):
        """Defines the system of ODE for SEIRC."""
        if len(y0)!=5:
            raise Exception("Error in y0: expected 5 initial values for SEIRC, got %i" % len(y0))
        
        seir_ode = super()._get_ODE(y0[:-1])
        dCdt = seir_ode[2] + seir_ode[3]  # dIdt + dRdt
        return seir_ode + (dCdt,)

def G1(data):
    """ Returns final number of cumulative cases.

    Args:
        data (numpy.array)
    """
    if len(data.shape)>1:
        return data[:, -1]
    else:
        return data[-1]
    
def G2(data, t):
    """ Returns point in time of peak infections.

    Args:
        data (array)
        t (array)
    """
    if len(data.shape)>1:
        return t[np.argmax(data, axis=1)]
    else:
        return t[data.argmax()]