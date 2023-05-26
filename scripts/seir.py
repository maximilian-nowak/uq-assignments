import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def plot(results):
    plt.figure(figsize=(10, 6))
    
    if len(results.T) == 5:
        S, E, I, R, C = results.T
        plt.plot(t, C, 'orange', label='Cases')
    else:
        S, E, I, R = results.T


    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'r', label='Exposed')
    plt.plot(t, I, 'g', label='Infected')
    plt.plot(t, R, 'm', label='Recovered')
    
    plt.title('SEIR Model')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

# ode system
def seir(y, t, *args):
    beta, alpha, gamma, N = args  # x0
    cumulative_cases = len(y)>4

    if len(y) < 3:
        raise Exception("Error in y: expected at least 3, got %i" % len(y))

    S, E, I = y[:3]
    dSdt = -beta*S*I/N
    dEdt = beta*S* I/N - alpha*E
    dIdt = alpha*E - gamma*I
    dRdt = gamma * I
    
    if cumulative_cases:
        dCdt = alpha * E
        return dSdt, dEdt, dIdt, dRdt, dCdt
    else:
        return dSdt, dEdt, dIdt, dRdt
     

beta = 14/9  # infection rate
alpha = 7/3  # incubation rate
gamma = 7/9  # recovery rate
i0 = 1000  # initial number of infected individuals

# Define initial conditions
N = 80e6
S0 = N - i0
E0 = 0
I0 = i0
R0 = 0
C0 = I0
y0 = [S0, E0, I0, R0, C0]

# simulation time
T = 60
t = np.linspace(0, T, 100)

x0 = (beta, alpha, gamma, N)
y0 = S0, E0, I0, R0, C0

sim_O1 = odeint(seir, y0, t, args=(x0))
plot(sim_O1)


