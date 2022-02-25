import numpy as np
from scipy.integrate import solve_ivp

# Specify the number of years. 
num_years = 5

# Specify the initial conditions. Group them into an array 'Z' which refers
# to the transformed populations (for numerical stability) with transformation
# z -> log(z).
S0 = 1500; E0 = 10; I0 = 1
TH0 = 100; TQ0 = 10**(-5) # simulates zero in log land
Z0 = np.log(np.array([S0, E0, I0, TH0, TQ0])).tolist()
Zsol = [Z0] 

# Define model parameters. 
parm = {}
parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
parm['mu'] = 0.2930; parm['eta'] = 1.3352; parm['nu'] = 2.3774
parm['a'] = 1; parm['b'] = 1; parm['d'] = 1
parm['gamma'] = 1.0755; parm['r'] = 0.8572
parm['beta'] = 1; parm['r_T'] = 20
parm['K'] = 1500; parm['xi'] = 23242

# Define the ODEs. Note: these rates of changes are in log form for numerical stability. 
def winter_odes(t, Z, parm):

    # Unpack the states, preallocate ODE evaluations. 
    Sz, Ez, Iz, THz, TQz = Z; ODEs = 5*[0]

    # Calculate ODEs
    dSz = -parm['mu']
    dEz = -parm['eta']
    dIz = -parm['eta']*(1-parm['c']) - parm['mu']*parm['c'] - parm['nu']*(np.exp(THz)/np.exp(Iz))/(parm['a']*parm['c'] + parm['b']*(1-parm['c']) + (np.exp(THz)/np.exp(Iz)))
    dTHz =  dIz
    dTQz = 0

def summer_odes(t, Z, parm):

    # Unpack the states, preallocate ODE evaluations. 
    Sz, Ez, Iz, THz, TQz = Z; ODEs = 5*[0]

    # Calculate ODEs
    dSz = np.exp(Ez - Sz)*parm['gamma']*parm['l']/(parm['d'] + parm['l']) - parm['mu']
    dEz = -(parm['mu'] + parm['gamma']*parm['l']/(parm['d'] + parm['l']))
    dIz = 0
    dTHz = 0 
    dTQz = 0

def autumn_odes(t, Z, parm):

    # Unpack the states, preallocate ODE evaluations. 
    Sz, Ez, Iz, THz, TQz = Z; ODEs = 5*[0]

    # Calculate ODEs
    dSz = -parm['beta']/parm['xi'] * np.exp(TQz) - parm['mu']
    dEz = -parm['beta']/parm['xi'] * np.exp(TQz) - parm['mu']
    dIz = parm['beta']/parm['xi'] * (np.exp(Sz) + np.exp(Ez)) * np.exp(TQz - Iz) - parm['mu']
    dTHz = parm['beta']*(np.exp(Sz) + np.exp(Ez) + np.exp(Iz))*np.exp(TQz - THz)
    dTQz = -parm['beta']*(np.exp(Sz) + np.exp(Ez) + np.exp(Iz))


for year in range(num_years):

    # Simulate winter.
    parm['c'] = np.exp(Zsol[-1][0]) / (np.exp(Zsol[-1][0]) + np.exp(Zsol[-1][1]))
    t_winter = [year, year + parm['omega']]
    X = solve_ivp(fun=winter_odes, t_span=t_winter, y0=Z0, args=(parm,))
    import pdb; pdb.set_trace()

    # Pulse into summer.

    # Simulate summer.

    # Pulse into autumn

    # Simulate autumn

    # Pulse into winter. 




