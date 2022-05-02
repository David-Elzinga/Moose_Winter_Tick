from SALib.sample import saltelli
import numpy as np
from tri_seasonal_model import simulate


def run_model(mu, eta, nu, loga, logb, gamma, logd, logbeta, r, r_T, xi):

    # Define model parameters. 
    parm = {}
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu'] = mu; parm['eta'] = eta; parm['nu'] = nu
    parm['a'] = 10**loga; parm['b'] = 10**logb; parm['d'] = 10**logd
    parm['gamma'] = gamma; parm['r'] = r
    parm['beta'] = 10**logbeta; parm['r_T'] = r_T
    parm['K'] = 1500; parm['xi'] = xi
    parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
    parm['psi'] = 0

    num_years = 100
    S0 = 800; E0 = 50; I0 = 120
    TH0 = np.exp(14.5); TQ0 = 0
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=100)
    var = np.var(Zwinter[-50:])
    return var
    
### Define the parameter space within the context of a problem dictionary ###
problem = {
    # number of parameters
    'num_vars' : 11,
    # parameter names
    'names' : ['mu', 'eta', 'nu', 'loga',
                'logb', 'gamma', 'logd', 'logbeta',
                'r', 'r_T', 'xi'], 
    # bounds for each corresponding parameter
    'bounds' : [[0.2402, 0.3458], [0.2402,2.9918], [2.0229, 2.9918], [-2,2],
                [-2,2], [0,2.150], [-2,2], [-4,-1],
                [0.8359, 0.8785], [14.8796, 85.9099], [1250, 95495]]
}

### Create an N*(2D+2) x num_var matrix of parameter values ###
param_values = saltelli.sample(problem, 2, calc_second_order=True)

run_model(*param_values[0])