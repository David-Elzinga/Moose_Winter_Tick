import os
import argparse
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from tri_seasonal_model import simulate


default_n = os.cpu_count()
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000,
                    help="obtain N*(2D+2) samples from parameter space")
parser.add_argument("-n", "--ncores", type=int, default = default_n,
                    help="number of cores, defaults to {} on this machine".format(default_n))



def run_model(mu, eta, nu, a_expn, b_expn, d_expn, gamma, r, beta_expn, r_T, xi, q):
    parm = {}
    # These parameters do not vary. 
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
    parm['K'] = 1500 

    # These parameters vary. 
    parm['mu'] = mu, parm['eta'] = eta, parm['nu'] = nu
    parm['a'] = 10**a_expn; parm['b'] = 10**b_expn; parm['d'] = 10**d_expn
    parm['gamma'] = gamma; parm['r'] = r
    parm['beta'] = 10*beta_expn; parm['r_T'] = r_T
    parm['xi'] = xi; parm['q'] = q

    # Run the model
    num_years = 500
    S0 = 800; E0 = 50; I0 = 120; TH0 = 10**8; TQ0 = 0
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=100, thresh=10)
    if len(Zwinter) < num_years: # In this case there was an extinction, so variance we assume is zero.
        var = 0
    else: # Not an extinction, record the variance over the final 100 years.
        var = np.var(Zwinter[-100:])
    return var

def main(N, ncores=None, pool=None):

    # Define the parameter space within the context of a problem dictionary
    problem = {
        # number of parameters
        'num_vars' : 12,
        # parameter names
        'names' : ['mu', 'eta', 'nu', 'a_expn', 'b_expn', 'd_expn','gamma', 'r', 'beta_expn', 'r_T', 'xi', 'q'], 
        # bounds for each corresponding parameter
        'bounds' : [[0.2402, 0.3458], [0.2402,2.9918], [2.0229, 2.9918], [2,5],
                    [2, 5], [-10, 0], [0, 2.150], [0.8359, 0.8785],
                    [-4, 0], [14.8796, 85.9099], [1250, 95495], [0, 1]]
    }

    # Create the parameter combinations. 
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    # Run the sensitivity analysis! 
    chunksize = param_values.shape[0]//ncores
    output = pool.starmap(run_model, param_values, chunksize=chunksize)

if __name__ == "__main__":
    args = parser.parse_args()
    with Pool(args.ncores) as pool:
        main(args.N, args.ncores, pool)