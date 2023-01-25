import os
import argparse
import pickle
import pandas as pd
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from tri_seasonal_model import simulate
import time

# N - 100,000

default_n = os.cpu_count()
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000,
                    help="obtain N*(2D+2) samples from parameter space")
parser.add_argument("-n", "--ncores", type=int, default = default_n,
                    help="number of cores, defaults to {} on this machine".format(default_n))
parser.add_argument("-m", "--mode", type=str, default = 'generate',
                    help="generate or analyze sobol samples")

def worker(param_values):
    std = run_model(param_values)
    return std

def run_model(param_values):

    parm = {}
    # These parameters do not vary. 
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
    parm['K'] = 1500 

    # These parameters vary. 
    parm['mu'] = param_values[0]; parm['eta'] = param_values[1]; parm['nu'] = param_values[2]/param_values[8]
    parm['d'] = 10**param_values[3]
    parm['gamma'] = param_values[4]; parm['r'] = param_values[5]
    parm['beta'] = 10**param_values[6]; parm['r_T'] = param_values[7]
    parm['xi'] = param_values[8]; parm['q'] = param_values[9]

    # Run the model
    num_years = 200
    S0 = 800; E0 = 50; I0 = 120; TH0 = 10**4; TQ0 = 0
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=3, thresh=10)
    if len(Zwinter) < num_years: # In this case there was an extinction, so deviation we assume is zero.
        std = 0
    else: # Not an extinction, record the standard deviation over the final 100 years.
        std = np.std(Zwinter[-100:])
    return std

def main(N, ncores=None, pool=None):

    # Define the parameter space within the context of a problem dictionary
    problem = {
        # number of parameters
        'num_vars' : 10,
        # parameter names
        'names' : ['mu', 'eta', 'nu', 'd_expn','gamma', 'r', 'beta_expn', 'r_T', 'xi', 'q'], 
        # bounds for each corresponding parameter
        'bounds' : [[0.2402, 0.3458], [0.2402,2.9918], [2.0229, 2.9918],
                    [-10, 0], [0, 2.150], [0.8359, 0.8785],
                    [-4, 0], [3.2588, 35.5531], [1250, 95495], [0, 1]]
    }

    # Create the parameter combinations. 
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    
    # Run the sensitivity analysis!
    t0 = time.time()
    output = pool.map(worker, param_values)
    t1 = time.time()

    print(t1-t0)

    # Save our results as a dictionary before we processes them.  
    with open("unprocessed_sobol.pickle", "wb") as f:
        result = {'output': output, 'param_values': param_values}
        pickle.dump(result, f)

def analyze_sobol():

    # Define the parameter space within the context of a problem dictionary
    problem = {
        # number of parameters
        'num_vars' : 10,
        # parameter names
        'names' : ['mu', 'eta', 'nu', 'd_expn','gamma', 'r', 'beta_expn', 'r_T', 'xi', 'q'], 
        # bounds for each corresponding parameter
        'bounds' : [[0.2402, 0.3458], [0.2402,2.9918], [2.0229, 2.9918],
                    [-10, 0], [0, 2.150], [0.8359, 0.8785],
                    [-4, 0], [3.2588, 35.5531], [1250, 95495], [0, 1]]
    }

    with open('unprocessed_sobol.pickle', 'rb') as handle:
        result = pickle.load(handle)
    
    output = np.array(result['output'])
    S2 = {}
    var_sens = sobol.analyze(problem, output, calc_second_order=True)
    S2['var'] = pd.DataFrame(var_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['var_conf'] = pd.DataFrame(var_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])

    var_sens = pd.DataFrame(var_sens,index=problem['names'])
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == 'generate':
        with Pool(args.ncores) as pool:
            main(args.N, args.ncores, pool)
    elif args.mode == 'analyze':
        analyze_sobol()
    else:
        print('Not a valid mode')