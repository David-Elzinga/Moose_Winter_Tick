import os
import argparse
import pickle
import pandas as pd
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from wildlife_management_model import simulate
import time

import matplotlib.pyplot as plt

# N = 2**14

default_n = os.cpu_count()
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000,
                    help="obtain N*(2D+2) samples from parameter space")
parser.add_argument("-n", "--ncores", type=int, default = default_n,
                    help="number of cores, defaults to {} on this machine".format(default_n))
parser.add_argument("-m", "--mode", type=str, default = 'generate',
                    help="generate or analyze sobol samples")

def worker(param_values):
    output = run_model(param_values)
    return output

def run_model(p):

    # These parameters do not vary. 
    parm = {}
    parm['mu_alpha'] = 0; parm['mu_omega'] = 0

    # These parameters vary. 
    parm['omega'] = p[0]; parm['alpha'] = p[1]; parm['tau'] = 1 - parm['alpha'] - parm['omega']
    parm['mu'] = p[2]; parm['nu'] = p[3];  parm['gamma'] = p[4]; parm['beta'] = 10**(p[5])
    parm['r_S'] = p[6]; parm['r_P'] = p[7]; parm['u'] = p[8]

    parm['eta'] = p[9]; parm['xi'] = p[10]; parm['q'] = p[11]; parm['K'] = p[12]; parm['r_T'] = p[13]
    parm['c'] = 10**(-p[14]); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']

    # Run the model
    num_years = 200
    S0 = 1000; E0 = 250; P0 = 250; H0 = 52000*250; Q0 = 0
    t0 = time.time()
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, P0, H0, Q0], p=parm, granularity=2, thresh=10)
    t1 = time.time()

    if t1 - t0 > 30:
        print(parm)

    if len(Zwinter) < num_years: # In this case there was an extinction, so deviation we assume is zero.
        std = 0
    else: # Not an extinction, record the standard deviation over the final 100 years.
        std = np.std(Zwinter[-100:])
    return std, Zwinter[-1]

def main(N, ncores=None, pool=None):

    # Define the parameter space within the context of a problem dictionary
    problem = {
        # number of parameters
        'num_vars' : 16,
        # parameter names
        'names' : ['omega', 'alpha', 'mu', 'nu', 'gamma', 'beta', 'r_S', 'r_P', 'u', 'eta', 'xi', 'q', 'K', 'r_T', 'c', 'dummy'], 
        # bounds for each corresponding parameter
        'bounds' : [
        [0.5014, 0.5836], # omega
        [0.0422, 0.1244], # alpha
        [0.2402, 0.3458], # mu
        [2.0223, 2.9918], # nu
        [1.4818, 2.2228], # gamma
        [-2, 1], # beta (log-scale)
        [0.6872, 1.0308], # r_S
        [0.3998, 0.5998], # r_P
        [0.25, 1], # u
        [0.9509, 1.4263], # eta
        [34800, 63600], # xi
        [0, 1], # q
        [1200, 1800], # K
        [3.2588, 35.5531], # r_T
        [-4, 0], # c (log-scale)
        [-1, 1] # dummy
        ]
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
        result = {'output_std': [k[0] for k in output], 'output_pop_size': [k[1] for k in output], 'param_values': param_values}
        pickle.dump(result, f)

def analyze_sobol():

    # Define the parameter space within the context of a problem dictionary
    problem = {
        # number of parameters
        'num_vars' : 16,
        # parameter names
        'names' : ['omega', 'alpha', 'mu', 'nu', 'gamma', 'beta', 'r_S', 'r_P', 'u', 'eta', 'xi', 'q', 'K', 'r_T', 'c', 'dummy'], 
        # bounds for each corresponding parameter
        'bounds' : [
        [0.5014, 0.5836], # omega
        [0.0422, 0.1244], # alpha
        [0.2402, 0.3458], # mu
        [2.0223, 2.9918], # nu
        [1.4818, 2.2228], # gamma
        [-2, 1], # beta (log-scale)
        [0.6872, 1.0308], # r_S
        [0.3998, 0.5998], # r_P
        [0.25, 1], # u
        [0.9509, 1.4263], # eta
        [34800, 63600], # xi
        [0, 1], # q
        [1200, 1800], # K
        [3.2588, 35.5531], # r_T
        [-4, 0], # c (log-scale)
        [-1, 1] # dummy
        ]
    }

    with open('unprocessed_sobol.pickle', 'rb') as handle:
        result = pickle.load(handle)
    
    output = np.array(result['output_std'])
    S2 = {}
    var_sens = sobol.analyze(problem, output, calc_second_order=True)
    S2['var'] = pd.DataFrame(var_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['var_conf'] = pd.DataFrame(var_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
    var_sens_std = pd.DataFrame(var_sens,index=problem['names'])
    var_sens_std = var_sens_std[var_sens_std['ST'] > 0]

    output = np.array(result['output_pop_size'])
    S2 = {}
    var_sens = sobol.analyze(problem, output, calc_second_order=True)
    S2['var'] = pd.DataFrame(var_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['var_conf'] = pd.DataFrame(var_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
    var_sens_pop_size = pd.DataFrame(var_sens,index=problem['names'])
    var_sens_pop_size = var_sens_pop_size[var_sens_pop_size['ST'] > 0]


    ind = np.arange(len(var_sens_std.index))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, var_sens_std['ST'], width, yerr=var_sens_std['ST_conf'],
                    label='Pop. SD.')
    rects2 = ax.bar(ind + width/2, var_sens_pop_size['ST'], width, yerr=var_sens_pop_size['ST_conf'],
                    label='Pop. Size')
    
    ax.set_ylabel('Total Sobol Index', fontsize=15)
    ax.set_xticks(ind)
    ax.set_xticklabels((r'$\omega$', r'$\alpha$', r'$\mu$', r'$\nu$', r'$\beta$', r'$r_S$', r'$r_P$', r'$u$', r'$\eta$', r'$\xi$', r'$q$', r'$K$', r'$r_T$'))
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)

    plt.savefig('sobol.pdf')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == 'generate':
        with Pool(args.ncores) as pool:
            main(args.N, args.ncores, pool)
    elif args.mode == 'analyze':
        analyze_sobol()
    else:
        print('Not a valid mode')