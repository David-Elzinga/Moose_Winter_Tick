import os
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import minimize
from wildlife_management_model import simulate

'''
This code generates heatmaps for the population volitality at various
q, r_P, and beta values. Its parsers specify the number of cores to use, if you 
are running or plotting the results, and the sample size of the grid. In addition 
to the volitality heatmaps, it uses a Mulitstart algorithm to calculate the cycle 
periods, although this bit was excluded from the manuscript for brevity. 
'''

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type=int, help="number of cores", default=os.cpu_count() - 2)
parser.add_argument("-m", "--mode", type=str, help="run/plot mode", default="run")
parser.add_argument("-s", "--samplesize", type=int, help="sample sizes per parameter", default=200)

def worker(param_values):

    # Unpack the parameters that change
    parm = {}
    parm['beta'] = 10**param_values[0]
    parm['r_T'] = param_values[1]
    parm['q'] = param_values[2]

    # These parameters do not vary. 
    parm['omega'] = 0.5425; parm['alpha'] = 0.0833; parm['tau'] = 1 - parm['alpha'] - parm['omega']
    parm['mu'] = 0.2835; parm['nu'] = 2.3772;  parm['gamma'] = 1.8523
    parm['r_S'] = 0.8590; parm['r_P'] = 0.4998; parm['u'] = 0.75

    parm['eta'] = 1.1886; parm['xi'] = 52000; parm['K'] = 1500
    parm['c'] = 10**(-2); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']; parm['epsilon'] = 0.01

    parm['mu_alpha'] = 0; parm['mu_omega'] = 0

    # Run the model
    num_years = 200
    S0 = 1000; E0 = 250; P0 = 250; H0 = 52000*250; Q0 = 0
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, P0, H0, Q0], p=parm, granularity=2, thresh=10)

    if len(Zwinter) < num_years: # In this case there was an extinction
        return (np.nan, len(Zwinter)) # std dev, ext time
    else: # Not an extinction, record an estimate of the parameters to fit the cycles
        return (np.std(Zwinter[-100:]), np.nan)

def main(pool, s):

    # Define the number of samples we want for each parameter within its range. 
    logbeta_vals =  np.linspace(-3, 1, s)
    r_T_vals = np.linspace(3.2588, 35.5531, s)
    q_vals = np.array([0.75, 0.9])

    # Construct a df to hold all parameter sets
    df = pd.DataFrame(list(product(logbeta_vals, r_T_vals, q_vals)), columns=['logbeta', 'r_T', 'q_vals'])

    # In parallel, run the model with each parameter combination, record the std dev. in population. 
    results = pool.map(worker, df.values)
    df['std'] = [r[0] for r in results]
    df['ext_time'] = [r[1] for r in results]

    # Pickle the results.
    with open('oscillations_grid.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot():
    
    # Open up the data
    with open('oscillations_grid.pickle', 'rb') as handle:
        df = pickle.load(handle)
        s = df['logbeta'].unique().shape[0]

    # Plot the results! 
    fig, axes = plt.subplots(1, 2, figsize=(9,4), sharex=True, sharey=True)
    for ax, q_val in zip(axes, df['q_vals'].unique()):
        sub_df = df[df['q_vals'] == q_val]
        CS_sd = ax.contourf(sub_df['logbeta'].values.reshape(s,s), sub_df['r_T'].values.reshape(s,s), sub_df['std'].values.reshape(s,s), cmap='cool')
        CS_ext_time = ax.contourf(sub_df['logbeta'].values.reshape(s,s), sub_df['r_T'].values.reshape(s,s), sub_df['ext_time'].values.reshape(s,s), cmap='autumn')
        ax.set_title(r'$q = $' + str(q_val))
        ax.text(-1, 20, "Extirpation")

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=0.15)

    clb_ext_time = fig.colorbar(CS_ext_time, ax=axes.ravel().tolist())
    clb_ext_time.ax.set_ylabel('Extirpation Year', fontsize=12, rotation=270, labelpad=20)

    clb_sd = fig.colorbar(CS_sd, ax=axes.ravel().tolist())
    clb_sd.ax.set_ylabel('SD in Pop.', fontsize=12, rotation=270, labelpad=20)

    axes[0].set_xlabel(r'Log Rate of Parasitism, $\log_{10}(\beta)$', fontsize=11, labelpad=10)
    axes[1].set_xlabel(r'Log Rate of Parasitism, $\log_{10}(\beta)$', fontsize=11, labelpad=10)
    axes[0].set_ylabel(r'Average Tick Reproduction, $r_T$', fontsize=11, rotation=90, labelpad=10)
    plt.savefig('oscillations.pdf')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == "run":
        with multiprocessing.Pool(processes=args.ncores) as pool:
            main(pool, args.samplesize)
    elif args.mode == 'plot':
        plot()