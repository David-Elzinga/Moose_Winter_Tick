import os
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import minimize
from tri_seasonal_model import simulate

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--ncores", type=int, help="number of cores", default=os.cpu_count() - 2)
parser.add_argument("-m", "--mode", type=str, help="run/plot mode", default="run")
parser.add_argument("-s", "--samplesize", type=int, help="sample sizes per parameter", default=20)

def worker(param_values):

    parm = {}
    # These parameters do not vary. 
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
    parm['K'] = 1500 
    parm['mu'] = 0.293; parm['eta'] = 1.3352; parm['nu'] = 2.3774/23337.7315
    parm['d'] = 10**(-5)
    parm['gamma'] = 1.0755; parm['r'] = 0.8572
    parm['xi'] = 23337.7315

    parm['beta'] = 10**param_values[0]
    parm['r_T'] = param_values[1]
    parm['q'] = param_values[2]

    # Run the model
    num_years = 200
    S0 = 800; E0 = 50; I0 = 120; TH0 = 10**4; TQ0 = 0
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=3, thresh=10)

    if len(Zwinter) < num_years: # In this case there was an extinction
        return (np.nan, np.nan, np.nan, np.nan) # std dev, a, b, c (fits)
    else: # Not an extinction, record an estimate of the parameters to fit the cycles
        
        def func(x_data, a, b, c):
            return a*np.cos(b*x_data) + c

        def cost(x):
            a = x[0]; b = x[1]; c = x[2]
            return sum((y_data - func(x_data, a, b, c))**2)

        a_guesses = np.linspace(0, 1500, 5)
        b_guesses = np.linspace(np.pi/50, np.pi, 5)
        c_guesses = np.linspace(0, 1000, 5)

        x_data = np.linspace(0, 100, 100)
        y_data = Zwinter[-100:]

        guesses = list(product(a_guesses, b_guesses, c_guesses))
        cost_of_guesses = []
        fits = []

        for guess in guesses:
            fit = minimize(cost, guess, method='Nelder-Mead', bounds=((None, None), (0, np.pi), (None, None)), tol=1e-6)
            fits.append(fit)
            cost_of_guesses.append(fit.fun)

        best_fit_indx = cost_of_guesses.index(min(cost_of_guesses))
        popt = fits[best_fit_indx].x
        return (np.std(Zwinter[-100:]), popt[0], popt[1], popt[2])

def main(pool, s):

    # Define the number of samples we want for each parameter within its range. 
    logbeta_vals =  np.linspace(-4, 0, s)
    r_T_vals = np.linspace(3.2588, 35.5531, s)
    q_vals = np.array([0.1, 0.5, 0.9])

    # Construct a df to hold all parameter sets
    df = pd.DataFrame(list(product(logbeta_vals, r_T_vals, q_vals)), columns=['logbeta', 'r_T', 'q_vals'])

    # In parallel, run the model with each parameter combination, record the std dev. in population. 
    results = pool.map(worker, df.values)
    df['std'] = [r[0] for r in results]
    df['a'] = [r[1] for r in results]
    df['b'] = [r[2] for r in results]
    df['c'] = [r[3] for r in results]

    # Pickle the results.
    with open('bifurcation_grid.pickle', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot():
    
    # Open up the data
    with open('bifurcation_grid.pickle', 'rb') as handle:
        df = pickle.load(handle)
        s = df['logbeta'].unique().shape[0]

    import pdb; pdb.set_trace()
    # Plot the results! 
    fig, axes = plt.subplots(1, 3, figsize=(9,4), sharex=True, sharey=True)
    for ax, q_val in zip(axes, df['q_vals'].unique()):
        sub_df = df[df['q_vals'] == q_val]
        CS = ax.contourf(sub_df['logbeta'].values.reshape(s,s), sub_df['r_T'].values.reshape(s,s), sub_df['std'].values.reshape(s,s), cmap='viridis')
        ax.set_title(r'$q = $' + str(q_val))
        ax.text(-2, 20, "Extirpation")

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=0.15)
    clb = fig.colorbar(CS, ax=axes.ravel().tolist())
    clb.ax.set_ylabel('SD in Pop.', fontsize=12, rotation=270, labelpad=30)
    axes[1].set_xlabel(r'$\log_{10}(\beta)$', fontsize=14, labelpad=10)
    axes[0].set_ylabel(r'$r_T$', fontsize=14, rotation=0, labelpad=20)
    plt.savefig('bifurcation_std.pdf')

    # Cycle length figure
    df.loc[df['std'] < 100 ,'b'] = np.nan
    df['cycle_length'] = 2*np.pi / np.abs(df['b'])
    df.loc[df['cycle_length'] > 100, 'cycle_length'] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(9,4), sharex=True, sharey=True)
    for ax, q_val in zip(axes, df['q_vals'].unique()):
        sub_df = df[df['q_vals'] == q_val]
        CS = ax.contourf(sub_df['logbeta'].values.reshape(s,s), sub_df['r_T'].values.reshape(s,s), sub_df['cycle_length'].values.reshape(s,s), cmap='viridis')
        ax.set_title(r'$q = $' + str(q_val))
        ax.text(-2, 20, "Extirpation")

    fig.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=0.15)
    clb = fig.colorbar(CS, ax=axes.ravel().tolist())
    clb.ax.set_ylabel('Cycle Length', fontsize=12, rotation=270, labelpad=30)
    axes[1].set_xlabel(r'$\log_{10}(\beta)$', fontsize=14, labelpad=10)
    axes[0].set_ylabel(r'$r_T$', fontsize=14, rotation=0, labelpad=20)
    plt.savefig('bifurcation_cycle_length.pdf')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == "run":
        with multiprocessing.Pool(processes=args.ncores) as pool:
            main(pool, args.samplesize)
    elif args.mode == 'plot':
        plot()