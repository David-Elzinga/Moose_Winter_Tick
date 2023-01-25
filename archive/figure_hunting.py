import pandas as pd
import numpy as np
import multiprocessing
from tri_seasonal_model import simulate
import matplotlib.pyplot as plt
from itertools import product

def worker(obj):

    # Specify the initial conditions. Group them into an array 'Z'. 
    S0 = 800; E0 = 1; I0 = 1
    TH0 = np.exp(4.6); TQ0 = 0
    Z0 = np.array([S0, E0, I0, TH0, TQ0])
    Zsol = [Z0]; tsol = [0]

    # Define model parameters. 
    parm = {}
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
    parm['K'] = 1500 
    parm['mu'] = 0.293; parm['eta'] = 1.3352; parm['nu'] = 2.3774/100.7315
    parm['d'] = 10**(-5)
    parm['gamma'] = 1.0755; parm['r'] = 0.8572
    parm['xi'] = 100.7315

    parm['beta'] = 10**(-2.6)
    parm['r_T'] = 13.0293
    parm['q'] = 0.9

    # Unpack the hunting params.
    print(obj)
    parm['mu_omega'] = obj[0]; parm['mu_alpha'] = obj[1]

    # Run the model
    num_years = 200
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=3, thresh = 10)

    if len(Zwinter) == num_years:
        return (max(Zwinter[100:]), min(Zwinter[100:]))
    else:
        return (np.nan, np.nan)

def main(pool):

    # Construct a grid of values for the hunting rates. 
    s = 60
    mu_omega_vals =  np.linspace(0, 1, s)
    mu_alpha_vals = np.linspace(0, 1, s)

    # Construct a df to hold all parameter sets
    df = pd.DataFrame(list(product(mu_omega_vals, mu_alpha_vals)), columns=['mu_omega', 'mu_alpha'])

    # Run all time series. Extract minimum and maximum population.
    results = pool.map(worker, df.values)
    df['max_pop'] = [r[0] for r in results]
    df['min_pop'] = [r[1] for r in results]

    # Plot! 
    fig, ax = plt.subplots(1, 2, figsize=(9,4), sharex=True, sharey=True)
    CS = ax[0].contourf(df['mu_omega'].values.reshape(s,s), df['mu_alpha'].values.reshape(s,s), df['min_pop'].values.reshape(s,s), cmap='viridis')
    ax[0].set_title('Minimum Herd Population')
    ax[0].set_xlabel(r'$\mu_\omega$', fontsize=20); ax[0].set_ylabel(r'$\mu_\alpha$', fontsize=20)
    cbar = fig.colorbar(CS, ax = ax[0])

    CS = ax[1].contourf(df['mu_omega'].values.reshape(s,s), df['mu_alpha'].values.reshape(s,s), df['max_pop'].values.reshape(s,s), cmap='viridis')
    ax[1].set_title('Maximum Herd Population')
    ax[1].set_xlabel(r'$\mu_\omega$', fontsize=20); ax[1].set_ylabel(r'$\mu_\alpha$', fontsize=20)
    cbar = fig.colorbar(CS, ax = ax[1])

    plt.savefig('fig_hunting_cycles.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)
