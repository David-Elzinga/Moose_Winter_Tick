import pandas as pd
import numpy as np
import multiprocessing
from wildlife_management_model import simulate
import matplotlib.pyplot as plt
from itertools import product

'''
This code generates heatmaps for the population volitality at various
autumn and winter harvesting rates. 
'''

def worker(obj):

    # Specify the initial conditions. Group them into an array 'Z'. 
    S0 = 1000; E0 = 250; P0 = 250; H0 = 52000*250; Q0 = 0
    Z0 = np.array([S0, E0, P0, H0, Q0])
    Zsol = [Z0]; tsol = [0]

    # Define model parameters. 
    parm = {}
    parm['omega'] = 0.5425; parm['alpha'] = 0.0833; parm['tau'] = 1 - parm['alpha'] - parm['omega']
    parm['mu'] = 0.2835; parm['nu'] = 2.3772;  parm['gamma'] = 1.8523; parm['beta'] = 10**(-2.1)
    parm['r_S'] = 0.8590; parm['r_P'] = 0.4998; parm['u'] = 0.75

    parm['eta'] = 1.1886; parm['xi'] = 52000; parm['q'] = 0.75; parm['K'] = 1500; parm['r_T'] = 13.0293
    parm['c'] = 10**(-2); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']; parm['epsilon'] = 0.01

    # Unpack the hunting params.
    #print(obj)
    parm['mu_omega'] = obj[0]; parm['mu_alpha'] = obj[1]

    # Run the model
    num_years = 200
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, P0, H0, Q0], p=parm, granularity=2, thresh = 10)

    if len(Zwinter) == num_years + 1:
        return (max(Zwinter[100:]), min(Zwinter[100:]))
    else:
        return (np.nan, np.nan)

def main(pool):

    # Construct a grid of values for the hunting rates. 
    s = 20
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
    ax[0].set_xlabel('Winter Harvesting Rate ' + r'$\mu_\omega$' + ' (moose/year)', fontsize=11); ax[0].set_ylabel('Autumn Harvesting Rate ' + r'$\mu_\alpha$' + ' (moose/year)', fontsize=11)
    cbar = fig.colorbar(CS, ax = ax[0])

    CS = ax[1].contourf(df['mu_omega'].values.reshape(s,s), df['mu_alpha'].values.reshape(s,s), df['max_pop'].values.reshape(s,s), cmap='viridis')
    ax[1].set_title('Maximum Herd Population')
    ax[1].set_xlabel('Winter Harvesting Rate ' + r'$\mu_\omega$' + ' (moose/year)', fontsize=11); ax[1].set_ylabel('Autumn Harvesting Rate ' + r'$\mu_\alpha$' + ' (moose/year)', fontsize=11)
    cbar = fig.colorbar(CS, ax = ax[1])

    plt.savefig('hunting_heatmap.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)
