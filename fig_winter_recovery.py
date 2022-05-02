import numpy as np
import multiprocessing
from tri_seasonal_model import simulate
import matplotlib.pyplot as plt

def worker(obj):

    # Specify the initial conditions. Group them into an array 'Z'. 
    S0 = 800; E0 = 50; I0 = 120
    TH0 = np.exp(14.5); TQ0 = 0
    Z0 = np.array([S0, E0, I0, TH0, TQ0])
    Zsol = [Z0]; tsol = [0]

    # Define model parameters. 
    parm = {}
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu'] = 0.2930; parm['eta'] = 1.3352; parm['nu'] = 2.3774
    parm['a'] = 1; parm['b'] = 1; parm['d'] = 1
    parm['gamma'] = 1.0755; parm['r'] = 0.8572
    parm['beta'] = 0.001; parm['r_T'] = 20 #0.001
    parm['K'] = 1500; parm['xi'] = 23242
    parm['mu_tau'] = 0; parm['mu_omega'] = 0; parm['mu_alpha'] = 0

    # Unpack the winter recovery rate.
    parm['psi'] = obj

    # Run the model
    tsol, Zsol = simulate(num_years=50, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm)
    
    return (Zsol[:,:3].sum(axis=1).max(), Zsol[:,:3].sum(axis=1).min())

def main(pool):
    # Construct a list of values for the recovery rate
    psi_vals = np.linspace(0,0.5,20)

    # Run all time series. Extract minimum and maximum population.
    pop_max_min = pool.map(worker, psi_vals)
    pop_max = [p[0] for p in pop_max_min]
    pop_min = [p[1] for p in pop_max_min]

    # Plot! 
    fig, axes = plt.subplots(1,2)
    curve_min = axes[0].plot(psi_vals,pop_min)
    curve_max = axes[1].plot(psi_vals,pop_max)

    axes[0].set_xlabel(r'$\psi$', fontsize=20); axes[1].set_xlabel(r'$\psi$', fontsize=20)
    axes[0].set_ylabel('Minimum Herd Size', fontsize=20); axes[1].set_ylabel('Maximum Herd Size', fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)
