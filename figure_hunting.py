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
    parm['beta'] = 0.0001; parm['r_T'] = 20 #0.001
    parm['K'] = 1500; parm['xi'] = 23242
    parm['mu_tau'] = 0

    # Unpack the hunting params.
    parm['mu_omega'], parm['mu_alpha'] = obj

    # Run the model
    tsol, Zsol = simulate(num_years=50, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm)
    
    return (Zsol[:,:3].sum(axis=1).max(), Zsol[:,:3].sum(axis=1).min())

def main(pool):
    # Construct a grid of values for the hunting rates. 
    n=20j; m = int(np.imag(n))
    mu_omega_vals, mu_alpha_vals = np.mgrid[0:3*0.2930:n, 0:3*0.2930:n]
    mu_omega_alpha_grid = np.vstack([mu_omega_vals.ravel(), mu_alpha_vals.ravel()]).T

    # Run all time series. Extract minimum and maximum population.
    pop_max_min = pool.map(worker, mu_omega_alpha_grid)
    pop_max = [p[0] for p in pop_max_min]
    pop_min = [p[1] for p in pop_max_min]

    # Plot! 
    fig, axes = plt.subplots(1,2)
    im_min = axes[0].pcolormesh(mu_omega_vals,mu_alpha_vals,np.array(pop_min).reshape(m,m), shading='auto')
    im_max = axes[1].pcolormesh(mu_omega_vals,mu_alpha_vals,np.array(pop_max).reshape(m,m), shading='auto')
    fig.colorbar(im_min, ax=axes[0])
    fig.colorbar(im_max, ax=axes[1])

    axes[0].set_xlabel(r'$\mu_\omega$', fontsize=20); axes[1].set_xlabel(r'$\mu_\omega$', fontsize=20)
    axes[0].set_ylabel(r'$\mu_\alpha$', fontsize=20); axes[1].set_ylabel(r'$\mu_\alpha$', fontsize=20)
    axes[0].set_title('Minimum Herd Size', fontsize=15); axes[1].set_title('Maximum Herd Size', fontsize=15)

    plt.show()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    main(pool)
