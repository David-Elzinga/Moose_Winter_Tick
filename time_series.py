import numpy as np
import analytical_climate_model as acm
import wildlife_management_model as wmm
import matplotlib.pyplot as plt
import time

'''
This code produces a time series plot for a given set of parameter values. The user 
should also choose the model type as either acm (Analytical Climate Model) or wmm 
(Wildlife Management Model)
'''

model_type = 'wmm'

# Define model parameters. 
parm = {}
parm['omega'] = 0.5425; parm['alpha'] = 0.0833; parm['tau'] = 1 - parm['alpha'] - parm['omega']
parm['mu'] = 0.2835; parm['nu'] = 2.3772;  parm['gamma'] = 1.8523; parm['beta'] = 10**(-3)
parm['r_S'] = 0.8590; parm['r_P'] = 0.4998; parm['u'] = 0.75

parm['eta'] = 1.1886; parm['xi'] = 52000; parm['q'] = 0.75; parm['K'] = 1500; parm['r_T'] = 13.0293
parm['c'] = 10**(-2); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']

parm['mu_alpha'] = 0; parm['mu_omega'] = 0

# Run the model! Specify the number of years
num_years = 200
if model_type == 'acm':

    # Specify the initial conditions. Group them into an array 'Z0'. 
    S0 = 1500; P0 = 500; Z0 = [S0, P0]

    # Run the model numerically
    tsol, Zsol, Zwinter_ODEs = acm.simulate(num_years, init_cond=Z0, parm=parm, granularity=2, thresh = 10)

    # Get the exact solution
    Zwinter_exact = acm.annual_map(num_years, init_cond=Z0, parm=parm, thresh=10)

    # Plot the time series (including within year variation)
    fig, axes = plt.subplots(1, 2, figsize=(5, 10), sharex=True)

    axes[0].plot(tsol, Zsol[:,0], 'b-', label=r'$S$')
    axes[0].plot(tsol, Zsol[:,1], color='firebrick', linestyle= '-', label=r'$P$')

    axes[0].legend(fontsize=15, loc='upper right')
    axes[0].set_xlabel('Time (Years)', fontsize=20)
    axes[0].set_ylabel('Moose Population', fontsize=20)

    axes[1].plot(tsol, Zsol[:,0] + Zsol[:,1], 'k-', label='Total Population (ODEs)')
    axes[1].plot(range(num_years + 1), Zwinter_exact, 'rs', label='Total Population (Map)')

    axes[1].legend(fontsize=15, loc='upper right')
    axes[1].set_xlabel('Time (Years)', fontsize=20)
    axes[1].set_ylabel('Moose Population', fontsize=20)

    plt.subplots_adjust(wspace=0.5)
    plt.show()

else:
    
    # Specify the initial conditions. Group them into an array 'Z0'. 
    S0 = 1000; E0 = 250; P0 = 250; H0 = 52000*250; Q0 = 0; Z0 = [S0, E0, P0, H0, Q0]

    # Run the model numerically
    tsol, Zsol, Zwinter_ODEs = wmm.simulate(num_years, init_cond=Z0, p=parm, granularity=2, thresh=10)

    # Plot the time series (including within year variation)
    fig, axes = plt.subplots(1, 2, figsize=(5, 10), sharex=True)

    axes[0].plot(tsol, Zsol[:,0], 'b-', label=r'$S$')
    axes[0].plot(tsol, Zsol[:,1], 'g-', label=r'$E$')
    axes[0].plot(tsol, Zsol[:,2], color='firebrick', linestyle= '-', label=r'$P$')

    axes[0].legend(fontsize=15, loc='upper right')
    axes[0].set_xlabel('Time (Years)', fontsize=20)
    axes[0].set_ylabel('Moose Population', fontsize=20)

    axes[1].plot(tsol, Zsol[:,3], 'b-', label=r'$H$')
    axes[1].plot(tsol, Zsol[:,4], 'g-', label=r'$Q$')

    axes[1].legend(fontsize=15, loc='upper right')
    axes[1].set_xlabel('Time (Years)', fontsize=20)
    axes[1].set_ylabel('Tick Population', fontsize=20)

    plt.subplots_adjust(wspace=0.5)
    plt.show()
