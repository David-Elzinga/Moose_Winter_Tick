import numpy as np
import analytical_climate_model as acm
import wildlife_management_model as wmm
import matplotlib.pyplot as plt
import time

model_type = 'wmm'

# Define model parameters. 
parm = {}
parm['omega'] = 0.5425; parm['alpha'] = 0.0833; parm['tau'] = 1 - parm['alpha'] - parm['omega']
parm['mu'] = 0.2835; parm['nu'] = 2.3772;  parm['gamma'] = 1.8523; parm['beta'] = 10**(0.5)
parm['r_S'] = 0.8590; parm['r_P'] = 0.4998; parm['u'] = 0.75

parm['eta'] = 1.1886; parm['xi'] = 52000; parm['q'] = 0.9; parm['K'] = 1500; parm['r_T'] = 8#13.0293
parm['c'] = 10**(-2); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']
parm['mu_alpha'] = 0; parm['mu_omega'] = 0

parm = {'mu_alpha': 0, 'mu_omega': 0, 'omega': 0.520665625, 'alpha': 0.048621875, 'tau': 0.43071249999999994, 'mu': 0.26825, 'nu': 2.9766515625, 'gamma': 2.0028156249999998, 'beta': 89.76871324473142, 'r_S': 0.9395312499999999, 'r_P': 0.471675, 'u': 0.28515625, 'eta': 1.121746875, 'xi': 58650.0, 'q': 0.890625, 'K': 1509.375, 'r_T': 4.7725953125, 'c': 2053.525026457146, 'beta_T': 89.76871324473142, 'beta_M': 0.001530583346031226, 'n': 200, 'H+/P+': 65107.43399583003, 'H+/S+': 13834.432872996773, 'H+/E+': 334904.52900515794, 'b': 0.9603301197018808, 'P-/H-': 0.0}
#print((1/parm['beta']) * (np.log(1 + parm['r_S'] - ((parm['r_P'] + 1 - np.exp(-parm['gamma']*parm['tau'])*(1 + parm['u']*parm['r_P']))*(np.exp(parm['mu']) - parm['r_S'] - 1))/(np.exp(parm['mu'] + parm['nu']*parm['omega']) - parm['r_P'] - 1)) - parm['mu']))

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
    t0 = time.time()
    tsol, Zsol, Zwinter_ODEs = wmm.simulate(num_years, init_cond=Z0, p=parm, granularity=2, thresh=10)
    t1 = time.time()
    print(t1- t0)

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
    axes[1].set_ylabel('Moose Population', fontsize=20)

    plt.subplots_adjust(wspace=0.5)
    plt.show()



# Update S
#((1 + parm['r_P'])*np.exp(parm['gamma']*parm['tau']) - parm['u']*parm['r_P'] - 1)*10*np.exp(-parm['alpha']*parm['beta'] - parm['gamma']*parm['tau'] - parm['nu']*parm['omega'] - parm['mu']) + (1 + parm['r_S'])*500*np.exp(-parm['alpha']*parm['beta'] - parm['mu'])

# Update P
#((np.exp(parm['alpha']*parm['beta']) - 1)*np.exp(parm['gamma']*parm['tau']) + parm['r_P']*((np.exp(parm['alpha']*parm['beta']) - 1)*np.exp(parm['gamma']*parm['tau']) + parm['u']) + 1)*10*np.exp(-parm['alpha']*parm['beta'] - parm['gamma']*parm['tau'] - parm['nu']*parm['omega'] - parm['mu']) + (np.exp(parm['alpha']*parm['beta']) - 1)*(1 + parm['r_S'])*500*np.exp(-parm['alpha']*parm['beta'] - parm['mu'])