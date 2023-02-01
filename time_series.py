import numpy as np
import analytical_climate_model as acm
import matplotlib.pyplot as plt

# Things to do: add rannge in plotting to be minimum of number of years specified and number of years ran before thresh extn

# Specify the initial conditions. Group them into an array 'Z0'. 
S0 = 100; P0 = 500; Z0 = [S0, P0]

# Define model parameters. 
parm = {}
parm['omega'] = 0.25; parm['alpha'] = 0.6; parm['tau'] = 1 - parm['alpha'] - parm['omega']
parm['mu'] = 0.293; parm['nu'] = 3;  parm['gamma'] = 0.755; parm['beta'] = 10**(0.5)
parm['r_S'] = 1.8572; parm['r_P'] = 1.7572; parm['u'] = 1

#(1 - np.exp(parm['mu']) + np.exp(parm['mu'] + parm['gamma']*parm['tau'])*(1 + parm['r_P'] - np.exp(parm['alpha']*parm['beta']) - np.exp(parm['nu']*parm['omega']) + np.exp(parm['alpha']*parm['beta'] + parm['nu']*parm['omega'] + parm['mu']) - parm['r_P']*np.exp(parm['alpha']*parm['beta'])) + parm['u']*parm['r_P']*(1-np.exp(parm['mu']))) / (np.exp(parm['gamma']*parm['tau'] + parm['nu']*parm['omega'] + parm['mu']) - parm['u']*parm['r_P'] - 1)

print((1/parm['beta']) * (np.log(1 + parm['r_S'] - ((parm['r_P'] + 1 - np.exp(-parm['gamma']*parm['tau'])*(1 + parm['u']*parm['r_P']))*(np.exp(parm['mu']) - parm['r_S'] - 1))/(np.exp(parm['mu'] + parm['nu']*parm['omega']) - parm['r_P'] - 1)) - parm['mu']))

# Run the model numerically
num_years = 30
tsol, Zsol, Zwinter_ODEs = acm.simulate(num_years, init_cond=Z0, parm=parm, granularity=10, thresh = 10)

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

# Update S
#((1 + parm['r_P'])*np.exp(parm['gamma']*parm['tau']) - parm['u']*parm['r_P'] - 1)*10*np.exp(-parm['alpha']*parm['beta'] - parm['gamma']*parm['tau'] - parm['nu']*parm['omega'] - parm['mu']) + (1 + parm['r_S'])*500*np.exp(-parm['alpha']*parm['beta'] - parm['mu'])

# Update P
#((np.exp(parm['alpha']*parm['beta']) - 1)*np.exp(parm['gamma']*parm['tau']) + parm['r_P']*((np.exp(parm['alpha']*parm['beta']) - 1)*np.exp(parm['gamma']*parm['tau']) + parm['u']) + 1)*10*np.exp(-parm['alpha']*parm['beta'] - parm['gamma']*parm['tau'] - parm['nu']*parm['omega'] - parm['mu']) + (np.exp(parm['alpha']*parm['beta']) - 1)*(1 + parm['r_S'])*500*np.exp(-parm['alpha']*parm['beta'] - parm['mu'])