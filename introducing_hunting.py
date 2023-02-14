import numpy as np
import analytical_climate_model as acm
import wildlife_management_model as wmm
import matplotlib.pyplot as plt
import time

model_type = 'wmm'

# Define model parameters. 
parm = {}
parm['omega'] = 0.5425; parm['alpha'] = 0.0833; parm['tau'] = 1 - parm['alpha'] - parm['omega']
parm['mu'] = 0.2835; parm['nu'] = 2.3772;  parm['gamma'] = 1.8523; parm['beta'] = 10**(-2)
parm['r_S'] = 0.8590; parm['r_P'] = 0.4998; parm['u'] = 0.75

parm['eta'] = 1.1886; parm['xi'] = 52000; parm['q'] = 0.75; parm['K'] = 1500; parm['r_T'] = 13.0293
parm['c'] = 10**(-2); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']

parm['mu_alpha'] = 0; parm['mu_omega'] = 0

# Specify the initial conditions. Group them into an array 'Z0'. 
S0 = 1000; E0 = 250; P0 = 250; H0 = 52000*250; Q0 = 0; Z0 = [S0, E0, P0, H0, Q0]

# Run the model numerically
tsol1, Zsol1, Zwinter_ODEs1 = wmm.simulate(100, init_cond=Z0, p=parm, granularity=2, thresh=10)

parm['mu_omega'] = 0.4

tsol2, Zsol2, Zwinter_ODEs2 = wmm.simulate(100, init_cond=Zsol1[-1], p=parm, granularity=2, thresh=10)

tsol = np.concatenate([np.array(tsol1), np.array(tsol2) + 100])
Zsol = np.vstack([Zsol1, Zsol2])

# Plot the time series (including within year variation)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

axes[0].plot(tsol, Zsol[:,0], 'b-', label=r'$S$')
axes[0].plot(tsol, Zsol[:,1], 'g-', label=r'$E$')
axes[0].plot(tsol, Zsol[:,2], color='firebrick', linestyle= '-', label=r'$P$')

axes[0].legend(fontsize=12, loc='upper right')
axes[0].set_xlabel('Time (Years)', fontsize=20)
axes[0].set_ylabel('Moose Population', fontsize=20)

axes[1].plot(tsol, Zsol[:,3], 'b-', label=r'$H$')
axes[1].plot(tsol, Zsol[:,4], 'g-', label=r'$Q$')

axes[1].legend(fontsize=12, loc='upper right')
axes[1].set_xlabel('Time (Years)', fontsize=20)
axes[1].set_ylabel('Tick Population', fontsize=20)

axes[0].axvline(100, linestyle=':', color='k')
axes[1].axvline(100, linestyle=':', color='k')

plt.subplots_adjust(wspace=0.5)
plt.savefig('introducing_hunting.pdf')
plt.show()