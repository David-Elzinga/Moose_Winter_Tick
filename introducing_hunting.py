import numpy as np
import analytical_climate_model as acm
import wildlife_management_model as wmm
import matplotlib.pyplot as plt
import time

'''
This code generates the figure in the manuscript of introducing winter hunting to lower
population volitality. The system runs for 100 years with no harvesting, and harvesting is then introduced
for the next 100 years. 
'''

model_type = 'wmm'

# Define model parameters. 
parm = {}
parm['omega'] = 0.5425; parm['alpha'] = 0.0833; parm['tau'] = 1 - parm['alpha'] - parm['omega']
parm['mu'] = 0.2835; parm['nu'] = 2.3772;  parm['gamma'] = 1.8523; parm['beta'] = 10**(-2.1)
parm['r_S'] = 0.8590; parm['r_P'] = 0.4998; parm['u'] = 0.75

parm['eta'] = 1.1886; parm['xi'] = 52000; parm['q'] = 0.75; parm['K'] = 1500; parm['r_T'] = 13.0293
parm['c'] = 10**(-2); parm['beta_T'] = parm['beta']/1; parm['beta_M'] = parm['beta']/parm['xi']
parm['epsilon'] = 0.01 

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
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

axes[0].plot(tsol, Zsol[:,0], 'b-', label=r'$S$')
axes[0].plot(tsol, Zsol[:,2], color='firebrick', linestyle= '-', label=r'$P$')
axes[0].plot(tsol, Zsol[:,1], 'g-', label=r'$E$')

axes[0].legend(fontsize=12, loc='upper right')
axes[0].set_ylabel('Moose Population', fontsize=18)

import pdb; pdb.set_trace()
axes[1].plot(tsol[tsol < 112], np.log(Zsol[:,4][tsol < 112]), 'g-', label=r'$Q$')
axes[1].plot(tsol[tsol < 112], np.log(Zsol[:,3][tsol < 112]), 'b-', label=r'$H$')

axes[1].legend(fontsize=12, loc='upper right')
axes[1].set_xlabel('Time (Years)', fontsize=18, labelpad=10)
axes[1].set_ylabel('Log Tick Population', fontsize=18)

axes[0].axvline(100, linestyle=':', color='k')
axes[1].axvline(100, linestyle=':', color='k')

plt.subplots_adjust(hspace=0.1)
plt.savefig('introducing_hunting.pdf')
plt.show()