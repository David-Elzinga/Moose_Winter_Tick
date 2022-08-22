import numpy as np
from tri_seasonal_model import simulate
import matplotlib.pyplot as plt

# Specify the initial conditions. Group them into an array 'Z'. 
S0 = 800; E0 = 50; I0 = 120; TH0 = 10**4; TQ0 = 0

# Define model parameters. 
parm = {}
parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
parm['mu_omega'] = 0.5; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
parm['K'] = 1500 
parm['mu'] = 0.293; parm['eta'] = 1.3352; parm['nu'] = 2.3774/23337.7315
parm['d'] = 10**(-5)
parm['gamma'] = 1.0755; parm['r'] = 0.8572
parm['xi'] = 23337.7315

parm['beta'] = 10**(-2.6)
parm['r_T'] = 13.0293
parm['q'] = 0.9

# Run the model
num_years = 100
tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=3, thresh = 10)

if len(Zwinter) < num_years: # In this case there was an extinction, so variance we assume is zero.
    std = 0
else: # Not an extinction, record the variance over the final 100 years.
    std = np.std(Zwinter[-100:])

print(std)

# Plot the time series (including within year variation)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(tsol, Zsol[:,0], 'b-', label=r'$S$')
axes[0].plot(tsol, Zsol[:,1], color='fuchsia', linestyle='-', label=r'$E$')
axes[0].plot(tsol, Zsol[:,2], color='firebrick', linestyle= '-', label=r'$I$')
axes[1].plot(tsol, np.log(Zsol[:,4] + 1), color='forestgreen', linestyle='-', label=r'$T_Q$')
axes[1].plot(tsol, np.log(Zsol[:,3] + 1), color='brown', linestyle='-', label=r'$T_H$')

#axes[0].legend(fontsize=15)
#axes[1].legend(fontsize=15)

axes[0].set_xlabel('Time (Years)', fontsize=20)
axes[1].set_xlabel('Time (Years)', fontsize=20)

axes[0].set_ylabel('Moose Population', fontsize=20)
axes[1].set_ylabel('Log Tick Population', fontsize=20)
plt.subplots_adjust(wspace=0.5)
plt.savefig('time_series.png', bbox_inches='tight')
#plt.show()

# Plot the time series (no within year variation - just winter population)
plt.plot(range(num_years), Zwinter, 'k-', label=r'$N$')
#plt.legend(fontsize=15)
plt.xlabel('Time (Years)', fontsize=20)
plt.ylabel('Moose Population', fontsize=20)
#plt.show()