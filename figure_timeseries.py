import numpy as np
from tri_seasonal_model import simulate
import matplotlib.pyplot as plt

# Specify the initial conditions. Group them into an array 'Z'. 
S0 = 800; E0 = 50; I0 = 120
TH0 = np.exp(7); TQ0 = 0

# Define model parameters. 
parm = {}
parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
parm['mu'] = 0.2930; parm['eta'] = 1.3352; parm['nu'] = 2.3774
parm['a'] = 10**4; parm['b'] = 10**3; parm['d'] = 1
parm['gamma'] = 1.0755; parm['r'] = 0.8572
parm['beta'] = 0.01; parm['r_T'] = 1.555
parm['K'] = 1500; parm['xi'] = 1000; parm['q'] = 0.5
parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0

# Run the model
num_years = 50
tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=100, thresh = 15)

# Plot the time series (including within year variation)
fig, axes = plt.subplots(1, 2)
axes[0].plot(tsol, Zsol[:,0], 'b-', label=r'$S$')
axes[0].plot(tsol, Zsol[:,1], color='fuchsia', linestyle='-', label=r'$E$')
axes[0].plot(tsol, Zsol[:,2], color='firebrick', linestyle= '-', label=r'$I$')
axes[1].plot(tsol, np.log(Zsol[:,4] + 1), color='forestgreen', linestyle='-', label=r'$T_Q$')
axes[1].plot(tsol, np.log(Zsol[:,3] + 1), color='brown', linestyle='-', label=r'$T_H$')

axes[0].legend(fontsize=15)
axes[1].legend(fontsize=15)

axes[0].set_xlabel('Time (Years)', fontsize=20)
axes[1].set_xlabel('Time (Years)', fontsize=20)

axes[0].set_ylabel('Moose Population', fontsize=20)
axes[1].set_ylabel('Log Tick Population', fontsize=20)
plt.show()

# Plot the time series (no within year variation - just winter population)
plt.plot(range(num_years), Zwinter, 'k-', label=r'$N$')
plt.legend(fontsize=15)
plt.xlabel('Time (Years)', fontsize=20)
plt.ylabel('Moose Population', fontsize=20)
plt.show()