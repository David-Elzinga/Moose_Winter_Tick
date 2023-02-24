import numpy as np
import matplotlib.pyplot as plt

'''
This code creates the bifurcation plot in the manuscript. It varies alpha and r_P 
to determine which of the extirpation global stability conditions pass/fail. 
'''

def cond_3_alpha(p):
    return 1/p['beta'] * np.log(1 - \
    np.exp(-p['T'] - p['mu'])*(1 + p['u']*p['r_P'] - np.exp(p['T'] + p['mu'] + p['Omega']))*(1 + p['r_S'] - np.exp(p['mu'])) \
    / (np.exp(p['mu'] + p['Omega']) - p['r_P'] - 1))

# Define parameter values
p = {}
p['omega'] = 0.5425; p['alpha'] = 0.0833; p['tau'] = 1 - p['alpha'] - p['omega']
p['mu'] = 0.2835; p['nu'] = 2.3772;  p['gamma'] = 1.8523; p['beta'] = 2.5595
p['r_S'] = 0.8590; p['r_P'] = 0.4998; p['u'] = 0.75

# Define parameter groupings
p['T'] = p['tau']*p['gamma'];
p['Omega'] = p['nu']*p['omega']

# Check the conditions for the extirpation eq. to be stable. 

# Check condition 1
if 1 + p['u']*p['r_P'] - np.exp(p['T'] + p['mu'] + p['Omega']) < 0:
    print('Condition 1 for Extirpation Passes')
else:
    print('Condition 1 for Extirpation Fails')

# Check condition 2
if 1 + p['r_S'] - np.exp(p['mu']) < 0:
    print('Condition 2 for Extirpation Passes')
else:
    print('Condition 2 for Extirpation Fails')

# Check condition 3
if 1 + p['r_P'] - np.exp(p['Omega'] + p['mu']) < 0:
    print('Condition 3 for Extirpation Part 1 Passes')
else:
    print('Condition 3 for Extirpation Part 1 Passes')

# Determine the critical value of rP from condition 1. If rP is greater than this value,
# the eq loses stability
cond_1_rP = (np.exp(p['T'] + p['mu'] + p['Omega']) - 1)/p['u']

# Determine the critical value of rP from condition 3. If rP is greater than this value,
# the eq loses stability
cond_3_rP = np.exp(p['Omega'] + p['mu']) - 1

# Calculate the largest value of rP such that both conditions pass (e.g. the min of the two)
rP_max = min(cond_1_rP, cond_3_rP)

# Determine the critical value of alpha. If alpha is less than this value, 
# the eq loses stability
rP_range_plot = np.linspace(0, rP_max, 10000)
cond_3_alpha = np.array([cond_3_alpha(p) for p['r_P'] in rP_range_plot])

fig, ax = plt.subplots(1,1)

ax.plot(rP_range_plot[cond_3_alpha < 1], cond_3_alpha[cond_3_alpha < 1], color='purple', linestyle='--', linewidth=2.5)
ax.fill_between(x=rP_range_plot[cond_3_alpha < 1], y1=cond_3_alpha[cond_3_alpha < 1], y2 = 1, color='purple', alpha=0.5)
ax.plot(0.4, 0.0833, color='orange', marker='.', markersize=20)

ax.hlines(y = 1, xmin = 0, xmax = 1000, linestyle='-', color='k')
ax.vlines(x = rP_max, ymin = 0, ymax = 1, linestyle='--', color='k')

ax.set_xlim(0, rP_max  + 0.5)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r'$r_P$', fontsize=15)
ax.set_ylabel(r'$\alpha$', rotation=0, fontsize=15, labelpad=20)

ax.text(x=1.25, y=0.55, s="Extirpation", size=15)
ax.text(x=2.5, y=0.05, s="Persistence", size=15)
ax.text(x=4.05, y=0.35, s="Persistence", rotation=90, size=15)

fig.savefig('bif_plot.pdf')
plt.show()