import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import mpltern

'''
This code plots the conditions 1 and 2.2 for extripation to be globally stable, at various levels of 
beta. Condition 2.1 is excluded  See the paper for the conditions. See this link for more info on 
tenary plots: https://mpltern.readthedocs.io/en/latest/index.html
'''

# Condition 1 (return T/F)
def cond_1(p):

    pt1 = p['u']*p['r_P'] < np.exp(p['T'] + p['tau'] + p['Omega']) - 1
    return pt1

# Condition 3 (return T/F)
def cond_3(p):
    pt1 = p['r_P'] < np.exp(p['Omega'] + p['mu']) - 1
    pt2 = p['alpha'] > 1/p['beta'] * np.log(1 - \
    np.exp(-p['T'] - p['mu'])*(1 + p['u']*p['r_P'] - np.exp(p['T'] + p['mu'] + p['Omega']))*(1 + p['r_S'] - np.exp(p['mu'])) \
    / (np.exp(p['mu'] + p['Omega']) - p['r_P'] - 1))

    return 1*(pt1 * pt2)

# Define parameter values
p = {}
p['omega'] = 0.5425; p['alpha'] = 0.0833; p['tau'] = 1 - p['alpha'] - p['omega']
p['mu'] = 0.2835; p['nu'] = 2.3772;  p['gamma'] = 1.8523; p['beta'] = 2.5595
p['r_S'] = 0.8287; p['r_P'] = 0.4998; p['u'] = 0.75

# Make the figure 
fig = plt.figure(figsize=(14.8, 4.8))

# Iterate over 3 beta values
for k, p['beta'] in enumerate([0.8*2.5595, 2.5595, 1.2*2.5595]):
    alphas = []; omegas = []; taus = [];
    cond_1s = []; cond_3s = [];

    # Consider alpha values up to 1 and omega values up to 1
    for p['alpha'] in np.linspace(0, 1, 100, endpoint=False):
        for p['omega'] in np.linspace(0, 1 - p['alpha'], 100):

            # Tau is defined based on alpha and omega
            p['tau'] = 1 - p['alpha'] - p['omega']

            # Define the other parameter groupings
            p['T'] = p['tau']*p['gamma'];
            p['Omega'] = p['nu']*p['omega']
            p['A'] = p['alpha']*p['beta']

            cond_1s.append(cond_1(p))
            cond_3s.append(cond_3(p))
            alphas.append(p['alpha']); omegas.append(p['omega']); taus.append(p['tau'])


    # Now add the plot and beautify! 
    vmin = 0.0
    vmax = 1
    ax = fig.add_subplot(1, 3, k + 1, projection='ternary')

    cond_1_cmap = matplotlib.colors.ListedColormap(['white', 'blue'])
    cs = ax.tripcolor(
        alphas, omegas, taus, cond_1s, shading='flat', alpha = 0.5, vmin=vmin, vmax=vmax, rasterized=True, cmap=cond_1_cmap, linewidth=0.0, antialiased=True, edgecolors='face')

    cond_3_cmap = matplotlib.colors.ListedColormap(['white', 'red'])
    cs = ax.tripcolor(
        alphas, omegas, taus, cond_3s, shading='flat', alpha = 0.5, vmin=vmin, vmax=vmax, rasterized=True, cmap=cond_3_cmap, linewidth=0.0, antialiased=True, edgecolors='face')
    
    ax.scatter(0.0833, 0.5425, 0.3742, color='orange', marker='o', s=60)

    ax.set_tlabel(r'Autumn, $\alpha$', fontsize=15, rotation=45)
    ax.set_llabel(r'Winter, $\omega$', fontsize=15)
    ax.set_rlabel(r'Summer ($\tau$)', fontsize=15)

    ax.taxis.set_label_rotation_mode('axis')
    ax.laxis.set_label_rotation_mode('axis')
    ax.raxis.set_label_rotation_mode('axis')

    ax.set_title(r'$\beta = $' + str(round(p['beta'], 4)), y=0, pad=-35, verticalalignment="top", fontsize=15)

# Save the plot!
plt.subplots_adjust(wspace=0.8)
fig.savefig('tri_plot.pdf', bbox_inches='tight')
plt.show()