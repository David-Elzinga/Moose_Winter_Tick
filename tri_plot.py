import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import mpltern

# https://mpltern.readthedocs.io/en/latest/index.html

def cond_1(p):

    pt1 = p['u']*p['r_P'] < np.exp(p['T'] + p['tau'] + p['Omega']) - 1
    return pt1

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
p['r_S'] = 0.8590; p['r_P'] = 0.4998; p['u'] = 0.75

fig = plt.figure(figsize=(14.8, 4.8))

for k, p['beta'] in enumerate([0.8*2.5595, 2.5595, 1.2*2.5595]):
    alphas = []; omegas = []; taus = [];
    cond_1s = []; cond_3s = [];
    for p['alpha'] in np.linspace(0, 1, 100, endpoint=False):
        for p['omega'] in np.linspace(0, 1 - p['alpha'], 100):
            p['tau'] = 1 - p['alpha'] - p['omega']
            p['T'] = p['tau']*p['gamma'];
            p['Omega'] = p['nu']*p['omega']
            p['A'] = p['alpha']*p['beta']

            cond_1s.append(cond_1(p))
            cond_3s.append(cond_3(p))
            alphas.append(p['alpha']); omegas.append(p['omega']); taus.append(p['tau'])

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

    ax.set_tlabel(r'$\alpha$', fontsize=15)
    ax.set_llabel(r'$\omega$', fontsize=15)
    ax.set_rlabel(r'$\tau$', fontsize=15)

    ax.taxis.set_label_rotation_mode('horizontal')
    ax.laxis.set_label_rotation_mode('horizontal')
    ax.raxis.set_label_rotation_mode('horizontal')

    ax.set_title(r'$\beta = $' + str(round(p['beta'], 4)), y=0, pad=-35, verticalalignment="top", fontsize=15)

plt.subplots_adjust(wspace=0.6)

fig.savefig('tri_plot.pdf', bbox_inches='tight')
plt.show()