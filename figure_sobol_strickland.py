import sys, os, time, warnings
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import pickle
import argparse
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from tri_seasonal_model import simulate

default_N = os.cpu_count()
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000,
                    help="obtain N*(2D+2) samples from parameter space")
parser.add_argument("-n", "--ncores", type=int,
                    help="number of cores, defaults to {} on this machine".format(default_N))
parser.add_argument("-o", "--filename", type=str, 
                    help="filename to write output to, no extension",
                    default='analysis')

def run_model(mu, eta, nu, loga, logb, gamma, logd, logbeta, r, r_T, xi):

    # Define model parameters. 
    parm = {}
    parm['alpha'] = 0.2532; parm['omega'] = 0.2819; parm['tau'] = 0.4649
    parm['mu'] = mu; parm['eta'] = eta; parm['nu'] = nu
    parm['a'] = 10**loga; parm['b'] = 10**logb; parm['d'] = 10**logd
    parm['gamma'] = gamma; parm['r'] = r
    parm['beta'] = 10**logbeta; parm['r_T'] = r_T
    parm['K'] = 1500; parm['xi'] = xi
    parm['mu_omega'] = 0; parm['mu_tau'] = 0; parm['mu_alpha'] = 0
    parm['psi'] = 0

    num_years = 100
    S0 = 800; E0 = 50; I0 = 120
    TH0 = np.exp(14.5); TQ0 = 0
    tsol, Zsol, Zwinter = simulate(num_years, init_cond=[S0, E0, I0, TH0, TQ0], parm=parm, granularity=100, thresh=15)
    if len(Zwinter) < num_years:
        var = 0
    else:
        var = np.var(Zwinter[-50:])
    return var


def main(N, filename, ncores=None, pool=None):

    ### Define the parameter space within the context of a problem dictionary ###
    problem = {
        # number of parameters
        'num_vars' : 11,
        # parameter names
        'names' : ['mu', 'eta', 'nu', 'loga',
                   'logb', 'gamma', 'logd', 'logbeta',
                   'r', 'r_T', 'xi'], 
        # bounds for each corresponding parameter
        'bounds' : [[0.2402, 0.3458], [0.2402,2.9918], [2.0229, 2.9918], [-2,2],
                    [-2,2], [0,2.150], [-2,2], [-4,-1],
                    [0.8359, 0.8785], [14.8796, 85.9099], [1250, 95495]]
    }

    ### Create an N*(2D+2) x num_var matrix of parameter values ###
    param_values = saltelli.sample(problem, N, calc_second_order=True)

    print(len(param_values))

    ### Run model ###
    print('Examining the parameter space.')
    if pool is not None:
        if ncores is None:
            poolsize = os.cpu_count()
        else:
            poolsize = ncores
        chunksize = param_values.shape[0]//poolsize
        output = pool.starmap(run_model, param_values, chunksize=chunksize)
    else:
        # This is a bad idea. Say so.
        print('Warning!!! Running in serial only! Multiprocessing pool not utilized.')
        output = []
        for k, params in enumerate(param_values):
            print(k/len(param_values))
            output.append(run_model(*params))

    ### Parse and save the output ###
    print('Saving and checking the results for errors...')
    # write data to temporary location in case anything obnoxious happens
    with open("raw_result_data.pickle", "wb") as f:
        result = {'output':output, 'param_values':param_values}
        pickle.dump(result, f)
    # Look for errors
    error_num = 0
    error_places = []
    for n, result in enumerate(output):
        if isinstance(result, Exception):
            error_num += 1
            error_places.append(n)
    if error_num > 0:
        print("Errors discovered in output.")
        print("Parameter locations: {}".format(error_places))
        print("Pickling errors...")
        err_output = []
        err_params = []
        for idx in error_places:
            err_output.append(output.pop(idx)) # pull out err output
            err_params.append(param_values[idx,:]) # reference err param values
        with open("err_results.pickle", "wb") as f:
            err_result = {'err_output':err_output, 'err_param_values':param_values}
            pickle.dump(err_result, f)
        print("Saving all other data in HDF5...")
        output = np.array(output)
        # first remove err param values
        param_values = param_values[[ii for ii in range(len(param_values)) if ii not in error_places],:]
        # convert to dataframe
        param_values = pd.DataFrame(param_values, columns=problem['names'])
        assert output.shape[0] == param_values.shape[0]
        store = pd.HDFStore('nonerr_results.h5')
        store['param_values'] = param_values
        store['raw_output'] = pd.DataFrame(output, 
                            columns=['var'])
        store.close()
        os.remove('raw_result_data.pickle')
        print("Please review output dump.")
        return
    
    # Save results in HDF5 as dataframe
    print('Parsing the results...')
    output = np.array(output)
    param_values = pd.DataFrame(param_values, columns=problem['names'])
    # Resave as dataframe in hdf5
    store = pd.HDFStore(filename+'.h5')
    store['param_values'] = param_values
    store['raw_output'] = pd.DataFrame(output, 
                          columns=['var'])
    os.remove('raw_result_data.pickle')

    import pdb; pdb.set_trace()
    ### Analyze the results and view using Pandas ###
    # Conduct the sobol analysis and pop out the S2 results to a dict
    S2 = {}
    var_sens = sobol.analyze(problem, output, calc_second_order=True)
    S2['var'] = pd.DataFrame(var_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['var_conf'] = pd.DataFrame(var_sens.pop('S2_conf'), index=problem['names'],
                           columns=problem['names'])
                           
    # Convert the rest to a pandas dataframe
    var_sens = pd.DataFrame(var_sens,index=problem['names'])

    ### Save the analysis ###
    print('Saving...')
    store['var_sens'] = var_sens
    for key in S2.keys():
        store['S2/'+key] = S2[key]
    # Save the bounds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store['bounds'] = pd.Series(problem['bounds'])
    store.close()
    
    # Plot and save
    import pdb; pdb.set_trace()



def load_data(filename):
    '''Load analysis data from previous run and return for examination'''

    return pd.HDFStore(filename)



def print_max_conf(store):
    '''Print off the max confidence interval for each variable in the store,
    for both first-order and total-order indices'''
    for var in ['cv_sens', 'peak_sens', 'width_sens', 'skew_sens', 'RmRp_sens']:
        print('----------- '+var+' -----------')
        print('S1_conf_max: {}'.format(store[var]['S1_conf'].max()))
        print('ST_conf_max: {}'.format(store[var]['ST_conf'].max()))
        print(' ')



def find_feasible_params(store):
    '''This will return a dataframe with parameters and corresponding 
    observable values that fall within the desired ranges (hard coded)'''
    data = pd.concat([store['param_values'], store['raw_output']], axis=1)
    data1 = data[data['skew']>=1]
    data1 = data1[data1['skew']<=2]
    if data1.shape[0] == 0:
        print('No parameters match skew requirement alone.')
        return data
    data2 = data1[data1['peak']<=4280]
    data2 = data2[data2['peak']>=950]
    if data2.shape[0] == 0:
        print('No parameters match both skew and peak requirements.')
        return data1
    data3 = data2[data2['c/v']*data2['v']<=0.00852]
    data3 = data3[data3['c/v']*data3['v']>=0.000508]
    if data3.shape[0] == 0:
        print('No parameters match skew, peak, and vel requirements.')
        return data2

    return data3


def plot_S1_ST_tbl_from_store(store, show=True, ext='pdf'):
    cv_sens = store['cv_sens']
    peak_sens = store['peak_sens']
    width_sens = store['width_sens']
    skew_sens = store['skew_sens']
    RmRp_sens = store['RmRp_sens']
    bounds = list(store['bounds'])
    
    plot_S1_ST_tbl(peak_sens, width_sens, skew_sens, cv_sens, RmRp_sens,
                   bounds, show, ext)



def plot_S1_ST_tbl(peak_sens, width_sens, skew_sens, cv_sens, RmRp_sens,
                   bounds=None, show=True, ext='pdf', startclr=None):

    # Gather the S1 and ST results
    all_names = ['max density', 'std width', 'skewness', 'c/v', r'$R^-/R^+$']
    all_results = [peak_sens, width_sens, skew_sens, cv_sens, RmRp_sens]
    names = []
    results = []
    for n, result in enumerate(all_results):
        # Keep only the ones actually passed
        if result is not None:
            results.append(result)
            names.append(all_names[n])
    S1 = pd.concat([result['S1'] for result in results[::-1]], keys=names[::-1], axis=1) #produces copy
    ST = pd.concat([result['ST'] for result in results[::-1]], keys=names[::-1], axis=1)

    # Gather the S1 and ST results without skew
    # S1 = pd.concat([cv_sens['S1'], peak_sens['S1'], width_sens['S1'], 
    #                RmRp_sens['S1']], 
    #                keys=['$c$', 'peak', 'std', '$R^-$'], axis=1) #produces copy
    # ST = pd.concat([cv_sens['ST'], peak_sens['ST'], width_sens['ST'], 
    #                RmRp_sens['ST']], 
    #                keys=['$c$', 'peak', 'std', '$R^-$'], axis=1)
    ##### Reorder (manual) #####
    order = ['N', 'Rplus', 'v', 'beta', 'eta/alpha', 'theta/beta' ,'logDelta', 
             'gamma', 'delta', 'loglam']
    bndorder = [0, 1, 2, -3, 5, 6, 4, -2, -1, 3]
    S1 = S1.reindex(order)
    ST = ST.reindex(order)
    if bounds is not None:
        new_bounds = [bounds[ii] for ii in bndorder]
        bounds = new_bounds

    ###### Change to greek, LaTeX #####
    for id in S1.index:
        # handle greek ratios
        if '/' in id:
            togreek = "$\\" + id[:id.find('/')+1] + "\\" + id[id.find('/')+1:] + r"$"
            S1.rename(index={id: togreek}, inplace=True)
        elif id == 'loglam':
            S1.rename(index={id: r'$\log(\lambda)$'}, inplace=True)
        elif id == 'logDelta':
            S1.rename(index={id: r'$\log(\Delta)$'}, inplace=True)
        elif id == 'Rplus':
            S1.rename(index={id: r'$R^{+}$'}, inplace=True)
        # all others
        elif id not in ['N', 'v']:
            S1.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
        
    for id in ST.index:
        if '/' in id:
            togreek = "$\\" + id[:id.find('/')+1] + "\\" + id[id.find('/')+1:] + r"$"
            ST.rename(index={id: togreek}, inplace=True)
        elif id == 'loglam':
            ST.rename(index={id: r'$\log(\lambda)$'}, inplace=True)
        elif id == 'logDelta':
            ST.rename(index={id: r'$\log(\Delta)$'}, inplace=True)
        elif id == 'Rplus':
            ST.rename(index={id: r'$R^{+}$'}, inplace=True)
        elif id not in ['N', 'v']:
            ST.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
    
    ###### Plot ######
    if bounds is not None:
        # setup for table
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,.35], wspace=.15, left=0.04,
                               right=0.975, bottom=0.15, top=0.915)
        axes = []
        for ii in range(2):
            axes.append(plt.subplot(gs[ii]))
    else:
        # setup without table
        fig, axes = plt.subplots(ncols=2, figsize=(13, 6))
    # Switch the last two colors so A is red
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    # colors = colors[:4]
    # clr = colors[-1]; colors[-1] = colors[-2]; colors[-2] = clr
    
    if startclr is not None:
        # Start at a different color
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = colors[startclr:]
        s1bars = S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8, color=colors)
        s2bars = ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, color=colors, legend=False)
    else:
        s1bars = S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8)
        s2bars = ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, legend=False)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=18, rotation=-40) #-25
        ax.tick_params(axis='y', labelsize=14)
        #ax.get_yaxis().set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].set_title('First-order indices', fontsize=26)
    axes[1].set_title('Total-order indices', fontsize=26)
    handles, labels = s1bars.get_legend_handles_labels()
    s1bars.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=16)
    # Create table
    if bounds is not None:
        columns = ('Value Range',)
        rows = list(S1.index)
        # turn bounds into strings of ranges
        cell_text = []
        for bnd in bounds:
            low = str(bnd[0])
            high = str(bnd[1])
            # concatenate, remove leading zeros
            if low != "0" and low != "0.0":
                low = low.lstrip("0")
            if high != "0" and high != "0.0":
                high = high.lstrip("0")
            # raise any minus signs
            if '-' in low:
                low = "\u00AF"+low[1:]
            if '-' in high:
                high = "\u00AF"+high[1:]
            cell_text.append([low+"-"+high])
        tbl_ax = plt.subplot(gs[2])
        the_table = tbl_ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                    loc='center')
        the_table.set_fontsize(18)
        the_table.scale(1,2.3)
        the_table.auto_set_column_width(0)
        tbl_ax.axis('off')
    #plt.tight_layout()
    # reposition table
    pos = tbl_ax.get_position()
    newpos = [pos.x0 + 0.02, pos.y0, pos.width, pos.height]
    tbl_ax.set_position(newpos)
    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.{}".format(time.strftime("%m_%d_%H%M"), ext))
    return (fig, axes)



def plot_shape_rel_from_store(store, show=True, ext='pdf'):
    cv_sens = store['cv_sens']
    peak_sens = store['peak_sens']
    width_sens = store['width_sens']
    skew_sens = store['skew_sens']
    RmRp_sens = store['RmRp_sens']
    bounds = list(store['bounds'])
    
    plot_shape_rel(peak_sens, width_sens, skew_sens, cv_sens, RmRp_sens,
                   bounds, show, ext)



def plot_shape_rel(peak_sens, width_sens, skew_sens, cv_sens, RmRp_sens,
                   bounds=None, show=True, ext='pdf'):

    # Gather the shape and rel results into two groups
    shape_results = [peak_sens, width_sens, skew_sens]
    shape_names = ['max density', 'std width', 'skewness']
    rel_names = ['c/v', r'$R^-/R^+$']
    rel_results = [cv_sens, RmRp_sens]

    S1_shape = pd.concat([result['S1'] for result in shape_results[::-1]], keys=shape_names[::-1], axis=1) #produces copy
    ST_shape = pd.concat([result['ST'] for result in shape_results[::-1]], keys=shape_names[::-1], axis=1)
    S1_rel = pd.concat([result['S1'] for result in rel_results[::-1]], keys=rel_names[::-1], axis=1)
    ST_rel = pd.concat([result['ST'] for result in rel_results[::-1]], keys=rel_names[::-1], axis=1)

    ##### Reorder parameters (manual) #####
    order = ['N', 'Rplus', 'v', 'beta', 'eta/alpha', 'theta/beta' ,'logDelta', 
             'gamma', 'delta', 'loglam']
    bndorder = [0, 1, 2, -3, 5, 6, 4, -2, -1, 3]
    S1_shape = S1_shape.reindex(order)
    ST_shape = ST_shape.reindex(order)
    S1_rel = S1_rel.reindex(order)
    ST_rel = ST_rel.reindex(order)
    if bounds is not None:
        new_bounds = [bounds[ii] for ii in bndorder]
        bounds = new_bounds

    ###### Change to greek, LaTeX #####
    for id in S1_shape.index:
        # handle greek ratios
        if '/' in id:
            togreek = "$\\" + id[:id.find('/')+1] + "\\" + id[id.find('/')+1:] + r"$"
            S1_shape.rename(index={id: togreek}, inplace=True)
            ST_shape.rename(index={id: togreek}, inplace=True)
            S1_rel.rename(index={id: togreek}, inplace=True)
            ST_rel.rename(index={id: togreek}, inplace=True)
        elif id == 'loglam':
            S1_shape.rename(index={id: r'$\log(\lambda)$'}, inplace=True)
            ST_shape.rename(index={id: r'$\log(\lambda)$'}, inplace=True)
            S1_rel.rename(index={id: r'$\log(\lambda)$'}, inplace=True)
            ST_rel.rename(index={id: r'$\log(\lambda)$'}, inplace=True)
        elif id == 'logDelta':
            S1_shape.rename(index={id: r'$\log(\Delta)$'}, inplace=True)
            ST_shape.rename(index={id: r'$\log(\Delta)$'}, inplace=True)
            S1_rel.rename(index={id: r'$\log(\Delta)$'}, inplace=True)
            ST_rel.rename(index={id: r'$\log(\Delta)$'}, inplace=True)
        elif id == 'Rplus':
            S1_shape.rename(index={id: r'$R^{+}$'}, inplace=True)
            ST_shape.rename(index={id: r'$R^{+}$'}, inplace=True)
            S1_rel.rename(index={id: r'$R^{+}$'}, inplace=True)
            ST_rel.rename(index={id: r'$R^{+}$'}, inplace=True)
        # all others
        elif id not in ['N', 'v']:
            S1_shape.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
            ST_shape.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
            S1_rel.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
            ST_rel.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
            
    ###### Plot ######
    if bounds is not None:
        # setup for table
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1,1,.35], 
                    wspace=.15, left=0.04, right=0.975, bottom=0.08, top=0.95)
        axes = []
        for ii in range(2):
            for jj in range(2):
                axes.append(plt.subplot(gs[ii,jj]))
    else:
        # setup without table
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(13, 12))

    s1bars_shape = S1_shape.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8)
    s2bars_shape = ST_shape.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, legend=False)
    # Start relative bars at third color
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[3:]
    s1bars_rel = S1_rel.plot.bar(stacked=True, ax=axes[2], rot=0, width=0.8, color=colors)
    s2bars_rel = ST_rel.plot.bar(stacked=True, ax=axes[3], rot=0, width=0.8, color=colors, legend=False)
        

    for ax in axes:
        ax.tick_params(axis='x', labelsize=18, rotation=-40) #-25
        ax.tick_params(axis='y', labelsize=14)
        #ax.get_yaxis().set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].set_title('First-order indices', fontsize=26)
    axes[1].set_title('Total-order indices', fontsize=26)
    
    # set legends
    for s1bars in [s1bars_shape, s1bars_rel]:
        handles, labels = s1bars.get_legend_handles_labels()
        s1bars.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=16)
    # Create table
    if bounds is not None:
        columns = ('Value Range',)
        rows = list(S1_shape.index)
        # turn bounds into strings of ranges
        cell_text = []
        for bnd in bounds:
            low = str(bnd[0])
            high = str(bnd[1])
            # concatenate, remove leading zeros
            if low != "0" and low != "0.0":
                low = low.lstrip("0")
            if high != "0" and high != "0.0":
                high = high.lstrip("0")
            # raise any minus signs
            if '-' in low:
                low = "\u00AF"+low[1:]
            if '-' in high:
                high = "\u00AF"+high[1:]
            cell_text.append([low+"-"+high])
        tbl_ax = plt.subplot(gs[:,2])
        the_table = tbl_ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                    loc='center')
        the_table.set_fontsize(18)
        the_table.scale(1,2.3)
        the_table.auto_set_column_width(0)
        tbl_ax.axis('off')
    #plt.tight_layout()
    # reposition table
    pos = tbl_ax.get_position()
    newpos = [pos.x0 + 0.02, pos.y0, pos.width, pos.height]
    tbl_ax.set_position(newpos)
    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.{}".format(time.strftime("%m_%d_%H%M"), ext))
    return (fig, axes)



def plot_RmRp_from_store(store, peak_trunc=None, down_samp_frac=None, xrange=None,
                         show=True, save_png=False, save_pdf=False, save_tif=False,
                         plot_point=True):
    '''Function to plot the R_minus/R_plus vs. lambda plot'''

    plt.close('all')
    title_size = 28 #18
    label_size = 24 #14, 21
    tick_size = 24 #14
    fig = plt.figure(figsize=(10,7.5)) #8,6
    gs = gridspec.GridSpec(1, 1, left=0.04, right=0.975, bottom=0.15, top=0.915)
    x = store['param_values']['loglam']
    y = store['raw_output']['Rm/Rp']
    c = store['param_values']['logDelta']
    if down_samp_frac is not None:
        sample_len = len(store['param_values']['loglam'])
        down_num = int(down_samp_frac*sample_len)
        idx = np.random.randint(0, len(store['param_values']['loglam']), down_num)
        x = x[idx]
        y = y[idx]
        c = c[idx]
    if peak_trunc is not None:
        plt.scatter(x[store['raw_output']['peak']<=peak_trunc],
                y[store['raw_output']['peak']<=peak_trunc],
                s=1, c=c[store['raw_output']['peak']<=peak_trunc],
                cmap='viridis_r')
        c_ticks = np.linspace(c[store['raw_output']['peak']<=peak_trunc].min(),
                              c[store['raw_output']['peak']<=peak_trunc].max(), 5,
                              endpoint=True)
        plt.title('Fraction of resources remaining,\ntruncated at peak={:,.0f}'.format(peak_trunc),
            fontsize=title_size)
    else:
        plt.scatter(x, y, s=1, c=c, cmap='viridis_r')
        c_ticks = np.linspace(c.min(), c.max(), 5, endpoint=True)
        plt.title('Fraction of resources remaining', fontsize=title_size)
    if plot_point:
        plt.plot(-5,10**-6,'ro')
    clb = plt.colorbar(ticks=c_ticks)
    clb.ax.set_title(r'$\log_{10}(\Delta)$', fontsize=label_size, pad=15)
    clb.ax.tick_params(labelsize=tick_size)
    plt.xlabel(r"$\log_{10}(\lambda)$", fontsize=label_size)
    plt.ylabel(r"$\frac{R^-}{R^+}$", fontsize=label_size+8, rotation=0)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-.1,0.45)
    if xrange is None:
        plt.xlim(x.min(),x.max())
    else:
        plt.xlim(xrange)
    plt.ylim(0,1)
    plt.tick_params(labelsize=tick_size)
    plt.tight_layout()
    if save_png:
        fig.savefig("resource_plot.png", dpi=300)
    if save_pdf:
        fig.savefig("resource_plot.pdf")
    if save_tif:
        fig.savefig("resource_plot.tif", dpi=300)
    if show:
        plt.show()



def plot_cv_from_store(store, peak_trunc=None, down_samp_frac=None, xrange=None,
                       show=True, save_png=False, save_pdf=False, save_tif=False,
                       plot_point=True):
    '''Function to plot the theta/beta vs. c/v plot'''

    plt.close('all')
    title_size = 32 #18
    label_size = 24 #14, 21
    tick_size = 24 #14
    fig = plt.figure(figsize=(10,7.5)) #8,6
    gs = gridspec.GridSpec(1, 1, left=0.04, right=0.975, bottom=0.15, top=0.915)
    x = store['param_values']['loglam']
    y = store['raw_output']['c/v']
    c = store['param_values']['logDelta']
    if down_samp_frac is not None:
        sample_len = len(store['param_values']['loglam'])
        down_num = int(down_samp_frac*sample_len)
        idx = np.random.randint(0, len(store['param_values']['loglam']), down_num)
        x = x[idx]
        y = y[idx]
        c = c[idx]
    if peak_trunc is not None:
        plt.scatter(x[store['raw_output']['peak']<=peak_trunc],
                y[store['raw_output']['peak']<=peak_trunc],
                s=1, c=c[store['raw_output']['peak']<=peak_trunc],
                cmap='viridis_r')
        c_ticks = np.linspace(c[store['raw_output']['peak']<=peak_trunc].min(),
                              c[store['raw_output']['peak']<=peak_trunc].max(), 5,
                              endpoint=True)
        plt.title('Fraction of collective speed,\ntruncated at peak={:,.0f}'.format(peak_trunc),
            fontsize=title_size)
    else:
        plt.scatter(x, y, s=1, c=c, cmap='viridis_r')
        c_ticks = np.linspace(c.min(), c.max(), 5, endpoint=True)
        plt.title('Fraction of collective speed', fontsize=title_size)
    if plot_point:
        plt.plot(-5,0.1325,'ro')
    clb = plt.colorbar(ticks=c_ticks)
    clb.ax.set_title(r'$\log_{10}(\Delta)$', fontsize=label_size, pad=15)
    clb.ax.tick_params(labelsize=tick_size)
    plt.xlabel(r"$\log_{10}(\lambda)$", fontsize=label_size)
    plt.ylabel(r"$\frac{c}{v}$", fontsize=label_size+10, rotation=0)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-.1,0.47)
    if xrange is None:
        plt.xlim(x.min(),x.max())
    else:
        plt.xlim(xrange)
    #plt.xscale("log")
    plt.ylim(0,y.max())
    plt.tick_params(labelsize=tick_size)
    plt.tight_layout()
    if save_png:
        fig.savefig("speed_plot.png", dpi=300)
    if save_pdf:
        fig.savefig("speed_plot.pdf")
    if save_tif:
        fig.savefig("speed_plot.tif", dpi=300)
    if show:
        plt.show()



def plot_skew_from_store(store, peak_trunc=1e4, down_samp_frac=None, xrange=None,
                         show=True, save_png=False, save_pdf=False, plot_point=True):
    '''Function to plot the R_minus/R_plus vs. lambda plot'''

    plt.close('all')
    # title_size = 18
    # label_size = 14
    # tick_size = label_size
    title_size = 36
    label_size = 24
    tick_size = 24
    fig = plt.figure(figsize=(10,7.5))
    gs = gridspec.GridSpec(1, 1, left=0.04, right=0.975, bottom=0.15, top=0.915)
    x = store['param_values']['loglam']
    y = store['raw_output']['skew']
    c = store['param_values']['logDelta']
    if down_samp_frac is not None:
        sample_len = len(store['param_values']['loglam'])
        down_num = int(down_samp_frac*sample_len)
        idx = np.random.randint(0, len(store['param_values']['loglam']), down_num)
        x = x[idx]
        y = y[idx]
        c = c[idx]
    if peak_trunc is not None:
        plt.scatter(x[store['raw_output']['peak']<=peak_trunc],
                y[store['raw_output']['peak']<=peak_trunc],
                s=1, c=c[store['raw_output']['peak']<=peak_trunc],
                cmap='viridis_r')
        c_ticks = np.linspace(c[store['raw_output']['peak']<=peak_trunc].min(),
                              c[store['raw_output']['peak']<=peak_trunc].max(), 5,
                              endpoint=True)
        plt.title('max density '+r'$\leq$'+'{:,.0f}   '.format(peak_trunc), fontsize=title_size)
    else:
        plt.scatter(x, y, s=1, c=c, cmap='viridis_r')
        c_ticks = np.linspace(c.min(), c.max(), 5, endpoint=True)
        plt.title('skewness plot', fontsize=title_size)
    if plot_point:
        plt.plot(-5,1.78,'ro')
    clb = plt.colorbar(ticks=c_ticks)
    clb.ax.set_title(r"$\log_{10}(\Delta)$", fontsize=label_size, pad=15)
    clb.ax.tick_params(labelsize=tick_size)
    plt.xlabel(r'$\log_{10}(\lambda)$', fontsize=label_size)
    plt.ylabel("skewness", fontsize=label_size, rotation=90)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.05,0.56)
    if xrange is None:
        plt.xlim(x.min(),x.max())
    else:
        plt.xlim(xrange)
    #plt.xscale("log")
    #plt.ylim(-8,-5)
    plt.tick_params(labelsize=tick_size)
    plt.tight_layout()
    if save_png:
        fig.savefig("skew_plot.png", dpi=300)
    if save_pdf:
        fig.savefig("skew_plot.pdf")
    if show:
        plt.show()



if __name__ == "__main__":
    args = parser.parse_args()
    if args.ncores is None:
        with Pool() as pool:
            main(args.N, args.filename, args.ncores, pool)
    elif args.ncores > 1:
        with Pool(args.ncores) as pool:
            main(args.N, args.filename, args.ncores, pool)
    else:
        main(args.N, args.filename)