import numpy as np
from scipy.integrate import solve_ivp

def annual_map(num_years, init_cond, parm, thresh):
    # Unpack the states
    S, P = init_cond
    Zsol = [S + P]

    # Use discrete map to go one year at a time. 
    for year in range(num_years):
        S_new = ((1 + parm['r_P'])*np.exp(parm['gamma']*parm['tau']) - parm['u']*parm['r_P'] - 1)*P*np.exp(-parm['alpha']*parm['beta'] - parm['gamma']*parm['tau'] - parm['nu']*parm['omega'] - parm['mu']) + (1 + parm['r_S'])*S*np.exp(-parm['alpha']*parm['beta'] - parm['mu'])
        P_new = ((np.exp(parm['alpha']*parm['beta']) - 1)*np.exp(parm['gamma']*parm['tau']) + parm['r_P']*((np.exp(parm['alpha']*parm['beta']) - 1)*np.exp(parm['gamma']*parm['tau']) + parm['u']) + 1)*P*np.exp(-parm['alpha']*parm['beta'] - parm['gamma']*parm['tau'] - parm['nu']*parm['omega'] - parm['mu']) + (np.exp(parm['alpha']*parm['beta']) - 1)*(1 + parm['r_S'])*S*np.exp(-parm['alpha']*parm['beta'] - parm['mu'])
        Zsol.append(S_new+P_new)
        S = S_new; P = P_new
        if S + P < thresh:
            break
    
    return Zsol

def simulate(num_years, init_cond, parm, granularity, thresh):

    # Set up lists to hold the populations and time. One list holds the populations for each time point (Zsol),
    # and the other holds the total population at the beginning of each winter (Zwinter). The population lists 
    # are ordered as S, P. 
    Z0 = np.array(init_cond)
    Zsol = [Z0]; tsol = [0]
    Zwinter = []

    # Define the ODEs.
    def winter_odes(t, Z, parm):
        # Unpack the states, preallocate ODE evaluations. 
        S, P = Z; ODEs = 2*[0]
        
        # Calculate ODEs
        dS = -parm['mu']*S 
        dP = -(parm['mu'] + parm['nu'])*P
        return [dS, dP]

    def summer_odes(t, Z, parm):
        # Unpack the states
        S, P = Z; ODEs = 2*[0]

        # Calculate ODEs
        dS = parm['gamma']*P - parm['mu']*S
        dP = -(parm['gamma'] + parm['mu'])*P
        return [dS, dP]
    
    def autumn_odes(t, Z, parm):
        # Unpack the states
        S, P = Z; ODEs = 2*[0]

        # Calculate ODEs
        dS = -(parm['beta'] + parm['mu'])*S
        dP = parm['beta']*S - parm['mu']*P
        return [dS, dP]
    
    # Simulate the years. 
    for year in range(num_years + 1):

        parm['n'] = year
        Zwinter.append(sum(Zsol[-1][:3])) 

        # Simulate winter
        t_winter = [year, year + parm['omega']]
        X = solve_ivp(fun=winter_odes, t_span=t_winter, t_eval=np.linspace(t_winter[0], t_winter[1], granularity), y0=Zsol[-1], args=(parm,), atol=1e-10, rtol=1e-10)
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:2]) < thresh: # if the moose population is extinct, stop.
            break

        # Pulse into summer.
        tsol.append(tsol[-1]); S, P = Zsol[-1]
        new_S = (1 + parm['r_S'])*S + (1 - parm['u'])*parm['r_P']*P 
        new_P = (1 + parm['u']*parm['r_P'])*P
        Zsol = np.vstack((Zsol, np.array([new_S, new_P])))

        # Simulate summer. 
        t_summer = [year + parm['omega'], year + parm['omega'] + parm['tau']]
        X = solve_ivp(fun=summer_odes, t_span=t_summer, t_eval=np.linspace(t_summer[0], t_summer[1], granularity), y0=Zsol[-1], args=(parm,), atol=1e-10, rtol=1e-10)
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:2]) < thresh:
            break

        # Pulse into autumn
        tsol.append(tsol[-1]); S, P = Zsol[-1]
        new_S = S
        new_P = P
        Zsol = np.vstack((Zsol, np.array([new_S, new_P])))

        # Simulate autumn
        t_autumn = [year + parm['omega'] + parm['tau'], year + 1]
        X = solve_ivp(fun=autumn_odes, t_span=t_autumn, t_eval=np.linspace(t_autumn[0], t_autumn[1], granularity), y0=Zsol[-1], args=(parm,), atol=1e-10, rtol=1e-10)
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh:
            break

        # Pulse into winter. 
        tsol.append(tsol[-1]); S, P = Zsol[-1]
        new_S = S
        new_P = P
        Zsol = np.vstack((Zsol, np.array([new_S, new_P])))

    return tsol, Zsol, Zwinter