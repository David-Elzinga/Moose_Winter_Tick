import numpy as np
from scipy.integrate import solve_ivp

def simulate(num_years, init_cond, parm, granularity, thresh):

    # Set up lists to hold the populations and time. One list holds the populations for each time point (Zsol),
    # and the other holds the total population at the beginning of each winter (Zwinter).
    Z0 = np.array(init_cond)
    Zsol = [Z0]; tsol = [0]
    Zwinter = []

    # Define the ODEs.
    def winter_odes(t, Z, parm):
        # Unpack the states, preallocate ODE evaluations. 
        S, E, I, TH, TQ = Z; ODEs = 5*[0]
        
        # Calculate ODEs
        dS = -(parm['mu'] + parm['mu_omega'])*S 
        dE = -(parm['eta'] + parm['mu_omega'])*E
        dI = -(parm['eta'] + parm['mu_omega'])*(1-parm['c'])*I - (parm['mu'] + parm['mu_omega'])*parm['c']*I - parm['nu']*parm['z_I']*I
        dTH =  parm['z_I'] * parm['q'] * dI + (1-parm['q'])/2 * (parm['z_S'] * dS + parm['z_E'] * dE)
        dTQ = 0
        return [dS, dE, dI, dTH, dTQ]

    def summer_odes(t, Z, parm):
        # Unpack the states
        S, E, I, TH, TQ = Z

        # Calculate ODEs
        dS = E*parm['gamma']*(parm['l'] / (parm['d'] + parm['l'])) - (parm['mu'] + parm['mu_tau'])*S
        dE = -E*parm['gamma']*(parm['l'] / (parm['d'] + parm['l'])) - (parm['mu'] + parm['mu_tau'])*E
        dI = 0
        dTH = 0 
        dTQ = 0
        return [dS, dE, dI, dTH, dTQ]

    def autumn_odes(t, Z, parm):
        # Unpack the states
        S, E, I, TH, TQ = Z; ODEs = 5*[0]
        N = S+E+I

        # Calculate ODEs
        dS = -parm['beta']/parm['xi'] * S*TQ - (parm['mu'] + parm['mu_alpha'])*S
        dE = -parm['beta']/parm['xi'] * E*TQ - (parm['mu'] + parm['mu_alpha'])*E
        dI = parm['beta']/parm['xi'] * (S + E) * TQ - (parm['mu'] + parm['mu_alpha'])*I
        dTH = parm['beta']*N*TQ
        dTQ = -parm['beta']*N*TQ
        return [dS, dE, dI, dTH, dTQ]
    
    # Simulate the years. 
    for year in range(num_years):

        parm['n'] = year
        Zwinter.append(sum(Zsol[-1][:3])) 

        # Simulate winter. First calculate c and z
        parm['c'] = Zsol[-1][0] / (Zsol[-1][0] + Zsol[-1][1])
        parm['z_I'] = Zsol[-1][3] / Zsol[-1][2]; parm['z_E'] = Zsol[-1][3] / Zsol[-1][1]; parm['z_S'] = Zsol[-1][3] / Zsol[-1][0]
        t_winter = [year, year + parm['omega']]
        X = solve_ivp(fun=winter_odes, t_span=t_winter, t_eval=np.linspace(t_winter[0], t_winter[1], granularity), y0=Zsol[-1], args=(parm,))
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh: # if the moose population is extinct, stop.
            break

        # Pulse into summer.
        tsol.append(tsol[-1])
        new_S = Zsol[-1][0]*(1+parm['r'] - sum(Zsol[-1][0:3])*parm['r']/parm['K']) 
        new_E = Zsol[-1][1] + Zsol[-1][2]
        new_I = 0; new_TH = 0
        new_TQ = parm['r_T']*Zsol[-1][3]
        Zsol = np.vstack((Zsol, np.array([new_S, new_E, new_I, new_TH, new_TQ])))

        # Simulate summer. First calculate l (grab before the pulse)
        if Zsol[-2][3] > 0.0001: 
            parm['l'] = Zsol[-2][2]/Zsol[-2][3]
        else: # if we divide by zero, there's no ticks anymore -> make d = 0 and l = 1 so that gamma -> gamma max.
            parm['d'] = 0; parm['l'] = 1
        t_summer = [year + parm['omega'], year + parm['omega'] + parm['tau']]
        X = solve_ivp(fun=summer_odes, t_span=t_summer, t_eval=np.linspace(t_summer[0], t_summer[1], granularity), y0=Zsol[-1], args=(parm,))
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh:
            break

        # Pulse into autumn
        tsol.append(tsol[-1])
        new_S = Zsol[-1][0]
        new_E = Zsol[-1][1]
        new_I = 0; new_TH = 0
        new_TQ = Zsol[-1][4]
        Zsol = np.vstack((Zsol, np.array([new_S, new_E, new_I, new_TH, new_TQ])))

        # Simulate autumn
        t_autumn = [year + parm['omega'] + parm['tau'], year + 1]
        X = solve_ivp(fun=autumn_odes, t_span=t_autumn, t_eval=np.linspace(t_autumn[0], t_autumn[1], granularity), y0=Zsol[-1], args=(parm,))
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh:
            break

        # Pulse into winter. 
        tsol.append(tsol[-1])
        new_S = Zsol[-1][0]
        new_E = Zsol[-1][1]
        new_I = Zsol[-1][2]
        new_TH = Zsol[-1][3]
        new_TQ = 0
        Zsol = np.vstack((Zsol, np.array([new_S, new_E, new_I, new_TH, new_TQ])))

    return tsol, Zsol, Zwinter