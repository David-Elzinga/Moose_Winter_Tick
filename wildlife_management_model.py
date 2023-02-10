import numpy as np
from scipy.integrate import solve_ivp

# Log scale the ticks for faster sims

def simulate(num_years, init_cond, p, granularity, thresh):

    # Set up lists to hold the populations and time. One list holds the populations for each time point (Zsol),
    # and the other holds the total population at the beginning of each winter (Zwinter). The population lists 
    # are ordered as S, E, P, H, Q. 
    Z0 = np.array(init_cond)
    Zsol = [Z0]; tsol = [0]
    Zwinter = []

    # Define the ODEs.
    def winter_odes(t, Z, p):
        # Unpack the states, preallocate ODE evaluations. 
        S, E, P, H, Q = Z; ODEs = 5*[0]
        
        # Calculate ODEs
        dS = -(p['mu'] + p['mu_omega'])*S 
        dE = -(p['mu'] + p['mu_omega'] + p['eta'])*E
        dP = -(p['mu'] + p['eta'])*(1 - p['b'])*P - p['mu']*p['b']*P - p['nu']*(p['H+/P+']/p['xi'])*P - p['mu_omega']*P
        dH = p['q']*p['H+/P+']*dP + (1 - p['q'])/2 * (p['H+/S+']*dS + p['H+/E+']*dE)
        dQ = 0                      
        return [dS, dE, dP, dH, dQ]

    def summer_odes(t, Z, p):
        # Unpack the states
        S, E, P, H, Q = Z; ODEs = 5*[0]

        # Calculate ODEs
        dS = p['gamma']*p['P-/H-']*E / (p['c'] + p['P-/H-']) - p['mu']*S
        dE = -p['gamma']*p['P-/H-']*E / (p['c'] + p['P-/H-']) - p['mu']*E
        dP = 0
        dH = 0
        dQ = 0
        return [dS, dE, dP, dH, dQ]
    
    def autumn_odes(t, Z, p):
        # Unpack the states
        S, E, P, H, Q = Z; ODEs = 5*[0]
        N = S + E + P

        # Calculate ODEs
        dS = -p['beta_M']*S*Q - (p['mu'] + p['mu_alpha'])*S
        dE = -p['beta_M']*E*Q - (p['mu'] + p['mu_alpha'])*E
        dP = p['beta_M']*(S + E)*Q - (p['mu'] + p['mu_alpha'])*P
        dH = p['beta_T']*N*Q
        dQ = -p['beta_T']*N*Q
        return [dS, dE, dP, dH, dQ]
    
    # Simulate the years. 
    for year in range(num_years + 1):

        p['n'] = year
        Zwinter.append(sum(Zsol[-1][:3])) 

        # Simulate winter
        S, E, P, H, Q = Zsol[-1]
        p['H+/P+'] = H/P; p['H+/S+'] = H/S; p['H+/E+'] = H/E
        p['b'] = S/(S+E)
        t_winter = [year, year + p['omega']]
        X = solve_ivp(fun=winter_odes, t_span=t_winter, t_eval=np.linspace(t_winter[0], t_winter[1], granularity), y0=Zsol[-1], args=(p,), atol=1e-100, rtol=1e-100)
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh: # if the moose population is extinct, stop.
            break

        # Pulse into summer.
        tsol.append(tsol[-1]); S, E, P, H, Q = Zsol[-1]
        N = S + E + P
        new_S = (1 + p['r_S'])*S - p['r_S']*S*N/p['K'] + p['r_P']*(1 - p['u'])*(E + P)*(1 - N/p['K'])
        new_E = E + P + p['u']*p['r_P']*(E + P)*(1 - N/p['K'])
        new_P = 0
        new_H = 0
        new_Q = H
        Zsol = np.vstack((Zsol, np.array([new_S, new_E, new_P, new_H, new_Q])))

        # Simulate summer. 
        S, E, P, H, Q = Zsol[-1]
        p['P-/H-'] = P/Q # Q = H-
        t_summer = [year + p['omega'], year + p['omega'] + p['tau']]
        X = solve_ivp(fun=summer_odes, t_span=t_summer, t_eval=np.linspace(t_summer[0], t_summer[1], granularity), y0=Zsol[-1], args=(p,), atol=1e-100, rtol=1e-100)
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh:
            break

        # Pulse into autumn
        tsol.append(tsol[-1]); S, E, P, H, Q = Zsol[-1]
        new_S = S
        new_E = E
        new_P = P
        new_H = H
        new_Q = p['r_T']*Q
        Zsol = np.vstack((Zsol, np.array([new_S, new_E, new_P, new_H, new_Q])))

        # Simulate autumn
        t_autumn = [year + p['omega'] + p['tau'], year + 1]
        X = solve_ivp(fun=autumn_odes, t_span=t_autumn, t_eval=np.linspace(t_autumn[0], t_autumn[1], granularity), y0=Zsol[-1], args=(p,), atol=1e-100, rtol=1e-100)
        Zsol = np.vstack((Zsol, X.y.T)); tsol = tsol + X.t.tolist()
        if sum(X.y.T[-1][:3]) < thresh:
            break

        # Pulse into winter. 
        tsol.append(tsol[-1]); S, E, P, H, Q = Zsol[-1]
        new_S = S
        new_E = E
        new_P = P
        new_H = H
        new_Q = 0
        Zsol = np.vstack((Zsol, np.array([new_S, new_E, new_P, new_H, new_Q])))

    return tsol, Zsol, Zwinter