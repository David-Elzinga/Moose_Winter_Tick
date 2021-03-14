import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

'''
Solves the pulse-connected TriSeasonal model and plots the result in 
a time series. 
'''

# Define the system of ODEs for each of the seasons. Note that in each case 
# the ODE solver expects the first two arguments to be t and x, and that 
# params is a dictionary passed to the solver using the set_f_params method
def winter_odes(t,x,params):
    '''
    Returns the time derivates of S,I during the winter season. 
    '''

    S = x[0]; I = x[1]
    dx = np.zeros(2)

    dx[0] = -params['mu']*S
    dx[1] = -(params['mu'] + params['nu'])*I

    return dx

def summer_odes(t,x,params):
    '''
    Returns the time derivates of S,I during the summer season. 
    '''

    S = x[0]; I = x[1]
    dx = np.zeros(2)

    dx[0] = params['gamma']*I -params['mu']*S
    dx[1] = -(params['mu'] + params['gamma'])*I

    return dx

def autumn_odes(t,x,params):
    '''
    Returns the time derivates of S,I during the autumn season. 
    '''

    S = x[0]; I = x[1]
    dx = np.zeros(2)

    dx[0] = -(params['mu'] + params['beta'])*S
    dx[1] = params['beta']*S - params['mu']*I

    return dx

def pulse_to_summer(x,params):
    '''
    Returns the pulsed population S,I after the winter to summer pulse
    '''

    S = x[0]; I = x[1]

    Sp = S*(1 + params['r'] - (params['r']/params['K'])*(S + params['q']*I))
    Ip = params['q']*I

    return Sp, Ip

# Specify the initial conditions for the moose population (susceptible and infected) and the number of years to simulate
S0 = 25
I0 = 5
num_years = 5

# Specify the parameter values for the model
params = {}
params['q'] = 0.3
params['r'] = 2
params['K'] = 50
params['mu'] = 0.1
params['gamma'] = 0.1
params['nu'] = 0.2
params['beta'] = 0.7
params['omega'] = 0.2
params['tau'] = 0.4
params['alpha'] = 0.4


r_tilde = -1 + (params['q'] * np.exp(params['mu']) * ( np.exp(params['gamma'] * params['tau']) - np.exp(params['alpha']*params['beta'] + params['gamma']*params['tau']) - 1 ) + np.exp(params['alpha']*params['beta'] + params['gamma'] * params['tau'] + params['nu']*params['omega'] + 2*params['mu'])) / (np.exp(params['gamma'] * params['tau'] + params['nu']*params['omega'] + params['mu']) - params['q'])

print(r_tilde)

# Assert that the sum of the legnths of the seasons (omega + tau + alpha) is 1. 
assert (params['omega'] + params['tau'] + params['alpha'] == 1), "the combined length of the seasons is not one."

# Preallocate the solution for each class. Also store the time values.
Ssol = []; Isol = []; times = []

# Create solver objects for both systems of ODEs. Specify the solvers. Pass the parameters.
winter_solver = ode(winter_odes); summer_solver = ode(summer_odes); autumn_solver = ode(autumn_odes)

winter_solver.set_integrator('dopri5'); summer_solver.set_integrator('dopri5'); autumn_solver.set_integrator('dopri5')

winter_solver.set_f_params(params); summer_solver.set_f_params(params); autumn_solver.set_f_params(params)

# The initial solution is the first solution to the system, record it. 
Ssol.append(S0)
Isol.append(I0)
times.append(0)

# Iterate over each year. During each year we solve the summer equations, then we solve the winter equations. 
for y in range(num_years):
    print(y)
    # Beginning in the winter season, solve the corresponding ODE system
    for n, t in enumerate(np.linspace(y, y + params['omega'], 1000)):
        
        # Check to see if this moment in time represents the pulse from autumn to winter. 
        if n > 0: 
            # This is not the time to pulse, but rather is inside the winter season. Use the ode solver to 
            # evaluate the solution at this time.  

            # For each time that we solve the system at, record the time value. 
            times.append(t)

            winter_solver.set_initial_value(x0,y)
            winter_solver.integrate(t)
            assert winter_solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(winter_solver.y[0])
            Isol.append(winter_solver.y[1])
        elif n == 0:
            # This represents the pulse from autumn to winter. be careful not to double count a single time, make sure to overwrite! 
            x0 = [Ssol[-1], Isol[-1]]

    # Continuing in the summer season, solve the corresponding ODE system
    for n, t in enumerate(np.linspace(y + params['omega'], y + params['omega'] + params['tau'], 1000)):

        # Check to see if this moment in time represents the pulse from winter to summer. 
        if n > 0: 
            # This is not the time to pulse, but rather is inside the summer season. Use the ode solver to 
            # evaluate the solution at this time. 

            # For each time that we solve the system at, record the time value. 
            times.append(t) 

            summer_solver.set_initial_value(x0,y + params['omega'])
            summer_solver.integrate(t)
            assert summer_solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(summer_solver.y[0])
            Isol.append(summer_solver.y[1])
        elif n == 0:
            # This represents the pulse from winter to summer. Perform the pulse, overwrite the most recent value of S and I to reflect the pulse happening. 
            Ssol[-1], Isol[-1] = pulse_to_summer([Ssol[-1],Isol[-1]], params)
            x0 = [Ssol[-1], Isol[-1]]

    # Continuing in the autumn season, solve the corresponding ODE system
    for n, t in enumerate(np.linspace(y + params['omega'] + params['tau'], y + params['omega'] + params['tau'] + params['alpha'], 1000)):
        
        # Check to see if this moment in time represents the pulse from summer to autumn. 
        if n > 0: 
            #import pdb; pdb.set_trace()
            # This is not the time to pulse, but rather is inside the autumn season. Use the ode solver to 
            # evaluate the solution at this time.  

            # For each time that we solve the system at, record the time value. 
            times.append(t)

            autumn_solver.set_initial_value(x0,y + params['omega'] + params['tau'])
            autumn_solver.integrate(t)
            assert autumn_solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(autumn_solver.y[0])
            Isol.append(autumn_solver.y[1])
        elif n == 0:
            # This represents the pulse from summer to autumn. be careful not to double count a single time, make sure to overwrite! 
            x0 = [Ssol[-1], Isol[-1]]

# Plot the solution! 
plt.plot(times,Ssol,times,Isol)
plt.legend(['S','I'])
plt.xlabel("$t$ (years)")
plt.ylabel("Number of Moose")
plt.title("TriSeasonal Model")
plt.show()