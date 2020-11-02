import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

'''
Solves the ODEs of the single-pulse system.
'''

# Define the ODEs equation systems. 

def SI_ODEs_Winter(t,x,params):
    '''
    This function returns the time derivates of S,I

    The ODE solver expects the first two arguments to be t and x

    The params argument represents a dictionary. The dictionary is
    passed into the solver using the set_f_params method
    '''

    S = x[0]; I = x[1]
    dx = np.zeros(2)

    dx[0] = -params['mu_W']*(1 - params['tau'])*S
    dx[1] = -(params['mu_W'] + params['nu'])*(1 - params['tau'])*I

    return dx

def Pulse(x,params):
    '''
    This function executes the pulse to the summer season, through the summer, and into the winter.
    '''
    S = x[0]; I = x[1]
    #import pdb; pdb.set_trace()
    # Define the beverton holt term for simplicity
    BH_Term = S + params['q']*I + (params['r']*(S + params['q']*I)) / (1 + params['kappa']*(S + params['q']*I))

    Sp = (1 - params['p'])*(BH_Term*np.exp(-params['mu_S']*params['tau']) - params['q']*I*np.exp(-params['tau']*(params['mu_S'] + params['gamma'])))
    Ip = params['q']*I*np.exp(-params['tau']*(params['mu_S'] + params['gamma'])) + params['p']*(BH_Term*np.exp(-params['mu_S']*params['tau']) - params['q']*I*np.exp(-params['tau']*(params['mu_S'] + params['gamma'])))

    return Sp, Ip


# Specify the initial conditions for the moose population (susceptible and infectious)
S0 = 10.3
I0 = 0

# Specify the parameter values for the model
params = {}
params['mu_S'] = 0.3
params['mu_W'] = 0.5
params['nu'] = 0.5
params['gamma'] = 0.1
params['kappa'] = 0.1
params['r'] =  1
params['p'] =  0.5
params['q'] = 0.2
params['tau'] = 0.5


# Specify the solution as x. Provide the initial condition x0.
x0 = np.array([S0,I0])

# Preallocate the solution for each class. Also store the time values.
Ssol = []; Isol = []; times = []

# Create solver objects for both systems of ODEs. Specify the solvers. Pass the parameters.
Winter_solver = ode(SI_ODEs_Winter)

Winter_solver.set_integrator('dopri5')

Winter_solver.set_f_params(params)

# Iterate over each year. During each year we solve the summer equations, then we solve the winter equations. 
num_years = 50
for y in range(num_years):

    # Solve the summer equations.
    for t in np.linspace(y, y + 1, 1000):

        times.append(t)
        
        # The initial solution in the summer equation is determined from the pulse from winter to summer equations (if y > 0),
        # otherwise (y = 0) it is the initial condition pre-specified. We allow a small cushion to avoid checking the equality of floats.
        if (t < y + 0.00001) and y > 0:
            Pulse_Result = Pulse([Ssol[-1], Isol[-1]], params)
            Ssol.append(Pulse_Result[0])
            Isol.append(Pulse_Result[1])
            x0 = [Ssol[-1], Isol[-1]]

        elif (t < y + 0.0001) and y == 0:
            Pulse_Result = Pulse([x0[0], x0[1]], params)
            Ssol.append(Pulse_Result[0])
            Isol.append(Pulse_Result[1])
            x0 = [Ssol[-1], Isol[-1]]

        else: # Assuming we are not at the beginning of the season, use the ODEs to evaluate the solution at this time. 
            Winter_solver.set_initial_value(x0,y)
            Winter_solver.integrate(t)
            assert Winter_solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(Winter_solver.y[0])
            Isol.append(Winter_solver.y[1])

# Plot the solution! 
plt.plot(times,Ssol,times,Isol)
plt.legend(['S','I'])
plt.xlabel("$t$ (years)")
plt.ylabel("Number of Moose")
plt.title("One Pulse Model")
plt.show()