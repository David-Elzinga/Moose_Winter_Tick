import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

'''
Solves the ODEs of the full system.
'''

# Define the ODEs equation systems. 
def SI_ODEs_Summer(t,x,params):
    '''
    This function returns the time derivates of S,I

    The ODE solver expects the first two arguments to be t and x

    The params argument represents a dictionary. The dictionary is
    passed into the solver using the set_f_params method
    '''

    S = x[0]; I = x[1]
    dx = np.zeros(2)

    dx[0] = -params['mu_S']*S + params['gamma']*I
    dx[1] = -params['mu_S']*I - params['gamma']*I

    return dx

def SI_ODEs_Winter(t,x,params):
    '''
    This function returns the time derivates of S,I

    The ODE solver expects the first two arguments to be t and x

    The params argument represents a dictionary. The dictionary is
    passed into the solver using the set_f_params method
    '''

    S = x[0]; I = x[1]
    dx = np.zeros(2)

    dx[0] = -params['mu_W']*S
    dx[1] = -(params['mu_W'] + params['nu'])*I

    return dx

def SW_Pulse(x,params):
    '''
    This function executes the pulse from the summer season to the winter season.
    '''

    S = x[0]; I = x[1]

    Sp = (1 - params['p'])*S
    Ip = I + params['p']*S

    return Sp, Ip

def WS_Pulse(x,params):
    '''
    This function executes the pulse from the winter season to the summer season.
    '''

    S = x[0]; I = x[1]

    Sp = S + (params['r']*(S + params['q']*I)) / (1 + params['kappa']*(S + params['q']*I))
    Ip = params['q']*I

    return Sp, Ip

# Specify the initial conditions for the moose population (susceptible and infectious)
S0 = 90
I0 = 0

# Specify the parameter values for the model
params = {}
params['mu_S'] = 0.3
params['mu_W'] = 0.5
params['nu'] = 0.5
params['gamma'] = 0.1
params['kappa'] = 0.1
params['r'] =   1
params['p'] =   0.5
params['q'] = 0.2
params['tau'] = 0.5

# Specify the solution as x. Provide the initial condition x0.
x0 = np.array([S0,I0])

# Preallocate the solution for each class. Also store the time values.
Ssol = []; Isol = []; times = []

# Create solver objects for both systems of ODEs. Specify the solvers. Pass the parameters.
Summer_solver = ode(SI_ODEs_Summer)
Winter_solver = ode(SI_ODEs_Winter)

Summer_solver.set_integrator('dopri5')
Winter_solver.set_integrator('dopri5')

Summer_solver.set_f_params(params)
Winter_solver.set_f_params(params)

# Iterate over each year. During each year we solve the summer equations, then we solve the winter equations. 
num_years = 50
for y in range(num_years):

    # Solve the summer equations.
    for t in np.linspace(y, y + params['tau'], 1000):

        times.append(t)
        # The initial solution in the summer equation is determined from the pulse from winter to summer equations (if y > 0),
        # otherwise (y = 0) it is the initial condition pre-specified. We allow a small cushion to avoid checking the equality of floats.
        if (t < y + 0.0001) and y > 0:
            WS_Pulse_Result = WS_Pulse([Ssol[-1], Isol[-1]], params)
            Ssol.append(WS_Pulse_Result[0])
            Isol.append(WS_Pulse_Result[1])
            x0 = [Ssol[-1], Isol[-1]]

        elif (t < y + 0.0001) and y == 0:
            WS_Pulse_Result = WS_Pulse([x0[0], x0[1]], params)
            Ssol.append(WS_Pulse_Result[0])
            Isol.append(WS_Pulse_Result[1])
            x0 = [Ssol[-1], Isol[-1]]

        else: # Assuming we are not at the beginning of the season, use the ODEs to evaluate the solution at this time. 
            Summer_solver.set_initial_value(x0,y)
            Summer_solver.integrate(t)
            assert Summer_solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(Summer_solver.y[0])
            Isol.append(Summer_solver.y[1])

    # Solve the winter equations. 
    for t in np.linspace(y + params['tau'], y + 1, 1000):

        times.append(t)

        # The initial solution in the winter equations is determined from the pulse from summer to winter equations. We allow
        # a small cushion to avoid checking the equality of floats.
        if (t < y + params['tau'] + 0.0001):
            SW_Pulse_Result = SW_Pulse([Ssol[-1], Isol[-1]], params)
            Ssol.append(SW_Pulse_Result[0])
            Isol.append(SW_Pulse_Result[1])
            x0 = [Ssol[-1], Isol[-1]]
        
        else: # Assuming we are not at the beginning of the season, use the ODEs to evaluate the solution at this time. 
            Winter_solver.set_initial_value(x0, y + params['tau'])
            Winter_solver.integrate(t)
            assert Winter_solver.successful(), "Solver did not converge at time {}.".format(t)
            Ssol.append(Winter_solver.y[0])
            Isol.append(Winter_solver.y[1])

# Plot the solution! 
plt.plot(times,Ssol,times,Isol)
plt.legend(['S','I'])
plt.xlabel("$t$ (years)")
plt.ylabel("Number of Moose")
plt.title("Two Pulse Model")
plt.show()