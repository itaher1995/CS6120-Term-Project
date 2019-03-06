"""
Creates NODE model based on torchdiffeq for project document classification
"""

# Import right packages

# Get batch data

# Create ODE network based on pytorch objects in ODEfunc class
	# Needs to have at least __init__ and forward functions
	# Where __init__ has the network declaration and forward has the forward propogation

# Some type of reporting class, if we're modeling after NODE peoples work

# train funciton
	# Initialize layers based one ODEfunc
	# Define loss criterion and optimizer
	# Leverage odeint (main funciton from project) to calculate prediction using time as the integral intervales (I still don't perfectly understand)

# test function
	# Used saved model to test things