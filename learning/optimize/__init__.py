# Make important classes available from here

# Problem class for making problem instances to optimize
from learning.optimize.problem import Problem

# Initial step size strategies
from learning.optimize.initialstep import (IncrPrevStep, FOChangeInitialStep,
                                           QuadraticInitialStep)

# Step size strategies (line search)
from learning.optimize.linesearch import (SetStepSize, BacktrackingLineSearch,
                                          WolfeLineSearch)

# Optimizers
from learning.optimize.optimizer import (SteepestDescent,
                                         SteepestDescentMomentum, BFGS)
