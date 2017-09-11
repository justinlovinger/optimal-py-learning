import numpy

from learning.optimize import (Problem, BacktrackingLineSearch,
                               WolfeLineSearch, BFGS, SteepestDescent,
                               SteepestDescentMomentum)
from learning.optimize import optimizer

from learning.testing import helpers


#########################
# BFGS
#########################
def test_bfgs_backtracking_line_search():
    check_optimize_sphere_function(
        BFGS(step_size_getter=BacktrackingLineSearch()))


def test_bfgs_eq():
    """Should satisfy certain requirements.

    min_{H_{k+1}} ||H_{k+1} - H_k||,
    subject to H_{k+1} = H_{k+1}^T and H_{k+1} y_k = s_k.
    """
    H_k = numpy.identity(2)
    s_k = numpy.array([-8.0, -8.0]) - numpy.array([10.0, 10.0])
    y_k = numpy.array([20., 20.]) - numpy.array([-16., -16.])

    H_kp1 = optimizer._bfgs_eq(H_k, s_k, y_k)

    # TODO: Check minimize condition (use scipy.minimize with constraints)
    assert helpers.approx_equal(H_kp1.T, H_kp1)
    assert helpers.approx_equal(H_kp1.dot(y_k), s_k)


############################
# Backtracking Line Search
############################
def test_steepest_descent_backtracking_line_search():
    check_optimize_sphere_function(
        SteepestDescent(step_size_getter=BacktrackingLineSearch()))


def test_steepest_descent_momentum_backtracking_line_search():
    check_optimize_sphere_function(
        SteepestDescentMomentum(step_size_getter=BacktrackingLineSearch()))


############################
# Wolfe Line Search
############################
def test_steepest_descent_wolfe_line_search():
    check_optimize_sphere_function(
        SteepestDescent(step_size_getter=WolfeLineSearch()))


######################
# Helpers
######################
def check_optimize_sphere_function(my_optimizer):
    # Attempt to optimize a simple sphere function
    f = lambda vec: vec[0]**2 + vec[1]**2
    df = lambda vec: numpy.array([2.0 * vec[0], 2.0 * vec[1]])

    problem = Problem(obj_func=f, jac_func=df)

    # Optimize
    vec = numpy.array([10, 10])
    iteration = 1
    obj_value = 1
    while obj_value > 1e-10 and iteration < 100:
        obj_value, vec = my_optimizer.next(problem, vec)

    assert obj_value <= 1e-10
