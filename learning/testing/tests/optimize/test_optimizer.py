###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2017 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import numpy

from learning import optimize
from learning.optimize import (Problem, BacktrackingLineSearch,
                               WolfeLineSearch, BFGS, LBFGS, SteepestDescent,
                               SteepestDescentMomentum)
from learning.optimize import optimizer

from learning.testing import helpers


#########################
# BFGS
#########################
def test_BFGS_wolfe_line_search():
    check_optimize_sphere_function(
        BFGS(step_size_getter=WolfeLineSearch()))

def test_BFGS_backtracking_line_search():
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


#########################
# L-BFGS
#########################
def test_LBFGS_wolfe_line_search():
    check_optimize_sphere_function(
        LBFGS(step_size_getter=WolfeLineSearch()))


def test_LBFGS_approx_equal_BFGS_infinite_num_remembered_iterations():
    # When LBFGS has no limit on remembered iterations, it should approximately
    # equal BFGS, given initial hessian is the same on all iterations
    # "During its first m - 1 iterations,
    # Algorithm 7.5 is equivalent to the BFGS algorithm of Chapter 6
    # if the initial matrix H_0 is the same in both methods,
    # and if L-BFGS chooses H_0^k = H_0 at each iteration."
    # ~ Numerical Optimization 2nd pp. 179

    # Rosenbrock function
    f = lambda vec: 100.0 * (vec[1] - vec[0]**2)**2 + (vec[0] - 1.0)**2
    df = lambda vec: numpy.array([2.0 * (200.0 * vec[0]**3 - 200.0 * vec[0] * vec[1] + vec[0] - 1.0), 200.0 * (vec[1] - vec[0]**2)])

    problem = Problem(obj_func=f, jac_func=df)

    # Optimize
    bfgs_vec = numpy.random.random(2)
    lbfgs_vec = numpy.copy(bfgs_vec)

    # Same identity hessian, for both optimizers
    bfgs_optimizer = BFGS(
        step_size_getter=WolfeLineSearch(),
        initial_hessian_func=optimizer.initial_hessian_identity)
    lbfgs_optimizer = LBFGS(
        step_size_getter=WolfeLineSearch(),
        num_remembered_iterations=float('inf'),
        initial_hessian_scalar_func=optimizer.initial_hessian_one_scalar)

    for i in range(10):
        _, bfgs_vec = bfgs_optimizer.next(problem, bfgs_vec)
        _, lbfgs_vec = lbfgs_optimizer.next(problem, lbfgs_vec)

        print i
        assert helpers.approx_equal(bfgs_vec, lbfgs_vec)


def test_LBFGS_non_smooth_gradient():
    """A non-smooth gradient can result in jac_diff == \vec{0} and 1 / 0."""
    my_optimizer = LBFGS(step_size_getter=optimize.SetStepSize(0.5))
    
    # Attempt to optimize a non-smooth function
    f = lambda vec: numpy.sum(numpy.abs(vec))
    df = lambda vec: numpy.sign(vec, dtype='float64')

    problem = Problem(obj_func=f, jac_func=df)

    # Optimize
    vec = numpy.array([10, 10])
    iteration = 1
    obj_value = 1
    while obj_value > 1e-10 and iteration < 1000:
        obj_value, vec = my_optimizer.next(problem, vec)
        iteration += 1

    assert obj_value <= 1e-10


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
    while obj_value > 1e-10 and iteration < 1000:
        obj_value, vec = my_optimizer.next(problem, vec)
        iteration += 1

    assert obj_value <= 1e-10
