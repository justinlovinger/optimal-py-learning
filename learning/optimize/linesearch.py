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
"""Strategies for determining step size during an optimization iteration."""

import logging
import itertools

import numpy

from learning.optimize import IncrPrevStep, QuadraticInitialStep


#########################
# Base Model
#########################
class StepSizeGetter(object):
    """Returns step size when called.

    Used by Optimizer.
    """

    def reset(self):
        """Reset parameters."""
        pass

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        raise NotImplementedError()


###############################
# StepSizeGetter implementations
###############################
class SetStepSize(StepSizeGetter):
    """Return a given step size every call.

    Simple and efficient. Not always effective.
    """

    def __init__(self, step_size):
        super(SetStepSize, self).__init__()

        self._step_size = step_size

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        return self._step_size


class BacktrackingLineSearch(StepSizeGetter):
    """Return step size found with backtracking line search."""

    def __init__(self, c_1=1e-4, decr_rate=0.9, initial_step_getter=None):
        super(BacktrackingLineSearch, self).__init__()

        self._c_1 = c_1
        self._decr_rate = decr_rate

        if initial_step_getter is None:
            # Slightly more than 1 step up
            initial_step_getter = IncrPrevStep(
                incr_rate=1.0 / self._decr_rate + 0.05, upper_bound=None)
        self._initial_step_getter = initial_step_getter

    def reset(self):
        """Reset parameters."""
        super(BacktrackingLineSearch, self).reset()
        self._initial_step_getter.reset()

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        initial_step = self._initial_step_getter(xk, obj_xk, jac_xk, step_dir,
                                                 problem)

        step_size = _backtracking_line_search(
            xk,
            obj_xk,
            jac_xk,
            step_dir,
            problem.get_obj,
            self._c_1,
            initial_step,
            decr_rate=self._decr_rate)

        self._initial_step_getter.update(step_size)

        return step_size


class WolfeLineSearch(StepSizeGetter):
    """Specialized algorithm for finding step size that satisfies strong wolfe conditions."""

    def __init__(self, c_1=1e-4, c_2=0.9, initial_step_getter=None):
        super(WolfeLineSearch, self).__init__()

        # "In practice, c_1 is chosen to be quite small, say c_1 = 10^-4"
        # ~Numerical Optimization (2nd) pp. 33
        self._c_1 = c_1

        # "Typical values of c_2 are 0.9 when the search direction p_k
        # is chosen by a Newton or quasi-Newton method,
        # and 0.1 when pk is obtained from a nonlinear conjugate gradient method."
        # ~Numerical Optimization (2nd) pp. 34
        self._c_2 = c_2

        if initial_step_getter is None:
            initial_step_getter = QuadraticInitialStep()
        self._initial_step_getter = initial_step_getter

    def reset(self):
        """Reset parameters."""
        super(WolfeLineSearch, self).reset()
        self._initial_step_getter.reset()

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        initial_step = self._initial_step_getter(xk, obj_xk, jac_xk, step_dir,
                                                 problem)

        step_size = _line_search_wolfe(xk, obj_xk, jac_xk, step_dir,
                                       problem.get_obj_jac, self._c_1,
                                       self._c_2, initial_step)

        self._initial_step_getter.update(step_size)
        return step_size


def _backtracking_line_search(parameters,
                              obj_xk,
                              jac_xk,
                              step_dir,
                              obj_func,
                              c_1,
                              initial_step,
                              decr_rate=0.9):
    """Return step size that satisfies the armijo rule.

    Discover step size by decreasing step size in small increments.

    args:
        parameters: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        obj_func: Function taking parameters and returning obj value at given parameters.
        c_1: Strictness parameter for Armijo rule.
    """
    if numpy.isnan(obj_xk):
        # Failsafe because _armijo_rule will never return True
        logging.warning(
            'nan objective value in _backtracking_line_search, defaulting to 1e-25 step size'
        )
        return 1e-25

    step_size = initial_step
    for i in itertools.count(start=1):
        if step_size < 1e-25:
            # Failsafe for numerical precision errors preventing _armijo_rule returning True
            # This can happen if gradient provides very little improvement
            # (or is in the wrong direction)
            logging.warning(
                '_backtracking_line_search failed Armijo with step_size ~= 1e-25, returning'
            )
            return step_size

        obj_xk_plus_ap = obj_func(parameters + step_size * step_dir)
        if _armijo_rule(step_size, obj_xk, jac_xk, step_dir, obj_xk_plus_ap,
                        c_1):
            assert step_size > 0
            return step_size

        # Did not satisfy, decrease step size and try again
        step_size *= decr_rate


WOLFE_INCR_RATE = 1.5


def _line_search_wolfe(parameters, obj_xk, jac_xk, step_dir, obj_jac_func, c_1,
                       c_2, initial_step):
    """Return step size that satisfies wolfe conditions.

    See Numerical Optimization (2nd) pp. 60

    This procedure first finds an interval containing an
    acceptable step length (or just happens upon such a length),
    then calls the zoom procedure to fine tune that interval
    until an acceptable step length is discovered.

    args:
        parameters: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        obj_jac_func: Function taking parameters and returning obj and jac at given parameters.
        c_1: Strictness parameter for Armijo rule.
        c_2: Strictness parameter for curvature condition.
    """
    if numpy.isnan(obj_xk):
        # Failsafe for erroneously calculated obj_xk (usually overflow or x/0)
        # TODO: Might need similar failsafe for nan in jac_xk or step_dir
        logging.warning(
            'nan objective value in _line_search_wolfe, defaulting to 1e-10 step size'
        )
        return 1e-10

    step_zero_obj = obj_xk
    step_zero_grad = jac_xk.dot(step_dir)

    # We need the current and previous step size for some operations
    prev_step_size = 0.0
    prev_step_obj = step_zero_obj

    step_size = initial_step
    for i in itertools.count(start=1):
        if i >= 100:
            # Failsafe for numerical precision errors preventing convergence
            # This can happen if gradient provides very little improvement
            # (or is in the wrong direction)
            logging.warning('Wolfe line search aborting after 100 iterations')
            return step_size

        # Evaluate objective and jacobian for most recent step size
        step_obj, step_grad = _step_size_obj_jac_func(step_size, parameters,
                                                      step_dir, obj_jac_func)

        # True if objective did not improve (step_obj >= prev_step_obj), after first iterations,
        # or armijo condition is False (step_obj > obj_xk + c_1*step_size*step_grad)
        if ((i > 1 and step_obj >= prev_step_obj)
                or (step_obj > obj_xk + c_1 * step_size * step_grad)):
            return _zoom_wolfe(prev_step_size, prev_step_obj, step_size,
                               parameters, obj_xk, step_zero_grad, step_dir,
                               obj_jac_func, c_1, c_2)

        # Check if step size is already an acceptable step length
        # True when gradient is sufficiently small (magnitude wise)
        elif numpy.abs(step_grad) <= -c_2 * step_zero_grad:
            return step_size

        # If objective value did not improve (first if statement)
        # and step size needs to increase (non-negative gradient)
        elif step_grad >= 0:
            return _zoom_wolfe(step_size, step_obj, prev_step_size, parameters,
                               obj_xk, step_zero_grad, step_dir, obj_jac_func,
                               c_1, c_2)

        # Increase step size, score current values for comparison to previous
        prev_step_size = step_size
        prev_step_obj = step_obj
        prev_step_grad = step_grad

        # Similar to zoom, we need to find a new trial step size
        # somewhere between current, and an arbitrary max
        # alpha_i < alpha_{i+1} < max
        # "The last step of the algorithm performs extrapolation to find
        # the next trial value alpha_{i+1}.
        # To implement this step we can use approaches like the interpolation
        # procedures above, or we can simply set alpha_{i+1} to some constant
        # multiple of alpha_i.
        # Whichever strategy we use,
        # it is important that the successive steps increase quickly enough to
        # reach the upper limit alpha_max in a finite number of iterations."
        # ~Numerical Optimization (2nd) pp. 61
        # Use multiply by constant strategy
        # TODO: Try other interpolation strategies
        step_size *= WOLFE_INCR_RATE


def _zoom_wolfe(step_size_low, step_size_low_obj, step_size_high, parameters,
                step_zero_obj, step_zero_grad, step_dir, obj_jac_func, c_1,
                c_2):
    """Zoom into acceptable step size within a given interval.

    Args:
        step_size_low: Step size with low objective value (good)
        step_size_high: Step size with high objective value (high)
    """
    # NOTE: lower objective values are better
    # (hence step_size_low better than step_size_high)
    # TODO: Optimize by caching values repeatedly used in inequalities

    for i in itertools.count(start=1):
        # Choose step size
        # NOTE: step_size should not be too close to low or high
        # TODO: Test other strategies (see Interpolation subsection of NumOpt)
        # "Interpolate (using quadratic, cubic, or bisection)
        # to find a trial step length alpha_j between alpha_lo and alpha_hi"
        # ~Numerical Optimization (2nd) pp. 61
        # Use bisection
        step_size = _bisect_value(
            min(step_size_low, step_size_high),
            max(step_size_low, step_size_high))
        assert step_size >= 0

        if i >= 100:
            # Failsafe for numerical precision errors preventing convergence
            # This can happen if gradient provides very little improvement
            # (or is in the wrong direction)
            logging.warning(
                'Wolfe line search (zoom) aborting after 100 iterations')
            return step_size

        step_obj, step_grad = _step_size_obj_jac_func(step_size, parameters,
                                                      step_dir, obj_jac_func)

        # If this step is worse, than the projection from initial parameters
        # (a.k.a. Armijo condition if False)
        # or this step is worse than the current high (bad) step size
        if (step_obj > step_zero_obj + c_1 * step_size * step_zero_grad
                or step_obj >= step_size_low_obj):
            # step_size is not an improvement
            # This step size is the new poor valued side of the interval
            step_size_high = step_size

        # step_size is an improvement
        else:
            # If this step size caused an improvement
            # (first if statement is false),
            # and step size gradient is sufficiently small (magnitude wise)
            if numpy.abs(step_grad) <= -c_2 * step_zero_grad:
                return step_size

            # If good step size is larger than bad step size,
            # and gradient is positive,
            # or vice versa
            if step_grad * (step_size_high - step_size_low) >= 0:
                # Set the current bad step size to the current good step size
                # Because step_size is better (and will be set so in a couple lines)
                step_size_high = step_size_low

            # Set step_size_low
            step_size_low = step_size
            step_size_low_obj = step_obj


def _bisect_value(min_, max_):
    """Return value half way between min and max."""
    return min_ + 0.5 * (max_ - min_)


def _step_size_obj_jac_func(step_size, parameters, step_dir, obj_jac_func):
    """Return objective value and gradient for step size."""
    step_obj, jac_xk_plus_ap = obj_jac_func(parameters + step_size * step_dir)
    # Derivative of step size objective function, is jacobian
    # dot step direction
    step_grad = jac_xk_plus_ap.dot(step_dir)

    return step_obj, step_grad


def _wolfe_conditions(step_size, parameters, obj_xk, jac_xk, step_dir,
                      obj_xk_plus_ap, jac_xk_plus_ap, c_1, c_2):
    """Return True if Wolfe conditions (Armijo rule and curvature condition) are met.

    args:
        step_size: a; Proposed step size.
        parameters: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        obj_xk_plus_ap: f(x_k + a_k p_k); Objective value at x_k + a_k p_k
        jac_xk_plus_ap: grad_f(x_k = a_k p_k); jacobian value at x_k + a_k p_k
        c_1: Strictness parameter for Armijo rule.
        c_2: Strictness parameter for curvature condition.
    """
    if not (0 < c_1 < c_2 < 1):
        raise ValueError('0 < c_1 < c_2 < 1')

    wolfe = (_armijo_rule(step_size, obj_xk, jac_xk, step_dir, obj_xk_plus_ap,
                          c_1)
             and _curvature_condition(jac_xk, step_dir, jac_xk_plus_ap, c_2))
    assert isinstance(
        wolfe,
        (numpy.bool_,
         bool)), '_wolfe_conditions should return bool, check parameters shape'
    return wolfe


def _armijo_rule(step_size, obj_xk, jac_xk, step_dir, obj_xk_plus_ap, c_1):
    """Return True if Armijo rule is met.

    Armijo rule:
    f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)

    Where all vectors all column matrices

    args:
        step_size: a; Proposed step size.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        obj_xk_plus_ap: f(x_k + a_k p_k); Objective value at x_k + a_k p_k
        c_1: Strictness parameter for Armijo rule.
    """
    # NOTE: x.dot(y) == col_matrix(x).T * col_matrix(y)
    return obj_xk_plus_ap <= obj_xk + (c_1 * step_size) * (jac_xk.dot(step_dir))


def _curvature_condition(jac_xk, step_dir, jac_xk_plus_ap, c_2):
    """Return True if curvature condition is met.

    Curvature condition:
    grad_f(x_k + a_k p_k)^T p_k  >= c_2 grad_f(x_k)^T p_k

    Where all vectors all column matrices

    args:
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        jac_xk_plus_ap: grad_f(x_k = a_k p_k); jacobian value at x_k + a_k p_k
        c_2: Strictness parameter for curvature condition.
    """
    # NOTE: x.dot(y) == col_matrix(x).T * col_matrix(y)
    return (jac_xk_plus_ap).dot(step_dir) >= c_2 * (jac_xk.dot(step_dir))
