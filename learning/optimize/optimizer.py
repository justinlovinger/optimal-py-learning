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
"""Numerical optimization methods."""

import logging

import numpy

from learning.optimize import WolfeLineSearch
from learning.optimize import IncrPrevStep, FOChangeInitialStep

JACOBIAN_NORM_BREAK = 1e-10


################################
# Base Model
################################
class Optimizer(object):
    """Optimizer for optimizing model parameters."""

    def __init__(self):
        self.jacobian = None  # Last computed jacobian
        self.hessian = None  # Last computed hessian

    def reset(self):
        """Reset optimizer parameters."""
        self.jacobian = None
        self.hessian = None

    def next(self, problem, parameters):
        """Return next iteration of this optimizer."""
        raise NotImplementedError()


################################
# Optimizer Implementations
################################
# TODO: Optimize optimizers by re-using objective and jacobians calculated
# during line searches. Such as by caching these values, and checking cache before
# calculating (Problem object can cache these (by parameters), but we need to make sure
# model do not make another instance of an Problem if the problem is the same)
class SteepestDescent(Optimizer):
    """Simple steepest descent with constant step size."""

    def __init__(self, step_size_getter=None):
        super(SteepestDescent, self).__init__()

        if step_size_getter is None:
            step_size_getter = WolfeLineSearch(
                initial_step_getter=FOChangeInitialStep())
        self._step_size_getter = step_size_getter

    def reset(self):
        """Reset optimizer parameters."""
        super(SteepestDescent, self).__init__()
        self._step_size_getter.reset()

    def next(self, problem, parameters):
        """Return next iteration of this optimizer."""
        obj_value, self.jacobian = problem.get_obj_jac(parameters)

        if numpy.linalg.norm(self.jacobian) < JACOBIAN_NORM_BREAK:
            logging.info('Optimizer converged with small jacobian')
            return obj_value, parameters

        step_size = self._step_size_getter(
            parameters, obj_value, self.jacobian, -self.jacobian, problem)

        # Take a step down the first derivative direction
        return obj_value, parameters - step_size * self.jacobian


class SteepestDescentMomentum(Optimizer):
    """Simple gradient descent with constant step size, and momentum."""

    def __init__(self, step_size_getter=None, momentum_rate=0.2):
        super(SteepestDescentMomentum, self).__init__()
        if step_size_getter is None:
            step_size_getter = WolfeLineSearch(
                initial_step_getter=FOChangeInitialStep())
        self._step_size_getter = step_size_getter

        self._momentum_rate = momentum_rate

        # Store previous step (step_size*direction) for momentum
        self._prev_step = None

    def reset(self):
        """Reset optimizer parameters."""
        super(SteepestDescentMomentum, self).reset()
        self._step_size_getter.reset()
        self._prev_step = None

    def next(self, problem, parameters):
        """Return next iteration of this optimizer."""
        obj_value, self.jacobian = problem.get_obj_jac(parameters)

        if numpy.linalg.norm(self.jacobian) < JACOBIAN_NORM_BREAK:
            logging.info('Optimizer converged with small jacobian')
            return obj_value, parameters

        # Setup step for this iteration (step_size*direction)
        # TODO: step_dir for this iteration should be -self.jacobian - self._momentum_rate*self._prev_jacobian
        # instead of adding self._momentum_rate * self._prev_step after this step
        # This allow the step size finder to account for momentum, and more accurately and efficiently converge
        step_dir = -self.jacobian
        step_size = self._step_size_getter(parameters, obj_value,
                                           self.jacobian, step_dir, problem)
        step = step_size * step_dir

        # Add steps from this and previous iteration
        next_parameters = parameters + step
        if self._prev_step is not None:
            next_parameters += self._momentum_rate * self._prev_step
        self._prev_step = step

        # Take a step down the first derivative direction
        return obj_value, next_parameters


class BFGS(Optimizer):
    """Quasi-Newton BFGS optimizer.

    Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    Ref: Numerical Optimization pp. 136
    """

    def __init__(self, step_size_getter=None):
        super(BFGS, self).__init__()

        if step_size_getter is None:
            step_size_getter = WolfeLineSearch(
                initial_step_getter=IncrPrevStep())
        self._step_size_getter = step_size_getter

        # BFGS Parameters
        self._prev_params = None
        self._prev_jacobian = None
        self._prev_inv_hessian = None

    def reset(self):
        """Reset optimizer parameters."""
        super(BFGS, self).reset()
        self._step_size_getter.reset()

        # Reset BFGS Parameters
        self._prev_params = None
        self._prev_jacobian = None
        self._prev_inv_hessian = None

    def next(self, problem, parameters):
        """Return next iteration of this optimizer."""
        obj_value, self.jacobian = problem.get_obj_jac(parameters)

        if numpy.linalg.norm(self.jacobian) < JACOBIAN_NORM_BREAK:
            logging.info('Optimizer converged with small jacobian')
            return obj_value, parameters

        approx_inv_hessian = self._get_approx_inv_hessian(
            parameters, self.jacobian)

        step_dir = -(approx_inv_hessian.dot(self.jacobian))

        step_size = self._step_size_getter(parameters, obj_value,
                                           self.jacobian, step_dir, problem)

        return obj_value, parameters + step_size * step_dir

    def _get_approx_inv_hessian(self, parameters, jacobian):
        """Calculate approx inv hessian for this iteration, and return it."""
        if self._prev_inv_hessian is None:
            # If first iteration, default to identity for approx inv hessian
            H_kp1 = numpy.identity(parameters.shape[0])
        else:
            H_kp1 = _bfgs_eq(self._prev_inv_hessian,
                             parameters - self._prev_params,
                             jacobian - self._prev_jacobian)

        # Save values from current iteration for next iteration
        self._prev_inv_hessian = H_kp1
        self._prev_params = parameters
        self._prev_jacobian = jacobian

        return H_kp1

def _bfgs_eq(H_k, s_k, y_k):
    """Apply the bfgs update rule to obtain the next approx inverse hessian.

    H_{k+1} = (I - p_k s_k y_k^T) H_k (I - p_k y_k s_k^T) + p_k s_k s_k^T
    where
    s_k = x_{k+1} - x_k (x = parameters)
    y_k = jac_f_{k+1} - jac_f_k
    p_k = 1 / (y_k^T s_k)

    Note that all vectors are column vectors (so vec.T is a row vector)

    Note that the current iteration is k+1, and k is the previous iteration.
    However s_k and y_k correspond to he current iteration (and previous).
    """
    # An implementation very close to the original, using matrices, and column matrices:
    # I = numpy.matrix(I)

    # H_k = numpy.matrix(H_k)
    # s_k = numpy.matrix(s_k).T
    # y_k = numpy.matrix(y_k).T

    # p_k = float(1.0 / (y_k.T * s_k))

    # p_k_times_s_k = p_k * s_k
    # return numpy.array(
    #     (I - p_k_times_s_k * y_k.T)
    #     * H_k
    #     * (I - p_k * y_k * s_k.T)
    #     + (p_k_times_s_k * s_k.T)
    # )

    # More efficient implementation with arrays and fast [:, None] transposes
    # Vectors are row vectors (1d, as given)
    I = numpy.identity(s_k.shape[0])

    # Calculate p_k with failsafe for divide by zero errors
    y_k_dot_s_k = y_k.dot(s_k) # y_k.dot(s_k) == y_k.dot(s_k[:, None])
    # Failsafe for divide by zero errors
    # y_k and s_k are change in jacobian and parameters respectively
    # If these values did not change, we can re-use previous inv hessian
    if y_k_dot_s_k == 0.0:
        return H_k
    p_k = 1.0 / y_k_dot_s_k

    p_k_times_s_k = p_k * s_k
    return (
        (I - p_k_times_s_k[:, None] * y_k)
        .dot(H_k)
        .dot(I - (p_k * y_k)[:, None] * (s_k))
        + (p_k_times_s_k[:, None] * s_k)
    )
