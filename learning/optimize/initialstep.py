"""Strategies for determining the initial step size before searching for best step size."""

import logging

import numpy


##########################
# Base Model
##########################
class InitialStepGetter(object):
    """Returns initial step size when called.

    Used by StepSizeGetter.
    """

    def reset(self):
        """Reset parameters."""
        pass

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return initial step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        raise NotImplementedError()

    def update(self, step_size):
        """Update parameters

        Optional. Some methods require remembering previously discovered step sizes.
        """
        pass


####################################
# InitialStepGetter implementations
####################################
class IncrPrevStep(InitialStepGetter):
    """Return initial step size by incrementing previous best.

    Effective for optimizers that converge superlinearly
    (such as Newton and quasi-newton (TODO: Confirm)),
    where 1 will eventually always be accepted.

    NOTE: Set upper_bound to None if using BacktrackingLineSearch
    with an optimizer that may requires step_size > 1,
    such as quasi-Newton methods.
    """

    def __init__(self, incr_rate=1.05, upper_bound=1.0):
        if incr_rate < 1.0:
            raise ValueError('incr_rate > 1 to increment')

        if upper_bound is not None and upper_bound < 0:
            raise ValueError('upper_bound must be positive')

        self._incr_rate = incr_rate
        self._upper_bound = upper_bound

        self._prev_step_size = 1.0

    def reset(self):
        """Reset parameters."""
        self._prev_step_size = 1.0

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return initial step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        initial_step = self._incr_rate * self._prev_step_size
        if self._upper_bound is not None:
            return min(self._upper_bound, initial_step)
        else:
            return initial_step

    def update(self, step_size):
        """Update parameters."""
        self._prev_step_size = step_size


class FOChangeInitialStep(InitialStepGetter):
    """Return initial step by assuming first order change is same as previous iteration.

    Popular for methods that do not produce well scaled search directions,
    such as steepest descent and conjugate gradient optimizers.

    alpha_0 = alpha_{k-1} (grad_f_{k-1}^T p_{k-1} / grad_f_k^T p_k)
    where alpha_0 is the initial step.

    Note that vectors are column matrices.
    """

    def __init__(self):
        self._prev_step_size = None
        self._prev_jac_dot_dir = None

    def reset(self):
        """Reset parameters."""
        self._prev_step_size = None
        self._prev_jac_dot_dir = None

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return initial step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        jac_dot_dir = jac_xk.dot(step_dir)

        if self._prev_step_size is None:
            # Default to 1 for first iteration
            initial_step = 1.0
        else:
            # Failsafe for divide by 0
            if jac_dot_dir == 0.0:
                logging.warning(
                    'jac_dot_dir == 0 in FOChangeInitialStep, defaulting to 1')
                return 1.0

            initial_step = self._prev_step_size * (
                self._prev_jac_dot_dir / jac_dot_dir)

        # For next iteration
        self._prev_jac_dot_dir = jac_dot_dir

        return initial_step

    def update(self, step_size):
        """Update parameters."""
        self._prev_step_size = step_size


class QuadraticInitialStep(InitialStepGetter):
    """Return initial step by interpolating a quadratic between prev and current obj value.

    Get initial step by minimizing a quadratic interpolated to
    previous objective value, current objective value, and derivative of step obj.

    alpha_0 = (2 (f_k - f_{k-1})) / phi'(0)
    where alpha_0 is the initial step,
    phi'(0) = grad_f_k^T step_dir

    Note that vectors are column matrices.
    """

    def __init__(self):
        self._prev_obj_value = None

    def reset(self):
        """Reset parameters."""
        self._prev_obj_value = None

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return initial step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        if self._prev_obj_value is None:
            # Default to 1 for first iteration
            initial_step = 1.0
        else:
            jac_dot_dir = jac_xk.dot(step_dir)
            # Failsafe for divide by 0
            if jac_dot_dir == 0.0:
                logging.warning(
                    'jac_dot_dir == 0 in QuadraticInitialStep, defaulting to 1'
                )
                return 1.0

            initial_step = ((2.0 *
                             (obj_xk - self._prev_obj_value)) / jac_dot_dir)

        # For next iteration
        self._prev_obj_value = obj_xk

        if numpy.isnan(initial_step):
            logging.warning('nan in objective of jacobian, in QuadraticInitialStep call, '\
                            'returning 1e-10')
            return 1e-10

        return initial_step
