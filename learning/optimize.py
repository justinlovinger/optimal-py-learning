"""Optimizers for optimizing model parameters."""

import functools
import operator
import itertools
import logging

import numpy

############################
# Problem
############################
class Problem(object):
    """Problem instance for optimizer.

    Functions are prioritized as follows:
        value(s): first priority, second_priority, (first_value_using_best_func, second_func)

        obj: obj_func, obj_jac_func, obj_hess_func, obj_jac_hess_func
        jac: jac_func, obj_jac_func, jac_hess_func, obj_jac_hess_func
        hess: hess_func, obj_hess_func, jac_hess_func, obj_jac_hess_func

        obj_jac: obj_jac_func, obj_jac_hess_func, (obj, jac)
        obj_hess: obj_hess_func, obj_jac_hess_func, (obj, hess)
        jac_hess: jac_hess_func, obj_jac_hess_func, (jac, hess)

        obj_jac_hess: obj_jac_hess_func, (obj_jac_func, hess), (obj_hess_func, jac),
            (obj, jac_hess_func), (obj, jac, hess)
    """
    def __init__(self, obj_func=None, jac_func=None, hess_func=None,
                 obj_jac_func=None, obj_hess_func=None, jac_hess_func=None,
                 obj_jac_hess_func=None):
        # Get objective function
        if obj_func is not None:
            self.get_obj = obj_func
        elif obj_jac_func is not None:
            self.get_obj = functools.partial(_call_return_index, obj_jac_func, 0)
        elif obj_hess_func is not None:
            self.get_obj = functools.partial(_call_return_index, obj_hess_func, 0)
        elif obj_jac_hess_func is not None:
            self.get_obj = functools.partial(_call_return_index, obj_jac_hess_func, 0)
        else:
            self.get_obj = _return_none

        # Get jacobian function
        if jac_func is not None:
            self.get_jac = jac_func
        elif obj_jac_func is not None:
            self.get_jac = functools.partial(_call_return_index, obj_jac_func, 1)
        elif jac_hess_func is not None:
            self.get_jac = functools.partial(_call_return_index, jac_hess_func, 0)
        elif obj_jac_hess_func is not None:
            self.get_jac = functools.partial(_call_return_index, obj_jac_hess_func, 1)
        else:
            self.get_jac = _return_none

        # Get hessian function
        if hess_func is not None:
            self.get_hess = hess_func
        elif obj_hess_func is not None:
            self.get_hess = functools.partial(_call_return_index, obj_hess_func, 1)
        elif jac_hess_func is not None:
            self.get_hess = functools.partial(_call_return_index, jac_hess_func, 1)
        elif obj_jac_hess_func is not None:
            self.get_hess = functools.partial(_call_return_index, obj_jac_hess_func, 2)
        else:
            self.get_hess = _return_none

        # Get objective and jacobian function
        if obj_jac_func is not None:
            self.get_obj_jac = obj_jac_func
        elif obj_jac_hess_func is not None:
            self.get_obj_jac = functools.partial(_call_return_indices, obj_jac_hess_func, (0, 1))
        else:
            self.get_obj_jac = functools.partial(_bundle, (self.get_obj, self.get_jac))

        # Get objective and hessian function
        if obj_hess_func is not None:
            self.get_obj_hess = obj_hess_func
        elif obj_jac_hess_func is not None:
            self.get_obj_hess = functools.partial(_call_return_indices, obj_jac_hess_func, (0, 2))
        else:
            self.get_obj_hess = functools.partial(_bundle, (self.get_obj, self.get_hess))

        # Get jacobian and hessian function
        if jac_hess_func is not None:
            self.get_jac_hess = jac_hess_func
        elif obj_jac_hess_func is not None:
            self.get_jac_hess = functools.partial(_call_return_indices, obj_jac_hess_func, (1, 2))
        else:
            self.get_jac_hess = functools.partial(_bundle, (self.get_jac, self.get_hess))

        # Get objective, jacobian, hessian function
        if obj_jac_hess_func is not None:
            self.get_obj_jac_hess = obj_jac_hess_func
        elif obj_jac_func is not None:
            self.get_obj_jac_hess = functools.partial(
                _bundle_add,
                (obj_jac_func, functools.partial(_tuple_result, self.get_hess))
            )
        elif obj_hess_func is not None:
            self.get_obj_jac_hess = functools.partial(
                _bundle_add_split, obj_hess_func, self.get_jac)
        elif jac_hess_func is not None:
            self.get_obj_jac_hess = functools.partial(
                _bundle_add,
                (functools.partial(_tuple_result, self.get_obj), jac_hess_func)
            )
        else:
            self.get_obj_jac_hess = functools.partial(
                _bundle, (self.get_obj, self.get_jac, self.get_hess))

def _call_return_indices(func, indices, *args, **kwargs):
    """Return indices of func called with *args and **kwargs.

    Use with functools.partial to wrap func.
    """
    values = func(*args, **kwargs)
    return [values[i] for i in indices]

def _call_return_index(func, index, *args, **kwargs):
    """Return index of func called with *args and **kwargs.

    Use with functools.partial to wrap func.
    """
    return func(*args, **kwargs)[index]

def _bundle_add(functions, *args, **kwargs):
    """Return results of functions, concatenated together, called with *args and **kwargs."""
    return reduce(operator.add, _bundle(functions, *args, **kwargs))

def _bundle(functions, *args, **kwargs):
    """Return result of each function, called with *args and **kwargs."""
    return [func(*args, **kwargs) for func in functions]

def _bundle_add_split(f_1, f_2, *args, **kwargs):
    """Return f_1[0], f_2, f_1[1], called with *args, **kwargs."""
    f_1_values = f_1(*args, **kwargs)
    return f_1_values[0], f_2(*args, **kwargs), f_1_values[1]

def _tuple_result(func, *args, **kwargs):
    return func(*args, **kwargs), # , makes tuple

def _return_none(*args, **kwargs):
    """Return None."""
    return None

################################
# Base Models
################################
class Optimizer(object):
    """Optimizer for optimizing model parameters."""

    def __init__(self):
        self.jacobian = None # Last computed jacobian
        self.hessian = None # Last computed hessian

    def reset(self):
        """Reset optimizer parameters."""
        self.jacobian = None
        self.hessian = None

    def next(self, problem, parameters):
        """Return next iteration of this optimizer."""
        raise NotImplementedError()

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
                initial_step_getter=FOChangeInitialStep()
            )
        self._step_size_getter = step_size_getter

    def reset(self):
        """Reset optimizer parameters."""
        super(SteepestDescent, self).__init__()
        self._step_size_getter.reset()

    def next(self, problem, parameters):
        """Return next iteration of this optimizer."""
        obj_value, self.jacobian = problem.get_obj_jac(parameters)

        step_size = self._step_size_getter(parameters, obj_value, self.jacobian,
                                           -self.jacobian, problem)

        # Take a step down the first derivative direction
        return obj_value, parameters - step_size*self.jacobian

class SteepestDescentMomentum(Optimizer):
    """Simple gradient descent with constant step size, and momentum."""
    def __init__(self, step_size_getter=None, momentum_rate=0.2):
        super(SteepestDescentMomentum, self).__init__()
        if step_size_getter is None:
            step_size_getter = WolfeLineSearch(
                initial_step_getter=FOChangeInitialStep()
            )
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

        # Setup step for this iteration (step_size*direction)
        step_dir = -self.jacobian
        step_size = self._step_size_getter(parameters, obj_value, self.jacobian,
                                           step_dir, problem)
        step = step_size*step_dir

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
                initial_step_getter=IncrPrevStep()
            )
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
        approx_inv_hessian = self._get_approx_inv_hessian(parameters, self.jacobian)

        step_dir = -(approx_inv_hessian.dot(self.jacobian))

        step_size = self._step_size_getter(parameters, obj_value, self.jacobian,
                                           step_dir, problem)

        return obj_value, parameters + step_size*step_dir

    def _get_approx_inv_hessian(self, parameters, jacobian):
        """Calculate approx inv hessian for this iteration, and return it."""
        if self._prev_inv_hessian is None:
            # If first iteration, default to identity for approx inv hessian
            H_kp1 = numpy.identity(parameters.shape[0])
        else:
            H_kp1 = _bfgs_eq(
                self._prev_inv_hessian,
                parameters - self._prev_params,
                jacobian - self._prev_jacobian
            )

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
    # import pdb; pdb.set_trace()
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
                incr_rate=1.0/self._decr_rate+0.05, upper_bound=None)
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
        initial_step = self._initial_step_getter(xk, obj_xk, jac_xk, step_dir, problem)

        step_size = _backtracking_line_search(
            xk, obj_xk, jac_xk, step_dir, problem.get_obj, self._c_1,
            initial_step, decr_rate=self._decr_rate)

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
            # Slightly more than 1 step up
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
        initial_step = self._initial_step_getter(xk, obj_xk, jac_xk, step_dir, problem)

        step_size = _line_search_wolfe(
            xk, obj_xk, jac_xk, step_dir, problem.get_obj_jac, self._c_1, self._c_2,
            initial_step)

        self._initial_step_getter.update(step_size)
        return step_size


def _backtracking_line_search(parameters, obj_xk, jac_xk, step_dir, obj_func, c_1,
                              initial_step, decr_rate=0.9):
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
        logging.warning('nan objective value in _backtracking_line_search, defaulting to 1e-10 step size')
        return 1e-10

    step_size = initial_step
    for i in itertools.count(start=1):
        if step_size < 1e-10:
            # Failsafe for numerical precision errors preventing _armijo_rule returning True
            # This can happen if gradient provides very little improvement
            # (or is in the wrong direction)
            logging.warning('_backtracking_line_search failed Armijo with step_size ~= 1e-10, returning')
            return step_size

        obj_xk_plus_ap = obj_func(parameters + step_size*step_dir)
        if _armijo_rule(step_size, obj_xk, jac_xk, step_dir, obj_xk_plus_ap, c_1):
            assert step_size > 0
            return step_size

        # Did not satisfy, decrease step size and try again
        step_size *= decr_rate

WOLFE_INCR_RATE = 1.5
def _line_search_wolfe(parameters, obj_xk, jac_xk, step_dir, obj_jac_func, c_1, c_2,
                       initial_step):
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
        logging.warning('nan objective value in _line_search_wolfe, defaulting to 1e-10 step size')
        return 1e-10

    step_zero_obj = obj_xk
    step_zero_grad = jac_xk.dot(step_dir)

    # We need the current and previous step size for some operations
    prev_step_size = 0.0
    prev_step_obj = step_zero_obj
    prev_step_grad = step_zero_grad

    step_size = initial_step
    for i in itertools.count(start=1):
        if i >= 100:
            # Failsafe for numerical precision errors preventing convergence
            # This can happen if gradient provides very little improvement
            # (or is in the wrong direction)
            logging.warning('Wolfe line search aborting after 100 iterations')
            return step_size

        # Evaluate objective and jacobian for most recent step size
        step_obj, step_grad = _step_size_obj_jac_func(
            step_size, parameters, step_dir, obj_jac_func)

        # True if objective did not improve (step_obj >= prev_step_obj), after first iterations,
        # or armijo condition is False (step_obj > obj_xk + c_1*step_size*step_grad)
        if ((i > 1 and step_obj >= prev_step_obj)
                or (step_obj > obj_xk + c_1*step_size*step_grad)):
            return _zoom_wolfe(prev_step_size, prev_step_obj, step_size, parameters,
                               obj_xk, step_zero_grad, step_dir, obj_jac_func, c_1, c_2)

        # Check if step size is already an acceptable step length
        # True when gradient is sufficiently small (magnitude wise)
        elif numpy.abs(step_grad) <= -c_2 * step_zero_grad:
            return step_size

        # If objective value did not improve (first if statement)
        # and step size needs to increase (non-negative gradient)
        elif step_grad >= 0:
            return _zoom_wolfe(step_size, step_obj, prev_step_size, parameters,
                               obj_xk, step_zero_grad, step_dir, obj_jac_func, c_1, c_2)

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
                step_zero_obj, step_zero_grad, step_dir, obj_jac_func, c_1, c_2):
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
            max(step_size_low, step_size_high)
        )
        assert step_size >= 0

        if i >= 100:
            # Failsafe for numerical precision errors preventing convergence
            # This can happen if gradient provides very little improvement
            # (or is in the wrong direction)
            logging.warning('Wolfe line search (zoom) aborting after 100 iterations')
            return step_size

        step_obj, step_grad = _step_size_obj_jac_func(
            step_size, parameters, step_dir, obj_jac_func)

        # If this step is worse, than the projection from initial parameters
        # or this step is worse than the current high (bad) step size
        if (step_obj > step_zero_obj + c_1*step_size*step_zero_grad
                or step_obj >= step_size_low_obj):
            # step_size is not an improvement
            # This step size is the new poor valued side of the interval
            step_size_high = step_size

        # step_size is an improvement
        else:
            # If this step size caused an improvement
            # (first if statement is false),
            # and step size gradient is sufficiently small (magnitude wise)
            if numpy.abs(step_grad) <= -c_2*step_zero_grad:
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
    return min_ + 0.5*(max_ - min_)

def _step_size_obj_jac_func(step_size, parameters, step_dir, obj_jac_func):
    """Return objective value and gradient for step size."""
    step_obj, jac_xk_plus_ap = obj_jac_func(parameters + step_size*step_dir)
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

    wolfe = (_armijo_rule(step_size, obj_xk, jac_xk, step_dir, obj_xk_plus_ap, c_1)
             and _curvature_condition(jac_xk, step_dir, jac_xk_plus_ap, c_2))
    assert isinstance(wolfe, (numpy.bool_, bool)), '_wolfe_conditions should return bool, check parameters shape'
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
    return obj_xk_plus_ap <= obj_xk + (c_1*step_size)*(jac_xk.dot(step_dir))

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
    return (jac_xk_plus_ap).dot(step_dir) >= c_2*(jac_xk.dot(step_dir))

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
        initial_step = self._incr_rate*self._prev_step_size
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
                logging.warning('jac_dot_dir == 0 in FOChangeInitialStep, defaulting to 1')
                return 1.0

            initial_step = self._prev_step_size * (self._prev_jac_dot_dir / jac_dot_dir)

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
                logging.warning('jac_dot_dir == 0 in QuadraticInitialStep, defaulting to 1')
                return 1.0

            initial_step = ((2.0 * (obj_xk - self._prev_obj_value))
                            / jac_dot_dir)

        # For next iteration
        self._prev_obj_value = obj_xk

        if numpy.isnan(initial_step):
            logging.warning('nan in objective of jacobian, in QuadraticInitialStep call, '\
                            'returning 1e-10')
            return 1e-10

        return initial_step
