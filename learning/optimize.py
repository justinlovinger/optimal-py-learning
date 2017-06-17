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

class GetStepSize(object):
    """Returns step size when called."""
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

################################
# Optimizer Implementations
################################
class SteepestDescent(Optimizer):
    """Simple steepest descent with constant step size."""
    def __init__(self, step_size_getter=None):
        super(SteepestDescent, self).__init__()

        if step_size_getter is None:
            # TODO: Backtracking armijo is not considered best for steepest descent
            # "This simple and popular strategy for terminating a line search is well suited
            # for Newton methods but is less appropriate for quasi-Newton and
            # conjugate gradient methods." ~ Numerical Optimization (2nd) pp. 56
            step_size_getter = BacktrackingStepSize()
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
            # TODO: Backtracking armijo is not considered best for steepest descent
            # "This simple and popular strategy for terminating a line search is well suited
            # for Newton methods but is less appropriate for quasi-Newton and
            # conjugate gradient methods." ~ Numerical Optimization (2nd) pp. 56
            step_size_getter = BacktrackingStepSize()
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
            # TODO: Backtracking armijo is not considered best for steepest descent
            # "This simple and popular strategy for terminating a line search is well suited
            # for Newton methods but is less appropriate for quasi-Newton and
            # conjugate gradient methods." ~ Numerical Optimization (2nd) pp. 56
            #
            # "The performance of the BFGS method can degrade if the line search
            # is not based on the Wolfe conditions.
            # For example, some software implements an Armijo backtracking line search
            # (see Section 3.1): The unit step length ak = 1 is tried first
            # and is successively decreased until the sufficient decrease
            # condition (3.6a) is satisfied. For this strategy,
            # there is no guarantee that the curvature condition y_k^T s_k > 0 (6.7)
            # will be satisfied by the chosen step,
            # since a step length greater than 1 may be required to satisfy this condition"
            #  ~ Numerical Optimization (2nd) pp. 143
            step_size_getter = BacktrackingStepSize()
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

    p_k = 1.0 / (y_k.dot(s_k)) # y_k.dot(s_k) == y_k.dot(s_k[:, None])

    p_k_times_s_k = p_k * s_k
    return (
        (I - p_k_times_s_k[:, None] * y_k)
        .dot(H_k)
        .dot(I - (p_k * y_k)[:, None] * (s_k))
        + (p_k_times_s_k[:, None] * s_k)
    )


###############################
# GetStepSize implementations
###############################
class SetStepSize(GetStepSize):
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

class BacktrackingStepSize(GetStepSize):
    """Return step size found with backtracking line search."""
    def __init__(self, c_1=1e-4, decr_rate=0.9):
        super(BacktrackingStepSize, self).__init__()

        self._c_1 = c_1
        self._decr_rate = decr_rate

        self._prev_step_size = 1.0

    def reset(self):
        """Reset parameters."""
        super(BacktrackingStepSize, self).reset()
        self._prev_step_size = 1.0

    def __call__(self, xk, obj_xk, jac_xk, step_dir, problem):
        """Return step size.

        xk: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        problem: Problem; Problem instance passed to Optimizer
        """
        # Initialize to one step up from previously best step size
        step_size = _backtracking_line_search(
            xk, obj_xk, jac_xk, step_dir, problem.get_obj, self._c_1,
            (self._prev_step_size/self._decr_rate), decr_rate=self._decr_rate)
        self._prev_step_size = step_size
        return step_size

def _optimize_until_wolfe(parameters, obj_xk, jac_xk, step_dir, obj_jac_func, c_1, c_2):
    """Return step size that satisfies wolfe conditions.

    Discover step size by metaheuristic optimization.

    NOTE: Experimental, and quite slow. Should not be used.

    args:
        parameters: x_k; Parameter values at current step.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        obj_jac_func: Function taking parameters and returning obj and jac at given parameters.
        c_1: Strictness parameter for Armijo rule.
        c_2: Strictness parameter for curvature condition.
    """
    import optimal

    MIN_STEP = 1e-10
    MAX_STEP = 10.0
    def decode(encoded_solution):
        return optimal.helpers.binary_to_float(encoded_solution, MIN_STEP, MAX_STEP)

    def fitness(step_size):
        # Fitness is distance from 0 derivative
        # NOTE: Should it just be -obj?
        obj_xk_plus_ap, jac_xk_plus_ap = obj_jac_func(parameters + step_size*step_dir)

        # Finished when wolfe conditions are met
        finished = _wolfe_conditions(step_size, parameters, obj_xk, jac_xk, step_dir,
                                     obj_xk_plus_ap, jac_xk_plus_ap, c_1, c_2)
        return -numpy.abs(jac_xk_plus_ap.T.dot(step_dir)), finished

    optimizer = optimal.GenAlg(13, population_size=4)
    optimizer.logging = False
    return optimizer.optimize(optimal.Problem(fitness, decode_function=decode))


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

def _line_search_wolfe(parameters, obj_xk, jac_xk, step_dir, obj_jac_func, c_1, c_2,
                       initial_step=1.0):
    """Return step size that satisfies wolfe conditions.

    Discover step size with gradient descent.

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
        # Failsafe because _armijo rule will never return True
        # TODO: Might need similar failsafe for nan in jac_xk or step_dir
        logging.warning('nan objective value in _line_search_wolfe, defaulting to 1e-10 step size')
        return 1e-10

    step_size = initial_step

    # Use gradient descent to find step size that satisfies wolfe conditions
    # TODO: Optimizer must have constraint that step_size > 0
    optimizer = SteepestDescent(step_size_getter=SetStepSize(0.1))
    # Used to extract jac_xk_plus_ap that is computed during step_size_obj_jac_func,
    # because it is also needed for _wolfe_conditions
    jac_xk_plus_ap = [None]
    def step_size_obj_jac_func(a):
        # NOTE: Note that we add step_dir, not subtract, ex. -jacobian is a step direction
        obj_val, jac_xk_plus_ap[0] = obj_jac_func(parameters + a*step_dir)
        return obj_val, jac_xk_plus_ap[0].T.dot(step_dir)
    problem = Problem(obj_jac_func=step_size_obj_jac_func)

    # Optimize step size until it matches wolfe conditions
    # obj_xk_plus_ap and jac_xk_plus_ap[0] is are for the step size when optimizer.next is called
    # (not the new one returned by optimizer.next)
    # That is why we store next_step_size, and set it equal to step_size before calling optimizer.next
    iteration = 1
    obj_xk_plus_ap, next_step_size = optimizer.next(problem, step_size)
    while not _wolfe_conditions(step_size, parameters, obj_xk, jac_xk, step_dir,
                                obj_xk_plus_ap, jac_xk_plus_ap[0], c_1, c_2):
        step_size = next_step_size
        obj_xk_plus_ap, next_step_size = optimizer.next(problem, step_size)

        iteration += 1
        if iteration > 1000:
            raise RuntimeError('Line search did not converge')

    assert step_size > 0
    return step_size

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

    args:
        step_size: a; Proposed step size.
        obj_xk: f(x_k); Objective value at x_k.
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        obj_xk_plus_ap: f(x_k + a_k p_k); Objective value at x_k + a_k p_k
        c_1: Strictness parameter for Armijo rule.
    """
    return obj_xk_plus_ap <= obj_xk + ((c_1*step_size)*jac_xk.T).dot(step_dir)

def _curvature_condition(jac_xk, step_dir, jac_xk_plus_ap, c_2):
    """Return True if curvature condition is met.

    Curvature condition:
    grad_f(x_k + a_k p_k)^T p_k  >= c_2 grad_f(x_k)^T p_k

    args:
        jac_xk: grad_f(x_k); First derivative (jacobian) at x_k.
        step_dir: p_k; Step direction (ex. jacobian in steepest descent) at x_k.
        jac_xk_plus_ap: grad_f(x_k = a_k p_k); jacobian value at x_k + a_k p_k
        c_2: Strictness parameter for curvature condition.
    """
    return (jac_xk_plus_ap.T).dot(step_dir) >= (c_2*jac_xk.T).dot(step_dir)
