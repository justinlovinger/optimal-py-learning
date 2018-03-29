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

import copy
import collections
import random
import operator

import numpy

from learning import Model
from learning.architecture import pbnn


class EmptyModel(Model):
    def activate(self, inputs):
        pass

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        output = self.activate(input_vec)
        if output is not None:
            return numpy.mean((target_vec - output)**2)


class SetOutputModel(EmptyModel):
    def __init__(self, output):
        super(SetOutputModel, self).__init__()

        self.output = numpy.array(output)

    def activate(self, inputs):
        return self.output


class ManySetOutputsModel(EmptyModel):
    def __init__(self, outputs):
        super(ManySetOutputsModel, self).__init__()

        self.outputs = [numpy.array(output) for output in outputs]

    def activate(self, inputs):
        return self.outputs.pop(0)


class RememberPatternsModel(EmptyModel):
    """Returns the output for a given input."""

    def __init__(self):
        super(RememberPatternsModel, self).__init__()

        self._inputs_output_dict = {}

    def reset(self):
        self._inputs_output_dict = {}

    def train(self, input_matrix, target_matrix, *args, **kwargs):
        for input_vec, target_vec in zip(input_matrix, target_matrix):
            self._inputs_output_dict[tuple(input_vec)] = numpy.array(
                target_vec)

    def activate(self, inputs):
        return numpy.array(self._inputs_output_dict[tuple(inputs)])


class SummationModel(EmptyModel):
    def activate(self, inputs):
        return numpy.sum(inputs, axis=1)


class WeightedSumModel(Model):
    """Model that returns stored targets weighted by inputs."""

    def __init__(self):
        super(WeightedSumModel, self).__init__()
        self._stored_targets = None

    def reset(self):
        super(WeightedSumModel, self).reset()

        self._stored_targets = None

    def activate(self, inputs):
        return pbnn._weighted_sum_rows(self._stored_targets,
                                       numpy.array(inputs))

    def train(self, input_matrix, target_matrix, *args, **kwargs):
        self._stored_targets = numpy.copy(target_matrix)


def approx_equal(a, b, tol=0.001):
    """Check if two numbers or lists are about the same.

    Useful to correct for floating point errors.
    """
    if isinstance(a, numpy.ndarray):
        a = list(a)
    if isinstance(b, numpy.ndarray):
        b = list(b)

    if isinstance(b, (list, tuple)) and not isinstance(a, (list, tuple)):
        return False

    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)):
            return False

        # Check that each element is approx equal
        if len(a) != len(b):
            return False

        for a_, b_ in zip(a, b):
            if not approx_equal(a_, b_, tol):
                return False

        return True
    else:
        return _approx_equal(a, b, tol)


def _approx_equal(a, b, tol=0.001):
    """Check if two numbers are about the same.

    Useful to correct for floating point errors.
    """
    return abs(a - b) < tol


class SaneEqualityArray(numpy.ndarray):
    """Numpy array with working == operator."""

    def __eq__(self, other):
        return (isinstance(other, numpy.ndarray) and self.shape == other.shape
                and numpy.array_equal(self, other))


def fix_numpy_array_equality(iterable):
    if isinstance(iterable, numpy.ndarray):
        return sane_equality_array(iterable)

    if isinstance(iterable,
                  str) or not isinstance(iterable, collections.Iterable):
        # Not iterable
        return iterable

    new_iterable = copy.deepcopy(iterable)

    for i, item in enumerate(new_iterable):
        if isinstance(new_iterable, dict):
            key = item
            item = new_iterable[key]

        # Recurse
        if isinstance(new_iterable, tuple):
            # Tuples do not directly support assignment
            new_iterable = _change_tuple(new_iterable, i,
                                         fix_numpy_array_equality(item))
        elif isinstance(new_iterable, dict):
            new_iterable[key] = fix_numpy_array_equality(item)
        else:
            new_iterable[i] = fix_numpy_array_equality(item)

    return new_iterable


def sane_equality_array(object):
    array = numpy.array(object)
    return SaneEqualityArray(array.shape, array.dtype, array)


def _change_tuple(tuple_, i, new_item):
    list_ = list(tuple_)
    list_[i] = new_item
    return tuple(list_)


def equal_ignore_order(a, b):
    """Check if two lists contain the same elements.

    Note: Use only when elements are neither hashable nor sortable!
    """
    # Fix numpy arrays
    copy_a = fix_numpy_array_equality(a)
    copy_b = fix_numpy_array_equality(b)

    # Check equality
    for element in copy_a:
        try:
            copy_b.remove(element)
        except ValueError:
            return False

    return not copy_b


############################
# Gradient checking
############################
# Numerical precision error of IEEE double precision floating-point numbers
# On forward difference gradient approximation, sqrt(ieee_double_error) is theoretically optimal.
# On central different gradient approximation, cube_root(ieee_double_error) is theoretically optimal.
# ~Numerical Optimization 2nd, pp. 196-197
IEEE_DOUBLE_ERROR = 1.1e-16  # u
FORWARD_DIFF_EPSILON = IEEE_DOUBLE_ERROR**0.5  # sqrt(u)
CENTRAL_DIFF_EPSILON = IEEE_DOUBLE_ERROR**(1. / 3) # cube_root(u)


def check_gradient(f, df, f_arg_tensor=None, epsilon=CENTRAL_DIFF_EPSILON, f_shape='scalar'):
    if f_arg_tensor is None:
        f_arg_tensor = numpy.random.rand(random.randint(2, 10))

    if f_shape == 'scalar':
        approx_func = _approximate_gradient_scalar
    elif f_shape == 'lin':  # TODO: Change to diag
        approx_func = _approximate_gradient_lin
    elif f_shape == 'jac':
        approx_func = _approximate_gradient_jac
    elif f_shape == 'jac-stack':
        approx_func = _approximate_gradient_jac_stack
    else:
        raise ValueError(
            "Invalid f_shape. Must be one of ('scalar', 'lin', 'jac').")

    try:
        assert approx_equal(
            df(f_arg_tensor),
            approx_func(f, f_arg_tensor, epsilon),
            tol=10 * epsilon)
    except AssertionError:
        actual = df(f_arg_tensor)
        expected = approx_func(f, f_arg_tensor, epsilon)
        print 'Actual Gradient:'
        print actual
        print 'Expected Gradient:'
        print expected
        if actual.shape != expected.shape:
            print 'Actual shape %s != expected shape %s' % (
                actual.shape, expected.shape)
        raise


def _approximate_gradient_scalar(f, x, epsilon):
    return numpy.array([
        _approximate_ith(i, f, x, epsilon)
        for i in range(reduce(operator.mul, x.shape))
    ]).reshape(x.shape)


def _approximate_gradient_lin(f, x, epsilon):
    return numpy.array([
        _approximate_ith(i, f, x, epsilon).ravel()[i]
        for i in range(reduce(operator.mul, x.shape))
    ]).reshape(x.shape)


def _approximate_ith(i, f, x, epsilon):
    # Ravel in case x is not a vector
    x_plus_i = x.copy().ravel()
    x_plus_i[i] += epsilon
    x_plus_i = x_plus_i.reshape(x.shape)

    x_minus_i = x.copy().ravel()
    x_minus_i[i] -= epsilon
    x_minus_i = x_minus_i.reshape(x.shape)

    return (f(x_plus_i) - f(x_minus_i)) / (2 * epsilon)


def _approximate_gradient_jac_stack(f, x, epsilon):
    """Return stack of Jacobians between each row of x and f(x)."""
    if len(x.shape) != 2 or len(f(x).shape) != 2:
        raise ValueError('Approximate Jacobian stack only defined for matrix input and output.')

    jacobian = _approximate_gradient_jac(f, x, epsilon)
    # Grab only the i^th second row of each row
    # When each row of x is independent, the off second rows will be 0s
    return numpy.array([jacobian[i, :, i, :] for i in range(x.shape[0])])


def _approximate_gradient_jac(f, x, epsilon):
    fx_shape = f(x).shape
    outs = reduce(operator.mul, fx_shape)

    # Jacobian has inputs on cols and outputs on rows
    jacobian = numpy.zeros((outs, reduce(operator.mul, x.shape)))
    for j in range(outs):
        for i in range(reduce(operator.mul, x.shape)):
            # Ravel in case x is not a vector
            x_plus_i = x.copy().ravel()
            x_plus_i[i] += epsilon
            x_plus_i = x_plus_i.reshape(x.shape)

            x_minus_i = x.copy().ravel()
            x_minus_i[i] -= epsilon
            x_minus_i = x_minus_i.reshape(x.shape)

            jacobian[j, i] = (f(x_plus_i).ravel()[j] - f(x_minus_i).ravel()[j]) / (
                2.0 * epsilon)
    return jacobian.reshape(fx_shape + x.shape)


# TODO: Add random tensor function that makes a random tensor of given shape (with negative and posative values),
# and use that instead of numpy.random.random.
# Because numpy.random.random only gives values in the range [0, 1]
