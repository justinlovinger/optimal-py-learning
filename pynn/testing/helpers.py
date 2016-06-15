import copy

import numpy

from pynn import network

class EmptyLayer(network.Layer):
    def activate(self, inputs):
        pass

    def reset(self):
        pass

    def get_prev_errors(self, all_inputs, all_errors, outputs):
        pass

    def update(self, inputs, outputs, errors):
        pass

class SetOutputLayer(EmptyLayer):
    def __init__(self, output):
        super(SetOutputLayer, self).__init__()

        self.output = numpy.array(output)

    def activate(self, inputs):
        return self.output

class RememberPatternsLayer(EmptyLayer):
    """Returns the output for a given input."""
    def __init__(self):
        super(RememberPatternsLayer, self).__init__()

        self._inputs_output_dict = {}

    def pre_training(self, patterns):
        for (inputs, targets) in patterns:
            self._inputs_output_dict[tuple(inputs)] = numpy.array(targets)

    def activate(self, inputs):
        return numpy.array(self._inputs_output_dict[tuple(inputs)])

    def reset(self):
        self._inputs_output_dict = {}

class SummationLayer(network.Layer):
    def activate(self, inputs):
        return numpy.sum(inputs, axis=1)

def approx_equal(a, b, tol=0.001):
    """Check if two numbers or lists are about the same.

    Useful to correct for floating point errors.
    """
    if isinstance(a, (list, tuple)):
        # Check that each element is approx equal
        if len(a) != len(b):
            return False

        for a_, b_ in zip(a, b):
            if not _approx_equal(a_, b_, tol):
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
        return (isinstance(other, numpy.ndarray) and self.shape == other.shape and 
            numpy.array_equal(self, other))

def sane_equality_array(object):
    array = numpy.array(object)
    return SaneEqualityArray(array.shape, array.dtype, array)

def _change_tuple(tuple_, i, new_item):
    list_ = list(tuple_)
    list_[i] = new_item
    return tuple(list_)

def fix_numpy_array_equality(iterable):
    new_iterable = copy.deepcopy(iterable)

    for i, item in enumerate(new_iterable):
        if isinstance(item, numpy.ndarray):
            if isinstance(new_iterable, tuple):
                # Tuples do not directly support assignment
                new_iterable = _change_tuple(new_iterable, i, sane_equality_array(item))
            else:
                new_iterable[i] = sane_equality_array(item)
        # Recurse
        elif hasattr(item, '__iter__'):
            if isinstance(new_iterable, tuple):
                # Tuples do not directly support assignment
                new_iterable = _change_tuple(new_iterable, i,
                                             fix_numpy_array_equality(item))
            else:
                new_iterable[i] = fix_numpy_array_equality(item)

    return new_iterable

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