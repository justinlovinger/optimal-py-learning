import copy
import collections

import numpy

from learning import Model
from learning.architecture import pbnn

class EmptyModel(Model):
    def activate(self, inputs):
        pass

    def reset(self):
        pass

    def _train_increment(self, input_vec, target_vec):
        """Train on a single input, target pair.

        Optional.
        Model must either override train_step or implement _train_increment.
        """
        output = self.activate(input_vec)
        if output is not None:
            return target_vec - output

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
            self._inputs_output_dict[tuple(input_vec)] = numpy.array(target_vec)

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
        self._stored_targets = None

    def activate(self, inputs):
        return pbnn._weighted_sum_rows(self._stored_targets, numpy.array(inputs))

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

    if isinstance(a, (list, tuple)):
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
        return (isinstance(other, numpy.ndarray) and self.shape == other.shape and
                numpy.array_equal(self, other))

def fix_numpy_array_equality(iterable):
    if isinstance(iterable, numpy.ndarray):
        return sane_equality_array(iterable)

    if isinstance(iterable, str) or not isinstance(iterable, collections.Iterable):
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
