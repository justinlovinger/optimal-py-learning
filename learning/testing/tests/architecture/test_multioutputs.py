import numpy
import pytest

from learning import Model
from learning.architecture import multioutputs

from learning.testing import helpers

class LearnOutput(Model):
    def __init__(self, learn_rate):
        super(LearnOutput, self).__init__()

        self.learn_rate = learn_rate

        self._output = None

        self.reset()

    def reset(self):
        self._output = 0

    def activate(self, inputs):
        return self._output

    def train_step(self, inputs, target):
        self._output += self.learn_rate*(target-self.activate(inputs))

def test_multioutputs_list_of_models():
    model = multioutputs.MultiOutputs([LearnOutput(1.0), LearnOutput(1.0)])
    model.train([[None]], numpy.array([[-1, 1]]))

    assert model.activate([]) == [-1, 1]

def test_multioutputs_activate():
    model = multioutputs.MultiOutputs(helpers.SetOutputModel(1), 2)
    assert model.activate([None]) == [1, 1]

def test_multioutputs_train():
    model = multioutputs.MultiOutputs(LearnOutput(1.0), 2)
    model.train([None], numpy.array([[-1, 1]]))

    assert model.activate([]) == [-1, 1]

def test_nested_multioutputs_train():
    model = multioutputs.MultiOutputs(multioutputs.MultiOutputs(LearnOutput(1.0), 2), 2)
    model.train([None], numpy.array([[[1, 2], [3, 4]]]))
    model.logging = False

    assert model.activate([]) == [[1, 2], [3, 4]]

def test_get_reward():
    assert multioutputs._get_reward(1.0, 0.0) == 1.0
    assert multioutputs._get_reward(1.0, 0.5) == 0.6

####################
# Col getting
####################
@pytest.mark.parametrize('type_func', [numpy.array, lambda x: x])
def test_matrix_col_1d(type_func):
    matrix = type_func(['s1t1', 's2t1'])

    assert (multioutputs._matrix_col(
        matrix, 0
    ) == 's1t1')

    assert (multioutputs._matrix_col(
        matrix, 1
    ) == 's2t1')

@pytest.mark.parametrize('type_func', [numpy.array, lambda x: x])
def test_matrix_col_2d(type_func):
    matrix = type_func([['s1t1', 's1t2'],
                        ['s2t1', 's2t2']])

    assert helpers.fix_numpy_array_equality(multioutputs._matrix_col(
        matrix, 0
    ) == helpers.fix_numpy_array_equality(type_func(['s1t1', 's2t1'])))

    assert helpers.fix_numpy_array_equality(multioutputs._matrix_col(
        matrix, 1
    ) == helpers.fix_numpy_array_equality(type_func(['s1t2', 's2t2'])))

@pytest.mark.parametrize('type_func', [numpy.array, lambda x: x])
def test_matrix_col_3d(type_func):
    matrix = type_func([[['s1t11', 's1t12'],
                         ['s1t21', 's1t22']],
                        [['s2t11', 's2t12'],
                         ['s2t21', 's2t22']]])

    assert helpers.fix_numpy_array_equality(multioutputs._matrix_col(
        matrix, 0
    ) == helpers.fix_numpy_array_equality(type_func([['s1t11', 's1t12'],
                                                     ['s2t11', 's2t12']])))

    assert helpers.fix_numpy_array_equality(multioutputs._matrix_col(
        matrix, 1
    ) == helpers.fix_numpy_array_equality(type_func([['s1t21', 's1t22'],
                                                     ['s2t21', 's2t22']])))

@pytest.mark.parametrize('type_func', [numpy.array, lambda x: x])
def test_transpose_rowcol_1d(type_func):
    # Should cause no change    
    matrix = type_func(['s1t1', 's2t1'])

    assert helpers.fix_numpy_array_equality(multioutputs._transpose_rowcol(
        matrix
    ) == helpers.fix_numpy_array_equality(type_func(['s1t1', 's2t1'])))

@pytest.mark.parametrize('type_func', [numpy.array, lambda x: x])
def test_transpose_rowcol_2d(type_func):
    matrix = type_func([['s1t1', 's1t2'],
                        ['s2t1', 's2t2']])

    assert helpers.fix_numpy_array_equality(multioutputs._transpose_rowcol(
        matrix
    ) == helpers.fix_numpy_array_equality(type_func([('s1t1', 's2t1'),
                                                     ('s1t2', 's2t2')])))

@pytest.mark.parametrize('type_func', [numpy.array, lambda x: x])
def test_transpose_rowcol_3d(type_func):
    matrix = type_func([[['s1t11', 's1t12'],
                         ['s1t21', 's1t22']],
                        [['s2t11', 's2t12'],
                         ['s2t21', 's2t22']]])

    assert helpers.fix_numpy_array_equality(multioutputs._transpose_rowcol(
        matrix
    ) == helpers.fix_numpy_array_equality(type_func([(['s1t11', 's1t12'],
                                                      ['s2t11', 's2t12']),
                                                     (['s1t21', 's1t22'],
                                                      ['s2t21', 's2t22'])])))
