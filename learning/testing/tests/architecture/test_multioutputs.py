import numpy

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

def test_multioutputs_model_constructor():
    model = multioutputs.MultiOutputs(2, lambda: LearnOutput(1.0))
    model.train([[None]], numpy.array([[-1, 1]]))

    assert model.activate([]) == [-1, 1]

def test_multioutputs_activate():
    model = multioutputs.MultiOutputs(2, helpers.SetOutputModel(1))
    assert model.activate([None]) == [1, 1]

def test_multioutputs_train():
    model = multioutputs.MultiOutputs(2, LearnOutput(1.0))
    model.train([None], numpy.array([[-1, 1]]))

    assert model.activate([]) == [-1, 1]

def test_nested_multioutputs_train():
    model = multioutputs.MultiOutputs(2, multioutputs.MultiOutputs(2, LearnOutput(1.0)))
    model.train([None], numpy.array([[[1, 2], [3, 4]]]))
    model.logging = False

    assert model.activate([]) == [[1, 2], [3, 4]]

def test_get_reward():
    assert multioutputs._get_reward(1.0, 0.0) == 1.0
    assert multioutputs._get_reward(1.0, 0.5) == 0.6

####################
# Col getting
####################
def test_matrix_col_1d():
    matrix = numpy.array(['s1t1', 's2t1'])

    assert (multioutputs._matrix_col(
        matrix, 0
    ) == 's1t1')

    assert (multioutputs._matrix_col(
        matrix, 1
    ) == 's2t1')

def test_matrix_col_2d():
    matrix = numpy.array([['s1t1', 's1t2'],
                          ['s2t1', 's2t2']])

    assert (multioutputs._matrix_col(
        matrix, 0
    ) == numpy.array(['s1t1', 's2t1'])).all()

    assert (multioutputs._matrix_col(
        matrix, 1
    ) == numpy.array(['s1t2', 's2t2'])).all()

def test_matrix_col_3d():
    matrix = numpy.array([[['s1t11', 's1t12'],
                           ['s1t21', 's1t22']],
                          [['s2t11', 's2t12'],
                           ['s2t21', 's2t22']]])

    assert (multioutputs._matrix_col(
        matrix, 0
    ) == numpy.array([['s1t11', 's1t12'],
                      ['s2t11', 's2t12']])).all()

    assert (multioutputs._matrix_col(
        matrix, 1
    ) == numpy.array([['s1t21', 's1t22'],
                      ['s2t21', 's2t22']])).all()

def test_transpose_rowcol_1d():
    # Should cause no change    
    matrix = numpy.array(['s1t1', 's2t1'])

    assert (multioutputs._transpose_rowcol(
        matrix
    ) == numpy.array(['s1t1', 's2t1'])).all()

def test_transpose_rowcol_2d():
    matrix = numpy.array([['s1t1', 's1t2'],
                          ['s2t1', 's2t2']])

    assert (multioutputs._transpose_rowcol(
        matrix
    ) == numpy.array([['s1t1', 's2t1'],
                      ['s1t2', 's2t2']])).all()

def test_transpose_rowcol_3d():
    matrix = numpy.array([[['s1t11', 's1t12'],
                           ['s1t21', 's1t22']],
                          [['s2t11', 's2t12'],
                           ['s2t21', 's2t22']]])

    assert (multioutputs._transpose_rowcol(
        matrix
    ) == numpy.array([[['s1t11', 's1t12'],
                       ['s2t11', 's2t12']],
                      [['s1t21', 's1t22'],
                       ['s2t21', 's2t22']]])).all()
