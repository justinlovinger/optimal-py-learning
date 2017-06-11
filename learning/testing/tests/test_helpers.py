import copy
import math

import numpy

from learning.testing import helpers

def test_SaneEqualityArray():
    assert helpers.sane_equality_array([0, 1, 2]) == helpers.sane_equality_array([0, 1, 2])
    assert helpers.sane_equality_array([0]) == helpers.sane_equality_array([0])

    assert not helpers.sane_equality_array([0]) == helpers.sane_equality_array([0, 1, 2])
    assert not helpers.sane_equality_array([0, 1, 2]) == helpers.sane_equality_array([0])

def test_fix_numpy_array_equality():
    complex_obj = [(numpy.array([0, 1, 2]), 'thing', []), numpy.array([0, 1]),
                   [numpy.array([0, 1]), numpy.array([0]), [0, 1, 2]]]

    assert helpers.fix_numpy_array_equality(complex_obj) == \
        [(helpers.sane_equality_array([0, 1, 2]), 'thing', []), helpers.sane_equality_array([0, 1]),
         [helpers.sane_equality_array([0, 1]), helpers.sane_equality_array([0]), [0, 1, 2]]]

###########################
# Gradient checking
###########################
def test_check_gradient():
    helpers.check_gradient(lambda x: x**2, lambda x: 2*x)
    helpers.check_gradient(lambda x: numpy.sqrt(x), lambda x: 1.0 / (2*numpy.sqrt(x)))


def test_check_gradient_jacobian():
    helpers.check_gradient(lambda x: numpy.array([x[0]**2*x[1], 5*x[0]+math.sin(x[1])]),
                           lambda x: numpy.array([[2*x[0]*x[1], x[0]**2       ],
                                                  [5.0,         math.cos(x[1])]]),
                           inputs=numpy.random.rand(2),
                           jacobian=True)
