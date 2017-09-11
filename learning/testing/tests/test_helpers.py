import copy

import numpy

from learning.testing import helpers

def test_SaneEqualityArray():
    assert helpers.sane_equality_array([0, 1, 2]) == helpers.sane_equality_array([0, 1, 2])
    assert helpers.sane_equality_array([0]) == helpers.sane_equality_array([0])

    assert not helpers.sane_equality_array([0]) == helpers.sane_equality_array([0, 1, 2])
    assert not helpers.sane_equality_array([0, 1, 2]) == helpers.sane_equality_array([0])

def test_fix_numpy_array_equality():
    complex = [(numpy.array([0, 1, 2]), 'thing', []), numpy.array([0, 1]),
               [numpy.array([0, 1]), numpy.array([0]), [0, 1, 2]]]

    assert helpers.fix_numpy_array_equality(complex) == \
        [(helpers.sane_equality_array([0, 1, 2]), 'thing', []), helpers.sane_equality_array([0, 1]),
        [helpers.sane_equality_array([0, 1]), helpers.sane_equality_array([0]), [0, 1, 2]]]