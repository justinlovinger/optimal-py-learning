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
import math

import numpy

from learning.testing import helpers


def test_SaneEqualityArray():
    assert helpers.sane_equality_array(
        [0, 1, 2]) == helpers.sane_equality_array([0, 1, 2])
    assert helpers.sane_equality_array([0]) == helpers.sane_equality_array([0])

    assert not helpers.sane_equality_array([0]) == helpers.sane_equality_array(
        [0, 1, 2])
    assert not helpers.sane_equality_array(
        [0, 1, 2]) == helpers.sane_equality_array([0])


def test_fix_numpy_array_equality():
    complex_obj = [(numpy.array([0, 1, 2]), 'thing', []),
                   numpy.array([0, 1]),
                   [numpy.array([0, 1]),
                    numpy.array([0]), [0, 1, 2]]]

    assert helpers.fix_numpy_array_equality(complex_obj) == \
        [(helpers.sane_equality_array([0, 1, 2]), 'thing', []), helpers.sane_equality_array([0, 1]),
         [helpers.sane_equality_array([0, 1]), helpers.sane_equality_array([0]), [0, 1, 2]]]


###########################
# Gradient checking
###########################
def test_check_gradient_scalar():
    helpers.check_gradient(
        lambda x: numpy.sum(x**2), lambda x: 2.0 * x, f_shape='scalar')


def test_check_gradient_lin():
    helpers.check_gradient(lambda x: x**2, lambda x: 2 * x, f_shape='lin')
    helpers.check_gradient(
        lambda x: numpy.sqrt(x),
        lambda x: 1.0 / (2 * numpy.sqrt(x)),
        f_shape='lin')


def test_check_gradient_jacobian():
    helpers.check_gradient(lambda x: numpy.array([x[0]**2*x[1], 5*x[0]+math.sin(x[1])]),
                           lambda x: numpy.array([[2*x[0]*x[1], x[0]**2       ],
                                                  [5.0,         math.cos(x[1])]]),
                           inputs=numpy.random.rand(2),
                           f_shape='jac')
