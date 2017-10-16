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

import numpy

from learning.optimize import linesearch


#########################
# Wolfe conditions
#########################
def test_wolfe_conditions():
    # This step at this initial x will minimize f, and must therefore satisfy wolfe
    assert _wolfe_test(numpy.array([1.0, 1.0]), 0.5)

    # This step size will not change obj value, and should not satisfy wolfe
    assert not _wolfe_test(numpy.array([1.0, 1.0]), 1.0)


def _wolfe_test(xk, step_size):
    f = lambda vec: vec[0]**2 + vec[1]**2
    df = lambda vec: numpy.array([2.0 * vec[0], 2.0 * vec[1]])

    return linesearch._wolfe_conditions(step_size, xk,
                                        f(xk),
                                        df(xk), -df(xk),
                                        f(xk - step_size * df(xk)),
                                        df(xk - step_size * df(xk)), 1e-4, 0.1)


def test_armijo_rule():
    # This step at this initial x will minimize f, and must therefore satisfy armijo
    assert _armijo_test(numpy.array([1.0, 1.0]), 0.5)

    # This step size will not change obj value, and should not satisfy armijo
    assert not _armijo_test(numpy.array([1.0, 1.0]), 1.0)


def _armijo_test(xk, step_size):
    f = lambda vec: vec[0]**2 + vec[1]**2
    df = lambda vec: numpy.array([2.0 * vec[0], 2.0 * vec[1]])

    return linesearch._armijo_rule(step_size,
                                   f(xk),
                                   df(xk), -df(xk),
                                   f(xk - step_size * df(xk)), 1e-4)


def test_curvature_condition():
    # This step at this initial x will minimize f, and must therefore satisfy curvature
    assert _curvature_test(numpy.array([1.0, 1.0]), 0.5)

    # Curvature condition requires that slope increases from initial 0.0, it should always fail
    # at 0.0
    assert not _curvature_test(numpy.array([1.0, 1.0]), 0.0)


def _curvature_test(xk, step_size):
    f = lambda vec: vec[0]**2 + vec[1]**2
    df = lambda vec: numpy.array([2.0 * vec[0], 2.0 * vec[1]])

    return linesearch._curvature_condition(
        df(xk), -df(xk), df(xk - step_size * df(xk)), 0.1)
