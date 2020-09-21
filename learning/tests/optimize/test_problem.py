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

from learning.optimize import Problem


# NOTE: Not all combinations are tested
##################################
# Problem._get_obj
##################################
def test_optimizer_get_obj_obj_func():
    problem = Problem(obj_func=lambda x: x)
    assert problem.get_obj(1) == 1


def test_optimizer_get_obj_obj_jac_func():
    problem = Problem(obj_jac_func=lambda x: (x, x + 1))
    assert problem.get_obj(1) == 1


def test_optimizer_get_obj_obj_jac_hess_func():
    problem = Problem(obj_jac_hess_func=lambda x: (x, x + 1, x + 2))
    assert problem.get_obj(1) == 1


##################################
# Problem._get_jac
##################################
def test_optimizer_get_jac_jac_func():
    problem = Problem(jac_func=lambda x: x + 1)
    assert problem.get_jac(1) == 2


def test_optimizer_get_jac_obj_jac_func():
    problem = Problem(obj_jac_func=lambda x: (x, x + 1))
    assert problem.get_jac(1) == 2


##################################
# Problem._get_hess
##################################
def test_optimizer_get_hess_hess_func():
    problem = Problem(hess_func=lambda x: x + 2)
    assert problem.get_hess(1) == 3


def test_optimizer_get_hess_obj_hess_func():
    problem = Problem(obj_hess_func=lambda x: (x, x + 2))
    assert problem.get_hess(1) == 3


##################################
# Problem._get_obj_jac
##################################
def test_optimizer_get_obj_jac_obj_jac_func():
    problem = Problem(obj_jac_func=lambda x: (x, x + 1))
    assert problem.get_obj_jac(1) == (1, 2)


def test_optimizer_get_obj_jac_obj_jac_hess_func():
    problem = Problem(obj_jac_hess_func=lambda x: (x, x + 1, x + 2))
    assert tuple(problem.get_obj_jac(1)) == (1, 2)


def test_optimizer_get_obj_jac_individual_obj_jac():
    problem = Problem(obj_func=lambda x: x, jac_func=lambda x: x + 1)
    assert tuple(problem.get_obj_jac(1)) == (1, 2)


##################################
# Problem._get_obj_jac_hess
##################################
def test_optimizer_get_obj_jac_hess_obj_jac_hess_func():
    problem = Problem(obj_jac_hess_func=lambda x: (x, x + 1, x + 2))
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)


def test_optimizer_get_obj_jac_hess_obj_jac_func_individual_hess():
    problem = Problem(
        obj_jac_func=lambda x: (x, x + 1), hess_func=lambda x: x + 2)
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)


def test_optimizer_get_obj_jac_hess_obj_hess_func_individual_jac():
    problem = Problem(
        obj_hess_func=lambda x: (x, x + 2), jac_func=lambda x: x + 1)
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)


def test_optimizer_get_obj_jac_hess_individual_obj_jac_hess():
    problem = Problem(
        obj_func=lambda x: x,
        jac_func=lambda x: x + 1,
        hess_func=lambda x: x + 2)
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)
