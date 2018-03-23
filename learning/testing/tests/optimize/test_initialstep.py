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
import random

import numpy

from learning.optimize import initialstep
from learning.optimize import IncrPrevStep, FOChangeInitialStep, QuadraticInitialStep

######################
# InitialStepGetter s
######################
# TODO: Test IncrPrevStep
# TODO: Test FOChangeInitialStep
# TODO: Test QuadraticInitialStep

def test_QuadraticInitialStep_increasing_objective_value():
    """QuadraticInitialStep should default to 1 if objective value increased between calls."""
    vec_size = random.randint(1, 10)

    initial_step_getter = QuadraticInitialStep()

    x0 = numpy.random.random(vec_size)
    obj_x0 = random.uniform(-1, 1)
    jac_x0 = numpy.random.random(vec_size)
    initial_step_getter(
        numpy.random.random(vec_size), obj_x0, jac_x0, -jac_x0, None)

    # Call again with greater obj
    # Should default to 1
    assert initial_step_getter(x0 - jac_x0, obj_x0 + random.uniform(1e-10, 1),
                               jac_x0, -jac_x0, None) == 1


def test_QuadraticInitialStep_ascent_direction():
    """QuadraticInitialStep should default to 1 if step direction is an ascent direction."""
    vec_size = random.randint(1, 10)

    initial_step_getter = QuadraticInitialStep()

    # First call is always a default value
    x0 = numpy.random.random(vec_size)
    obj_x0 = random.uniform(-1, 1)
    jac_x0 = numpy.random.random(vec_size)
    initial_step_getter(
        numpy.random.random(vec_size), obj_x0, jac_x0, -jac_x0, None)

    # Call again with ascent direction (jac_xk.dot(step_dir) > 0)
    # Should default to 1
    assert initial_step_getter(x0 - jac_x0, obj_x0 - random.uniform(1e-10, 1),
                               jac_x0, jac_x0, None) == 1
