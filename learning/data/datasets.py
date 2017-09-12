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

import os
import random

import numpy

from learning.data import process

def filename_relative(name):
    return os.path.join(os.path.dirname(__file__), 'datasets', name)

def get_calhousing():
    return process.get_data(filename_relative('cal_housing_full.data'), 0, 
                            classification=False)

def get_cancer_diagnostic():
    return process.get_data(filename_relative('wdbc.data'), 2,
                            attr_end_pos=None, target_pos=1)

def get_cancer_wisconsin():
    return process.get_data(filename_relative('breast-cancer-wisconsin.data'), 1)

def get_haberman():
    return process.get_data(filename_relative('haberman.data'), 0)

def get_iris():
    return process.get_data(filename_relative('iris.data'), 0)

def get_lenses():
    return process.get_data(filename_relative('lenses.data'), 1)

def get_yeast():
    return process.get_data(filename_relative('yeast.data'), 1)

def get_xor():
    return numpy.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ]), numpy.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

def get_and():
    return numpy.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ]), numpy.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ])

def get_random_classification(num_points, num_dimensions, num_classes):
    # Random values in range [-1, 1]
    random_inputs = (numpy.random.random((num_points, num_dimensions))*2.0)-1.0

    # Make random targets
    random_targets = []
    for _ in range(num_points):
        # One hot targets
        random_target = [0.0]*num_classes
        random_target[random.randint(0, num_classes-1)] = 1.0

        random_targets.append(random_target)
    random_targets = numpy.array(random_targets)

    return random_inputs, random_targets

def get_random_regression(num_points, num_dimensions, num_targets):
    # Random values in range [-1, 1]
    random_inputs = (numpy.random.random((num_points, num_dimensions))*2.0)-1.0
    random_targets = (numpy.random.random((num_points, num_targets))*2.0)-1.0

    return random_inputs, random_targets
