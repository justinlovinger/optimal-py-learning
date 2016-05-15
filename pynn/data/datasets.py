import os
import random

import numpy

from pynn.data import process

def filename_relative(name):
    return os.path.join(os.path.dirname(__file__), 'datasets', name)

def get_calhousing():
    patterns = process.get_data(filename_relative('cal_housing_full.data'), 0, 
                                classification=False)
    process.normalize_targets(patterns)

    return patterns

def get_cancer_diagnostic():
    return process.get_data(filename_relative('wdbc.data'), 2, None, 1)

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
    return [
            [[-1,-1], [0]],
            [[-1,1], [1]],
            [[1,-1], [1]],
            [[1,1], [0]]
           ]

def get_random(num_points, num_dimensions, num_classes):
    random_inputs = numpy.random.random((num_points, num_dimensions))

    # Bundle random target
    dataset = []
    for input in random_inputs:
        # One hot targets
        random_target = [0.0]*num_classes
        random_target[random.randint(0, num_classes-1)] = 1.0

        dataset.append((input, random_target))

    return dataset