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


#########################
# Real datasets
#########################
def get_iris():
    r"""Return the iris classification dataset.

    Type: Classification
    Samples: 150
    Attributes: 4
    Classes: 3

    Source: http://archive.ics.uci.edu/ml/datasets/Iris

    Recommended citations:
    @misc{Lichman:2013,
        author={M. Lichman},
        year={2013},
        title={{UCI} Machine Learning Repository},
        note={\url{http://archive.ics.uci.edu/ml}},
        institution={University of California, Irvine, School of Information and Computer Sciences}
    }
    """
    return process.get_data(_filename_relative('iris.data'), 0)


def get_cancer_diagnostic():
    r"""Return the Wisconsin breast cancer diagnostic dataset.

    Type: Classification
    Samples: 569
    Attributes: 30
    Classes: 2

    Source: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

    Recommended citations:
    @misc{Lichman:2013,
        author={M. Lichman},
        year={2013},
        title={{UCI} Machine Learning Repository},
        note={\url{http://archive.ics.uci.edu/ml}},
        institution={University of California, Irvine, School of Information and Computer Sciences}
    }
    """
    return process.get_data(
        _filename_relative('wdbc.data'), 2, attr_end_pos=None, target_pos=1)


def get_cancer_original():
    r"""Return the original Wisconsin breast cancer dataset.

    Type: Classification
    Samples: 683
    Attributes: 9
    Classes 2

    Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
    
    @misc{Lichman:2013,
        author={M. Lichman},
        year={2013},
        title={{UCI} Machine Learning Repository},
        note={\url{http://archive.ics.uci.edu/ml}},
        institution={University of California, Irvine, School of Information and Computer Sciences}
    }
    """
    return process.get_data(
        _filename_relative('breast-cancer-wisconsin.data'), 1)


def get_calhousing():
    r"""Return the California housing regression dataset.

    Type: Regression
    Samples: 20640
    Attributes: 8
    Targets: 1

    Source: http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

    Recommended citations:
    @misc{calhousing,
        title={California Housing},
        author={Pace, R. Kelley},
        note={\url{http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html}}
    }

    @article{pace1997sparse,
        title={Sparse spatial autoregressions},
        author={Pace, R Kelley and Barry, Ronald},
        journal={Statistics \& Probability Letters},
        volume={33},
        number={3},
        pages={291--297},
        year={1997},
        publisher={Elsevier}
    }

    @article{huang2005generalized,
        title={A generalized growing and pruning RBF (GGAP-RBF) neural network for function approximation},
        author={Huang, Guang-Bin and Saratchandran, Paramasivan and Sundararajan, Narasimhan},
        journal={IEEE Transactions on Neural Networks},
        volume={16},
        number={1},
        pages={57--67},
        year={2005},
        publisher={IEEE}
    }

    @article{huang2006universal,
        title={Universal approximation using incremental constructive feedforward networks with random hidden nodes},
        author={Huang, Guang-Bin and Chen, Lei and Siew, Chee Kheong and others},
        journal={IEEE Trans. Neural Networks},
        volume={17},
        number={4},
        pages={879--892},
        year={2006}
    }
    """
    return process.get_data(
        _filename_relative('cal_housing.data'), 0, classification=False)


def get_haberman():
    r"""Return the Haberman's survival classification dataset.

    Type: Classification
    Samples: 306
    Attributes: 3
    Classes: 2

    Source: https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
    @misc{Lichman:2013,
        author={M. Lichman},
        year={2013},
        title={{UCI} Machine Learning Repository},
        note={\url{http://archive.ics.uci.edu/ml}},
        institution={University of California, Irvine, School of Information and Computer Sciences}
    }
    """
    return process.get_data(_filename_relative('haberman.data'), 0)


def get_lenses():
    r"""Return the lenses classification dataset.

    Type: Classification
    Samples: 24
    Attributes: 4
    Classes: 3

    Source: https://archive.ics.uci.edu/ml/datasets/Lenses

    Recommended citations:
    @misc{Lichman:2013,
        author={M. Lichman},
        year={2013},
        title={{UCI} Machine Learning Repository},
        note={\url{http://archive.ics.uci.edu/ml}},
        institution={University of California, Irvine, School of Information and Computer Sciences}
    }
    """
    return process.get_data(_filename_relative('lenses.data'), 1)


def get_yeast():
    r"""Return the yeast classification dataset.

    Type: Classification
    Samples: 1484
    Attributes: 8
    Classes: 10

    Source: https://archive.ics.uci.edu/ml/datasets/Yeast

    Recommended citations:
    @misc{Lichman:2013,
        author={M. Lichman},
        year={2013},
        title={{UCI} Machine Learning Repository},
        note={\url{http://archive.ics.uci.edu/ml}},
        institution={University of California, Irvine, School of Information and Computer Sciences}
    }
    """
    return process.get_data(_filename_relative('yeast.data'), 1)


def _filename_relative(name):
    return os.path.join(os.path.dirname(__file__), 'datasets', name)


######################
# Toy datasets
######################
def get_xor():
    """Return dataset for XOR function."""
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
    """Return dataset for AND function."""
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


def get_random_classification(num_points, num_attributes, num_classes):
    """Return a random classification dataset.

    Attributes are continuous, in the range [-1, 1].
    """
    # Random values in range [-1, 1]
    random_inputs = (numpy.random.random(
        (num_points, num_attributes)) * 2.0) - 1.0

    # Make random targets
    random_targets = []
    for _ in range(num_points):
        # One hot targets
        random_target = [0.0] * num_classes
        random_target[random.randint(0, num_classes - 1)] = 1.0

        random_targets.append(random_target)
    random_targets = numpy.array(random_targets)

    return random_inputs, random_targets


def get_random_regression(num_points, num_attributes, num_targets):
    """Return a random regression dataset.

    Attributes are continuous, in the range [-1, 1].
    Targets are continuous, in the range [-1, 1].
    """
    # Random values in range [-1, 1]
    random_inputs = (numpy.random.random(
        (num_points, num_attributes)) * 2.0) - 1.0
    random_targets = (numpy.random.random(
        (num_points, num_targets)) * 2.0) - 1.0

    return random_inputs, random_targets
