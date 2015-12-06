import numpy

import network

def distance(vec_a, vec_b):
    return numpy.linalg.norm(numpy.subtract(vec_a, vec_b))

class SOM(network.Layer):
    pass