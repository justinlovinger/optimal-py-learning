import numpy

def distance(vec_a, vec_b):
    diff = numpy.subtract(vec_a, vec_b)
    return numpy.sqrt(diff.dot(diff))