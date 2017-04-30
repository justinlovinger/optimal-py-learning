import numpy

def distance(vec_a, vec_b):
    # TODO: fix so it works with matrix inputs
    diff = numpy.subtract(vec_a, vec_b)
    return numpy.sqrt(diff.dot(diff))

def protvecdiv(vec_a, vec_b):
    """Divide vec_a by vec_b.

    When vec_b_i == 0, return 0 for component i.
    """
    with numpy.errstate(divide='raise'):
        try:
            # Try to quickly divide vectors
            return vec_a / vec_b
        except FloatingPointError:
            # Fallback to dividing component at a time
            # Slower, but lets us handle divide by 0
            result_vec = numpy.zeros(vec_a.shape)
            for i in range(vec_a.shape[0]):
                try:
                    result_vec[i] = vec_a[i] / vec_b[i]
                except FloatingPointError:
                    pass # Already 0 from numpy.zeros
            return result_vec
