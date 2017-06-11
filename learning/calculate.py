import numpy

def distance(vec_a, vec_b):
    # TODO: fix so it works with matrix inputs
    diff = numpy.subtract(vec_a, vec_b)
    return numpy.sqrt(diff.dot(diff))

def protvecdiv(vec_a, vec_b):
    """Divide vec_a by vec_b.

    When vec_b_i == 0, return 0 for component i.
    """
    with numpy.errstate(divide='raise', invalid='raise'):
        try:
            # Try to quickly divide vectors
            return vec_a / vec_b
        except FloatingPointError:
            # Fallback to dividing component at a time
            # Slower, but lets us handle divide by 0
            # TODO: Use procedure in preprocess.normalzie instead
            #   Divide, then np_matrix[~ numpy.isfinite(np_matrix)] = 0.0
            result_vec = numpy.zeros(vec_a.shape)
            for i in range(vec_a.shape[0]):
                try:
                    result_vec[i] = vec_a[i] / vec_b[i]
                except FloatingPointError:
                    pass # Already 0 from numpy.zeros
            return result_vec

#####################################
# Common math and transfer functions
#####################################
def tanh(x):
    """Sigmoid like function using tanh"""
    return numpy.tanh(x)

def dtanh(y):
    """Derivative of sigmoid above"""
    return 1.0 - y**2

def gaussian(x, variance=1.0):
    return numpy.exp(-(x**2/variance))

def dgaussian(x, y, variance=1.0):
    return -2.0*x*y / variance

def relu(x):
    """Return ln(1 + e^x) for each input value."""
    return numpy.log(1 + numpy.e**x)

def drelu(x):
    """Return the derivative of the softplus relu function for y."""
    # NOTE: Can be optimized by caching numpy.e**(x) and returning e^x / (e^x + 1)
    return 1.0 / (1.0 + numpy.e**(-x))

def softmax(x):
    """Return the softmax of vector x."""
    exp_ = numpy.exp(x)
    return exp_ / numpy.sum(exp_)

def dsoftmax(y):
    """Return the derivative of the softmax function for y."""
    # see http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network
    # Compute matrix J, n x n, with y_i(1 - y_j) on the diagonals
    # and - y_i y_j on the non-diagonals
    # When getting erros multiply by error vector (J \vec{e})

    # Start with - y_i y_j matrix, then replace diagonal with y_i(1 - y_j)
    jacobian = -y[:, None] * y
    jacobian[numpy.diag_indices(y.shape[0])] = y*(1 - y)
    return jacobian
