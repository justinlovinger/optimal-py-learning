import numpy

from learning import calculate

def test_protvecdiv_no_zero():
    assert (calculate.protvecdiv(
        numpy.array([1.0, 2.0, 3.0]), numpy.array([2.0, 2.0, 2.0]))
            == numpy.array([0.5, 1.0, 1.5])).all()

def test_protvecdiv_zero_den():
    # Returns 0 for position with 0 denominator
    # Also for 0 / 0
    assert (calculate.protvecdiv(
        numpy.array([1.0, 2.0, 0.0]), numpy.array([2.0, 0.0, 0.0]))
            == numpy.array([0.5, 0.0, 0.0])).all()
