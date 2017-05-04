import pytest

from learning.architecture import rbf
from learning.data import datasets

########################
# Integration tests
########################
def test_rbf():
    # Run for a couple of iterations
    # assert that new error is less than original
    nn = rbf.RBF(2, 4, 2, scale_by_similarity=True)
    pat = datasets.get_xor()

    error = nn.avg_mse(*pat)
    nn.train(*pat, iterations=10)
    assert nn.avg_mse(*pat) < error


pytest.mark.slowtest()
def test_rbf_convergence():
    # Run until convergence
    # assert that network can converge
    nn = rbf.RBF(2, 4, 2, learn_rate=0.75, scale_by_similarity=True)
    pat = datasets.get_xor()

    nn.train(*pat, retries=5, error_break=0.002)
    assert nn.avg_mse(*pat) <= 0.02
