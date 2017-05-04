from learning import PBNN
from learning.data import datasets

def test_pbnn_convergence():
    # Run until convergence
    # assert that network can converge
    model = PBNN()
    pat = datasets.get_xor()

    model.train(*pat)
    assert model.avg_mse(*pat) <= 0.02
