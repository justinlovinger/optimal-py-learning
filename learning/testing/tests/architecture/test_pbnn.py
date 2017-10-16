from learning import datasets, validation, PBNN


def test_pbnn_convergence():
    # Run until convergence
    # assert that network can converge
    model = PBNN()
    dataset = datasets.get_xor()

    model.train(*dataset)
    assert validation.get_error(model, *dataset) <= 0.02
