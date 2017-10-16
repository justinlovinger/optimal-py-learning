from learning import datasets
from learning.architecture import som


def test_som_reduces_distances():
    # SOM functions correctly if is moves neurons towards inputs
    input_matrix, target_matrix = datasets.get_xor()

    # Small initial weight range chosen so network isn't "accidentally"
    # very close to inputs initially (which could cause test to fail)
    som_ = som.SOM(2, 4, initial_weights_range=0.25)

    # Convenience function
    def min_distances():
        all_closest = []
        for inp_vec in input_matrix:
            distances = som_.activate(inp_vec)
            all_closest.append(min(distances))
        return all_closest

    # Train SOM
    # Assert that distances have decreased
    all_closest = min_distances()
    som_.train(input_matrix, target_matrix, iterations=20)
    new_closest = min_distances()
    print all_closest
    print new_closest
    for old_c, new_c in zip(all_closest, new_closest):
        assert new_c < old_c
