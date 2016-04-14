import numpy

from pynn import network
from pynn.architecture import som
from pynn.data import datasets

def test_som_reduces_distances():
    # SOM functions correctly if is moves neurons towards inputs
    pat = datasets.get_xor()
    for p in pat:
        p[1].extend([1.0]*3) # So num targets lines up

    # Small initial weight range chosen so network isn't "accidentally"
    # very close to inputs initially (which could cause test to fail)
    nn = network.Network([som.SOM(2, 4, initial_weights_range=0.25)])

    # Convenience function
    def min_distances():
        all_closest = []
        for p in pat:
            distances = nn.activate(p[0])
            all_closest.append(min(distances))
        return all_closest

    # Train SOM
    # Assert that distances have decreased
    all_closest = min_distances()
    nn.train(pat, 10)
    new_closest = min_distances()
    print all_closest
    print new_closest
    for old_c, new_c in zip(all_closest, new_closest):
        assert new_c < old_c