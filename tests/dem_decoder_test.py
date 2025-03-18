import unittest
import numpy as np
import stim
from decoder.dem_decoder import stim_pauli_string_to_binary_symplectic

class TestPauliStrings(unittest.TestCase):

    def test_identity_all_false(self):
        """When called on all identity, we should get a list of False."""

        nq = 3
        pstring = stim.PauliString(nq)
        bitstring = stim_pauli_string_to_binary_symplectic(pstring)
        target_bitstring = np.array([False] * (2 * nq), dtype=bool)
        assert np.all(bitstring == target_bitstring)
    
    def test_single_x(self):
        """Test the string +XII."""

        nq = 3
        pstring = stim.PauliString("+XII")
        bitstring = stim_pauli_string_to_binary_symplectic(pstring)
        target_bitstring = np.array([True] + [False] * (2 * nq - 1), dtype=bool)
        assert np.all(bitstring == target_bitstring)

    def test_single_y(self):
        """Test the string +IYI."""

        pstring = stim.PauliString("+IYI")
        bitstring = stim_pauli_string_to_binary_symplectic(pstring)
        raw_target = [False, True, False, False, True, False]
        target_bitstring = np.array(raw_target, dtype=bool)
        assert np.all(bitstring == target_bitstring)


if __name__ == "__main__":
    unittest.main()