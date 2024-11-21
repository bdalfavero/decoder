import unittest
import cirq
from decoder.surface_decoder import error_to_syndrome

class TestErrorToSyndrome(unittest.TestCase):

    def test_upper_corner_x(self):
        q = cirq.GridQubit(0, 0)
        err = cirq.PauliString({q: cirq.X})
        z_syndrome, x_syndrome = error_to_syndrome(3, err)
        z_syndrome_target = [False] * 6
        x_syndrome_target = [True] + [False] * 5
        self.assertTrue(z_syndrome == z_syndrome_target and x_syndrome == x_syndrome_target)

    def test_upper_corner_y(self):
        q = cirq.GridQubit(0, 0)
        err = cirq.PauliString({q: cirq.Y})
        z_syndrome, x_syndrome = error_to_syndrome(3, err)
        z_syndrome_target = [True] + [False] * 5
        x_syndrome_target = [True] + [False] * 5
        self.assertTrue(z_syndrome == z_syndrome_target and x_syndrome == x_syndrome_target)
    
    def test_bottom_row_z(self):
        qs = cirq.GridQubit.rect(1, 5, top=4)
        err = cirq.PauliString({
            qs[0]: cirq.Z, qs[2]: cirq.Z, qs[4]: cirq.Z
        })
        z_syndrome, x_syndrome = error_to_syndrome(3, err)
        z_syndrome_target = [False] * 3 + [True] * 3
        x_syndrome_target = [False] * 6
        self.assertTrue(z_syndrome == z_syndrome_target and x_syndrome == x_syndrome_target)
    
    def test_y_middle(self):
        q = cirq.GridQubit(2, 2)
        err = cirq.PauliString({q: cirq.Y})
        z_syndrome, x_syndrome = error_to_syndrome(3, err)
        z_syndrome_target = [False, True, False, False, True, False]
        x_syndrome_target = [False, False, True, True, False, False]
        self.assertTrue(z_syndrome == z_syndrome_target and x_syndrome == x_syndrome_target)
    
    def test_neighboring_x_cancels(self):
        # If there are X errors on two qubits for a stabilizer, then 
        # that stabilizer should register nothing. However, the 
        # next one will trigger.
        # For example, consider the error XXI,II,III,II,III.
        # The top-row x stabilizers should register [False, True].
        q1 = cirq.GridQubit(0, 0)
        q2 = cirq.GridQubit(0, 2)
        err = cirq.PauliString({q1: cirq.X, q2: cirq.X})
        z_syndrome, x_syndrome = error_to_syndrome(3, err)
        print(z_syndrome, x_syndrome)
        z_syndrome_target = [False] * 6
        x_syndrome_target = [False, True] + [False] * 4
        self.assertTrue(z_syndrome == z_syndrome_target and x_syndrome == x_syndrome_target)

if __name__ == "__main__":
    unittest.main()