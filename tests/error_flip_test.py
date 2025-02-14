import unittest
import cirq
from decoder.surface_decoder import is_error_logical_bit_flip

class TestBitFlips(unittest.TestCase):

    def test_logical_x(self):
        left_qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(2, 0),
            cirq.GridQubit(4, 0)
        ]
        logical_x = cirq.PauliString({q: cirq.X for q in left_qubits})
        self.assertTrue(is_error_logical_bit_flip(3, logical_x))

    def test_logical_z(self):
        top_qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(0, 2),
            cirq.GridQubit(0, 4)
        ]
        logical_z = cirq.PauliString({q: cirq.Z for q in top_qubits})
        self.assertFalse(is_error_logical_bit_flip(3, logical_z))
    
    def test_logical_y(self):
        left_qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(2, 0),
            cirq.GridQubit(4, 0)
        ]
        logical_x = cirq.PauliString({q: cirq.X for q in left_qubits})
        top_qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(0, 2),
            cirq.GridQubit(0, 4)
        ]
        logical_z = cirq.PauliString({q: cirq.Z for q in top_qubits})
        logical_y = logical_x * logical_z
        self.assertTrue(is_error_logical_bit_flip(3, logical_y))
    
    def test_x_times_stabilizer(self):
        left_qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(2, 0),
            cirq.GridQubit(4, 0)
        ]
        logical_x = cirq.PauliString({q: cirq.X for q in left_qubits})
        stabilizer_qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(0, 2),
            cirq.GridQubit(1, 1)
        ]
        stabilizer = cirq.PauliString({q: cirq.X for q in stabilizer_qubits})
        error_string = stabilizer * logical_x
        self.assertTrue(is_error_logical_bit_flip(3, error_string))

if __name__ == "__main__":
    unittest.main()