import numpy as np
import pandas as pd
import cirq
from decoder.surface_decoder import decode_representative

def sample_surface_error(d: int, p: float) -> cirq.PauliString:
    """Sample an error in the surface code."""

    assert p >= 0.0 and p <= 1.0, "p must be a valid probability."

    qubits_per_side: int = 2 * d - 1
    pauli_dict = {}
    for i in range(qubits_per_side):
        for j in range(qubits_per_side):
            r = np.random.rand() # Error probability.
            if r <= p:
                s = np.random.rand() # 1/3 chance of X, Y, or Z.
                if s < 1.0 / 3:
                    pauli_dict[cirq.GridQubit(i, j)] = cirq.X
                elif s >= 1.0 / 3 and s < 2.0 / 3:
                    pauli_dict[cirq.GridQubit(i, j)] = cirq.Y
                else:
                    pauli_dict[cirq.GridQubit(i, j)] = cirq.Z
    return cirq.PauliString(pauli_dict)


def main() -> None:
    ps = np.linspace(1e-4, 0.1, num=10)
    count_arr = np.zeros((ps.size, 4), dtype=int)
    for i, p in enumerate(ps):
        err_classes = []
        for _ in range(1000):
            err = sample_surface_error(3, p)
            err_class = decode_representative(3, err, p)
            err_classes.append(err_class)
        # Count the errors.
        count_dict = {}
        for j in range(4):
            count_arr[i, j] += err_classes.count(j)
    
    df = pd.DataFrame(data=count_arr, index=pd.Index(ps, name="p"), columns=['I','X','Y','Z'])
    df.to_csv("../data/error_counts.csv")

if __name__ == "__main__":
    main()