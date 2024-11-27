from typing import NewType, Callable
import numpy as np
import cirq

"""Error models mapping Pauli operators to their probabilities."""

ErrorModel = Callable[[cirq.PauliString], float]

def independent_depolarizing_noise(pauli: cirq.PauliString, p_depol: float) -> float:
    """A symmetric depolarizing channel acting on each qubit independently."""

    if len(pauli.values()) == 0:
        # This is the identity.
        return 1.0 - p_depol
    elif len(pauli.values()) == 1:
        if list(pauli.values())[0] in [cirq.X, cirq.Y, cirq.Z]:
            # This is X, Y, or Z:
            return p_depol / 3.0
        else:
            raise ValueError(
                f"Value {list(pauli.values())[0]} on qubit {list(pauli.keys())[0]} is invalid."
            )
    else:
        raise ValueError(f"String {pauli} has too high a weight. It should be weight 1.")


def independent_bit_flip_noise(pauli: cirq.PauliString, p_flip: float) -> float:
    """A bit flip channel acting on each qubit independently."""

    if len(pauli.values()) == 0:
        # This is the identity.
        return 1.0 - p_flip
    elif len(pauli.values()) == 1:
        if list(pauli.values())[0] in [cirq.Y, cirq.Z]:
            # In a bit flip channel, Y or Z errors will not happen.
            return 0.0
        elif list(pauli.values())[0] == cirq.X:
            return  p_flip / 3.0
        else:
            raise ValueError(
                f"Value {list(pauli.values())[0]} on qubit {list(pauli.keys())[0]} is invalid."
            )
    else:
        raise ValueError(f"String {pauli} has too high a weight. It should be weight 1.")


def sample_surface_error(d: int, p: float, bit_flip=True) -> cirq.PauliString:
    """Sample an error in the surface code."""

    assert p >= 0.0 and p <= 1.0, "p must be a valid probability."

    qubits_per_side: int = 2 * d - 1
    pauli_dict = {}
    for i in range(qubits_per_side):
        for j in range(qubits_per_side):
            if ((i % 2 == 0) and (j % 2 == 0)) or ((i % 2 != 0) and (j % 2 != 0)):
                r = np.random.rand() # Error probability.
                if r <= p:
                    if bit_flip:
                        pauli_dict[cirq.GridQubit(i, j)] = cirq.X
                    else:
                        s = np.random.rand() # 1/3 chance of X, Y, or Z.
                        if s < 1.0 / 3:
                            pauli_dict[cirq.GridQubit(i, j)] = cirq.X
                        elif s >= 1.0 / 3 and s < 2.0 / 3:
                            pauli_dict[cirq.GridQubit(i, j)] = cirq.Y
                        else:
                            pauli_dict[cirq.GridQubit(i, j)] = cirq.Z
    return cirq.PauliString(pauli_dict)
