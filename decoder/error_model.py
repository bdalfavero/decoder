from typing import NewType, Callable
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