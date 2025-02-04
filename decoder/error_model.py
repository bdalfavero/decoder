from typing import List, Dict
from copy import deepcopy
import numpy as np
import cirq

"""Error models mapping Pauli operators to their probabilities."""

class ErrorModel:
    """A representative of the probability for errors on qubits."""

    def __init__(self, qubits: List[cirq.Qid]):
        self.qubits = qubits

    def __call__(self, pauli: cirq.PauliString):
        """Map a qubit and a Pauli to a probability."""

        raise NotImplementedError("Base class does not implement the call method.")


class IndependentErrorModel(ErrorModel):
    """A model of independent Pauli errors on each qubit."""

    def __init__(self, probability_dict: Dict[cirq.PauliString, float]):
        super().__init__(list(probability_dict.keys()))
        self.probability_dict = probability_dict
    
    def __call__(self, pauli: cirq.PauliString):
        """Get the probability of a single qubit Pauli error."""

        assert len(pauli.qubits) in [0, 1], f"Error must be single qubt, but got {len(pauli.qubits)}"
        if len(pauli.qubits) == 1:
            q = pauli.qubits[0]
            coefficientless_pauli = cirq.PauliString({q: pauli[q]})
        else:
            coefficientless_pauli = cirq.PauliString()
        return self.probability_dict[coefficientless_pauli]


def independent_depolarizing_model(qs: List[cirq.Qid], p_depol: float) -> IndependentErrorModel:
    """Builds an independent (symmetric) depolarizing model."""

    probability_dict = {}
    for q in qs:
        probability_dict[cirq.PauliString({q: cirq.I})] = 1.0 - p_depol
        probability_dict[cirq.PauliString({q: cirq.X})] = p_depol / 3.0
        probability_dict[cirq.PauliString({q: cirq.Y})] = p_depol / 3.0
        probability_dict[cirq.PauliString({q: cirq.Z})] = p_depol / 3.0
    return IndependentErrorModel(probability_dict)


def independent_bit_flip_model(qs: List[cirq.Qid], p: float) -> IndependentErrorModel:
    """A model where each qubit indpendently experiences bit-flip noise."""

    probability_dict = {}
    for q in qs:
        probability_dict[cirq.PauliString({q: cirq.I})] = 1.0 - p
        probability_dict[cirq.PauliString({q: cirq.X})] = p
        probability_dict[cirq.PauliString({q: cirq.Y})] = 0.0
        probability_dict[cirq.PauliString({q: cirq.Z})] = 0.0
    return IndependentErrorModel(probability_dict)


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

if __name__ == "__main__":
    qs = cirq.LineQubit.range(9)
    model = independent_depolarizing_model(qs, 0.5)
    print(model(cirq.PauliString({qs[0]: cirq.X})))