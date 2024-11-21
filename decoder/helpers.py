import cirq

def raise_pauli_to_power(pauli: cirq.PauliString, pow: int) -> cirq.PauliString:
    """Raise a weight-one Pauli string to a power.
    This function exists because the default way to exponentiate Paulis
    results in a GateOperation.
    
    Arguments:
    pauli: a cirq PauliString of weight one.
    pow: an integer power.
    
    Returns:
    Pauli raised to that power."""

    assert len(pauli.qubits) == 1, "Pauli string must be weight one."
    q = pauli.qubits[0]
    if pow % 2 == 0:
        return cirq.PauliString({q: cirq.I})
    else:
        return pauli