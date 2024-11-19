from typing import List
import numpy as np
import networkx as nx
import cirq
from decoder.surface_decoder import decode_representative

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


def is_error_logical_bit_flip(d: int, err: cirq.PauliString) -> bool:
    """Decide if an error will flip the logical bit, i.e. the error has an X or Y logical component.
    The test for this is whether there is a chain of Xs and Ys that stretches from the left edge
    of the lattice to the right edge."""

    # Get a new error with only Xs and Ys, no Zs.
    xy_err_dict = {}
    for qid, op in err.items():
        if op == cirq.X or op == cirq.Y:
            xy_err_dict[qid] = op
    xy_err = cirq.PauliString(xy_err_dict)
    qubits_per_side: int = 2 * d - 1
    qs = cirq.GridQubit.rect(qubits_per_side, qubits_per_side)
    assert set(err.keys()).issubset(set(qs))
    # Make a graph where the vertices are qubits.
    # Start with a graph containing all data qubits.
    qubit_graph = nx.Graph()
    for q in qs:
        qubit_graph.add_node(q)
    for i in range(qubits_per_side): # Over rows
        for j in range(qubits_per_side): # Over cols
            if ((i % 2 == 0) and (j % 2 == 0)) or ((i % 2 != 0) and (j % 2 != 0)):
                if j != 0: # West
                    qubit_graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(i, j-2))
                if j < qubits_per_side - 2: # East
                    qubit_graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(i, j+2))
                if (i != 0) and (j != 0): # Northwest
                    qubit_graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(i-1, j-1))
                if (i != 0) and (j != qubits_per_side - 1): # Northeast
                    qubit_graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(i-1, j+1))
                if (i != qubits_per_side - 1) and (j != qubits_per_side - 1): # Southeast
                    qubit_graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j+1))
                if (i != qubits_per_side - 1) and (j != 0): # Southwest
                    qubit_graph.add_edge(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j-1))
    # Then take out the qubits that are not in the support of the X,Y portion of the error.
    for qubit in qs:
        if qubit not in xy_err.keys():
            qubit_graph.remove_node(qubit)
    # See if there are any paths between a left edge node and a right edge node.
    # Get the sets of qubits on the left and right (ragged) edges of the code that are in the error.
    left_qubits = set(cirq.GridQubit.rect(qubits_per_side, 2)).intersection(set(xy_err.keys()))
    right_qubits = set(cirq.GridQubit.rect(qubits_per_side, 2, left=qubits_per_side-2)).intersection(set(xy_err.keys()))
    # If no edge qubits are in the support of the error, then the error is obviously not a bit-flip.
    if len(left_qubits) == 0 or len(right_qubits) == 0:
        return False
    path_exists = False
    for left_q in left_qubits:
        for right_q in right_qubits:
            all_paths = nx.all_simple_paths(qubit_graph, left_q, right_q)
            if len(list(all_paths)) != 0 and not path_exists:
                path_exists = True
    return path_exists



def main() -> None:
    d: int = 3
    ps: np.ndarray = np.linspace(1e-2, 0.4, num=10)
    top_qs = [
        cirq.GridQubit(0, 0), cirq.GridQubit(0, 2), cirq.GridQubit(0, 4)
    ]
    #err = sample_surface_error(d, ps[-1], False)
    err = cirq.PauliString({q: cirq.X for q in top_qs})
    print(is_error_logical_bit_flip(d, err))

if __name__ == "__main__":
    main()