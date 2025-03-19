from typing import Tuple, List
from copy import deepcopy
from functools import reduce
import numpy as np
import networkx as nx
import quimb.tensor as qtn
import stim

def equality_tensor(nlegs: int, d: int = 2, prefix: str = "k") -> qtn.Tensor:
    """Build a tensor that effects a delta function."""

    assert d >= 2, f"Equality tensor must have at least two legs, but d={d} was passed."

    data = np.zeros((d,) * nlegs, dtype=float)
    for i in range(d):
        data[(i,) * nlegs] = 1.0
    indices = [prefix + str(i) for i in range(nlegs)]
    return qtn.Tensor(data=data, inds=indices, tags=["Delta"])


def parity_tensor(nlegs: int, prefix: str = "k") -> qtn.Tensor:
    """Build a tensor that effects the parity function (+ in Chubb's paper)."""

    data = np.zeros((2,) * nlegs, dtype=float)
    for inds, _ in np.ndenumerate(data):
        b = [bool(i) for i in inds]
        parity = not reduce(lambda x, y: x ^ y, b)
        data[inds] = int(parity)
    indices = [prefix + str(i) for i in range(nlegs)]
    return qtn.Tensor(data=data, inds=indices, tags=["Parity"])


def probability_tensor(ps: np.ndarray, indices: Tuple[str, str] = ("a", "b")) -> qtn.Tensor:
    """Build a tensor with probabilities of single-qubit
    errors conditioned on the binary-symplectic representation
    of the error, i.e.
    P_00 = p_I, P_01 = P_Z, P_10 = P_X, P_11 = P_Y.
    
    Arguments:
    ps - Array [P_I, P_X, P_Y, P_Z].
    indices - A tuple of two strings, telling the indices of the tensor.
    
    Returns:
    tensor - The specified probability tensor."""

    assert ps.size == 4
    assert abs(1.0 - np.sum(ps)) <= 1e-4, \
        f"Probabilities for tensor must sum to 1, but they sum to {np.sum(ps)}."

    data = np.zeros((2, 2), dtype=float)
    data[0, 0] = ps[0]
    data[0, 1] = ps[3]
    data[1, 0] = ps[1]
    data[1, 1] = ps[2]
    return qtn.Tensor(data=data, inds=indices, tags=["Prob"])


def stim_pauli_string_to_binary_symplectic(pstring: stim.PauliString) -> np.ndarray:
    """Convert a Pauli string to its binary-symplectic form."""

    bits = [False] * (2 * len(pstring)) # Initialize as all I.
    for i, p in enumerate(pstring):
        if p == 1 or p == 2:
            # p is X or Y.
            bits[i] = True
        if p == 2 or p == 3:
            bits[i + len(pstring)] = True
    return np.array(bits, dtype=bool)


def boolean_list_to_basis_tensors(b: List[bool], prefix: str = "k") -> List[qtn.Tensor]:
    """Turn a sequence of bits into a product of tensors representing a computational
    basis state, e.g. 
    [False False] -> [1 0 0 0], [False True] -> [0 1 0 0],
    [True False] -> [0 0 1 0], [True True] -> [0 0 0 1].
    
    Arguments:
    b - List of booleans
    prefix - Prefix string for the tensor indices."""

    tensors = []
    for i, bi in enumerate(b):
        if bi:
            data = np.array([0.0, 1.0])
        else:
            data = np.array([1.0, 0.0])
        tensors.append(qtn.Tensor(data, inds=(prefix + f"{i}",)))
    return tensors


class DEMDecoder:
    """TN decoder in the detector picture."""

    def __init__(
        self,
        probabilities: np.array,
        stabilizers: List[stim.PauliString],
        logicals: List[stim.PauliString]
    ) -> None:
        """Initialize the deocder with probabilities, stabilizers, and logical operators.
        The list of logicals for k=1 should be [Logical X, Logical Z.]"""

        self._probabilities = probabilities
        self._stabilizers = stabilizers
        self._logicals = logicals

    @property
    def network(self) -> qtn.TensorNetwork:
        """Build a tensor network to decode a set of detectors."""

        nq = self._probabilities.shape[0]
        binary_stabilizers = [stim_pauli_string_to_binary_symplectic(s) for s in self._stabilizers]
        binary_logicals = [stim_pauli_string_to_binary_symplectic(l) for l in self._logicals]

        # Loop through the edges of the graph, assigning indices to check and equality tensors.
        detector_tensor_inds = [[f"m{i}"] for i in range(len(self._stabilizers))]
        logical_tensor_inds = [[f"l{i}"] for i in range(len(self._logicals))]
        x_equ_tensor_inds = [[f"x{i}"] for i in range(nq)]
        z_equ_tensor_inds = [[f"z{i}"] for i in range(nq)]
        bond_counter = 0
        for i, s in enumerate(binary_stabilizers):
            for q in range(nq):
                if s[q + nq]:
                    #graph.add_edge(f"x{q}", f"m{i}")
                    x_equ_tensor_inds[q] += [f"h{bond_counter}"]
                    detector_tensor_inds[i] += [f"h{bond_counter}"]
                    bond_counter += 1
                if s[q]:
                    #graph.add_edge(f"z{q}", f"m{i}")
                    z_equ_tensor_inds[q] += [f"h{bond_counter}"]
                    detector_tensor_inds[i] += [f"h{bond_counter}"]
                    bond_counter += 1
        for i, l in enumerate(binary_logicals):
            for q in range(nq):
                if l[q]:
                    z_equ_tensor_inds[q] += [f"h{bond_counter}"]
                    logical_tensor_inds[i] += [f"h{bond_counter}"]
                    bond_counter += 1
                if l[i + nq]:
                    x_equ_tensor_inds[q] += [f"h{bond_counter}"]
                    logical_tensor_inds[i] += [f"h{bond_counter}"]
                    bond_counter += 1
        parity_tensor_inds = detector_tensor_inds + logical_tensor_inds
        equality_tensor_inds = x_equ_tensor_inds + z_equ_tensor_inds

        # Convert index lists to equality and parity tensors.
        parity_tensors: List[qtn.Tensor] = []
        for p_inds in parity_tensor_inds:
            tensor = parity_tensor(len(p_inds))
            tensor.modify(inds=p_inds)
            parity_tensors.append(tensor.copy(deep=True))
        equality_tensors: List[qtn.Tensor] = []
        for equ_inds in equality_tensor_inds:
            tensor = equality_tensor(len(equ_inds))
            tensor.modify(inds=equ_inds)
            equality_tensors.append(tensor.copy(deep=True))

        # Build tensors for the Pauli error probabilities.
        probability_tensors = []
        for i in range(nq):
            tensor = probability_tensor(self._probabilities[i, :], (f"x{i}", f"z{i}"))
            probability_tensors.append(tensor)

        return qtn.TensorNetwork(probability_tensors + parity_tensors + equality_tensors)

    def decode_syndrome(self, m: List[bool]) -> int:
        """Decoder a syndrome by predicting logical operator flips."""

        assert len(m) == len(self._stabilizers), \
            f"Syndrome has {len(m)} bits, but code has {len(self._stabilizers)} stabilizers."
        assert len(self._logicals) == 2, "At k=1, code must have two logical generators."

        # Make tensors to contract with the "m" indices.
        m_tensors = boolean_list_to_basis_tensors(m, "m")

        # Get the probability for each error class in {I, X, Y, Z}
        class_probabilities = []
        for i in range(4):
            if i == 0:
                l_tensors = boolean_list_to_basis_tensors([False, False], "l") # Logical I
            elif i == 1:
                l_tensors = boolean_list_to_basis_tensors([True, False], "l") # Logical X
            elif i == 2:
                l_tensors = boolean_list_to_basis_tensors([True, True], "l") # Logical Y
            else:
                l_tensors = boolean_list_to_basis_tensors([False, True], "l") # Logical Z
            network = self.network & m_tensors & l_tensors
            class_probabilities.append(network.contract())

        return np.argmax(class_probabilities)

    def decode_batch(self, m: np.ndarray) -> List[int]:
        """Decode a batch of syndromes.
        
        Arguments:
        m - (number of shots) x (number of syndrome bits) boolean array
        
        Returns:
        classes - List of classes for each shot."""

        assert m.shape[1] == len(self._stabilizers), \
            f"Must have {len(self._stabilizers)} columns, not {m.shape[1]}"

        classes: List[int] = []
        for i in range(m.shape[0]):
            syndrome = m[i, :].tolist()
            classes.append(self.decode_syndrome(syndrome))
        return classes


if __name__ == "__main__":
    # Prepare 5 qubit code deocder.
    stabilizers = []
    stabilizers.append(stim.PauliString("+XZZXI"))
    stabilizers.append(stim.PauliString("+IXZZX"))
    stabilizers.append(stim.PauliString("+XIXZZ"))
    stabilizers.append(stim.PauliString("+ZXIXZ"))

    logicals = []
    logicals.append(stim.PauliString("+XZIZI"))
    logicals.append(stim.PauliString("+IZIXX"))

    # Give each qubit depolarizing noise.
    p = 0.05
    single_q_prob = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0])
    probabilities = np.zeros((5, 4), dtype=float)
    for i in range(5):
        probabilities[i, :] = single_q_prob

    decoder = DEMDecoder(probabilities, stabilizers, logicals)
    print(decoder.decode_syndrome([False, False, False, True]))
