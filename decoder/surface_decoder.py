"""A tensor network decoder for the surface code, following Bravyi and Chubb.
For now, assumes i.i.d. noise."""

from typing import List, Dict, Optional
import numpy as np
import cirq
import quimb.tensor as qtn
from decoder.error_model import ErrorModel, independent_depolarizing_noise

def kron_tensor(d: int, inds: List[str]) -> qtn.Tensor:
    """A tensor that implements the Kronecker functions delta_{i,j,k,l} or delta_{i,j,k},
    depending on how many indices are passed.
    
    Arguments:
    d - integer, dimension of each leg in the tensor.
    inds - a list of indices (strings) for quimb.
    
    Returns:
    delta - a quimb Tensor encoding the Kronecker delta."""

    delta = np.zeros((d,) * len(inds), dtype=float)
    # Get a list of indices to set to one.
    diag_inds = [(i,) * len(inds) for i in range(d)]
    for di in diag_inds:
        delta[di] = 1.0
    return qtn.Tensor(delta, inds=inds)


def error_tensor(
    err: cirq.PauliString, err_model: ErrorModel, inds: Dict[str, Optional[str]]
) -> qtn.Tensor:
    """The tensor encoding probabilities of errors in Bravyi and Chubb.
    
    Arguments:
    err - A PauliString acting on all the qubits in the code.
    err_model - an ErrorModel the decoder uses.
    inds - dictionary of indices for north, south, east, and west.
    If the tensor has no index there, then we set that string to none.
    e.g. {"n": None, "s": "i", "e": "j", "w": None}
    is the tensor at the top left corner.
    
    Returns:
    err_tensor - a quimb Tensor for the TN decoder."""

    assert len(err.qubits) <= 1, "Error must have only one qubit."
    assert set(inds.keys()) == set(["n", "s", "e", "w"])

    if len(err.qubits) == 1:
        q = err.qubits[0]
    else:
        # TODO this does not work for non-iid noise - maybe qubit should be a keyword argument.
        q = cirq.LineQubit(0)
    pauli_x: cirq.PauliString = cirq.PauliString({q: cirq.X})
    pauli_y: cirq.PauliString = cirq.PauliString({q: cirq.Y})
    pauli_z: cirq.PauliString = cirq.PauliString({q: cirq.Z})

    # Count how many indices are not none.
    num_indices = 0
    indices_not_none: List[str] = []
    for _, val in inds.items():
        if val is not None:
            num_indices += 1
            indices_not_none.append(val)
    assert num_indices in [2, 3, 4]
    # The indices in this tensor will be ordered (north, south, east, west)
    err_tensor = np.zeros((2,) * num_indices)

    if num_indices == 2:
        # These are the corner tensors
        # One index points north or south, and the other east/west.
        err_tensor[0, 0] = err_model(err)
        err_tensor[0, 1] = err_model(err * pauli_x)
        err_tensor[1, 0] = err_model(err * pauli_z)
        err_tensor[1, 1] = err_model(err * pauli_y)
        ind_list = []
        if inds["n"] is None:
            ind_list.append(inds["s"])
        else:
            ind_list.append(inds["n"])
        if inds["e"] is None:
            ind_list.append(inds["w"])
        else:
            ind_list.append(inds["e"])
    elif num_indices == 3:
        # These are edge tensors. Either there are both north and south indices
        # and only the east or west index, or there are both east and west, but only
        # one of north or south.
        if inds["e"] is None or inds["w"] is None:
            # Both north and south are present.
            # The third index is east or west.
            err_tensor[0, 0, 0] = err_model(err)
            err_tensor[0, 0, 1] = err_model(err * pauli_x)
            err_tensor[0, 1, 0] = err_model(err * pauli_z)
            err_tensor[0, 1, 1] = err_model(err * pauli_y)
            err_tensor[1, 0, 0] = err_model(err * pauli_z)
            err_tensor[1, 0, 1] = err_model(err * pauli_y)
            err_tensor[1, 1, 0] = err_model(err)
            err_tensor[1, 1, 1] = err_model(err * pauli_x)
            if inds["e"] is None:
                ind_list = [inds["n"], inds["s"], inds["w"]]
            else:
                ind_list = [inds["n"], inds["s"], inds["e"]]
        else:
            # Both east and west are present, and only one
            # of north or south is here.
            err_tensor[0, 0, 0] = err_model(err)
            err_tensor[0, 0, 1] = err_model(err * pauli_x)
            err_tensor[0, 1, 0] = err_model(err * pauli_x)
            err_tensor[0, 1, 1] = err_model(err)
            err_tensor[1, 0, 0] = err_model(err * pauli_z)
            err_tensor[1, 0, 1] = err_model(err * pauli_y)
            err_tensor[1, 1, 0] = err_model(err * pauli_y)
            err_tensor[1, 1, 1] = err_model(err * pauli_z)
            if inds["n"] is None:
                ind_list = [inds["s"], inds["e"], inds["w"]]
            else:
                ind_list = [inds["n"], inds["e"], inds["w"]]
    else:
        # All edges are present.
        err_tensor[0, 0, 0, 0] = err_model(err)
        err_tensor[0, 0, 0, 1] = err_model(err * pauli_x)
        err_tensor[0, 0, 1, 0] = err_model(err * pauli_x)
        err_tensor[0, 0, 1, 1] = err_model(err)
        err_tensor[0, 1, 0, 0] = err_model(err * pauli_z)
        err_tensor[0, 1, 0, 1] = err_model(err * pauli_y)
        err_tensor[0, 1, 1, 0] = err_model(err * pauli_y)
        err_tensor[0, 1, 1, 1] = err_model(err * pauli_z)
        err_tensor[1, 0, 0, 0] = err_model(err * pauli_z)
        err_tensor[1, 0, 0, 1] = err_model(err * pauli_y)
        err_tensor[1, 0, 1, 0] = err_model(err * pauli_y)
        err_tensor[1, 0, 1, 1] = err_model(err * pauli_z)
        err_tensor[1, 1, 0, 0] = err_model(err)
        err_tensor[1, 1, 0, 1] = err_model(err * pauli_x)
        err_tensor[1, 1, 1, 0] = err_model(err * pauli_x)
        err_tensor[1, 1, 1, 1] = err_model(err)
        ind_list = [inds["n"], inds["s"], inds["e"], inds["w"]]
    return qtn.Tensor(err_tensor, inds=ind_list)


def build_network_for_error_class(err: cirq.PauliString, d: int, p_depol: float) -> qtn.TensorNetwork:
    """Builds the 2D PEPS for the RBIM partition function.
    
    Arguments:
    err - A Pauli string acting on qubits of the code. Represents the 'base error' to which
    stabilizers are added.
    d - distance of the code.
    p_depol - depolarizing probability
    
    Returns:
    peps - tensor network corresponding to the partition function."""

    assert (p_depol >= 0.0) and (p_depol <= 1.0), "Probability must be valid."
    assert d > 1

    # Make a list of horizontal and vertical indices.
    tensor_per_side = 2 * d - 1
    horizontal_indices: List[List[str]] = []
    vertical_indices: List[List[str]] = []
    current_i: int = 0
    current_j: int = 0
    for i in range(tensor_per_side): # over rows
        this_row_horizontal_inds: List[str] = []
        this_row_vertical_inds: List[str] = []
        for j in range(tensor_per_side - 1): # over cols
            this_row_horizontal_inds.append(f"i{current_i}")
            this_row_vertical_inds.append(f"j{current_j}")
            current_i += 1
            current_j += 1
        # There is always one more vertical link than horizontal.
        # Compensate by adding another link.
        this_row_vertical_inds.append(f"j{current_j}")
        current_j += 1
        horizontal_indices.append(this_row_horizontal_inds)
        vertical_indices.append(this_row_vertical_inds)
    
    # Build each of the tensors.
    tensors: List[qtn.Tensor] = []
    for i in range(tensor_per_side): # over rows
        for j in range(tensor_per_side): # over cols
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                # This is an error tensors.
                data_qubit: cirq.Qid = 
                error_on_qubit: cirq.PauliString = 
                model = lambda err_pstring: independent_depolarizing_noise(err_pstring, p_depol)
                inds = 
                tensors.append(error_tensor(error_on_qubit, model, inds))
            else:
                # This is a Kronecker delta tensor.
                tensors.append()


if __name__ == "__main__":
    q = cirq.LineQubit(0)
    err = cirq.PauliString({q: cirq.X})
    build_network_for_error_class(err, 3)
