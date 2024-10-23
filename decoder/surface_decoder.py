from typing import List, Dict, Optional
import numpy as np
import cirq
import quimb.tensor as qtn
from decoder.error_model import ErrorModel, independent_depolarizing_noise

"""A tensor network decoder for the surface code, following Bravyi and Chubb.
For now, assumes i.i.d. noise."""

def kron_tensor(d: int, inds: List[str]) -> qtn.Tensor:
    """A tensor that implements the Kronecker functions \delta_{i,j,k,l} or \delta_{i,j,k},
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


def error_tensor(err: cirq.PauliString, err_model: ErrorModel, inds: Dict[str, Optional[str]]) -> qtn.Tensor:
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
    pauli_x = cirq.PauliString({q: cirq.X})
    pauli_y = cirq.PauliString({q: cirq.Y})
    pauli_z = cirq.PauliString({q: cirq.Z})

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


if __name__ == "__main__":
    #delta = kron_tensor(2, ["i", "j", "k"])
    q = cirq.LineQubit(0)
    err = cirq.PauliString({q: cirq.X})
    model = lambda err_str: independent_depolarizing_noise(err_str, 0.1)
    err_tensor = error_tensor(err, model, {"n": "i", "s": "k", "e": "j", "w": None})
    print(err_tensor)