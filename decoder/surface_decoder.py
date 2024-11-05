"""A tensor network decoder for the surface code, following Bravyi and Chubb.
For now, assumes i.i.d. noise."""

from typing import List, Dict, Optional
import numpy as np
import cirq
import quimb.tensor as qtn
from decoder.error_model import ErrorModel, independent_depolarizing_noise
from decoder.helpers import raise_pauli_to_power

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
    err: cirq.PauliString, err_model: ErrorModel, num_inds: int,
    north_south_is_x: bool
) -> qtn.Tensor:
    """The tensor encoding probabilities of errors in Bravyi and Chubb.

    The tensor passed will have indices that are a subset of {i, j, k, l},
    following the clockwise order of Eqs 39-41 of Bravyi et al.
    If num_inds == 2, we return a tensor with indices (j, k).
    If num_inds == 3, we return a tensor with indices (i, j, k).
    If num_inds == 4, we return a tensor with indices (i, j, k, l).
    
    Arguments:
    err - A PauliString acting on just the data qubit for this tensor.
    err_model - an ErrorModel the decoder uses.
    num_inds - number of indices in the tensor.
    north_south_is_x - whether the north and south indices correspond to X
    (if True is passed) and the east and west to Z, or the north and south 
    correspond to Z while the east and west are X.
    
    Returns:
    err_tensor - a quimb Tensor for the TN decoder."""

    assert len(err.qubits) <= 1, "Error must have only one qubit."

    if len(err.qubits) == 1:
        q = err.qubits[0]
    else:
        q = cirq.LineQubit(0)
    pauli_x: cirq.PauliString = cirq.PauliString({q: cirq.X})
    pauli_z: cirq.PauliString = cirq.PauliString({q: cirq.Z})

    if num_inds == 2:
        # The first index is the horizontal one, and the second is the vertical one.
        tensor_data = np.zeros((2, 2), dtype=complex)
        for j in range(2): # East index.
            for k in range(2): # South index.
                if north_south_is_x:
                    pz_to_j = raise_pauli_to_power(pauli_z, j)
                    px_to_k = raise_pauli_to_power(pauli_x, k)
                    tensor_data[j, k] = err_model(err * pz_to_j * px_to_k)
                else:
                    #tensor_data[j, k] = err_model(err * pauli_x ** j * pauli_z ** k)
                    pz_to_k = raise_pauli_to_power(pauli_z, k)
                    px_to_j = raise_pauli_to_power(pauli_x, j)
                    tensor_data[j, k] = err_model(err * pz_to_k * px_to_j)
        return qtn.Tensor(tensor_data, inds=["j", "k"])
    elif num_inds == 3:
        tensor_data = np.zeros((2, 2, 2), dtype=complex)
        for i in range(2): # North index.
            for j in range(2): # East index.
                for k in range(2): # South index.
                    if north_south_is_x:
                        pz_to_j = raise_pauli_to_power(pauli_z, j)
                        px_to_ik = raise_pauli_to_power(pauli_x, (i+k))
                        #tensor_data[i, j, k] = err_model(err * pauli_z ** j * pauli_x ** (i + k))
                        tensor_data[i, j, k] = err_model(err * pz_to_j * px_to_ik) 
                    else:
                        px_to_j = raise_pauli_to_power(pauli_x, j)
                        pz_to_ik = raise_pauli_to_power(pauli_z, (i+k))
                        #tensor_data[i, j, k] = err_model(err * pauli_x ** j * pauli_z ** (i + k))
                        tensor_data[i, j, k] = err_model(err * px_to_j * pz_to_ik)
        return qtn.Tensor(tensor_data, inds=["i", "j", "k"])
    elif num_inds == 4:
        tensor_data = np.zeros((2, 2, 2, 2), dtype=complex)
        for i in range(2): # North index.
            for j in range(2): # East index.
                for k in range(2): # South index.
                    for l in range(2): # West index.
                        if north_south_is_x:
                            pz_to_jl = raise_pauli_to_power(pauli_z, j + l)
                            px_to_ik = raise_pauli_to_power(pauli_x, i + k)
                            #tensor_data[i, j, k, l] = err_model(err * pauli_z ** (j + l) * pauli_x ** (i + k))
                            tensor_data[i, j, k, l] = err_model(err * pz_to_jl * px_to_ik)
                        else:
                            px_to_jl = raise_pauli_to_power(pauli_x, j + l)
                            pz_to_ik = raise_pauli_to_power(pauli_z, i + k)
                            #tensor_data[i, j, k, l] = err_model(err * pauli_x ** (j + l) * pauli_z ** (i + k))
                            tensor_data[i, j, k, l] = err_model(err * px_to_jl * pz_to_ik)
        return qtn.Tensor(tensor_data, inds=["i", "j", "k", "l"])
    else:
        raise ValueError(f"Number of indices {num_inds} is invalide. Must be 2, 3, or 4.")


def build_network_for_error_class(qs: List[cirq.Qid], err: cirq.PauliString, d: int, p_depol: float) -> qtn.TensorNetwork:
    """Builds the 2D PEPS for the RBIM partition function.
    
    Arguments:
    qs - List of a data qubits for the code.
    err - A Pauli string acting on qubits of the code. Represents the 'base error' by which
    stabilizers and logicals are multiplied.
    d - distance of the code.
    p_depol - depolarizing probability
    
    Returns:
    peps - tensor network corresponding to the partition function."""

    assert set(err.keys()).issubset(set(qs)), "Error qubits must be a subset of the data qubits."
    assert (p_depol >= 0.0) and (p_depol <= 1.0), "Probability must be valid."
    assert d >= 3, "Distance must be greater than 3."

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
            this_row_horizontal_inds.append(f"k{current_i}")
            this_row_vertical_inds.append(f"l{current_j}")
            current_i += 1
            current_j += 1
        # There is always one more vertical link than horizontal.
        # Compensate by adding another link.
        this_row_vertical_inds.append(f"l{current_j}")
        current_j += 1
        horizontal_indices.append(this_row_horizontal_inds)
        vertical_indices.append(this_row_vertical_inds)
    
    # Build each of the tensors.
    data_qubit_index = 0 # Track which data qubit we are currently building a tensor for.
    tensors: List[qtn.Tensor] = []
    for i in range(tensor_per_side): # over rows
        for j in range(tensor_per_side): # over cols
            # Get a list of the indices for this tensor.
            # These are ordered [north, south, east, west].
            inds: List[str] = []
            if i != 0: # North
                inds.append(vertical_indices[i-1][j])
            if i != tensor_per_side - 1: # South
                inds.append(vertical_indices[i][j])
            if j != tensor_per_side - 1: # East
                inds.append(horizontal_indices[i][j])
            if j != 0: # East
                inds.append(horizontal_indices[i][j-1])
            #print(i, j, inds)
            # Build the tensor (either an error tensor or a kroncker tensor).
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                # This is an error tensor.
                if qs[data_qubit_index] in err.keys():
                    local_err = cirq.PauliString({qs[data_qubit_index]: err[qs[data_qubit_index]]})
                else:
                    local_err = cirq.PauliString({qs[data_qubit_index]: cirq.I})
                # Tensor is either "h" (horizontal) or "v" (vertical)
                # If the tensor is "h," then east and west are X, north and south are Z.
                # If the tensor is "v," then east and west are Z, north and south are X.
                tensor_is_h = (i % 2 == 0) and (j % 2 == 0)
                model = lambda e: independent_depolarizing_noise(e, p_depol)
                #breakpoint()
                tensor = error_tensor(local_err, model, len(inds), not tensor_is_h)
                # The above function assigns generic indices. Change these for specific tensors.
                if len(inds) == 2:
                    tensor.reindex({"i": inds[0], "j": inds[1]}, inplace=True)
                elif len(inds) == 3:
                    if i == 0 or i == tensor_per_side - 1:
                        # Top or bottom edge.
                        # This tensor has either North or South, 
                        # and both East and West.
                        # i and k are East and West, j is North or South.
                        ind_dict = {"i": inds[1], "j": inds[0], "k": inds[2]}
                    else:
                        # Left or right edge.
                        # This tensor has both North and South indices,
                        # And either one of East or West.
                        # i and k are North and South.
                        # j is either East or West.
                        ind_dict = {"i": inds[0], "j": inds[2], "k": inds[1]}
                    tensor.reindex(ind_dict, inplace=True)
                else:
                    tensor.reindex(dict(zip(["i", "j", "k", "l"], inds)), inplace=True)
                if tensor_is_h:
                    tensor.add_tag("h")
                else:
                    tensor.add_tag("v")
                data_qubit_index += 1
            else:
                # This is a Kronecker delta tensor.
                tensor = kron_tensor(2, inds)
                tensor.add_tag("s")
            tensor.add_tag(f"i{i}")
            tensor.add_tag(f"j{j}")
            tensors.append(tensor)
    return qtn.TensorNetwork(tensors)


if __name__ == "__main__":
    qs = cirq.LineQubit.range(13)
    q = cirq.LineQubit(0)
    err = cirq.PauliString({q: cirq.X})
    build_network_for_error_class(qs, err, 3, 0.1)
