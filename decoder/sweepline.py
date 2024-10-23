from typing import List, Tuple
from copy import deepcopy
import numpy as np
from numba import njit
from scipy.linalg import svd
import quimb.tensor as qtn

@njit
def decompose_tensor_as_MPS(
    tensor: qtn.Tensor, 
    max_bond: int, 
    bond_ind_prefix: str='k'
) -> qtn.TensorNetwork:
    """Turn a single tensor into a MPS using the SVD."""

    mps_tensors = []
    working_tensor = deepcopy(tensor)
    for i, ind in enumerate(tensor.inds):
        if i < len(tensor.inds) - 1:
            # If we are not working on the last index,
            # separate the next leg, along with the MPS
            # bond from the last step, from the rest of the 
            # legs of the tensor.
            if i == 0:
                left_inds = [ind]
            else:
                left_inds = [ind, f"k{i-1}"]
            left_tensor, right_tensor = working_tensor.split(
                left_inds, get='tensors', absorb='right',
                max_bond=max_bond, bond_ind=f"k{i}"
            )
            mps_tensors.append(left_tensor)
            working_tensor = right_tensor
        else:
            mps_tensors.append(working_tensor)
    mps = qtn.TensorNetwork(mps_tensors)
    return mps


@njit
def sweepline_contract(
    tensor_net: qtn.TensorNetwork,
    coords: List[Tuple[float, float]],
    max_bond: int,
    mps_bond_prefix: str='k'
) -> qtn.Tensor:
    """Contract the tensor network using the sweepline algorithm from Chubb."""

    tensors_and_coords = [(t, x, y) for t, (x, y) in zip(tensor_net.tensors, coords)]
    tensors_and_coords.sort(key=lambda t: t[1]) # sort by y.
    # start the MPS using the first tensor in the network.
    tnet = qtn.TensorNetwork()
    for i, (tensor, x, y) in enumerate(tensors_and_coords):
        if i == 0:
            tnet.add_tensor(tensor)
        elif i == len(tensor_net.tensors) - 1:
            tnet.add_tensor(tensor)
            final_tensor = tnet.contract()
        else:
            tnet.add_tensor(tensor)
            contracted = tnet.contract()
            tnet = decompose_tensor_as_MPS(contracted, max_bond, mps_bond_prefix)
    return final_tensor