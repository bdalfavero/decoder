from typing import List
import numpy as np
import jax
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator

def contract_2d_network(
    rows: int, cols: int, tn: qtn.TensorNetwork, chi: int,
    backend: str = "numpy", mps_order: str="lrp",
    mpo_order: str="lrud"
) -> float:
    """Contract the 2D network corresponding to the surface code decoder.
    This is done using the MPS-MPO-MPS method as detailed in the Bravyi decoding paper.
    We treat the first and last columns as MPS's, and each intervening column as an MPO.
    Contract each of the MPO's into the leftmost MPS/column, compressing after each contraction.
    Then overlap the rightmost column."""

    assert chi >= 1, "chi must be a positive integer."
    assert rows * cols == len(tn.tensors), "Network must have the correct number of tensors."
    assert backend in ["numpy", "jax"], "Backend must be numpy or jax."

    if backend == "jax":
        for t in tn.tensors:
            t.modify(apply=lambda d: jax.numpy.array(d))

    # Check the number of tensors in each column.
    for i in range(cols):
        tag = f"col{i}"
        assert tag in tn.tag_map.keys(), f"Network must have the tag {tag}"
        col_tensor_count = len([tn.tensors[i] for i in tn.tag_map[tag]])
        assert col_tensor_count == rows, \
            f"Column tag {tag} must have {rows} tensors, but has {col_tensor_count}."

    # Extract the first column.
    first_column_tensors = [tn.tensors[i] for i in tn.tag_map["col0"]]
    for t in first_column_tensors:
        print(t.inds, t.tags)
    first_col_tensor_data = [t.data for t in first_column_tensors]
    evolving_mps = qtn.tensor_1d.MatrixProductState(
        first_col_tensor_data, shape=mps_order,
        site_ind_id='k{}'
    )
    breakpoint()
    # Contract all of the other columns in, up to the last one.
    for i in range(1, cols - 1):
        print("i=", i)
        mpo_tensors: List[qtn.Tensor] = [tn.tensors[k] for k in tn.tag_map[f"col{i}"]]
        for t in mpo_tensors:
            print(t.inds, t.tags)
        mpo_tensor_data: List[np.ndarray] = [t.data for t in mpo_tensors]
        mpo = qtn.tensor_1d.MatrixProductOperator(
            mpo_tensor_data, shape=mpo_order,
            upper_ind_id='k{}', lower_ind_id='b{}'
        )
        evolving_mps.gate_with_mpo(mpo, max_bond=chi, inplace=True)
        breakpoint()
    # Compute the overlap with the last column.
    print("Last column.")
    last_col_tensors = [tn.tensors[i] for i in tn.tag_map[f"col{cols-1}"]]
    for t in last_col_tensors:
        print(t.inds, t.tags)
    last_col_tensor_data = [t.data for t in last_col_tensors]
    last_col_mps = qtn.tensor_1d.MatrixProductState(
        last_col_tensor_data, shape=mps_order,
        site_ind_id='k{}'
    )
    result = evolving_mps @ last_col_mps
    breakpoint()
    return result

if __name__ == "__main__":
    tn = qtn.TN2D_rand(4, 4, 2, x_tag_id="row{}", y_tag_id="col{}")
    #for i in range(4):
    #    print("i=", i)
    #    col_tensors = [tn.tensor_map[i] for i in tn.tag_map[f"col{i}"]]
    #    for t in col_tensors:
    #        print(t.inds)
    real_result = tn.contract()
    my_result = contract_2d_network(4, 4, tn, 1000000, mps_order="prl", mpo_order="durl")
    print(abs(real_result - my_result))