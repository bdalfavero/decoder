from typing import List
import jax
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator

def put_mps_tensors_in_lrp_order(mps_tensors: List[qtn.Tensor]):
    """Put the MPS tensors in the lrp (left, right, physical) order
    that the QuIMB MPS constructor expects. We use the 'row' tag
    to define the order of tensors in the MPS."""

    tn = qtn.TensorNetwork(mps_tensors)
    output_tensor_list: List[qtn.Tensor] = [] # The list of tensors with swapped axes.
    for i in range(len(mps_tensors)):
        assert f"row{i}" in [tag for ten in mps_tensors for tag in ten.tags]
        site_i_tensor = tn.tensors[list(tn.tag_map[f"row{i}"])[0]]
        if i == 0:
            right_tensor = tn.tensors[list(tn.tag_map[f"row{i+1}"])[0]]
            physical_ind = list(set(site_i_tensor.inds) - set(right_tensor.inds))[0]
            bond_ind = list(set(site_i_tensor.inds) & set(right_tensor.inds))[0]
            swapped_tensor = site_i_tensor.copy().transpose((bond_ind, physical_ind))
        elif i == len(mps_tensors - 1):
            left_tensor = tn.tag_map[f"row{i-1}"]
        else:
            right_tensor = tn.tag_map[f"row{i+1}"]
            left_tensor = tn.tag_map[f"row{i-1}"]
    return output_tensor_list


def contract_2d_network(
    rows: int, cols: int, tn: qtn.TensorNetwork, chi: int,
    backend: str = "numpy"
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
        assert tag in tn.tag_map.keys(), "Network must have the tag col0"
        col_tensor_count = len([tn.tensors[i] for i in tn.tag_map[tag]])
        assert col_tensor_count == rows, \
            f"Column tag {tag} must have {rows} tensors, but has {col_tensor_count}."

    # Extract the first column.
    first_column_tensors = [tn.tensor_map[i] for i in tn.tag_map["col0"]]
    evolving_mps = qtn.tensor_1d.MatrixProductState([t.data for t in first_column_tensors])
    # Contract all of the other columns in, up to the last one.
    for i in range(1, cols - 1):
        mpo_tensors: List[qtn.Tensor] = [tn.tensor_map[k] for k in tn.tag_map[f"col{i}"]]
        mpo = qtn.tensor_1d.MatrixProductOperator([t.data for t in mpo_tensors])
        evolving_mps.gate_with_mpo(mpo, inplace=True, max_bond=chi)
    # Compute the overlap with the last column.
    last_col_tensors = [tn.tensor_map[i] for i in tn.tag_map[f"col{cols-1}"]]
    last_col_mps = qtn.tensor_1d.MatrixProductState([t.data for t in last_col_tensors])
    result = evolving_mps @ last_col_mps
    return result

if __name__ == "__main__":
    tn = qtn.TN2D_rand(3, 3, D=2, y_tag_id="col{}", x_tag_id="row{}")
    #exact_result = tn.contract()
    #my_result = contract_2d_network(3, 3, tn, 1_000_000)
    #print(exact_result - my_result)
    first_col_tensors = [tn.tensor_map[i] for i in tn.tag_map["col0"]]
    ordered_tensors = put_mps_tensors_in_lrp_order(first_col_tensors)