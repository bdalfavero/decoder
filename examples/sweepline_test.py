import sys
import cupy as cp
from cupy.random import rand
from cuquantum import contract
sys.path.append("../decoder/")
from sweepline import sweepline_contract

a = rand(5, 4, 3)
b = rand(4, 3, 2)
c = rand(2, 1)
inds = ["ijk", "jkl", "lm"]

contracted_indices, contracted_tensor = sweepline_contract([a, b, c], inds)

exact_contract = contract("ijk,jkl,lm->im", a, b, c)

assert contracted_tensor.shape == exact_contract.shape
assert cp.allclose(contracted_tensor, exact_contract)