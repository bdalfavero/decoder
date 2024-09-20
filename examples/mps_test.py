import sys
import cupy
from cupy.random import rand
from cuquantum import contract
sys.path.append("../decoder/")
from sweepline import tensor_to_mps

a = rand(3, 3, 4)
mps_tensors, mps_inds = tensor_to_mps(a, "ghi")

print(mps_inds)

a_contracted = contract('aA,AbB,Bc->abc', *mps_tensors)
assert cupy.allclose(a_contracted, a)