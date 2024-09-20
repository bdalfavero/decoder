from typing import List
import sys
import cupy as cp
from cupy.random import rand
from cuquantum import contract
sys.path.append("../decoder/")
from sweepline import sweepline_contract, get_free_indices

l = 3
chi = 5
tensors: List[cp.array] = []
inds: List[str] = []
for i in range(l):
    for j in range(l):
        ten = rand()

descriptor: str = ','.join(inds) + '->' + get_free_indices(inds)