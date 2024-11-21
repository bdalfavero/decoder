from typing import List
import sys
import cupy as cp
from cupy.random import rand
from cuquantum import contract
sys.path.append("../decoder/")
from sweepline import sweepline_contract, get_free_indices

@dataclass
class TwoDTensorIndices:

    def __init__(self, north, east, south, west):
        self.north = north
        self.east = east
        self.south = south
        self.west = west

l = 2
chi = 5
tensors: List[cp.array] = []
inds: List[str] = []
index_offset = 0 # Offset for index characters.
for i in range(l):
    for j in range(l):
        pass

descriptor: str = ','.join(inds) + '->' + get_free_indices(inds)