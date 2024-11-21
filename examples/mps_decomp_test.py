import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState
from decoder.sweepline import sweepline_contract, decompose_tensor_as_MPS

a = np.random.rand(3, 4, 5)
t = qtn.Tensor(a, ['b', 'c', 'd'])

mps = decompose_tensor_as_MPS(t, 1_000)
assert np.allclose(mps.contract().data, t.data)