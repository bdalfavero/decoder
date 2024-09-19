import numpy as np
import scipy.linalg as la
import quimb.tensor as qtn
from decoder.sweepline import sweepline_contract

a1 = np.random.rand(2, 3)
t1 = qtn.Tensor(a1, ['a', 'b'])
a2 = np.random.rand(3, 100, 50).astype(float)
t2 = qtn.Tensor(a2, ['b', 'c', 'd'])
a3 = np.random.rand(100, 50).astype(float)
t3 = qtn.Tensor(a3, ['c', 'd'])

tensor_network = t1 & t2 & t3
coords = [(0, 0), (0, 1), (0, 2)]

for max_bond in range(1, 100):
    result = sweepline_contract(tensor_network, coords, max_bond, 'k')
    builtin_result = tensor_network.contract()
    assert result.shape == builtin_result.data.shape
    result_reshaped = result.data.reshape(result.data.size)
    builtin_reshaped = builtin_result.data.reshape(builtin_result.data.size)
    print(max_bond, la.norm(result_reshaped - builtin_reshaped))