import numpy as np
from scipy.linalg import norm
import quimb.tensor as qtn
from decoder.sweepline import sweepline_contract

L = 3
chi = 5
tn = qtn.tensor_2d.PEPS.rand(L, L, chi, phys_dim=2)
coords = []
for i in range(L):
    for j in range(L):
        coords.append((i, j))

exact_contraction = tn.contract()

for max_bond in range(100):
    approx_contraction = sweepline_contract(tn, coords, max_bond)
    print(max_bond, norm(approx_contraction.data - exact_contraction.data))
    #print(np.max(approx_contraction.data - exact_contraction.data))