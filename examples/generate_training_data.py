import numpy as np
import pandas as pd
import cirq
from decoder.surface_decoder import decode_representative
from decoder.error_model import independent_depolarizing_model

def dense_pauli_to_symplectic_array(ps: cirq.DensePauliString) -> np.ndarray:
    """Convert a pauli string"""

    nq = len(ps)
    symplectic_array = np.zeros(2 * nq, dtype=bool)
    for i, p in enumerate(ps.pauli_mask):
        if p == 1 or p == 2:
            symplectic_array[i] = True
        if p == 3 or p == 2:
            symplectic_array[i + nq] = True
    return symplectic_array

d = 3
qs = cirq.GridQubit.rect(2 * d + 1, 2 * d + 1)
data_qs = []
for q in qs:
    if q.row % 2 == q.col % 2:
        data_qs.append(q)

model = independent_depolarizing_model(data_qs, 0.1)

# Generate many random Pauli representatives, and decode each one.
nsamples = 10_000
paulis = []
error_classes = []
for _ in range(nsamples):
    mask = np.random.randint(0, high=3, size=len(data_qs))
    pstring = cirq.DensePauliString(mask)
    sparse_pstring = pstring.sparse(data_qs)
    error_class = decode_representative(3, sparse_pstring, model)
    paulis.append(pstring)
    error_classes.append(error_class)

binary_array = np.zeros((nsamples, len(2 * data_qs)), dtype=bool)
for i, ps in enumerate(paulis):
    binary_array[i, :] = dense_pauli_to_symplectic_array(ps)

# Save this to a file.
binary_cols = ["b" + str(i) for i in range(2 * len(data_qs))]
df = pd.DataFrame(binary_array, columns=binary_cols)
df["class"] = np.array(error_classes, dtype=int)
df.index.name = "i"
df.to_csv("training_data.csv")
