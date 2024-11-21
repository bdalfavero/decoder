from typing import List
import numpy as np
import networkx as nx
import cirq
from decoder.surface_decoder import decode_representative, is_error_logical_bit_flip
from decoder.error_model import sample_surface_error

def main() -> None:
    d: int = 3
    ps: np.ndarray = np.linspace(1e-2, 0.4, num=10)
    top_qs = [
        cirq.GridQubit(0, 0), cirq.GridQubit(0, 2), cirq.GridQubit(0, 4)
    ]
    #err = sample_surface_error(d, ps[-1], False)
    err = cirq.PauliString({q: cirq.X for q in top_qs})
    print(is_error_logical_bit_flip(d, err))

if __name__ == "__main__":
    main()