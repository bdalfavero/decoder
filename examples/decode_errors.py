from typing import List
import numpy as np
import networkx as nx
import cirq
from decoder.surface_decoder import decode_representative, is_error_logical_bit_flip
from decoder.error_model import sample_surface_error, independent_depolarizing_noise

def main() -> None:
    d: int = 3
    ps: np.ndarray = np.linspace(1e-2, 0.4, num=10)
    p = ps[0]
    model = lambda e: independent_depolarizing_noise(e, p)
    err = sample_surface_error(d, p, False)
    # Get the error class (I, X, Y, or Z) and see if this error is a logical bit flip.
    predicted_err_class: int = decode_representative(d, err, p, model)
    error_flips_bit: bool = is_error_logical_bit_flip(d, err)
    # If the error class is X or Y, the correction flips the bit back after the error.
    err_class_flips_bit: bool = predicted_err_class == 1 or predicted_err_class ==2 
    print(err_class_flips_bit ^ error_flips_bit)

if __name__ == "__main__":
    main()