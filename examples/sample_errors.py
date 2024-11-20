import numpy as np
import pandas as pd
import cirq
from decoder.error_model import independent_bit_flip_noise, independent_depolarizing_noise
from decoder.error_model import sample_surface_error
from decoder.surface_decoder import decode_representative

def main() -> None:
    ps = np.linspace(1e-3, 0.4, num=10)
    count_arr = np.zeros((ps.size, 4), dtype=int)
    for i, p in enumerate(ps):
        #model = lambda e: independent_bit_flip_noise(e, p)
        model = lambda e: independent_depolarizing_noise(e, p)
        err_classes = []
        for _ in range(1000):
            err = sample_surface_error(3, p, False)
            err_class = decode_representative(3, err, p, model)
            err_classes.append(err_class)
        # Count the errors.
        count_dict = {}
        for j in range(4):
            count_arr[i, j] += err_classes.count(j)
    
    df = pd.DataFrame(data=count_arr, index=pd.Index(ps, name="p"), columns=['I','X','Y','Z'])
    df.to_csv("../data/error_counts.csv")

if __name__ == "__main__":
    main()