import numpy as np
import pandas as pd


class SpecralDegreeMatrix:
    def __init__(self, input_similarity, output_degree):
        self.input_file = input_similarity
        self.output_file = output_degree

    def run(self):
        W_df = pd.read_csv(self.input_file)
        W_numeric = W_df.select_dtypes(include=[np.number])
        if W_numeric.shape[1] == 0:
            # fallback: coerce all to numeric
            W_numeric = W_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        W = W_numeric.values

        # Kalkulasi degree matrix
        degree = np.sum(W, axis=1)
        D = np.diag(degree)

        # Simpan hasil ke file CSV
        pd.DataFrame(D).to_csv(self.output_file, index=False)
        print(f"Degree matrix telah disimpan ke {self.output_file}")
