import numpy as np
import pandas as pd


class SpectralNormalizedLaplacian:
    def __init__(self, input_laplacian, input_degree, output_lapnorm):
        self.input_file = input_laplacian
        self.degree_file = input_degree
        self.output_file = output_lapnorm

    def run(self):
        # Load Laplacian matrix robustly
        L_df = pd.read_csv(self.input_file)
        L_numeric = L_df.select_dtypes(include=[np.number])
        if L_numeric.shape[1] == 0:
            L_numeric = L_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        L = L_numeric.values

        # Load degree matrix dari file
        D_df = pd.read_csv(self.degree_file)
        D_numeric = D_df.select_dtypes(include=[np.number])
        if D_numeric.shape[1] == 0:
            D_numeric = D_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        degree_matrix = D_numeric.values
        # Jika degree diberikan sebagai vector/col, ubah menjadi array 1D
        if degree_matrix.ndim == 1 or degree_matrix.shape[0] != degree_matrix.shape[1]:
            degree = degree_matrix.flatten()
        else:
            degree = np.diag(degree_matrix).copy()

        # Handle zero-degree nodes
        degree[degree == 0] = 1e-10

        # Hitung D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

        # Hitung normalized Laplacian (Lnorm = D^(-1/2) L D^(-1/2))
        Lnorm = D_inv_sqrt @ L @ D_inv_sqrt

        # Paksa diagonal menjadi 1 (stabilitas numerik)
        np.fill_diagonal(Lnorm, 1.0)

        # Simpan hasil ke file CSV
        pd.DataFrame(Lnorm).to_csv(self.output_file, index=False)
        print(f"Normalized Laplacian telah disimpan ke {self.output_file}")
