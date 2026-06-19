import pandas as pd
import numpy as np


class SpectralEigenvector:
    def __init__(self, input_lapnorm, output_eigenvectors, k=2):
        self.input_file = input_lapnorm
        self.output_file = output_eigenvectors
        self.k = int(k)

    def run(self):
        # Baca matriks Laplacian dari file CSV
        lapnorm_df = pd.read_csv(self.input_file)
        lapnorm_matrix = lapnorm_df.values
        # Validasi matriks berbentuk kotak
        n_rows, n_cols = lapnorm_matrix.shape
        if n_rows != n_cols:
            raise ValueError(
                f"Laplacian matrix must be square, got shape {lapnorm_matrix.shape}"
            )
        n = n_rows
        # Hitung eigenvalue dan eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(lapnorm_matrix)
        # Tentukan k yang valid tidak boleh >= n
        if self.k <= 0:
            raise ValueError("k must be a positive integer")
        k_use = min(self.k, n - 1)  # ambil paling banyak n-1 untuk clustering
        if k_use != self.k:
            print(
                f"[Warning] Requested k={self.k} terlalu besar untuk n={n}, menggunakan k={k_use} instead"
            )
        # Ambil k eigenvector terkecil
        smallest_eigenvectors = eigenvectors[:, :k_use]
        # Simpan hasil ke file CSV
        pd.DataFrame(smallest_eigenvectors).to_csv(self.output_file, index=False)
        print(f"Eigenvector telah disimpan ke {self.output_file}")
