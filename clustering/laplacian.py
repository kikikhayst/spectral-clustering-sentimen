import numpy as np
import pandas as pd


class SpectralLaplacian:
    def __init__(self, similarity_file, degree_file, output_file):
        self.similarity_file = similarity_file
        self.degree_file = degree_file
        self.output_file = output_file

    def run(self):
        # Membaca matriks similarity
        W_df = pd.read_csv(self.similarity_file)
        W_numeric = W_df.select_dtypes(include=[np.number])
        if W_numeric.shape[1] == 0:
            W_numeric = W_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        similarity_matrix = W_numeric.values

        # Membaca matriks degree
        D_df = pd.read_csv(self.degree_file)
        D_numeric = D_df.select_dtypes(include=[np.number])
        if D_numeric.shape[1] == 0:
            D_numeric = D_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        degree_matrix = D_numeric.values

        # Menghitung matriks Laplacian: L = D - W
        laplacian_matrix = degree_matrix - similarity_matrix

        # Simpan hasil ke file CSV
        pd.DataFrame(laplacian_matrix).to_csv(self.output_file, index=False)
        print(f"Laplacian telah disimpan ke {self.output_file}")
