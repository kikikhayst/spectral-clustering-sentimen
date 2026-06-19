import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class SpectralSimilarity:
    def __init__(self, input_tfidf_vectorizer, output_similarity, sigma=None):
        self.input_file = input_tfidf_vectorizer
        self.output_file = output_similarity
        self.sigma = sigma

    def _extract_tfidf_matrix(self, df):
        metadata_columns = [
            "review_text",
            "sentiment_score",
            "positive_words_count",
            "negative_words_count",
            "sentiment_label",
            "Rating",
            "Product URL",
        ]

        # Ambil kolom TF-IDF
        tfidf_columns = [col for col in df.columns if col not in metadata_columns]
        if len(tfidf_columns) == 0:
            raise ValueError("TF-IDF tidak ditemukan di file input!")
        tfidf_matrix = df[tfidf_columns].values
        return tfidf_matrix

    def _calculate_sigma_heuristic(self, distances):
        # Ambil median dari semua jarak (tidak termasuk diagonal 0)
        non_zero_distances = distances[distances > 0]
        if len(non_zero_distances) > 0:
            sigma = np.median(non_zero_distances)
        else:
            sigma = 1.0  # fallback value
        return sigma

    def _gaussian_kernel(self, distances, sigma):
        # Gaussian/RBF kernel: exp(-distances^2 / (2 * sigma^2))
        gamma = 1.0 / (2 * sigma**2) if sigma != 0 else 1.0
        similarity_matrix = np.exp(-gamma * distances**2)
        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix

    def run(self):
        df = pd.read_csv(self.input_file)
        tfidf_matrix = self._extract_tfidf_matrix(df)
        print(f"Data shape: {tfidf_matrix.shape}")
        print(f"Value range: [{tfidf_matrix.min():.4f}, {tfidf_matrix.max():.4f}]")
        distances = euclidean_distances(tfidf_matrix)  # menghitung jarak Euclidean

        # Menentukan sigma (bandwidth)
        if self.sigma is None:
            sigma = self._calculate_sigma_heuristic(distances)
            print(f"Sigma dihitung otomatis: {sigma:.4f}")
        else:
            sigma = self.sigma
            print(f"Menggunakan sigma yang ditentukan: {sigma:.4f}")

        # Menghitung Gaussian similarity
        similarity_matrix = self._gaussian_kernel(distances, sigma)

        # Simpan hasil ke file CSV
        similarity_df = pd.DataFrame(similarity_matrix)
        similarity_df.to_csv(self.output_file, index=False)
        print(f"Gaussian similarity telah disimpan ke {self.output_file}")
        return similarity_matrix, sigma
