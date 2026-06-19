import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class KMeansClustering:
    def __init__(self, input_eigenvectors, output_clusters, n_clusters=2):
        self.input_file = input_eigenvectors
        self.output_file = output_clusters
        self.n_clusters = int(n_clusters)

    def run(self):
        # eigenvector dan pertahankan index
        df_preview = pd.read_csv(self.input_file, nrows=0)
        if "Unnamed: 0" in df_preview.columns:
            eigen_df = pd.read_csv(self.input_file, index_col=0)
        else:
            eigen_df = pd.read_csv(self.input_file)

        eigen_vectors = eigen_df.values
        if eigen_vectors.size == 0:
            raise ValueError("Empty eigenvector file")

        # Row-normalize
        Vn = normalize(eigen_vectors, axis=1, norm="l2")

        # n_init=50: Lebih banyak inisialisasi untuk hasil yang lebih stabil
        # max_iter=500: Iterasi untuk konvergensi
        # tol=1e-4: Tolerance yang ketat untuk konvergensi lebih baik
        # algorithm='lloyd': Algoritma klasik yang stabil
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=50,
            max_iter=500,
            tol=1e-4,
            algorithm="lloyd",
        )
        clusters = kmeans.fit_predict(Vn)

        # Simpan clusters dengan index
        out_df = pd.DataFrame({"cluster": clusters}, index=eigen_df.index)
        out_df.to_csv(self.output_file, index=True)

        # Hitung jumlah masing-masing cluster dan total data
        total_data = len(clusters)
        unique_clusters, cluster_counts = (
            pd.Series(clusters).value_counts().sort_index().index,
            pd.Series(clusters).value_counts().sort_index().values,
        )

        print(f"Total data yang diuji: {total_data}")
        for cluster_id, count in zip(unique_clusters, cluster_counts):
            print(f"Jumlah data di cluster {cluster_id}: {count}")

        print(f"Clustering telah disimpan ke {self.output_file}")
