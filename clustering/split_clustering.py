import os
import pandas as pd
from clustering.tfidf_modular import (
    TermFrequency,
    InverseDocumentFrequency,
    TFIDFVectorizer,
)
from clustering.gaussian_similarity import SpectralSimilarity
from clustering.cosine_similarity import SpectralSimilarity as CosineSimilarity
from clustering.degree import SpecralDegreeMatrix
from clustering.laplacian import SpectralLaplacian
from clustering.lap_norm import SpectralNormalizedLaplacian
from clustering.eigen import SpectralEigenvector
from clustering.clustering import KMeansClustering


class SplitLabelClustering:

    def __init__(
        self,
        labeled_split_dir,
        output_dir,
        similarity_type="_w_gaussian",
        n_clusters=2,
        n_components=2,
    ):
        self.labeled_split_dir = labeled_split_dir
        self.output_dir = output_dir
        self.similarity_type = similarity_type
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.clustering_results = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_clustering_for_label(self, label_file, label_name):
        print(f"\n")
        print(f"Melakukan Clustering untuk data {label_name.upper()}")
        print(f"")

        # Tentukan similarity module
        if self.similarity_type == "_w_gaussian":
            similarity_module = SpectralSimilarity
        elif self.similarity_type == "_w_cosine":
            similarity_module = CosineSimilarity
        else:
            similarity_module = SpectralSimilarity

        # Setup output paths
        output_tf = os.path.join(
            self.output_dir, f"tokopedia_tf_{label_name}{self.similarity_type}.csv"
        )
        output_idf = os.path.join(
            self.output_dir, f"tokopedia_idf_{label_name}{self.similarity_type}.csv"
        )
        output_tfidf = os.path.join(
            self.output_dir, f"tokopedia_tfidf_{label_name}{self.similarity_type}.csv"
        )
        output_similarity = os.path.join(
            self.output_dir,
            f"tokopedia_similarity_matrix_{label_name}{self.similarity_type}.csv",
        )
        output_degree = os.path.join(
            self.output_dir,
            f"tokopedia_degree_matrix_{label_name}{self.similarity_type}.csv",
        )
        output_laplacian = os.path.join(
            self.output_dir,
            f"tokopedia_laplacian_matrix_{label_name}{self.similarity_type}.csv",
        )
        output_lapnorm = os.path.join(
            self.output_dir,
            f"tokopedia_normalized_laplacian_matrix_{label_name}{self.similarity_type}.csv",
        )
        output_eigenvectors = os.path.join(
            self.output_dir,
            f"tokopedia_eigenvectors_{label_name}{self.similarity_type}.csv",
        )
        output_clusters = os.path.join(
            self.output_dir,
            f"tokopedia_clusters_{label_name}{self.similarity_type}.csv",
        )

        try:
            # Proses TF-IDF
            print(f"\nMenghitung Term Frequency untuk {label_name}...")
            tf_calculator = TermFrequency(label_file, output_tf, "review_text")
            tf_calculator.run()

            print(f"Menghitung Inverse Document Frequency untuk {label_name}...")
            idf_calculator = InverseDocumentFrequency(
                label_file, output_idf, "review_text"
            )
            idf_calculator.run()

            print(f"Melakukan TF-IDF Vectorization untuk {label_name}...")
            tfidf_vectorizer = TFIDFVectorizer(
                label_file,
                output_tfidf,
                "review_text",
                use_precomputed=True,
                input_tf=output_tf,
                input_idf=output_idf,
            )
            tfidf_vectorizer.run()

            # Proses Spectral Clustering
            print(f"Melakukan Similarity Calculation untuk {label_name}...")
            spectral_similarity = similarity_module(output_tfidf, output_similarity)
            spectral_similarity.run()

            print(
                f"Melakukan Degree & Laplacian Calculation untuk {label_name}..."
            )
            spectral_degree = SpecralDegreeMatrix(output_similarity, output_degree)
            spectral_degree.run()

            spectral_laplacian = SpectralLaplacian(
                output_similarity, output_degree, output_laplacian
            )
            spectral_laplacian.run()

            spectral_lapnorm = SpectralNormalizedLaplacian(
                output_laplacian, output_degree, output_lapnorm
            )
            spectral_lapnorm.run()

            eigenvector = SpectralEigenvector(
                output_lapnorm, output_eigenvectors, k=self.n_components
            )
            eigenvector.run()

            print(f"Melakukan K-Means Clustering untuk {label_name}...")
            spectral_clustering = KMeansClustering(
                output_eigenvectors, output_clusters, n_clusters=self.n_clusters
            )
            spectral_clustering.run()

            # Simpan hasil clustering
            self.clustering_results[label_name] = {
                "input_file": label_file,
                "output_clusters": output_clusters,
                "output_tfidf": output_tfidf,
                "output_eigenvectors": output_eigenvectors,
            }

            print(f"\nClustering untuk {label_name} berhasil!")
            return True

        except Exception as e:
            print(f"\nError saat clustering {label_name}: {str(e)}")
            return False

    def run(self):
        # Cari file labeled terpisah di direktori
        labeled_files = {
            "positif": os.path.join(
                self.labeled_split_dir, "tokopedia_labeled_positif.csv"
            ),
            "negatif": os.path.join(
                self.labeled_split_dir, "tokopedia_labeled_negatif.csv"
            ),
            "all": os.path.join(self.labeled_split_dir, "tokopedia_labeled_all.csv"),
        }

        success_count = 0
        for label_name, label_file in labeled_files.items():
            if os.path.exists(label_file):
                if self.run_clustering_for_label(label_file, label_name):
                    success_count += 1
            else:
                print(f"\n⚠ File tidak ditemukan: {label_file}")

        print(f"\n")
        print(
            f"Clustering selesai {success_count}/{len(labeled_files)} label berhasil diproses"
        )
        print(f"\n")

        return self.clustering_results
