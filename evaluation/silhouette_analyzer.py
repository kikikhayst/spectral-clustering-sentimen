import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import os


class SilhouetteAnalyzer:
    """Analisis Silhouette Score untuk evaluasi kualitas clustering"""

    def __init__(self, output_dir):

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.results = {}

    def calculate_silhouette(self, X, labels, label_name, similarity_type):
        try:
            # Hitung Silhouette Score keseluruhan
            score = silhouette_score(X, labels)

            # Hitung Silhouette Score per sample
            sample_scores = silhouette_samples(X, labels)

            # Statistik per cluster
            unique_labels = np.unique(labels)
            cluster_stats = []

            for cluster_id in unique_labels:
                mask = labels == cluster_id
                cluster_samples = sample_scores[mask]

                cluster_stats.append(
                    {
                        "Cluster": int(cluster_id),
                        "Size": np.sum(mask),
                        "Mean Silhouette": np.mean(cluster_samples),
                        "Min Silhouette": np.min(cluster_samples),
                        "Max Silhouette": np.max(cluster_samples),
                        "Std Silhouette": np.std(cluster_samples),
                    }
                )

            cluster_df = pd.DataFrame(cluster_stats)

            # Simpan hasil
            result_key = f"{label_name}_{similarity_type}"
            self.results[result_key] = {
                "score": score,
                "sample_scores": sample_scores,
                "cluster_stats": cluster_df,
                "n_clusters": len(unique_labels),
                "n_samples": len(X),
            }

            return {
                "score": score,
                "cluster_stats": cluster_df,
                "n_clusters": len(unique_labels),
            }

        except Exception as e:
            print(f"Error saat hitung Silhouette: {e}")
            return None

    def analyze_single_label(
        self, eigenvectors_file, clusters_file, label_name, similarity_type
    ):
        print("\n")
        print(f"Menganalisis Silhouette - {label_name} ({similarity_type})")

        try:
            # Load data
            X = pd.read_csv(eigenvectors_file, index_col=0).values
            clusters_df = pd.read_csv(clusters_file, index_col=0)
            labels = clusters_df.iloc[
                :, 0
            ].values  # Kolom pertama adalah cluster assignment

            # Hitung Silhouette
            result = self.calculate_silhouette(X, labels, label_name, similarity_type)

            if result:
                score = result["score"]
                cluster_df = result["cluster_stats"]
                n_clusters = result["n_clusters"]

                # Interpretasi score
                if score > 0.7:
                    interpretation = "EXCELLENT - Strong structure"
                elif score > 0.5:
                    interpretation = "GOOD - Well separated clusters"
                elif score > 0.3:
                    interpretation = "FAIR - Reasonable separation"
                elif score > 0:
                    interpretation = "WEAK - Overlapping clusters"
                else:
                    interpretation = "POOR - Clusters may be artificial"

                print(f"\nAnalisis selesai")
                print(f"Overall Silhouette Score: {score:.4f}")
                print(f"Interpretation: {interpretation}")
                print(f"Number of clusters: {n_clusters}")
                print(f"\nPer-Cluster Statistics:")
                print(cluster_df.to_string(index=False))

                return {
                    "score": score,
                    "interpretation": interpretation,
                    "cluster_stats": cluster_df,
                }
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def generate_report(self, output_file="silhouette_report.txt"):
        """Generate laporan Silhouette Score lengkap"""
        report_path = os.path.join(self.output_dir, output_file)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write("SILHOUETTE COEFFICIENT ANALYSIS REPORT\n")
            f.write("\n\n")

            f.write("INTERPRETATION GUIDE:\n")
            f.write("\n")
            f.write("  > 0.7:  EXCELLENT - Strong and well-separated clusters\n")
            f.write("  0.5-0.7: GOOD - Well separated clusters\n")
            f.write("  0.3-0.5: FAIR - Reasonable cluster separation\n")
            f.write("  0-0.3:   WEAK - Overlapping clusters\n")
            f.write("  < 0:     POOR - Clusters may be artificial\n\n")

            f.write("\n")
            f.write("ANALYSIS RESULTS\n")
            f.write("\n\n")

            for key, result in self.results.items():
                f.write(f"\n{key}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Overall Silhouette Score: {result['score']:.4f}\n")
                f.write(f"Number of Clusters: {result['n_clusters']}\n")
                f.write(f"Number of Samples: {result['n_samples']}\n")
                f.write(f"\nCluster-wise Statistics:\n")
                f.write(result["cluster_stats"].to_string(index=False))
                f.write("\n\n")

        print(f"\nReport tersimpan: {report_path}")
        return report_path

    def print_summary(self):
        print("\n")
        print("SILHOUETTE ANALYSIS SUMMARY")

        for key, result in self.results.items():
            score = result["score"]

            # Determine color/emoji based on score
            if score > 0.7:
                quality = "EXCELLENT"
            elif score > 0.5:
                quality = "GOOD"
            elif score > 0.3:
                quality = "FAIR"
            elif score > 0:
                quality = "WEAK"
            else:
                quality = "POOR"

            print(f"\n{key}:")
            print(f"  Score: {score:.4f} ({quality})")
            print(f"  Clusters: {result['n_clusters']}")
