import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Set font untuk mendukung Unicode
rcParams["font.sans-serif"] = ["DejaVu Sans"]


class ClusteringScatterPlotter:

    def __init__(
        self,
        clustering_results,
        output_dir,
        similarity_type="_w_gaussian",
        n_clusters=2,
    ):
        self.clustering_results = clustering_results
        self.output_dir = output_dir
        self.similarity_type = similarity_type
        self.n_clusters = n_clusters
        self.similarity_name = (
            "gaussian" if "_gaussian" in similarity_type else "cosine"
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_single_label(self, label_name, label_results, figsize=(12, 8)):
        try:
            print(f"\nMembuat scatter plot untuk label {label_name.upper()}...")

            # Load data clusters dan eigenvectors
            clusters_file = label_results["output_clusters"]
            eigenvectors_file = label_results["output_eigenvectors"]

            if not os.path.exists(clusters_file) or not os.path.exists(
                eigenvectors_file
            ):
                print(f"File tidak ditemukan untuk label {label_name}")
                return False

            clusters_df = pd.read_csv(clusters_file)
            eigenvectors_df = pd.read_csv(eigenvectors_file)

            # Ambil 2 eigenvector pertama untuk dimensionality reduction
            if eigenvectors_df.shape[1] < 2:
                print(
                    f"Eigenvectors tidak cukup untuk scatter plot (hanya {eigenvectors_df.shape[1]} kolom)"
                )
                return False

            x = eigenvectors_df.iloc[:, 0].values  # Eigenvector 1
            y = eigenvectors_df.iloc[:, 1].values  # Eigenvector 2
            clusters = clusters_df["cluster"].values  # Cluster assignments

            # Buat figure
            fig, ax = plt.subplots(figsize=figsize)

            # Define color palette
            unique_clusters = np.unique(clusters)
            colors = sns.color_palette("husl", len(unique_clusters))
            cluster_colors = {
                cluster: colors[i] for i, cluster in enumerate(unique_clusters)
            }

            # Plot data points
            for cluster in unique_clusters:
                mask = clusters == cluster
                ax.scatter(
                    x[mask],
                    y[mask],
                    c=[cluster_colors[cluster]],
                    label=f"Cluster {int(cluster)}",
                    s=50,
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=0.5,
                )

            # Konfigurasi plot
            ax.set_xlabel("Eigenvector 1", fontsize=12, fontweight="bold")
            ax.set_ylabel("Eigenvector 2", fontsize=12, fontweight="bold")
            ax.set_title(
                f"Clustering Results - {label_name.upper()} Reviews",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")

            # Add statistics
            stats_text = (
                f"Total Documents: {len(clusters)}\nClusters: {len(unique_clusters)}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

            # Simpan figure dengan nama yang berisi similarity type dan n_clusters
            output_file = os.path.join(
                self.output_dir,
                f"clustering_scatter_{label_name}_{self.similarity_name}_{self.n_clusters}k.png",
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Scatter plot untuk {label_name} tersimpan: {output_file}")
            return True

        except Exception as e:
            print(f"Error membuat scatter plot untuk {label_name}: {str(e)}")
            return False

    def plot_comparison(self, figsize=(16, 6)):
        try:
            print("\nMembuat comparison scatter plot...")

            fig, axes = plt.subplots(1, len(self.clustering_results), figsize=figsize)

            # Jika hanya 1 label, axes bukan array
            if len(self.clustering_results) == 1:
                axes = [axes]

            for idx, (label_name, label_results) in enumerate(
                self.clustering_results.items()
            ):
                clusters_file = label_results["output_clusters"]
                eigenvectors_file = label_results["output_eigenvectors"]

                if not os.path.exists(clusters_file) or not os.path.exists(
                    eigenvectors_file
                ):
                    continue

                clusters_df = pd.read_csv(clusters_file)
                eigenvectors_df = pd.read_csv(eigenvectors_file)

                if eigenvectors_df.shape[1] < 2:
                    continue

                x = eigenvectors_df.iloc[:, 0].values
                y = eigenvectors_df.iloc[:, 1].values
                clusters = clusters_df["cluster"].values

                ax = axes[idx]

                # Plot
                unique_clusters = np.unique(clusters)
                colors = sns.color_palette("husl", len(unique_clusters))
                cluster_colors = {
                    cluster: colors[i] for i, cluster in enumerate(unique_clusters)
                }

                for cluster in unique_clusters:
                    mask = clusters == cluster
                    ax.scatter(
                        x[mask],
                        y[mask],
                        c=[cluster_colors[cluster]],
                        label=f"Cluster {int(cluster)}",
                        s=50,
                        alpha=0.6,
                        edgecolors="black",
                        linewidth=0.5,
                    )

                ax.set_xlabel("Eigenvector 1", fontsize=11, fontweight="bold")
                ax.set_ylabel("Eigenvector 2", fontsize=11, fontweight="bold")
                ax.set_title(
                    f"{label_name.upper()} Reviews\n({len(clusters)} documents, {len(unique_clusters)} clusters)",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3, linestyle="--")

            plt.subplots_adjust(
                left=0.08, right=0.95, top=0.92, bottom=0.12, wspace=0.3
            )

            # Simpan figure dengan nama yang berisi similarity type dan n_clusters
            output_file = os.path.join(
                self.output_dir,
                f"clustering_scatter_comparison_{self.similarity_name}_{self.n_clusters}k.png",
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Comparison scatter plot tersimpan: {output_file}")
            return True

        except Exception as e:
            print(f"Error membuat comparison plot: {str(e)}")
            return False

    def run(self):
        print("\nVISUALISASI HASIL CLUSTERING")

        success_count = 0

        # Plot untuk setiap label
        for label_name, label_results in self.clustering_results.items():
            if self.plot_single_label(label_name, label_results):
                success_count += 1

        # Plot comparison
        if len(self.clustering_results) > 1:
            if self.plot_comparison():
                success_count += 1

        print("\n")
        print(
            f"Visualisasi selesai! {success_count}/{len(self.clustering_results)} label berhasil divisualisasikan"
        )

        return True
