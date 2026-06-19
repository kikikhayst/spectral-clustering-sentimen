import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import rcParams

# Set font untuk mendukung Unicode
rcParams["font.sans-serif"] = ["Arial"]


class WordCloudVisualizer:

    def __init__(self, output_dir="evaluation/wordcloud_outputs"):
        self.output_dir = output_dir
        self._create_output_dir()

    def _create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Output directory dibuat: {self.output_dir}")

    def visualize_single_wordcloud(
        self, text_data, title="WordCloud", figsize=(15, 8), filename=None
    ):
        # Convert list ke string jika perlu
        if isinstance(text_data, list):
            text_data = " ".join(str(item) for item in text_data)

        # Create wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
        ).generate(str(text_data))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.axis("off")
        plt.tight_layout()

        # Save if filename provided
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✓ WordCloud tersimpan: {filepath}")

        plt.show()

    def visualize_csv_wordcloud(
        self, csv_path, text_column, title="WordCloud from CSV", filename=None
    ):
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV loaded: {csv_path}")
            print(f"Data shape: {df.shape}")
        except FileNotFoundError:
            print(f"CSV file tidak ditemukan: {csv_path}")
            return

        # Check column exists
        if text_column not in df.columns:
            print(f"Kolom '{text_column}' tidak ditemukan di CSV. Kolom tersedia:")
            print(df.columns.tolist())
            return

        # Combine text data
        text_data = " ".join(str(item) for item in df[text_column].dropna().tolist())

        # Create wordcloud
        self.visualize_single_wordcloud(text_data, title, filename=filename)

    def visualize_by_label(self, csv_path, text_column, label_column, figsize=(20, 12)):
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV loaded: {csv_path}")
        except FileNotFoundError:
            print(f"CSV file tidak ditemukan: {csv_path}")
            return

        # Check columns exist
        if text_column not in df.columns or label_column not in df.columns:
            print(f"Kolom tidak ditemukan di CSV. Kolom tersedia:")
            print(df.columns.tolist())
            return

        # Get unique labels
        unique_labels = df[label_column].unique()
        n_labels = len(unique_labels)

        # Create subplots
        fig, axes = plt.subplots(1, n_labels, figsize=figsize)
        if n_labels == 1:
            axes = [axes]

        # Create wordcloud untuk setiap label
        for idx, label in enumerate(unique_labels):
            # Filter data by label
            label_data = df[df[label_column] == label]
            text_data = " ".join(
                str(item) for item in label_data[text_column].dropna().tolist()
            )

            # Create wordcloud
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color="white",
                colormap="plasma",
                max_words=80,
                relative_scaling=0.5,
                min_font_size=10,
            ).generate(text_data)

            # Display on subplot
            axes[idx].imshow(wordcloud, interpolation="bilinear")
            axes[idx].set_title(
                f"WordCloud - {label} (n={len(label_data)})",
                fontsize=14,
                fontweight="bold",
            )
            axes[idx].axis("off")

        plt.tight_layout()

        # Save
        filename = f"wordcloud_by_label_{label_column}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"WordCloud tersimpan: {filepath}")

        plt.show()

    def visualize_by_cluster(
        self,
        clustered_csv_path,
        text_column,
        cluster_column="cluster",
        figsize=(20, 12),
        similarity_type="_w_cosine",
        n_clusters=None,
    ):
        # Read CSV
        try:
            df = pd.read_csv(clustered_csv_path)
            print(f"CSV loaded: {clustered_csv_path}")
        except FileNotFoundError:
            print(f"CSV file tidak ditemukan: {clustered_csv_path}")
            return

        # Check columns exist
        if text_column not in df.columns or cluster_column not in df.columns:
            print(f"Kolom tidak ditemukan di CSV. Kolom tersedia:")
            print(df.columns.tolist())
            return

        # Get unique clusters
        unique_clusters = sorted(df[cluster_column].unique())
        n_clusters_actual = len(unique_clusters)
        if n_clusters is None:
            n_clusters = n_clusters_actual

        # Create subplots
        fig, axes = plt.subplots(1, n_clusters_actual, figsize=figsize)
        if n_clusters_actual == 1:
            axes = [axes]

        # Create wordcloud untuk setiap cluster
        colormap_list = ["viridis", "plasma", "inferno", "magma", "cividis"]

        for idx, cluster in enumerate(unique_clusters):
            # Filter data by cluster
            cluster_data = df[df[cluster_column] == cluster]
            text_data = " ".join(
                str(item) for item in cluster_data[text_column].dropna().tolist()
            )

            # Select colormap
            colormap = colormap_list[idx % len(colormap_list)]

            # Create wordcloud
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color="white",
                colormap=colormap,
                max_words=80,
                relative_scaling=0.5,
                min_font_size=10,
            ).generate(text_data)

            # Display on subplot
            axes[idx].imshow(wordcloud, interpolation="bilinear")
            axes[idx].set_title(
                f"{n_clusters}cluster_{similarity_type.replace('_w_', '')} - Cluster {cluster} (n={len(cluster_data)})",
                fontsize=14,
                fontweight="bold",
            )
            axes[idx].axis("off")

        plt.tight_layout()

        # Save
        filename = f"wordcloud_by_cluster_{n_clusters}cluster{similarity_type}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"WordCloud tersimpan: {filepath}")

        plt.show()

    def visualize_by_cluster_with_merge(
        self,
        labeled_csv_path,
        clusters_csv_path,
        text_column,
        cluster_column="cluster",
        label_name=None,
        figsize=(20, 12),
        similarity_type="_w_cosine",
        n_clusters=2,
    ):
        # Read both CSV files
        try:
            df_labeled = pd.read_csv(labeled_csv_path)
            print(f"Labeled CSV loaded: {labeled_csv_path}")
        except FileNotFoundError:
            print(f"File tidak ditemukan: {labeled_csv_path}")
            return

        try:
            df_clusters = pd.read_csv(clusters_csv_path)
            print(f"Clusters CSV loaded: {clusters_csv_path}")
        except FileNotFoundError:
            print(f"File tidak ditemukan: {clusters_csv_path}")
            return

        # Check text column exists
        if text_column not in df_labeled.columns:
            print(f"Kolom '{text_column}' tidak ditemukan di labeled CSV")
            print(f"Kolom tersedia: {df_labeled.columns.tolist()}")
            return

        # Check cluster column exists
        if cluster_column not in df_clusters.columns:
            print(f"Kolom '{cluster_column}' tidak ditemukan di clusters CSV")
            print(f"Kolom tersedia: {df_clusters.columns.tolist()}")
            return

        # Merge data (reset index untuk alignment)
        if len(df_labeled) != len(df_clusters):
            print(
                f"Warning: Data size mismatch (labeled: {len(df_labeled)}, clusters: {len(df_clusters)})"
            )

        df_labeled = df_labeled.reset_index(drop=True)
        df_clusters = df_clusters.reset_index(drop=True)

        # Merge by index
        df_merged = df_labeled.copy()
        df_merged[cluster_column] = df_clusters[cluster_column]

        print(f"Data merged: {len(df_merged)} records")

        # Get unique clusters
        unique_clusters = sorted(df_merged[cluster_column].unique())
        n_clusters = len(unique_clusters)

        print(f"Found {n_clusters} clusters")

        # Create subplots
        fig, axes = plt.subplots(1, n_clusters, figsize=figsize)
        if n_clusters == 1:
            axes = [axes]

        # Create wordcloud untuk setiap cluster
        colormap_list = ["viridis", "plasma", "inferno", "magma", "cividis"]

        for idx, cluster in enumerate(unique_clusters):
            # Filter data by cluster
            cluster_data = df_merged[df_merged[cluster_column] == cluster]
            text_data = " ".join(
                str(item) for item in cluster_data[text_column].dropna().tolist()
            )

            # Select colormap
            colormap = colormap_list[idx % len(colormap_list)]

            # Create wordcloud
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color="white",
                colormap=colormap,
                max_words=80,
                relative_scaling=0.5,
                min_font_size=10,
            ).generate(text_data)

            # Display on subplot
            axes[idx].imshow(wordcloud, interpolation="bilinear")
            axes[idx].set_title(
                f"{n_clusters}cluster_{similarity_type.replace('_w_', '')} - Cluster {cluster} (n={len(cluster_data)})",
                fontsize=14,
                fontweight="bold",
            )
            axes[idx].axis("off")

        plt.tight_layout()

        # Save dengan nama yang berbeda berdasarkan label
        if label_name:
            filename = f"wordcloud_by_cluster_{label_name}_{n_clusters}cluster{similarity_type}.png"
        else:
            filename = f"wordcloud_by_cluster_merged_{n_clusters}cluster{similarity_type}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"WordCloud tersimpan: {filepath}")

        plt.show()

    def create_comparison_wordcloud(
        self,
        csv_path_list,
        text_column,
        label_list,
        figure_title="WordCloud Comparison",
    ):
        n_files = len(csv_path_list)

        # Create subplots
        fig, axes = plt.subplots(1, n_files, figsize=(20, 8))
        if n_files == 1:
            axes = [axes]

        # Create wordcloud untuk setiap file
        colormap_list = ["viridis", "plasma", "inferno", "magma", "cividis"]

        for idx, (csv_path, label) in enumerate(zip(csv_path_list, label_list)):
            try:
                df = pd.read_csv(csv_path)
                text_data = " ".join(
                    str(item) for item in df[text_column].dropna().tolist()
                )

                colormap = colormap_list[idx % len(colormap_list)]

                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color="white",
                    colormap=colormap,
                    max_words=80,
                    relative_scaling=0.5,
                    min_font_size=10,
                ).generate(text_data)

                axes[idx].imshow(wordcloud, interpolation="bilinear")
                axes[idx].set_title(label, fontsize=14, fontweight="bold")
                axes[idx].axis("off")

                print(f"Processed: {label} ({len(df)} records)")

            except FileNotFoundError:
                print(f"File tidak ditemukan: {csv_path}")
                axes[idx].text(0.5, 0.5, "File Not Found", ha="center", va="center")
                axes[idx].axis("off")

        plt.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        # Save
        filename = f"wordcloud_comparison_{n_files}files.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Comparison WordCloud tersimpan: {filepath}")

        plt.show()


if __name__ == "__main__":
    # Example usage
    visualizer = WordCloudVisualizer()