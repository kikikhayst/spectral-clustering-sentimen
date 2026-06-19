import os
import pandas as pd
from evaluation.wordcloud_visualization import WordCloudVisualizer


class WordCloudRunner:

    def __init__(self, output_dir="results", wordcloud_output_dir="evaluation/wordcloud_outputs"):
        self.output_dir = output_dir
        self.wordcloud_output_dir = wordcloud_output_dir
        self.visualizer = WordCloudVisualizer(output_dir=wordcloud_output_dir)

    def run(self, similarity_type="_w_cosine", n_clusters=2):
        print("\n")
        print("MEMULAI WORDCLOUD VISUALIZATION")

        # Labels untuk diproses
        labels = ["positif", "negatif", "all"]

        # 1. Visualisasi WordCloud by Label (dari labeled data)
        print("\nMembuat WordCloud by Sentiment Label")
        self._visualize_by_sentiment_label()

        # 2. Visualisasi WordCloud by Cluster untuk setiap label
        print("\nMembuat WordCloud by Cluster")
        for label in labels:
            self._visualize_by_cluster_for_label(label, similarity_type, n_clusters)

        # 3. Visualisasi WordCloud dari Text/Review yang sudah diproses
        print("\nMembuat WordCloud dari Text yang sudah diproses")
        self._visualize_processed_text()

        print("\n")
        print("WORDCLOUD VISUALIZATION SELESAI")

    def _visualize_by_sentiment_label(self):
        labeled_file = os.path.join(self.output_dir, "tokopedia_labeled_all.csv")

        if not os.path.exists(labeled_file):
            print(f"File tidak ditemukan: {labeled_file}")
            return

        try:
            print(f"Membuat WordCloud by Sentiment Label dari: {labeled_file}")
            self.visualizer.visualize_by_label(
                csv_path=labeled_file,
                text_column="review_text",
                label_column="sentiment_label",
                figsize=(20, 8),
            )
            print("WordCloud by Sentiment Label selesai")
        except Exception as e:
            print(f"Error: {str(e)}")

    def _visualize_by_cluster_for_label(self, label, similarity_type, n_clusters):
        try:
            # Tentukan file paths
            if label == "all":
                labeled_file = os.path.join(self.output_dir, "tokopedia_labeled_all.csv")
            else:
                labeled_file = os.path.join(self.output_dir, f"tokopedia_labeled_{label}.csv")

            clusters_file = os.path.join(
                self.output_dir,
                f"tokopedia_clusters_{label}{similarity_type}.csv",
            )

            # Validasi file existence
            if not os.path.exists(labeled_file):
                print(
                    f"Labeled file tidak ditemukan untuk label '{label}': {labeled_file}"
                )
                return

            if not os.path.exists(clusters_file):
                print(
                    f"Clusters file tidak ditemukan untuk label '{label}': {clusters_file}"
                )
                return

            print(f"Membuat WordCloud by Cluster untuk label '{label}'...")

            # Menggunakan merge method untuk combine text dari labeled + cluster assignments
            self.visualizer.visualize_by_cluster_with_merge(
                labeled_csv_path=labeled_file,
                clusters_csv_path=clusters_file,
                text_column="review_text",
                cluster_column="cluster",
                label_name=label,
                figsize=(20, 10),
                similarity_type=similarity_type,
                n_clusters=n_clusters,
            )

            print(f"WordCloud by Cluster untuk label '{label}' selesai")

        except Exception as e:
            print(f"Error untuk label '{label}': {str(e)}")

    def _visualize_processed_text(self):
        try:
            # Cek file-file preprocessing yang tersedia
            preprocessing_files = [
                ("tokopedia_lemmatized.csv", "review_text", "Lemmatized Text"),
                (
                    "tokopedia_stopword_removed.csv",
                    "review_text",
                    "Stopword Removed Text",
                ),
                ("tokopedia_deduplicated.csv", "review_text", "Deduplicated Text"),
            ]

            for filename, text_column, title_prefix in preprocessing_files:
                filepath = os.path.join(self.output_dir, filename)

                if os.path.exists(filepath):
                    print(f"Membuat WordCloud dari {filename}...")
                    output_filename = f"wordcloud_{filename.replace('.csv', '')}.png"

                    try:
                        self.visualizer.visualize_csv_wordcloud(
                            csv_path=filepath,
                            text_column=text_column,
                            title=f"WordCloud - {title_prefix}",
                            filename=output_filename,
                        )
                        print(f"WordCloud untuk {filename} selesai")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                else:
                    print(f"File tidak ditemukan: {filename}")

        except Exception as e:
            print(f"Error dalam visualisasi processed text: {str(e)}")

    def generate_report(self):
        report = f"""Gambar tersimpan di: {self.wordcloud_output_dir}"""
        return report
