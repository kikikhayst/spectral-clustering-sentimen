import os
import pandas as pd
from collections import Counter
import json
import re
import string


class TopWordsAnalyzer:

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.top_words_dir = os.path.join(output_dir, "top_words_analysis")
        
        if not os.path.exists(self.top_words_dir):
            os.makedirs(self.top_words_dir)
        
        self.analysis_results = {}
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        stopwords = set()
        stop_dict_file = "custom_dict/combined_stop_words.txt"
        
        if os.path.exists(stop_dict_file):
            try:
                with open(stop_dict_file, 'r', encoding='utf-8') as f:
                    stopwords = set(word.strip().lower() for word in f.readlines() if word.strip())
                print(f"✓ Loaded {len(stopwords)} stopwords")
            except Exception as e:
                print(f"⚠ Gagal load stopwords: {e}")
        else:
            print(f"File stopwords tidak ditemukan: {stop_dict_file}")
        
        return stopwords

    def _preprocess_text(self, text):
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words
        words = text.split()
        
        # Filter stopwords dan empty strings
        words = [word for word in words if word and word not in self.stopwords]
        
        return words
        self.stopwords = self._load_stopwords()

    def analyze_top_words(self, similarity_type="_w_cosine", n_clusters=2):
        print("\n")
        print("MEMULAI ANALISIS TOP 10 KATA PER CLUSTER")

        labels = ["positif", "negatif", "all"]

        for label in labels:
            print(f"\n[Analisis {label.upper()}]")
            self._analyze_label(label, similarity_type, n_clusters)

        print("\n")
        print("ANALISIS TOP 10 KATA SELESAI")

    def _analyze_label(self, label, similarity_type, n_clusters):
        
        try:
            # Tentukan file paths
            if label == "all":
                labeled_file = os.path.join(self.output_dir, "tokopedia_labeled_all.csv")
            else:
                labeled_file = os.path.join(
                    self.output_dir, f"tokopedia_labeled_{label}.csv"
                )

            clusters_file = os.path.join(
                self.output_dir,
                f"tokopedia_clusters_{label}{similarity_type}.csv",
            )

            # Validasi file existence
            if not os.path.exists(labeled_file):
                print(f"File labeled tidak ditemukan: {labeled_file}")
                return

            if not os.path.exists(clusters_file):
                print(f"File clusters tidak ditemukan: {clusters_file}")
                return

            # Load data
            df_labeled = pd.read_csv(labeled_file)
            df_clusters = pd.read_csv(clusters_file)

            # Merge berdasarkan index
            df_merged = pd.concat(
                [df_labeled[["review_text"]], df_clusters[["cluster"]]], axis=1
            )

            # Extract top words untuk setiap cluster
            top_words_dict = {}
            label_result = {
                "label": label,
                "similarity_type": similarity_type,
                "n_clusters": n_clusters,
                "clusters": {}
            }

            for cluster_id in range(n_clusters):
                cluster_texts = df_merged[df_merged["cluster"] == cluster_id][
                    "review_text"
                ]
                
                if len(cluster_texts) == 0:
                    print(f"File cluster {cluster_id} tidak memiliki data")
                    continue

                # Combine all text in this cluster
                all_text = " ".join(cluster_texts.astype(str))
                
                # Preprocess text (remove punctuation, lowercase, remove stopwords)
                words = self._preprocess_text(all_text)
                
                # Count word frequencies
                word_counts = Counter(words)
                
                # Get top 10 words
                top_10 = word_counts.most_common(10)
                
                top_words_dict[cluster_id] = top_10
                
                # Store in results
                label_result["clusters"][f"cluster_{cluster_id}"] = {
                    "doc_count": len(cluster_texts),
                    "word_count": len(words),
                    "top_10_words": [
                        {"word": word, "frequency": freq} for word, freq in top_10
                    ]
                }
                
                print(f"  Cluster {cluster_id}: {len(cluster_texts)} dokumen, "
                      f"{len(words)} total kata")
                print(f"    Top 10 kata: {', '.join([word for word, _ in top_10])}")

            self.analysis_results[label] = label_result
            
            # Save hasil analisis ke file JSON
            self._save_analysis_json(label, similarity_type, label_result)
            
            # Save hasil analisis ke file CSV
            self._save_analysis_csv(label, similarity_type, top_words_dict, n_clusters)

        except Exception as e:
            print(f"Error untuk label '{label}': {str(e)}")

    def _save_analysis_json(self, label, similarity_type, result):
        output_file = os.path.join(
            self.top_words_dir,
            f"top_words_{label}{similarity_type}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Hasil JSON disimpan: {output_file}")

    def _save_analysis_csv(self, label, similarity_type, top_words_dict, n_clusters):
        # Format data untuk CSV (each cluster gets a column)
        csv_data = []
        max_words = 10
        
        for word_rank in range(max_words):
            row = {}
            for cluster_id in range(n_clusters):
                if cluster_id in top_words_dict and word_rank < len(top_words_dict[cluster_id]):
                    word, freq = top_words_dict[cluster_id][word_rank]
                    row[f"Cluster_{cluster_id}_Word"] = word
                    row[f"Cluster_{cluster_id}_Freq"] = freq
                else:
                    row[f"Cluster_{cluster_id}_Word"] = ""
                    row[f"Cluster_{cluster_id}_Freq"] = ""
            
            csv_data.append(row)
        
        df_csv = pd.DataFrame(csv_data)
        output_file = os.path.join(
            self.top_words_dir,
            f"top_words_{label}{similarity_type}.csv"
        )
        
        df_csv.to_csv(output_file, index_label="Rank", encoding='utf-8')
        print(f"Hasil CSV disimpan: {output_file}")

    def generate_report(self):
        report = []
        report.append("\n")
        report.append("LAPORAN TOP 10 KATA PER CLUSTER")


        for label, result in self.analysis_results.items():
            report.append(f"\n")
            report.append(f"LABEL: {result['label'].upper()}")
            report.append(f"Similarity Type: {result['similarity_type']}")
            report.append(f"Number of Clusters: {result['n_clusters']}")
            report.append(f"{'─' * 80}")

            for cluster_key, cluster_data in result["clusters"].items():
                cluster_num = cluster_key.split("_")[-1]
                report.append(f"\n  {cluster_key.upper()}")
                report.append(f"Jumlah Dokumen: {cluster_data['doc_count']}")
                report.append(f"Total Kata: {cluster_data['word_count']}")
                report.append(f"Top 10 Kata:")

                for rank, word_data in enumerate(cluster_data["top_10_words"], 1):
                    word = word_data["word"]
                    freq = word_data["frequency"]
                    percentage = (freq / cluster_data['word_count']) * 100
                    report.append(
                        f"    {rank:2d}. {word:30s} (Frekuensi: {freq:5d}, "
                        f"Persentase: {percentage:6.2f}%)"
                    )

        report.append("\n")
        
        return "\n".join(report)

    def print_summary(self):
        report = self.generate_report()
        print(report)
