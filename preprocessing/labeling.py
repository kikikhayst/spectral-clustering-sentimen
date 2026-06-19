import pandas as pd
import re
import os


class ReviewLabeler:
    def __init__(self, input_deduplicated, output_labeled, column_deduplicated):
        self.input_file = input_deduplicated
        self.output_file = output_labeled
        self.text_column = column_deduplicated

    # Load kamus kata positif dan negatif
    def load_lexicon(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        return set(words)

    positive_words = load_lexicon("custom_dict/combined_positive_words.txt")
    negative_words = load_lexicon("custom_dict/combined_negative_words.txt")

    # Fungsi untuk membersihkan teks
    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)  # Hapus tanda baca
        text = re.sub(r"\s+", " ", text).strip()  # Hapus spasi berlebih
        return text

    # Fungsi menghitung skor sentimen
    @classmethod
    def calculate_sentiment_score(cls, text):
        text = cls.clean_text(text)
        words = text.split()
        pos_count = 0
        neg_count = 0
        for word in words:
            if word in cls.positive_words:
                pos_count += 1
            elif word in cls.negative_words:
                neg_count += 1
        total_words = len(words)
        if total_words == 0:
            return 0, 0, 0
        # Hitung persentase kata positif dan negatif
        pos_score = pos_count / total_words
        neg_score = neg_count / total_words
        # Hitung skor sentimen (positif - negatif)
        sentiment_score = pos_score - neg_score
        return sentiment_score, pos_count, neg_count

    # Fungsi untuk menentukan label berdasarkan skor sentimen
    @staticmethod
    def get_sentiment_label(score, pos_count=0, neg_count=0):
        if score >= 0:
            return "positif"
        else:
            return "negatif"

    def split_by_label(self, labeled_df, output_dir):
        """Pisahkan data berdasarkan label sentimen dan simpan ke file terpisah"""
        # Pisahkan berdasarkan label dan simpan langsung di output_dir
        labels = labeled_df["sentiment_label"].unique()
        split_files = {}
        for label in labels:
            label_df = labeled_df[labeled_df["sentiment_label"] == label]
            output_file = os.path.join(output_dir, f"tokopedia_labeled_{label}.csv")
            label_df.to_csv(output_file, index=False, encoding="utf-8")
            split_files[label] = output_file
            print(f"  - {label}: {len(label_df)} data -> {output_file}")
        return split_files

    def run(self):
        results = []
        # Load data ulasan
        df = pd.read_csv(self.input_file)
        for row in df.iterrows():
            review = row[1][self.text_column]
            score, pos_count, neg_count = self.calculate_sentiment_score(review)
            label = self.get_sentiment_label(score, pos_count, neg_count)
            results.append(
                {
                    "review_text": review,
                    "sentiment_score": score,
                    "positive_words_count": pos_count,
                    "negative_words_count": neg_count,
                    "sentiment_label": label,
                }
            )
        labeled_df = pd.DataFrame(results)
        # Simpan hasil ke file CSV
        labeled_df.to_csv(self.output_file, index=False, encoding="utf-8")

        # Hitung jumlah masing-masing sentiment label dan total data
        total_data = len(labeled_df)
        label_counts = labeled_df["sentiment_label"].value_counts().sort_index()
        print(f"Total data yang diuji: {total_data}")
        for label, count in label_counts.items():
            print(f"Jumlah data dengan label '{label}': {count}")
        print(f"Labeling telah disimpan ke {self.output_file}")

        # Pisahkan file berdasarkan label
        print("\nMemisahkan data berdasarkan label sentimen:")
        output_dir = os.path.dirname(self.output_file)
        return self.split_by_label(labeled_df, output_dir)
