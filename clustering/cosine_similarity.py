import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SpectralSimilarity:
    def __init__(self, input_tfidf_vectorizer, output_similarity):
        self.input_file = input_tfidf_vectorizer
        self.output_file = output_similarity

    def run(self):
        # ambil kolom numerik TF-IDF
        df = pd.read_csv(self.input_file)

        # Kolom metadata yang dikecualikan
        metadata_columns = [
            "review_text",
            "sentiment_score",
            "positive_words_count",
            "negative_words_count",
            "sentiment_label",
            "Rating",
            "Product URL",
        ]

        # Ambil kolom TF-IDF saja
        tfidf_columns = [col for col in df.columns if col not in metadata_columns]
        tfidf_matrix = df[tfidf_columns].values

        if tfidf_matrix.shape[1] == 0:
            print("Warning: No TF-IDF features found!")
            return

        # Hitung matriks similaritas
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Simpan hasil ke file CSV
        similarity_df = pd.DataFrame(similarity_matrix)
        similarity_df.to_csv(self.output_file, index=False)
        print(f"Cosine similarity telah disimpan ke {self.output_file}")
