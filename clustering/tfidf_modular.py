import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class TermFrequency:
    def __init__(self, input_labeled, output_tf, column_review):
        self.input_file = input_labeled
        self.output_file = output_tf
        self.text_column = column_review
        self.vectorizer = None
        self.tf_matrix = None
        self.feature_names = None

    def run(self):
        # Baca text, hitung TF (normalized by total words per doc), dan simpan.
        if not pd.io.common.file_exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        df = pd.read_csv(self.input_file)
        df[self.text_column] = df[self.text_column].fillna("")
        text_data = df[self.text_column].tolist()

        # Gunakan CountVectorizer untuk hitung raw term frequency
        self.vectorizer = CountVectorizer()
        self.tf_matrix = self.vectorizer.fit_transform(text_data)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Konversi ke dense array
        tf_dense = self.tf_matrix.toarray()

        # Normalisasi: bagi setiap row dengan jumlah total kata dalam dokumen
        # TF(t,d) = count(t,d) / total_words(d)
        total_words_per_doc = tf_dense.sum(axis=1, keepdims=True)
        # Hindari division by zero
        total_words_per_doc = np.where(total_words_per_doc == 0, 1, total_words_per_doc)
        tf_normalized = tf_dense / total_words_per_doc
        # Konversi ke DataFrame dan simpan
        tf_df = pd.DataFrame(tf_normalized, columns=self.feature_names)
        tf_df.to_csv(self.output_file, index=False)
        print(f"Term Frequency telah disimpan ke {self.output_file}")


class InverseDocumentFrequency:
    # Hitung IDF dari text dan simpan ke CSV.

    def __init__(self, input_labeled, output_idf, column_review):
        self.input_file = input_labeled
        self.output_file = output_idf
        self.text_column = column_review
        self.vectorizer = None
        self.idf_values = None
        self.feature_names = None

    def run(self):
        # Baca text, hitung IDF, dan simpan.
        if not pd.io.common.file_exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        df = pd.read_csv(self.input_file)
        df[self.text_column] = df[self.text_column].fillna("")
        text_data = df[self.text_column].tolist()
        # TfidfVectorizer untuk hitung IDF
        self.vectorizer = SklearnTfidfVectorizer(
            use_idf=True, sublinear_tf=False, norm=None
        )
        self.vectorizer.fit(text_data)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.idf_values = self.vectorizer.idf_
        # Simpan IDF sebagai DataFrame
        idf_df = pd.DataFrame(
            {col: [val] for col, val in zip(self.feature_names, self.idf_values)}
        )
        idf_df.to_csv(self.output_file, index=False)
        print(f"Inverse Document Frequency telah disimpan ke {self.output_file}")


class TFIDFVectorizer:
    # Gabung TF × IDF atau hitung TF-IDF langsung, simpan hasilnya.

    def __init__(
        self,
        input_labeled,
        output_tfidf,
        column_review,
        use_precomputed=False,
        input_tf=None,
        input_idf=None,
    ):
        self.input_file = input_labeled
        self.output_file = output_tfidf
        self.text_column = column_review
        self.use_precomputed = use_precomputed
        self.input_tf = input_tf
        self.input_idf = input_idf

    def run(self):
        # Baca data ulasan
        df = pd.read_csv(self.input_file)

        if self.use_precomputed and self.input_tf and self.input_idf:
            # Gabung TF dan IDF yang sudah dihitung sebelumnya
            print("Menggunakan TF dan IDF yang sudah dihitung...")
            tf_df = pd.read_csv(self.input_tf)
            idf_df = pd.read_csv(self.input_idf)
            # IDF ada 1 baris, repeat untuk setiap dokumen
            idf_values = idf_df.iloc[0].values
            # TF-IDF = TF × IDF
            tfidf_matrix = tf_df.values * idf_values
            tfidf_df = pd.DataFrame(tfidf_matrix, columns=tf_df.columns)
        else:
            # Hitung TF-IDF dari scratch
            print("Menghitung TF-IDF dari scratch...")
            df[self.text_column] = df[self.text_column].fillna("")
            text_data = df[self.text_column].tolist()
            vectorizer = SklearnTfidfVectorizer(norm="l2", sublinear_tf=False)
            tfidf_matrix = vectorizer.fit_transform(text_data)
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
            )

        # Gabung dengan metadata (sentiment_label, dll)
        result_df = pd.concat([df, tfidf_df], axis=1)
        result_df.to_csv(self.output_file, index=False)
        print(f"TF-IDF vectorizer telah disimpan ke {self.output_file}")
        print(f"Total kata yang terdeteksi: {tfidf_df.shape[1]}")
