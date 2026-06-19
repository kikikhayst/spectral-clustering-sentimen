import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
import re


class ReviewLemmatizer:
    def __init__(
        self,
        input_normalized,
        output_lemmatized,
        column_normalized,
        root_dict_file,
    ):
        self.input_file = input_normalized
        self.output_file = output_lemmatized
        self.text_column = column_normalized
        self.root_dict = root_dict_file

    def load_root_words(self, file_path):
        # Load daftar kata dasar dari file
        with open(file_path, "r", encoding="utf-8") as file:
            root_words = [word.strip() for word in file.readlines()]
        return root_words

    def create_enhanced_dictionary(self, custom_words):
        # Membuat dictionary gabungan antara kamus Sastrawi dan kamus custom
        dictionary = ArrayDictionary(custom_words)
        for word in custom_words:
            dictionary.add(word)
        return dictionary

    def enhanced_stemmer(self, dictionary):
        # Membuat stemmer dengan dictionary yang diperluas
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stemmer.dictionary = dictionary
        return stemmer

    def clean_and_validate_text(self, text, stemmer, dictionary):
        # Membersihkan teks dan memvalidasi kata berdasarkan kamus
        if pd.isna(text):
            return text

        # Daftar awalan/akhiran yang harus dihapus jika berdiri sendiri
        prefixes = r"\b(di|ke|se|ter|me|mem|men|meng|meny|pen|peng|peny|ber|bel|per)\b"
        suffixes = r"\b(kan|an|i|nya|kah|lah|pun|tah|wan|wati)\b"

        # Pembersihan dasar
        text = re.sub(r"[^\w\s]", " ", text)  # Hapus tanda baca
        text = re.sub(r"\s+", " ", text).strip()  # Normalisasi spasi

        # Hapus komponen tidak perlu
        text = re.sub(prefixes, "", text)
        text = re.sub(suffixes, "", text)
        text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Huruf tunggal
        text = re.sub(r"\b\d+\b", "", text)  # Angka

        # Filter kata berdasarkan kamus
        words = text.split()
        valid_words = []

        for word in words:
            # Cek apakah kata ada di kamus atau bentuk dasarnya ada
            stemmed_word = stemmer.stem(word)
            if dictionary.contains(word) or dictionary.contains(stemmed_word):
                valid_words.append(
                    stemmed_word if dictionary.contains(stemmed_word) else word
                )

        return " ".join(valid_words)

    def run(self):
        # Load kamus tambahan
        custom_words = self.load_root_words(self.root_dict)
        # Buat dictionary gabungan
        dictionary = self.create_enhanced_dictionary(custom_words)
        stemmer = self.enhanced_stemmer(dictionary)
        # Proses file CSV
        df = pd.read_csv(self.input_file)
        df["lemmatized_text"] = df[self.text_column].apply(
            lambda x: self.clean_and_validate_text(x, stemmer, dictionary)
        )
        df = df[df["lemmatized_text"].str.len() > 0]
        df.drop(columns=[self.text_column], inplace=True, errors="ignore")

        # Simpan hasil ke file CSV
        df.to_csv(self.output_file, index=False, encoding="utf-8")
        print(f"Lemmatization telah disimpan ke {self.output_file}")
