import pandas as pd
import re
import json


class Normalizer:
    def __init__(
        self, input_cleaned, output_normalized, column_cleaned, slang_dict_file
    ):
        self.input_file = input_cleaned
        self.output_file = output_normalized
        self.text_column = column_cleaned
        self.slang_dict = self.load_slang_dict(slang_dict_file)

    @staticmethod
    def load_slang_dict(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading slang dict: {e}")
            return {}

    def normalize_text(self, text):
        # Normalisasi teks
        if pd.isna(text) or not isinstance(text, str):
            return ""
        normalized_words = []
        for word in text.split():
            # Pengulangan karakter
            word = re.sub(r"(.)\1{2,}", r"\1", word.lower())
            # Ganti kata slang dari kamus
            word = self.slang_dict.get(word, word)
            normalized_words.append(word)
        return " ".join(normalized_words)

    def run(self):
        df = pd.read_csv(self.input_file).dropna(subset=[self.text_column])
        df["normalized_text"] = df[self.text_column].apply(self.normalize_text)
        # Simpan hasil ke file CSV
        df = df.drop(columns=[self.text_column])
        df.to_csv(self.output_file, index=False)
        print(f"Normalization telah disimpan ke {self.output_file}")
