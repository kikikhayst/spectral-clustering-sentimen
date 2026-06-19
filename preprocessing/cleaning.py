import pandas as pd
import re


class ReviewCleaner:
    def __init__(self, input_casefolding, output_cleaned, column_casefolding):
        self.input_file = input_casefolding
        self.output_file = output_cleaned
        self.text_column = column_casefolding

    @staticmethod
    def clean_text(text):
        if pd.isna(text) or isinstance(text, float):
            return ""
        text = str(text)  # Konversi ke string
        text = re.sub(r"\d+", "", text)  # Hapus angka
        text = re.sub(r"[^\w\s]", " ", text)  # Hapus tanda baca
        text = text.encode("ascii", "ignore").decode("ascii")  # Hapus emoticon
        text = " ".join(text.split())  # Normalisasi spasi
        return text.strip()

    def run(self):
        df = pd.read_csv(self.input_file)
        df = df.dropna(subset=[self.text_column])  # Hapus NaN

        # Hapus non-string
        df = df[df[self.text_column].apply(lambda x: isinstance(x, str))]
        df["cleaned_text"] = df[self.text_column].apply(self.clean_text)
        df_output = df[["cleaned_text"]]
        df_output.to_csv(self.output_file, index=False)
        print(f"Cleaning telah disimpan ke {self.output_file}")
