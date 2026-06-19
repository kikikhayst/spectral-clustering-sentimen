import pandas as pd
import re


class ReviewStopWordRemover:
    def __init__(
        self,
        input_lemmatizer,
        output_stopword_remover,
        column_lemmatized,
        stop_dict_file,
    ):
        self.input_file = input_lemmatizer
        self.output_file = output_stopword_remover
        self.text_column = column_lemmatized
        self.stop_dict = stop_dict_file

    @staticmethod
    def load_custom_stopwords(filepath):
        # Load custom stopwords dari file
        with open(filepath, "r", encoding="utf-8-sig") as file:
            stopwords = set()
            for line in file:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
                    stopwords.add(word + ".")
                    stopwords.add(word + ",")
                    stopwords.add(word + "?")
                    stopwords.add(word + "!")
            return stopwords

    def create_stopword_remover(self, custom_stopwords_file):
        custom_stopwords = ReviewStopWordRemover.load_custom_stopwords(
            custom_stopwords_file
        )
        stopword_pattern = (
            r"\b(?:" + "|".join(map(re.escape, custom_stopwords)) + r")\b"
        )
        return re.compile(stopword_pattern, flags=re.IGNORECASE)

    @staticmethod
    def stopword_removal(text, pattern):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"(\w)([.,!?])", r"\1 \2", text)
        # Remove stopwords
        text = pattern.sub("", text)
        # Clean up results
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def verify_stopword_removal(text_series, custom_stopwords_file):
        custom_stopwords = ReviewStopWordRemover.load_custom_stopwords(
            custom_stopwords_file
        )
        remaining_stopwords = set()

        for text in text_series:
            words = re.findall(r"\b\w+\b", str(text).lower())
            for word in words:
                if word in custom_stopwords:
                    remaining_stopwords.add(word)

        print("Verification Results:")
        print(f"Total unique stopwords in dictionary: {len(custom_stopwords)}")
        print(f"Stopwords that remain: {len(remaining_stopwords)}")
        print("Sample remaining stopwords:", list(remaining_stopwords)[:20])

    def run(self):
        # Proses file CSV
        df = pd.read_csv(self.input_file)
        stopword_pattern = self.create_stopword_remover(self.stop_dict)
        df["stopword_text"] = df[self.text_column].apply(
            lambda x: self.stopword_removal(x, stopword_pattern)
        )
        df.drop(self.text_column, axis=1, inplace=True)
        # Simpan hasil ke file CSV
        df.to_csv(self.output_file, index=False)
        print(f"Stopword removal telah disimpan ke {self.output_file}")
