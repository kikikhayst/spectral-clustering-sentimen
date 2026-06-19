import pandas as pd


class ReviewDeduplicator:
    def __init__(
        self, input_stopword_removed, output_deduplicated, column_stopword_removed
    ):
        self.input_file = input_stopword_removed
        self.output_file = output_deduplicated
        self.text_column = column_stopword_removed

    def run(self):
        df = pd.read_csv(self.input_file)
        initial_count = len(df)
        print(f"Jumlah data awal: {initial_count}")

        # Handling duplicate ulasan berdasarkan exact string matching (case-sensitive)
        # Hanya menghapus jika ulasan PERSIS sama (kata dan urutan sama)
        # Contoh: "bagus sekali" = "bagus sekali" (dihapus)
        # Contoh: "enak murah mantab" ≠ "murah enak mantab" (TIDAK dihapus)
        df_deduplicated = df.drop_duplicates(subset=[self.text_column], keep="first")
        final_count = len(df_deduplicated)
        print(f"Jumlah data setelah deduplicate: {final_count}")
        print(f"Jumlah duplikat yang dihapus: {initial_count - final_count}")

        df_deduplicated = df_deduplicated.rename(
            columns={self.text_column: "deduplicated_text"}
        )
        df_deduplicated.to_csv(self.output_file, index=False)
        print(f"Deduplicate telah disimpan ke {self.output_file}")
