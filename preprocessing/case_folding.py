import pandas as pd


class CaseFolding:
    def __init__(self, input_review: str, output_casefolding: str, column_review: str):
        self.input_file = input_review
        self.output_file = output_casefolding
        self.text_column = column_review

    def run(self):
        df = pd.read_csv(self.input_file, usecols=[self.text_column])
        # melakukan case folding
        df_output = pd.DataFrame({"case_folded": df[self.text_column].str.lower()})
        # menyimpan hasil
        df_output.to_csv(self.output_file, index=False)
        print(f"Case folding telah disimpan ke {self.output_file}")
