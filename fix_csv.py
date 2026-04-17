import pandas as pd
import csv

input_path = "data/fake_job_postings.csv"
output_path = "data/fake_job_postings_clean.csv"

df = pd.read_csv(
    input_path,
    engine="python",
    on_bad_lines="skip",
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())

df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Clean file saved to: {output_path}")