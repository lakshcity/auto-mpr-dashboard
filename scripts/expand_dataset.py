import pandas as pd

INPUT_PATH = "data/cases_training.csv"
OUTPUT_PATH = "data/cases_training_25k.csv"
TARGET_ROWS = 25000

# Try common enterprise encodings
encodings_to_try = ["utf-8", "latin-1", "cp1252"]

df = None
for enc in encodings_to_try:
    try:
        df = pd.read_csv(INPUT_PATH, encoding=enc)
        print(f"Loaded CSV using encoding: {enc}")
        break
    except UnicodeDecodeError:
        print(f"Failed with encoding: {enc}")

if df is None:
    raise ValueError("Could not decode CSV with known encodings")

original_rows = len(df)
multiplier = (TARGET_ROWS // original_rows) + 1

df_large = pd.concat([df] * multiplier, ignore_index=True)
df_large = df_large.iloc[:TARGET_ROWS]

df_large.to_csv(OUTPUT_PATH, index=False)

print(f"Original rows: {original_rows}")
print(f"New rows: {len(df_large)}")
print(f"Saved expanded dataset to: {OUTPUT_PATH}")
