import bz2
import pandas as pd

test_file = "data/test.ft.txt.bz2"

test_data = []

with bz2.open(test_file, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        label, text = line.strip().split(" ", 1)

        test_data.append({
            "label": label,
            "text": text
        })

        if i == 19999:
            break

test_df = pd.DataFrame(test_data)

print("Test dataset shape:", test_df.shape)
print(test_df.head())
print(test_df["label"].value_counts())
# Load test data
test_data = []

with bz2.open("data/test.ft.txt.bz2", "rt", encoding="utf-8") as f:
    for line in f:
        label, text = line.strip().split(" ", 1)

        test_data.append({
            "label": label,
            "text": text
        })

test_df = pd.DataFrame(test_data)

print("Test dataset shape:", test_df.shape)
print(test_df.head())