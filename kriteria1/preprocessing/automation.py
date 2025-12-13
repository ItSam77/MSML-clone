import pandas as pd

df = pd.read_csv("data.csv")

quality_map = {
    "low" : "0",
    "medium" : "1",
    "high" : "2"
}
df["quality_label"] = df["quality_label"].map(quality_map)
df["quality_label"] = df["quality_label"].astype(int)

### Harusnya disini preprocessing more deeper, cuman ini contoh aja

df.to_csv("preprocess.csv")