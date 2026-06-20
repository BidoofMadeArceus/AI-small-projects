import pandas as pd

df = pd.read_csv("RainChanceData.csv")
df = df.dropna(subset=["PRCP"])

df["Rain"] = (df["PRCP"] > 0).astype(int)
