import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("RainChanceData.csv")
df = df.dropna(subset=["PRCP"])
df = df.dropna(subset=["TMAX"])
df = df.dropna(subset=["TMIN"])

df["Rain"] = (df["PRCP"] > 0).astype(int)

X1 = df["TMAX"]
X2 = df["TMIN"]
y = df["Rain"]
