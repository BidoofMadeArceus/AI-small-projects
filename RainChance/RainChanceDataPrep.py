import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("RainChance/RainChanceData.csv")
df = df.dropna(subset=["PRCP"])
df = df.dropna(subset=["TMAX"])
df = df.dropna(subset=["TMIN"])

df["Rain"] = (df["PRCP"] > 0).astype(int)

X1 = df["TMAX"]
X1 = X1/10
X2 = df["TMIN"]
X2 = X2/10
y = df["Rain"]

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1,
    X2,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
