import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("RainChance/RainChanceData.csv")
df = df.dropna(subset=["PRCP"])
df = df.dropna(subset=["TMAX"])
df = df.dropna(subset=["TMIN"])
df = df.dropna(subset=["TAVG"])

df["Rain"] = (df["PRCP"] > 0).astype(int)

X11 = df["TMAX"]
X11 = X11/10
X12 = df["TMIN"]
X12 = X12/10
X2 = df["TAVG"]
X2 = X2/10
X1 = X11 - X12
X1 = round(X1, 1)
X2 = round(X2, 1)
Scale1 = X1
Scale2 = X2
X1 = (X1 - Scale1.min())/(Scale1.max() - Scale1.min())
X2 = (X2 - Scale2.min())/(Scale2.max() - Scale2.min())
y = df["Rain"]

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1,
    X2,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
