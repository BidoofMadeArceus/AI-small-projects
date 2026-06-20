from math import e
import pandas as pd

weight = 0
bias = 0

for loop in range(1000):
  wErrors = []
  bErrors = []
  for i in range(len(X)):
    z = bias + (weight * X[i])
    predict = 1 / (1 + (e ** (0 - z)))
    wError = (predict - y[i]) * X[i]
    bError = (predict - y[i])
    wErrors.append(wError)
    bErrors.append(bError)
  wSlope = (sum(wErrors)) / len(X)
  bSlope = (sum(bErrors)) / len(X)
  weight -= (wSlope * l)
  bias -= (bSlope * l)

print("Weight: " + str(weight))
print("Bias: " + str(bias))
