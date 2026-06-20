import numpy as np
from math import e

def train(X, y, w, b, l):
  for loop in range(1000):
    wErrors = []
    bErrors = []
    for i in range(len(X)):
      z = b + (w * X[i])
      predict = 1 / (1 + (e ** (0 - z)))
      wError = (predict - y[i]) * X[i]
      bError = (predict - y[i])
      wErrors.append(wError)
      bErrors.append(bError)
    wSlope = (sum(wErrors)) / len(X)
    bSlope = (sum(bErrors)) / len(X)
    w -= (wSlope * l)
    b -= (bSlope * l)
  return w,b
