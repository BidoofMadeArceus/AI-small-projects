import pandas as pd
from math import e
import numpy as np
from RainChanceDataPrep import X1_test
from RainChanceDataPrep import X2_test
from RainChanceDataPrep import y_test

errors = []

for i in range(len(X1_test)):
  z = (X1_test.iloc[i] * weight_1) + (X2_test.iloc[i] * weight_2) + bias
  predict = 1 / (1 + e ** (-z))
  error = (y_test[i].iloc * np.log10(predict)) + ((1 - y_test[i].iloc) * np.log10(1 - predict))
  errors.append(error)

LogLoss = (-1) * (sum(errors) / len(X1_test))

print("LogLoss: " + str(LogLoss))
