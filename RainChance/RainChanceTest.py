from math import e
import numpy as np
from RainChanceDataPrep import X1_test
from RainChanceDataPrep import X2_test
from RainChanceDataPrep import y_test
from RainChanceTrain import weight_1
from RainChanceTrain import weight_2
from RainChanceTrain import bias

errors = []

for i in range(len(X1_test)):
  z = (X1_test[i] * weight_1) + (X2_test[i] * weight_2)) + bias
  predict = 1 / (1 + e ** (-z))
  error = (y_test[i] * np.log(predict)) + ((1 - y_test[i]) * np.log(1 - predict))
  errors.append(error)

LogLoss = (-1) * (sum(errors) / len(X1_test))

print("LogLoss: " + str(LogLoss))

for i in range(10):
    z = (X1_test[i] * weight_1) + (X2_tes[i] * weight_2)) + bias
    p = 1 / (1 + e**(-z))
    print("z= " + str(z) + " " + "p= " + str(p) + " " + "y= " + str(y_test[i]))
