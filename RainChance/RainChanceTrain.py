from math import e
from RainChanceDataPrep import X1_train
from RainChanceDataPrep import X2_train
from RainChanceDataPrep import y_train

weight_1 = 0
weight_2 = 0
bias = 0
learn_rate = 0.001
print("Weights: " + str(weight_1) + ", " + str(weight_2))
print("Bias: " + str(bias))
print(X1_train.min(), X1_train.max())
print(X2_train.min(), X2_train.max())

print("training...")

for i in range(1000):
  W1_errors = []
  W2_errors = []
  b_errors = []
  for i in range(len(X1_train)):
    z = bias + (weight_1 * X1_train.iloc[i]) + (weight_2 * X2_train.iloc[i]))
    predict = 1 / (1 + e**(0-z))
    w1_error = (predict - y_train.iloc[i]) * X1_train.iloc[i]
    w2_error = (predict - y_train.iloc[i]) * X2_train.iloc[i]
    b_error = predict - y_train.iloc[i]
    W1_errors.append(w1_error)
    W2_errors.append(w2_error)
    b_errors.append(b_error)
  W1slope = sum(W1_errors) / len(X1_train)
  W2slope = sum(W2_errors) / len(X1_train)
  Bslope = sum(b_errors) / len(X1_train)
  weight_1 -= (learn_rate * W1slope)
  weight_2 -= (learn_rate * W2slope)
  bias -= (learn_rate * Bslope)

print("training complete")

print("Weights: " + str(weight_1) + ", " + str(weight_2))
print("Bias: " + str(bias))
