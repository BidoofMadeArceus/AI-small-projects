from math import e
from RainChanceDataPrep import X1_train
from RainChanceDataPrep import X2_train
from RainChanceDataPrep import X3_train
from RainChanceDataPrep import X4_train
from RainChanceDataPrep import X5_train
from RainChanceDataPrep import y_train

weight_1 = 0
weight_2 = 0
weight_3 = 0
weight_4 = 0
weight_5 = 0
bias = 0
learn_rate = 0.01
print("Weights: " + str(weight_1) + ", " + str(weight_2) + ", " + str(weight_4) + ", " + str(weight_5))
print("Bias: " + str(bias))
print(X1_train.min(), X1_train.max())
print(X2_train.min(), X2_train.max())
print(X3_train.min(), X3_train.max())
print(X4_train.min(), X4_train.max())
print(X5_train.min(), X5_train.max())

print("training...")

for i in range(1000):
  W1_errors = []
  W2_errors = []
  W3_errors = []
  W4_errors = []
  W5_errors = []
  b_errors = []
  for i in range(len(X1_train)):
    z = bias + (weight_1 * X1_train.iloc[i]) + (weight_2 * X2_train.iloc[i]) + (weight_3 * X3_train.iloc[i]) + (weight_4 * X4_train.iloc[i]) + (weight_5 * X5_train.iloc[i])
    predict = 1 / (1 + e**(0-z))
    w1_error = (predict - y_train.iloc[i]) * X1_train.iloc[i]
    w2_error = (predict - y_train.iloc[i]) * X2_train.iloc[i]
    w3_error = (predict - y_train.iloc[i]) * X3_train.iloc[i]
    w4_error = (predict - y_train.iloc[i]) * X4_train.iloc[i]
    w5_error = (predict - y_train.iloc[i]) * X5_train.iloc[i]
    b_error = predict - y_train.iloc[i]
    W1_errors.append(w1_error)
    W2_errors.append(w2_error)
    W3_errors.append(w3_error)
    W4_errors.append(w4_error)
    W5_errors.append(w5_error)
    b_errors.append(b_error)
  W1slope = sum(W1_errors) / len(X1_train)
  W2slope = sum(W2_errors) / len(X1_train)
  W3slope = sum(W3_errors) / len(X1_train)
  W4slope = sum(W4_errors) / len(X1_train)
  W5slope = sum(W5_errors) / len(X1_train)
  Bslope = sum(b_errors) / len(X1_train)
  weight_1 -= (learn_rate * W1slope)
  weight_2 -= (learn_rate * W2slope)
  weight_3 -= (learn_rate * W3slope)
  weight_4 -= (learn_rate * W4slope)
  weight_5 -= (learn_rate * W5slope)
  bias -= (learn_rate * Bslope)
  print("Weights: " + str(weight_1) + ", " + str(weight_2) + ", " + str(weight_3) + ", " + str(weight_4) + ", " + str(weight_5))
  print("Bias: " + str(bias))

print("training complete")

print("Weights: " + str(weight_1) + ", " + str(weight_2) + ", " + str(weight_3) + ", " + str(weight_4) + ", " + str(weight_5))
print("Bias: " + str(bias))
