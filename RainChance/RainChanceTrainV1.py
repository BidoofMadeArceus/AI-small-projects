from math import e
from RainChanceDataPrep import X1_train
from RainChanceDataPrep import X2_train
from RainChanceDataPrep import y_train

def train(X1_train,X2_train,y_train,weight1,weigh2,bias,learn_rate):
  print("training..."
  for i in range(10000):
  W1_errors = []
  W2_errors = []
  b_errors = []
  for i in range(len(X1_train):
    z = b + (weight_1 * X1_test[i]) + (weight_2 * X2_test[i])
    predict = 1 / (1 + e**(0-z)
    w1_error = (predict - y[i]) * X1_train[i]
    w2_error = (predict - y[i]) * X2_train[i]
    b_error = predict - y_train[i]
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
return weight1,weight2,bias

weight1 = 0
weight2 = 0
bias = 0
learn_rate = 0
