from RainChanceTrain import weight_1
from RainChanceTrain import weight_2
from RainChanceTrain import bias
from RainChanceDataPrep import Scale1
from RainChanceDataPrep import Scale2
from math import e

TempRange = int(input("temperature range: "))
AvgTemp = int(input("average temperature: "))

TempRange = (TempRange - Scale1.min())/(Scale1.max() - Scale1.min())
TempRange = (AvgTemp - Scale2.min())/(Scale2.max() - Scale2.min())

z = bias + (weight_1 * TempRange) + (weight_2 * AvgTemp)
predict = 1 / (1 + e ** (0-z))

print(predict)
